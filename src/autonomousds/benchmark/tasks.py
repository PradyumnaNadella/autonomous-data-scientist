"""Benchmark tasks: download, split into agent-visible data + hidden labels, score."""
from __future__ import annotations

import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from sklearn import metrics
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[3]
TASKS_DIR = ROOT / "benchmark" / "tasks"
DATA_DIR = ROOT / "data"

# metric name -> (function, higher_is_better). ROC AUC is not supported yet:
# it needs probability predictions, and agents currently submit labels.
METRICS = {
    "accuracy": (metrics.accuracy_score, True),
    "rmse": (metrics.root_mean_squared_error, False),
    "mae": (metrics.mean_absolute_error, False),
}


@dataclass
class Task:
    """One benchmark task, loaded from benchmark/tasks/<name>.yaml."""
    name: str
    competition: str
    train_file: str
    target: str
    task_type: str            # "classification" or "regression"
    metric: str
    id_column: str | None = None
    holdout_fraction: float = 0.2
    seed: int = 42

    def __post_init__(self):
        if self.task_type not in ("classification", "regression"):
            raise ValueError(f"task_type must be classification or regression, "
                             f"got '{self.task_type}'")
        if self.metric not in METRICS:
            raise ValueError(f"Unsupported metric '{self.metric}'. "
                             f"Supported: {sorted(METRICS)}")
        self.auto_id = self.id_column is None      # dataset has no ID column
        if self.auto_id:
            self.id_column = "row_id"

    @property
    def dir(self) -> Path:
        return DATA_DIR / self.name

    @property
    def agent_dir(self) -> Path:      # what agents are allowed to see
        return self.dir / "agent"

    @property
    def hidden_dir(self) -> Path:     # answers, never shown to agents
        return self.dir / "hidden"

    @property
    def higher_is_better(self) -> bool:
        return METRICS[self.metric][1]


def load_task(name: str) -> Task:
    cfg = yaml.safe_load((TASKS_DIR / f"{name}.yaml").read_text())
    return Task(**cfg)


def find_train_file(task: Task) -> Path:
    """Locate the training CSV anywhere inside data/<task>/raw.

    Args:
        task: The benchmark task.

    Returns:
        Path to the training file.

    Raises:
        FileNotFoundError: If the file is not in the download.
    """
    matches = sorted((task.dir / "raw").rglob(task.train_file))
    if not matches:
        raise FileNotFoundError(f"'{task.train_file}' not found in {task.dir / 'raw'}")
    return matches[0]


def download(task: Task) -> None:
    """Download and unzip the competition data, unless it is already there.

    Args:
        task: The benchmark task.
    """
    raw = task.dir / "raw"
    if raw.exists() and any(raw.rglob(task.train_file)):
        return
    raw.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "competitions", "download",
                    "-c", task.competition, "-p", str(raw)], check=True)
    for z in raw.rglob("*.zip"):
        with zipfile.ZipFile(z) as f:
            f.extractall(z.parent)
        z.unlink()


def prepare(task: Task) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training file into agent-visible data and a hidden test set.

    Args:
        task: The benchmark task.

    Returns:
        The agent training rows and the hidden test rows (with labels).
    """
    download(task)
    df = pd.read_csv(find_train_file(task))
    if task.target not in df.columns:
        raise ValueError(f"Target '{task.target}' not in {task.train_file}")
    if task.auto_id:
        df.insert(0, task.id_column, range(len(df)))
    elif task.id_column not in df.columns:
        raise ValueError(f"ID column '{task.id_column}' not in {task.train_file}; "
                         "remove id_column from the YAML to create one automatically")
    stratify = df[task.target] if task.task_type == "classification" else None
    train, hold = train_test_split(df, test_size=task.holdout_fraction,
                                   random_state=task.seed, stratify=stratify)

    task.agent_dir.mkdir(parents=True, exist_ok=True)
    task.hidden_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(task.agent_dir / "train.csv", index=False)
    hold.drop(columns=[task.target]).to_csv(task.agent_dir / "test.csv", index=False)
    hold[[task.id_column, task.target]].to_csv(task.hidden_dir / "test_labels.csv",
                                               index=False)
    return train, hold


def score(task: Task, predictions: pd.DataFrame) -> float:
    """predictions: DataFrame with the id column and the target column."""
    labels = pd.read_csv(task.hidden_dir / "test_labels.csv")
    merged = labels.merge(predictions, on=task.id_column, how="left",
                          suffixes=("_true", "_pred"))
    if merged[f"{task.target}_pred"].isna().any():
        raise ValueError("Predictions are missing for some ids")
    fn, _ = METRICS[task.metric]
    return float(fn(merged[f"{task.target}_true"], merged[f"{task.target}_pred"]))
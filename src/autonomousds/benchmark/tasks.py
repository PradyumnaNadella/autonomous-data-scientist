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

# metric name -> (function, higher_is_better)
METRICS = {
    "accuracy": (metrics.accuracy_score, True),
    "roc_auc": (metrics.roc_auc_score, True),
    "rmse": (metrics.root_mean_squared_error, False),
}


@dataclass
class Task:
    name: str
    competition: str
    train_file: str
    target: str
    id_column: str
    task_type: str            # "classification" or "regression"
    metric: str
    holdout_fraction: float = 0.2
    seed: int = 42

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


def download(task: Task) -> None:
    raw = task.dir / "raw"
    if (raw / task.train_file).exists():
        return                        # already downloaded
    raw.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "competitions", "download",
                    "-c", task.competition, "-p", str(raw)], check=True)
    for z in raw.glob("*.zip"):
        with zipfile.ZipFile(z) as f:
            f.extractall(raw)
        z.unlink()


def prepare(task: Task) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split Kaggle's train.csv into agent data and a hidden test set."""
    download(task)
    df = pd.read_csv(task.dir / "raw" / task.train_file)
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
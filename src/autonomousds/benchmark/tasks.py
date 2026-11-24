"""Benchmark tasks: download, split into agent-visible data + hidden labels, score."""
from __future__ import annotations

import re
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

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
METRIC_TASK_TYPE = {"accuracy": "classification", "rmse": "regression", "mae": "regression"}
SOURCES = ("kaggle", "csv")
NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024


@dataclass
class Task:
    """One benchmark task, loaded from benchmark/tasks/<name>.yaml.

    Attributes:
        name: Task name; lowercase letters, digits, "-" and "_" (max 64 chars).
        target: Column to predict.
        task_type: "classification" or "regression".
        metric: Metric name, one of METRICS.
        source: "kaggle" (needs competition) or "csv" (needs url or path).
        competition: Kaggle competition slug.
        url: http(s) URL of a CSV file.
        path: Local path of a CSV file.
        train_file: File name of the training CSV inside data/<name>/raw.
        id_column: Row ID column; None creates "row_id" automatically.
        holdout_fraction: Share of rows kept as the hidden test set.
        seed: Random seed for the split.
    """
    name: str
    target: str
    task_type: str
    metric: str
    source: str = "kaggle"
    competition: str | None = None
    url: str | None = None
    path: str | None = None
    train_file: str = "train.csv"
    id_column: str | None = None
    holdout_fraction: float = 0.2
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate the fields and set up the automatic ID column.

        Raises:
            ValueError: If any field is invalid or fields contradict each other.
        """
        if not NAME_PATTERN.fullmatch(self.name):
            raise ValueError(f"Invalid task name '{self.name}': use lowercase letters, "
                             "digits, '-' and '_' (max 64 characters)")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got '{self.source}'")
        if self.source == "kaggle" and not self.competition:
            raise ValueError("source 'kaggle' needs 'competition'")
        if self.source == "csv" and bool(self.url) == bool(self.path):
            raise ValueError("source 'csv' needs exactly one of 'url' or 'path'")
        if self.url and urlparse(self.url).scheme not in ("http", "https"):
            raise ValueError(f"url must start with http:// or https://, got '{self.url}'")
        if Path(self.train_file).name != self.train_file:
            raise ValueError(f"train_file must be a file name, not a path: '{self.train_file}'")
        if self.task_type not in ("classification", "regression"):
            raise ValueError(f"task_type must be classification or regression, "
                             f"got '{self.task_type}'")
        if self.metric not in METRICS:
            raise ValueError(f"Unsupported metric '{self.metric}'. "
                             f"Supported: {sorted(METRICS)}")
        if METRIC_TASK_TYPE[self.metric] != self.task_type:
            raise ValueError(f"Metric '{self.metric}' is for "
                             f"{METRIC_TASK_TYPE[self.metric]} tasks, not {self.task_type}")
        if not 0 < self.holdout_fraction < 1:
            raise ValueError("holdout_fraction must be between 0 and 1")
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
    """Load a task from benchmark/tasks/<name>.yaml.

    Args:
        name: Task name.

    Returns:
        The validated task.

    Raises:
        FileNotFoundError: If no YAML exists for this name.
    """
    path = TASKS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown task '{name}': {path} does not exist")
    return Task(**yaml.safe_load(path.read_text()))


def save_task(task: Task, overwrite: bool = False) -> Path:
    """Write a task to benchmark/tasks/<name>.yaml.

    Empty fields are left out, and so is id_column when it was created automatically.

    Args:
        task: The task to save.
        overwrite: Replace an existing YAML with the same name.

    Returns:
        Path of the written YAML file.

    Raises:
        FileExistsError: If the YAML exists and overwrite is False.
    """
    path = TASKS_DIR / f"{task.name}.yaml"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists")
    cfg = {k: v for k, v in asdict(task).items() if v is not None}
    if task.auto_id:
        cfg.pop("id_column", None)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


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


def _download_kaggle(competition: str, raw: Path) -> None:
    """Download and unzip a Kaggle competition's files.

    Args:
        competition: Kaggle competition slug.
        raw: Folder to download into.
    """
    subprocess.run(["kaggle", "competitions", "download",
                    "-c", competition, "-p", str(raw)], check=True)
    for z in raw.rglob("*.zip"):
        with zipfile.ZipFile(z) as f:
            f.extractall(z.parent)
        z.unlink()


def _download_url(url: str, dest: Path, max_bytes: int = MAX_DOWNLOAD_BYTES) -> None:
    """Download a file over http(s), refusing files larger than max_bytes.

    Writes to "<dest>.part" first and renames at the end, so an interrupted
    download never leaves a half-written file that looks complete.

    Args:
        url: http(s) URL to download.
        dest: Where to save the file.
        max_bytes: Size limit in bytes.

    Raises:
        ValueError: If the URL is not http(s) or the file is too large.
        urllib.error.URLError: If the download fails (bad URL, 404, no network).
    """
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"Only http(s) URLs are allowed, got '{url}'")
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "autonomousds"})
    try:
        with urllib.request.urlopen(request, timeout=60) as resp, part.open("wb") as out:
            size = 0
            while chunk := resp.read(1 << 20):
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"Download is larger than {max_bytes // 2**20} MB")
                out.write(chunk)
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)


def download(task: Task) -> None:
    """Get the raw data into data/<task>/raw, unless it is already there.

    Args:
        task: The benchmark task.

    Raises:
        FileNotFoundError: If a local CSV path does not exist.
    """
    raw = task.dir / "raw"
    if raw.exists() and any(raw.rglob(task.train_file)):
        return
    raw.mkdir(parents=True, exist_ok=True)
    if task.source == "kaggle":
        _download_kaggle(task.competition, raw)
    elif task.path:
        src = Path(task.path).expanduser()
        if not src.is_file():
            raise FileNotFoundError(f"CSV file not found: {src}")
        shutil.copyfile(src, raw / task.train_file)
    else:
        _download_url(task.url, raw / task.train_file)


def prepare(task: Task) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training file into agent-visible data and a hidden test set.

    Args:
        task: The benchmark task.

    Returns:
        The agent training rows and the hidden test rows (with labels).

    Raises:
        ValueError: If the target or ID column is missing or unusable.
    """
    download(task)
    df = pd.read_csv(find_train_file(task))
    if task.target not in df.columns:
        raise ValueError(f"Target '{task.target}' not in {task.train_file}. "
                         f"Columns: {list(df.columns)}")
    missing = df[task.target].isna()
    if missing.any():
        print(f"Dropping {int(missing.sum())} rows with no '{task.target}' value")
        df = df[~missing].reset_index(drop=True)
    if task.task_type == "regression" and not pd.api.types.is_numeric_dtype(df[task.target]):
        raise ValueError(f"Regression target '{task.target}' is not numeric")
    if task.task_type == "classification":
        counts = df[task.target].value_counts()
        rare = counts[counts < 2]
        if not rare.empty:
            raise ValueError(f"Classes with fewer than 2 rows (can't split): {list(rare.index)}")

    if task.auto_id:
        if task.id_column in df.columns:
            raise ValueError(f"Dataset already has a '{task.id_column}' column; "
                             "set id_column explicitly")
        df.insert(0, task.id_column, range(len(df)))
    elif task.id_column not in df.columns:
        raise ValueError(f"ID column '{task.id_column}' not in {task.train_file}; "
                         "remove id_column from the YAML to create one automatically")
    ids = df[task.id_column]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError(f"ID column '{task.id_column}' must be unique and non-empty")

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

def create_csv_task(name: str, target: str, task_type: str, metric: str,
                    url: str | None = None, path: str | None = None,
                    id_column: str | None = None,
                    force: bool = False) -> tuple[Task, pd.DataFrame, pd.DataFrame]:
    """Create a CSV task, prepare its data, then save its YAML.

    The YAML is saved only after prepare() succeeds, so a failed attempt leaves
    no broken task behind. A task that already has benchmark results is never
    replaced, so results always match the data they were produced on.

    Args:
        name: Task name.
        target: Column to predict.
        task_type: "classification" or "regression".
        metric: Metric name, one of METRICS.
        url: http(s) URL of the CSV (give url or path).
        path: Local path of the CSV (give url or path).
        id_column: Row ID column; None creates "row_id" automatically.
        force: Replace an existing task that has no results yet.

    Returns:
        The task, the agent training rows and the hidden test rows.

    Raises:
        FileExistsError: If the task exists (without force) or already has results.
        ValueError: If the settings or the data are invalid.
        OSError: If the CSV cannot be downloaded or read.
    """
    task = Task(name=name, target=target, task_type=task_type, metric=metric,
                source="csv", url=url,
                path=str(Path(path).expanduser().resolve()) if path else None,
                id_column=id_column)
    results_dir = ROOT / "benchmark" / "results" / task.name
    if results_dir.exists():
        raise FileExistsError(f"Task '{task.name}' already has results; choose a different name")
    exists = (TASKS_DIR / f"{task.name}.yaml").exists() or task.dir.exists()
    if exists and not force:
        raise FileExistsError(f"Task '{task.name}' already exists")

    shutil.rmtree(task.dir, ignore_errors=True)
    try:
        train, hold = prepare(task)
    except Exception:
        shutil.rmtree(task.dir, ignore_errors=True)
        raise
    save_task(task, overwrite=True)
    return task, train, hold

def score(task: Task, predictions: pd.DataFrame) -> float:
    """predictions: DataFrame with the id column and the target column."""
    labels = pd.read_csv(task.hidden_dir / "test_labels.csv")
    merged = labels.merge(predictions, on=task.id_column, how="left",
                          suffixes=("_true", "_pred"))
    if merged[f"{task.target}_pred"].isna().any():
        raise ValueError("Predictions are missing for some ids")
    fn, _ = METRICS[task.metric]
    return float(fn(merged[f"{task.target}_true"], merged[f"{task.target}_pred"]))
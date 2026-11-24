"""Read benchmark results and run artifacts from disk, safely."""
from __future__ import annotations

import json
import re
from pathlib import Path

from autonomousds.benchmark.tasks import ROOT, load_task

RESULTS_DIR = ROOT / "benchmark" / "results"
SUMMARY_DIR = ROOT / "benchmark" / "summary"
# Where run folders can live: local runs, and published copies (for deployment)
ARTIFACT_DIRS = [ROOT / "runs", ROOT / "benchmark" / "showcase"]
BASELINES = {"direct", "automl"}
VIEWABLE_FILES = {"features.py", "train_model.py", "graph.mmd", "experiments.jsonl"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class NotFound(Exception):
    """Raised when a task, run or file does not exist."""


def _safe(name: str) -> str:
    """Reject names that could escape their folder (e.g. '../.env').

    Args:
        name: A task name, run id or file name from a URL.

    Returns:
        The same name, if it is safe.

    Raises:
        NotFound: If the name contains anything but letters, digits, _ . -
    """
    if not SAFE_NAME.match(name) or name in {".", ".."}:
        raise NotFound(name)
    return name


def _read_json(path: Path):
    if not path.is_file():
        raise NotFound(str(path.name))
    return json.loads(path.read_text())


def _run_dir(task: str, run_id: str) -> Path:
    """Find a run folder in any artifact location."""
    for base in ARTIFACT_DIRS:
        path = base / _safe(task) / _safe(run_id)
        if path.is_dir():
            return path
    raise NotFound(run_id)


def list_tasks() -> list[dict]:
    """All tasks that have a summary.

    Returns:
        One dict per task: name, task_type, metric, higher_is_better.
    """
    tasks = []
    for path in sorted(SUMMARY_DIR.glob("*.json")):
        task = load_task(path.stem)
        tasks.append({"name": task.name, "task_type": task.task_type,
                      "metric": task.metric, "higher_is_better": task.higher_is_better})
    return tasks


def task_summary(task: str) -> list[dict]:
    """Per-system summary metrics for a task (from build_report.py).

    Args:
        task: Task name.

    Returns:
        The list of system summaries.
    """
    return _read_json(SUMMARY_DIR / f"{_safe(task)}.json")


def list_runs(task: str, system: str | None = None) -> list[dict]:
    """Short records of every agent run for a task, newest first.

    Args:
        task: Task name.
        system: Optional system filter, e.g. "pipeline".

    Returns:
        One dict per run with its key numbers.
    """
    runs = []
    for path in sorted((RESULTS_DIR / _safe(task)).glob("*.json")):
        if path.stem in BASELINES or (system and path.stem != system):
            continue
        for r in _read_json(path).get("runs", []):
            runs.append({"run_id": r["run_id"], "system": path.stem,
                         "max_steps": r.get("max_steps"),
                         "submitted": r.get("submitted"),
                         "holdout_score": r.get("holdout_score"),
                         "seconds": r.get("seconds"),
                         "est_cost_usd": r.get("est_cost_usd"),
                         "critic_verdicts": r.get("critic_verdicts")})
    return sorted(runs, key=lambda r: r["run_id"].rsplit("-", 2)[-2:], reverse=True)


def run_detail(task: str, run_id: str) -> dict:
    """Everything needed to replay one run: record, transcript and code steps.

    Args:
        task: Task name.
        run_id: Run id.

    Returns:
        Dict with record, transcript, steps and available files.
    """
    record = next((r for path in (RESULTS_DIR / _safe(task)).glob("*.json")
                   if path.stem not in BASELINES
                   for r in _read_json(path).get("runs", [])
                   if r["run_id"] == run_id), None)
    if record is None:
        raise NotFound(run_id)

    folder = _run_dir(task, run_id)
    transcript = (json.loads((folder / "transcript.json").read_text())
                  if (folder / "transcript.json").is_file() else [])
    steps = []
    log = folder / "sandbox_log.jsonl"
    if log.is_file():
        for line in log.read_text().splitlines():
            s = json.loads(line)
            code = folder / f"{s['step']}.py"
            s["code"] = code.read_text() if code.is_file() else ""
            steps.append(s)
    files = sorted(n for n in VIEWABLE_FILES if (folder / n).is_file())
    return {"record": record, "transcript": transcript, "steps": steps, "files": files}


def run_file(task: str, run_id: str, name: str) -> str:
    """Text of one viewable file from a run (code, graph, experiment log).

    Args:
        task: Task name.
        run_id: Run id.
        name: File name; must be in VIEWABLE_FILES.

    Returns:
        The file's text.
    """
    if name not in VIEWABLE_FILES:
        raise NotFound(name)
    path = _run_dir(task, run_id) / name
    if not path.is_file():
        raise NotFound(name)
    return path.read_text()
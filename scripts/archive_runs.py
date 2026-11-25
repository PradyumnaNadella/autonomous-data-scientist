"""Copy run artifacts between runs/ (local, not in git) and benchmark/runs/ (committed).

build_report.py recomputes metrics from run folders and the LLM call log, but
GitHub runners start empty. "export" keeps a small committed copy of every run
(code, outputs, transcripts, predictions; never datasets or models); "restore"
puts it back before the report is rebuilt.

Usage:
    python scripts/archive_runs.py export    # runs/ -> benchmark/runs/
    python scripts/archive_runs.py restore   # benchmark/runs/ -> runs/
"""
import argparse
import shutil
from pathlib import Path

from autonomousds.benchmark.tasks import ROOT

RUNS = ROOT / "runs"
ARCHIVE = ROOT / "benchmark" / "runs"
LLM_LOG = ROOT / "logs" / "llm_calls.jsonl"
ARCHIVE_LOG = ARCHIVE / "llm_calls.jsonl"
KEEP_SUFFIXES = {".py", ".txt", ".json", ".jsonl", ".mmd", ".png", ".md"}
KEEP_NAMES = {"predictions.csv"}
MAX_FILE_BYTES = 2 * 1024 * 1024


def _keep(path: Path, base: Path) -> bool:
    """Decide whether a file belongs in the archive.

    Skips API job files (_jobs/), AutoGluon's model folder (<task>/automl/),
    caches, large files and anything that is not code, text, logs or predictions.
    Baseline predictions (<task>/baselines/<system>/predictions.csv) are kept.

    Args:
        path: File to check.
        base: Folder the file is inside (runs/ or benchmark/runs/).

    Returns:
        True if the file should be archived or restored.
    """
    parts = path.relative_to(base).parts
    if parts[0] == "_jobs" or "__pycache__" in parts:
        return False
    if len(parts) > 1 and parts[1] == "automl":     # runs/<task>/automl/: models
        return False
    return (path.is_file()
            and (path.suffix in KEEP_SUFFIXES or path.name in KEEP_NAMES)
            and path.stat().st_size <= MAX_FILE_BYTES)


def _merge_lines(src: Path, dest: Path) -> int:
    """Append lines from src to dest that dest does not already contain.

    Args:
        src: Source JSONL file.
        dest: Destination JSONL file (created if missing).

    Returns:
        Number of lines added.
    """
    if not src.is_file():
        return 0
    existing = set(dest.read_text().splitlines()) if dest.is_file() else set()
    new = [line for line in src.read_text().splitlines() if line and line not in existing]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a") as f:
        f.writelines(line + "\n" for line in new)
    return len(new)


def export() -> None:
    """Copy kept files from runs/ to benchmark/runs/ and archive the LLM call log."""
    count = 0
    for path in RUNS.rglob("*"):
        if path.parent == RUNS or not _keep(path, RUNS):
            continue
        dest = ARCHIVE / path.relative_to(RUNS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        count += 1
    lines = _merge_lines(LLM_LOG, ARCHIVE_LOG)
    print(f"Exported {count} files and {lines} new LLM log lines to {ARCHIVE}")


def restore() -> None:
    """Copy archived files back into runs/ (never overwriting) and rebuild the LLM log."""
    count = 0
    for path in ARCHIVE.rglob("*"):
        if path == ARCHIVE_LOG or not _keep(path, ARCHIVE):
            continue
        dest = RUNS / path.relative_to(ARCHIVE)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            count += 1
    lines = _merge_lines(ARCHIVE_LOG, LLM_LOG)
    print(f"Restored {count} files and {lines} LLM log lines from {ARCHIVE}")


def main() -> None:
    """Parse the command and run export or restore."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["export", "restore"])
    {"export": export, "restore": restore}[p.parse_args().command]()


if __name__ == "__main__":
    main()
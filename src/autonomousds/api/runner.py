"""Run agent systems as background jobs, with a time limit and status tracking.

A job is one agent run started through the API. Its status is stored in
runs/_jobs/<run_id>.json (and its output in <run_id>.log), so it survives API
restarts. A Runner decides where jobs execute: LocalRunner runs them on this
machine, one at a time; a GitHub Actions runner can be added later.
"""
from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from autonomousds.benchmark.tasks import ROOT, Task, load_task
from autonomousds.systems.common import append_result, new_run_id

JOBS_DIR = ROOT / "runs" / "_jobs"
RESULTS_DIR = ROOT / "benchmark" / "results"
RUN_ID_PREFIX = {"single_agent": "single", "pipeline": "pipeline"}
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_id_lock = threading.Lock()


def _now() -> str:
    """Current UTC time as an ISO string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _job_path(run_id: str) -> Path:
    """Path of a job's status file.

    Args:
        run_id: Run id.

    Returns:
        runs/_jobs/<run_id>.json

    Raises:
        ValueError: If the run id could escape the jobs folder.
    """
    if not SAFE_ID.fullmatch(run_id):
        raise ValueError(f"Invalid run id '{run_id}'")
    return JOBS_DIR / f"{run_id}.json"


def read_job(run_id: str) -> dict | None:
    """Read a job's status.

    Args:
        run_id: Run id.

    Returns:
        The job dict, or None if no job has this id.
    """
    try:
        path = _job_path(run_id)
    except ValueError:
        return None
    return json.loads(path.read_text()) if path.is_file() else None


def _update_job(job: dict, **changes) -> dict:
    """Apply changes to a job and save it atomically (write temp file, then rename).

    Args:
        job: The job dict.
        **changes: Fields to set.

    Returns:
        The updated job.
    """
    job.update(changes)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    path = _job_path(job["run_id"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=2))
    tmp.replace(path)
    return job


def create_job(task: Task, system: str, max_steps: int | None,
               time_limit_seconds: int) -> dict:
    """Create a queued job with a new, unused run id.

    Args:
        task: The task to solve.
        system: "single_agent" or "pipeline".
        max_steps: Step budget (single_agent only), else None.
        time_limit_seconds: The run is stopped after this long.

    Returns:
        The new job.
    """
    with _id_lock:   # run ids have 1-second resolution; avoid duplicates
        run_id = new_run_id(RUN_ID_PREFIX[system], task)
        while _job_path(run_id).exists() or (ROOT / "runs" / task.name / run_id).exists():
            time.sleep(1)
            run_id = new_run_id(RUN_ID_PREFIX[system], task)
        return _update_job({"run_id": run_id, "task": task.name, "system": system,
                            "max_steps": max_steps,
                            "time_limit_seconds": time_limit_seconds,
                            "status": "queued", "created_at": _now()})


def find_record(task_name: str, system: str, run_id: str) -> dict | None:
    """Find a run's record in benchmark/results/<task>/<system>.json.

    Args:
        task_name: Task name.
        system: System name.
        run_id: Run id.

    Returns:
        The record, or None if the run wrote none.
    """
    path = RESULTS_DIR / task_name / f"{system}.json"
    if not path.is_file():
        return None
    return next((r for r in json.loads(path.read_text()).get("runs", [])
                 if r.get("run_id") == run_id), None)


def remove_containers(run_id: str) -> int:
    """Force-remove sandbox containers a run left behind.

    DockerSandbox names every container sbx-<run_id>-<random>, so the name
    filter matches only this run's containers.

    Args:
        run_id: Run id.

    Returns:
        How many containers were removed.
    """
    out = subprocess.run(["docker", "ps", "-aq", "--filter", f"name=sbx-{run_id}-"],
                         capture_output=True, text=True)
    ids = out.stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
    return len(ids)


def record_failure(job: dict, reason: str) -> None:
    """Count a stopped or crashed run as a failed run in the benchmark results.

    Does nothing if the run already wrote its own record.

    Args:
        job: The job.
        reason: Why the run failed, e.g. "time limit (30 min)".
    """
    if find_record(job["task"], job["system"], job["run_id"]):
        return
    record = {"run_id": job["run_id"], "submitted": False, "holdout_score": None,
              "seconds": job.get("seconds"), "failed_reason": reason, "trigger": "api"}
    if job["system"] == "single_agent":
        record["max_steps"] = job["max_steps"]
    append_result(load_task(job["task"]), job["system"], record)


def _kill_worker(pid: int | None, run_id: str) -> None:
    """Kill a leftover worker process, only if it really is this run's worker.

    Args:
        pid: Process id saved in the job, if any.
        run_id: Run id the worker must belong to.
    """
    if not pid:
        return
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
    except OSError:
        return                       # process no longer exists
    if "autonomousds.api.worker" in cmdline and run_id in cmdline:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class Runner(ABC):
    """Where jobs execute. Subclasses: LocalRunner (now), GitHub Actions (later)."""

    def start(self) -> None:
        """Called once when the API starts."""

    @abstractmethod
    def submit(self, job: dict) -> None:
        """Schedule a queued job.

        Args:
            job: A job from create_job().
        """


class LocalRunner(Runner):
    """Runs jobs on this machine, one at a time, each in its own process."""

    def __init__(self) -> None:
        self._queue: queue.Queue[dict] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Clean up jobs left by a previous API process, then start the worker thread."""
        if self._thread:
            return
        self._recover()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="local-runner")
        self._thread.start()

    def submit(self, job: dict) -> None:
        """Add a job to the queue.

        Args:
            job: A job from create_job().
        """
        self._queue.put(job)

    def run_now(self, job: dict) -> dict:
        """Run a job immediately in the foreground, without the queue.

        Args:
            job: A job from create_job().

        Returns:
            The finished job.
        """
        self._run(job)
        return job

    def _recover(self) -> None:
        """Mark jobs that were queued or running when the API stopped as failed."""
        for path in sorted(JOBS_DIR.glob("*.json")):
            job = json.loads(path.read_text())
            if job["status"] == "queued":       # never started: not a system failure
                _update_job(job, status="failed", finished_at=_now(),
                            error="API restarted before the run started")
            elif job["status"] == "running":    # stopped mid-run: counts as failure
                _kill_worker(job.get("pid"), job["run_id"])
                remove_containers(job["run_id"])
                reason = "API stopped during the run"
                _update_job(job, status="failed", finished_at=_now(), error=reason, pid=None)
                record_failure(job, reason)

    def _loop(self) -> None:
        """Run queued jobs forever, one after another."""
        while True:
            job = self._queue.get()
            try:
                self._run(job)
            except Exception as e:      # keep the thread alive for the next job
                _update_job(job, status="failed", finished_at=_now(),
                            error=f"Runner error: {e}", pid=None)

    def _run(self, job: dict) -> None:
        """Run one job in a worker process, enforce the time limit, record the outcome.

        Args:
            job: The job to run.
        """
        log_path = JOBS_DIR / f"{job['run_id']}.log"
        cmd = [sys.executable, "-u", "-m", "autonomousds.api.worker",
               "--task", job["task"], "--system", job["system"], "--run-id", job["run_id"]]
        if job["max_steps"]:
            cmd += ["--max-steps", str(job["max_steps"])]

        start = time.monotonic()
        with log_path.open("w") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            _update_job(job, status="running", started_at=_now(), pid=proc.pid)
            try:
                code = proc.wait(timeout=job["time_limit_seconds"])
                reason = None if code == 0 else f"crashed (exit code {code}); see log"
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                reason = f"time limit ({job['time_limit_seconds'] // 60} min)"
        seconds = round(time.monotonic() - start, 1)
        removed = remove_containers(job["run_id"])

        if reason is None:
            record = find_record(job["task"], job["system"], job["run_id"]) or {}
            _update_job(job, status="done", finished_at=_now(), seconds=seconds, pid=None,
                        submitted=record.get("submitted"),
                        holdout_score=record.get("holdout_score"))
        else:
            _update_job(job, status="failed", finished_at=_now(), seconds=seconds,
                        pid=None, error=reason, containers_removed=removed)
            record_failure(job, reason)
        self._build_report(job, log_path)

    def _build_report(self, job: dict, log_path: Path) -> None:
        """Refresh summaries and README so the new run shows up in the API.

        Args:
            job: The finished job.
            log_path: Job log; the report's output is appended to it.
        """
        with log_path.open("a") as log:
            r = subprocess.run([sys.executable, "scripts/build_report.py"], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, timeout=300)
        if r.returncode != 0:
            _update_job(job, report_error="build_report.py failed; see log")
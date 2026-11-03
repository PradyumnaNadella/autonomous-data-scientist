"""Run untrusted Python code inside an isolated Docker container."""
from __future__ import annotations

import shutil
import subprocess
import uuid
import json
import time
from dataclasses import dataclass
from pathlib import Path

IMAGE = "autonomousds-sandbox:latest"
ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT / "runs"
MAX_OUTPUT_CHARS = 5000   # keep agent prompts small


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def _trim(text: str) -> str:
    """Keep the start and end of long output."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    return text[:half] + "\n...[output trimmed]...\n" + text[-half:]


class DockerSandbox:
    def __init__(self, run_id: str | None = None, timeout_seconds: int = 120,
                 memory: str = "2g", cpus: str = "2"):
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.timeout = timeout_seconds
        self.memory = memory
        self.cpus = cpus
        self.workdir = RUNS_DIR / self.run_id
        self.workdir.mkdir(parents=True, exist_ok=True)

    def add_file(self, src: str | Path, name: str | None = None) -> str:
        """Copy a file (e.g. the dataset) into the run folder."""
        src = Path(src)
        dest = self.workdir / (name or src.name)
        shutil.copy(src, dest)
        return dest.name

    def list_files(self) -> list[str]:
        return sorted(p.name for p in self.workdir.iterdir())

    def run_python(self, code: str) -> ExecResult:
        step = f"step_{uuid.uuid4().hex[:6]}"
        script = f"{step}.py"
        (self.workdir / script).write_text(code)
        container = f"sbx-{self.run_id}-{uuid.uuid4().hex[:6]}"

        cmd = [
            "docker", "run", "--rm", "--name", container,
            "--network", "none",
            "--memory", self.memory,
            "--cpus", self.cpus,
            "--pids-limit", "256",
            "--read-only",
            "--tmpfs", "/tmp:size=256m",
            "-v", f"{self.workdir}:/workspace",
            "-w", "/workspace",
            IMAGE, "python", script,
        ]
        start = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=self.timeout)
            result = ExecResult(_trim(p.stdout), _trim(p.stderr),
                                p.returncode, False)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", container], capture_output=True)
            result = ExecResult("", f"Timed out after {self.timeout}s", -1, True)

        self._save_log(step, result, time.time() - start)
        return result

    def _save_log(self, step: str, result: ExecResult, seconds: float) -> None:
        # Human-readable output next to the script
        (self.workdir / f"{step}.out.txt").write_text(
            f"exit_code: {result.exit_code}  timed_out: {result.timed_out}  "
            f"seconds: {seconds:.1f}\n\n"
            f"--- STDOUT ---\n{result.stdout}\n\n--- STDERR ---\n{result.stderr}\n"
        )
        # One line per step, for code and the run viewer later
        with (self.workdir / "sandbox_log.jsonl").open("a") as f:
            f.write(json.dumps({
                "step": step,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "seconds": round(seconds, 1),
                "stdout": result.stdout,
                "stderr": result.stderr,
            }) + "\n")
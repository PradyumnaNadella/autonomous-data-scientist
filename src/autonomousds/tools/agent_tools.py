"""Tools the agents can call. Each set of tools is bound to one run's sandbox."""
from __future__ import annotations

import pandas as pd
from langchain_core.tools import tool

from autonomousds.benchmark.tasks import Task
from autonomousds.sandbox.docker_runner import DockerSandbox

MAX_READ_CHARS = 3000
HIDDEN_FILES = ("step_", "sandbox_log")     # sandbox bookkeeping, not for the agent


def setup_run(task: Task, run_id: str | None = None,
              timeout_seconds: int = 120) -> DockerSandbox:
    """Create a sandbox run folder containing only the data agents may see."""
    sb = DockerSandbox(run_id=run_id, timeout_seconds=timeout_seconds)
    sb.add_file(task.agent_dir / "train.csv")
    sb.add_file(task.agent_dir / "test.csv")
    return sb


def make_tools(sb: DockerSandbox, task: Task):
    state = {"submitted": None}             # filled when predictions are accepted

    @tool(parse_docstring=True)
    def run_python(code: str) -> str:
        """Run Python code in an isolated sandbox and return its output.

        The working directory contains train.csv and test.csv. Each call is a NEW
        process: variables do NOT persist between calls, so save anything needed
        later to files (e.g. features.parquet, model.joblib). Available libraries:
        pandas, numpy, scikit-learn, lightgbm, xgboost, matplotlib, joblib.
        No internet access.

        Args:
            code: Complete Python script to run. Use print() to see results.
        """
        r = sb.run_python(code)
        if r.timed_out:
            status = "TIMED OUT"
        else:
            status = "OK" if r.ok else f"ERROR (exit code {r.exit_code})"
        return (f"[{status}]\n--- stdout ---\n{r.stdout or '(empty)'}\n"
                f"--- stderr ---\n{r.stderr or '(empty)'}")

    @tool
    def list_files() -> str:
        """List the files in the working directory."""
        files = [f for f in sb.list_files() if not f.startswith(HIDDEN_FILES)]
        return "\n".join(files) or "(no files)"

    @tool(parse_docstring=True)
    def read_file(name: str) -> str:
        """Read the first 3000 characters of a text file in the working directory.

        Args:
            name: File name in the working directory, e.g. "notes.txt".
        """
        path = (sb.workdir / name).resolve()
        if path.parent != sb.workdir.resolve() or not path.is_file():
            return f"Error: '{name}' not found in the working directory."
        return path.read_text(errors="replace")[:MAX_READ_CHARS]

    @tool(parse_docstring=True)
    def submit_predictions(filename: str = "predictions.csv") -> str:
        """Check and submit the final predictions CSV.

        Call this once, when your predictions are final. If it is not accepted,
        fix the file using the returned message and call again.

        Args:
            filename: CSV in the working directory with one row per test.csv row
                and two columns: the id column and the target column.
        """
        path = sb.workdir / filename
        if not path.is_file():
            return f"Not accepted: '{filename}' does not exist."
        try:
            preds = pd.read_csv(path)
        except Exception as e:
            return f"Not accepted: could not read the file ({e})."

        test = pd.read_csv(task.agent_dir / "test.csv")   # original, not the agent's copy
        idc, tgt = task.id_column, task.target
        problems = [f"missing column '{c}'" for c in (idc, tgt) if c not in preds.columns]
        if not problems:
            if len(preds) != len(test):
                problems.append(f"has {len(preds)} rows, expected {len(test)}")
            missing = set(test[idc]) - set(preds[idc])
            if missing:
                problems.append(f"{len(missing)} test ids have no prediction")
            if preds[tgt].isna().any():
                problems.append(f"'{tgt}' contains missing values")
            if task.task_type == "classification":
                allowed = set(pd.read_csv(task.agent_dir / "train.csv")[tgt].astype(str))
                bad = set(preds[tgt].dropna().astype(str)) - allowed
                if bad:
                    problems.append(f"invalid values {sorted(map(str, bad))[:5]}; "
                                    f"allowed: {sorted(map(str, allowed))}")
        if problems:
            return "Not accepted: " + "; ".join(problems)
        state["submitted"] = filename
        return "Accepted. Predictions are final."

    return [run_python, list_files, read_file, submit_predictions], state
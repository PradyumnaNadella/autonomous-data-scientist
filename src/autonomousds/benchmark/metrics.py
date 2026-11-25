"""Evaluation metrics: model quality per run, process metrics for agent systems,
and per-system summaries. Works for any task and any system."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pandas as pd
from sklearn import metrics as skm

from autonomousds.benchmark.tasks import ROOT, Task
from autonomousds.llm.models import LOG_PATH

RUNS_DIR = ROOT / "runs"


def baseline_predictions_path(task: Task, system: str) -> Path:
    """Where a baseline system saves its predictions.

    Args:
        task: The benchmark task.
        system: Baseline name, e.g. "direct" or "automl".

    Returns:
        Path to runs/<task>/baselines/<system>/predictions.csv.
    """
    return RUNS_DIR / task.name / "baselines" / system / "predictions.csv"


def model_metrics(task: Task, predictions: pd.DataFrame) -> dict:
    """Score predictions against the hidden labels with a full set of metrics.

    Args:
        task: The benchmark task.
        predictions: DataFrame with the id column and the target column.

    Returns:
        Classification: accuracy, balanced_accuracy, precision, recall, f1
        (macro-averaged, so multi-class works). Regression: rmse, mae, r2.
    """
    labels = pd.read_csv(task.hidden_dir / "test_labels.csv")
    m = labels.merge(predictions[[task.id_column, task.target]], on=task.id_column,
                     how="left", suffixes=("_true", "_pred"))
    y_true, y_pred = m[f"{task.target}_true"], m[f"{task.target}_pred"]
    if y_pred.isna().any():
        raise ValueError("Predictions are missing for some ids")

    if task.task_type == "classification":
        y_true, y_pred = y_true.astype(str), y_pred.astype(str)
        avg = {"average": "macro", "zero_division": 0}
        out = {"accuracy": skm.accuracy_score(y_true, y_pred),
               "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),
               "precision": skm.precision_score(y_true, y_pred, **avg),
               "recall": skm.recall_score(y_true, y_pred, **avg),
               "f1": skm.f1_score(y_true, y_pred, **avg)}
    else:
        y_pred = pd.to_numeric(y_pred)
        out = {"rmse": skm.root_mean_squared_error(y_true, y_pred),
               "mae": skm.mean_absolute_error(y_true, y_pred),
               "r2": skm.r2_score(y_true, y_pred)}
    return {k: round(float(v), 4) for k, v in out.items()}


def execution_metrics(workdir: Path) -> dict:
    """Code-execution reliability for one agent run, from sandbox_log.jsonl.

    A failure counts as recovered if any later execution in the run succeeded.

    Args:
        workdir: The run folder.

    Returns:
        Dict with executions, failed_executions, timeouts, recovered_failures.
    """
    log = workdir / "sandbox_log.jsonl"
    if not log.exists():
        return {}
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    ok = [r["exit_code"] == 0 and not r["timed_out"] for r in rows]
    failures = [i for i, good in enumerate(ok) if not good]
    return {"executions": len(rows),
            "failed_executions": len(failures),
            "timeouts": sum(bool(r["timed_out"]) for r in rows),
            "recovered_failures": sum(1 for i in failures if any(ok[i + 1:]))}


def check_metrics(workdir: Path) -> dict:
    """Count how often code-level checks sent an agent back, from transcript.json.

    Args:
        workdir: The run folder.

    Returns:
        Dict with checks_fired, check_missing_train_model and
        check_experiments_not_logged (empty if there is no transcript).
    """
    path = workdir / "transcript.json"
    if not path.exists():
        return {}
    marker = "Your work is incomplete. You must:"
    segments = [m["text"].split(marker)[1].split("Your previous report:")[0]
                for m in json.loads(path.read_text())
                if m.get("type") == "human" and marker in m.get("text", "")]
    return {"checks_fired": len(segments),
            "check_missing_train_model": sum("train_model.py" in s for s in segments),
            "check_experiments_not_logged": sum("log_experiment" in s for s in segments)}


def truncation_count(run_id: str) -> int:
    """Number of LLM responses in a run that were cut off by the output limit.

    Args:
        run_id: The run to check.

    Returns:
        Count of truncated responses (0 if the log predates truncation logging).
    """
    if not LOG_PATH.exists():
        return 0
    return sum(1 for l in LOG_PATH.read_text().splitlines()
               if (c := json.loads(l))["run_id"] == run_id and c.get("truncated"))


def _cv_optimism(task: Task, cv: float | None, test: float | None) -> float | None:
    """How much the CV estimate overstated the hidden-test result (positive = optimistic)."""
    if cv is None or test is None:
        return None
    return round(cv - test if task.higher_is_better else test - cv, 4)


def enrich_run(task: Task, record: dict) -> dict:
    """Add model and process metrics to one agent run's record.

    Args:
        task: The benchmark task.
        record: One entry from benchmark/results/<task>/<system>.json.

    Returns:
        A copy of the record with model_metrics, cv_score, cv_optimism,
        total_tokens, execution, check and truncation metrics.
    """
    r = dict(record)
    workdir = RUNS_DIR / task.name / r["run_id"]
    preds = workdir / "predictions.csv"
    if r.get("submitted") and preds.exists():
        r["model_metrics"] = model_metrics(task, pd.read_csv(preds))

    scores = [e["cv_score"] for e in r.get("experiments", [])
              if isinstance(e.get("cv_score"), (int, float))]
    if scores:
        r["cv_score"] = max(scores) if task.higher_is_better else min(scores)
        r["cv_optimism"] = _cv_optimism(
            task, r["cv_score"], r.get("model_metrics", {}).get(task.metric))

    r["total_tokens"] = r.get("input_tokens", 0) + r.get("output_tokens", 0)
    r.update(execution_metrics(workdir))
    r.update(check_metrics(workdir))
    r["truncated_responses"] = truncation_count(r["run_id"])
    return r


def enrich_baseline(task: Task, system: str, data: dict) -> dict:
    """Turn a baseline results file into a single run record with full metrics.

    Args:
        task: The benchmark task.
        system: "direct" or "automl".
        data: Contents of benchmark/results/<task>/<system>.json.

    Returns:
        A run record comparable to agent runs.
    """
    best = next((x for x in data.get("results", []) if x["model"] == data.get("best_model")), {})
    r = {"run_id": system, "submitted": True, "best_model": data.get("best_model"),
         "cv_score": best.get("cv_score"),
         "seconds": data.get("seconds") or sum(x.get("seconds", 0) for x in data.get("results", []))}
    preds = baseline_predictions_path(task, system)
    if preds.exists():
        r["model_metrics"] = model_metrics(task, pd.read_csv(preds))
        r["cv_optimism"] = _cv_optimism(task, r["cv_score"], r["model_metrics"][task.metric])
    return r


def summarize_system(task: Task, name: str, runs: list[dict]) -> dict:
    """Aggregate enriched runs of one system into summary metrics.

    Args:
        task: The benchmark task.
        name: Display name of the system.
        runs: Enriched run records.

    Returns:
        Summary: success rate, main-metric statistics, mean of every model
        metric, CV optimism, time, cost, reliability and review metrics.
    """
    done = [r for r in runs if r.get("model_metrics")]
    scores = [r["model_metrics"][task.metric] for r in done]
    best, worst = (max, min) if task.higher_is_better else (min, max)

    def mean(key, rows=runs):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    def total(key):
        vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
        return sum(vals) if vals else None

    costs = [r["est_cost_usd"] for r in runs if isinstance(r.get("est_cost_usd"), (int, float))]
    failed, recovered = total("failed_executions"), total("recovered_failures")
    return {
        "system": name,
        "runs": len(runs),
        "successful_runs": len(done),
        "success_rate": round(len(done) / len(runs), 3) if runs else None,
        "score_mean": round(statistics.mean(scores), 4) if scores else None,
        "score_std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0 if scores else None,
        "score_worst": worst(scores) if scores else None,
        "score_best": best(scores) if scores else None,
        "model_metrics_mean": ({k: mean(k, [r["model_metrics"] for r in done])
                                for k in done[0]["model_metrics"]} if done else {}),
        "cv_mean": mean("cv_score", done),
        "cv_optimism_mean": mean("cv_optimism", done),
        "seconds_median": (statistics.median([r["seconds"] for r in runs if r.get("seconds")])
                           if any(r.get("seconds") for r in runs) else None),
        "cost_mean_usd": round(sum(costs) / len(costs), 4) if costs else None,
        "cost_per_success_usd": round(sum(costs) / len(done), 4) if costs and done else None,
        "llm_calls_mean": mean("llm_calls"),
        "tokens_mean": mean("total_tokens"),
        "failed_executions_mean": mean("failed_executions"),
        "recovery_rate": round(recovered / failed, 3) if failed and recovered is not None else None,
        "timeouts_total": total("timeouts"),
        "truncated_total": total("truncated_responses"),
        "approval_rate": mean("approved"),
        "revisions_mean": mean("revisions"),
        "checks_fired_total": total("checks_fired"),
    }
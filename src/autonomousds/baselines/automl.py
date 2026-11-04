"""AutoML baseline: AutoGluon with a fixed time budget. No LLM."""
from __future__ import annotations

import json
import time

import pandas as pd
from autogluon.tabular import TabularPredictor

from autonomousds.benchmark.tasks import ROOT, Task, score

RESULTS_DIR = ROOT / "benchmark" / "results"
MODELS_DIR = ROOT / "runs" / "automl"      # big model files, not in Git

# our metric name -> AutoGluon's metric name
AG_METRIC = {"accuracy": "accuracy", "roc_auc": "roc_auc",
             "rmse": "root_mean_squared_error"}


def run_automl(task: Task, time_limit: int = 300) -> dict:
    train = pd.read_csv(task.agent_dir / "train.csv").drop(columns=[task.id_column])
    test = pd.read_csv(task.agent_dir / "test.csv")
    print(f"Task: {task.name} | metric: {task.metric} | time limit: {time_limit}s")

    start = time.time()
    predictor = TabularPredictor(
        label=task.target,
        eval_metric=AG_METRIC[task.metric],
        path=str(MODELS_DIR / task.name),
        verbosity=1,
    ).fit(train, time_limit=time_limit, presets="best_quality")
    secs = time.time() - start

    preds = pd.DataFrame({
        task.id_column: test[task.id_column],
        task.target: predictor.predict(test.drop(columns=[task.id_column])),
    })
    holdout = score(task, preds)

    # Score every model on the hidden test (evaluation only, never training)
    labels = pd.read_csv(task.hidden_dir / "test_labels.csv")
    test_labeled = test.merge(labels, on=task.id_column).drop(columns=[task.id_column])
    board = predictor.leaderboard(test_labeled, silent=True)

    sign = -1 if task.metric == "rmse" else 1   # AutoGluon makes error metrics negative
    results = [{
        "model": r["model"],
        "cv_score": round(float(sign * r["score_val"]), 4),
        "holdout_score": round(float(sign * r["score_test"]), 4),
        "fit_seconds": round(float(r["fit_time"]), 1),
    } for _, r in board.iterrows()]

    best = predictor.model_best
    print(f"  models trained: {len(results)} | best (chosen by validation): {best}")
    for r in results[:5]:
        print(f"  {r['model']:28s} validation={r['cv_score']:.4f}  hidden_test={r['holdout_score']:.4f}")
    print(f"  FINAL hidden_test={holdout:.4f}  ({secs:.0f}s)")

    out = {
        "task": task.name, "system": "automl", "metric": task.metric,
        "time_limit_seconds": time_limit, "preset": "best_quality",
        "best_model": best, "holdout_score": round(holdout, 4),
        "seconds": round(secs, 1), "results": results,
    }
    out_dir = RESULTS_DIR / task.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "automl.json").write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_dir / 'automl.json'}")
    return out
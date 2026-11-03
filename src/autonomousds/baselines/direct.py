"""Direct baseline: apply each ML family with default settings. No LLM, no tuning."""
from __future__ import annotations

import json
import time

import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from autonomousds.benchmark.tasks import ROOT, Task, score

RESULTS_DIR = ROOT / "benchmark" / "results"
MAX_CATEGORIES = 50          # drop text columns with more unique values
CV_SCORING = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "rmse": "neg_root_mean_squared_error",
}


def models_for(task: Task) -> dict:
    s = task.seed
    if task.task_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(random_state=s, n_jobs=-1),
            "lightgbm": LGBMClassifier(random_state=s, verbose=-1),
            "mlp": MLPClassifier(max_iter=500, random_state=s),
        }
    return {
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(random_state=s, n_jobs=-1),
        "lightgbm": LGBMRegressor(random_state=s, verbose=-1),
        "mlp": MLPRegressor(max_iter=500, random_state=s),
    }


def build_features(task: Task, train: pd.DataFrame, test: pd.DataFrame):
    """Generic preprocessing with no knowledge of the dataset."""
    X = train.drop(columns=[task.target, task.id_column])
    X_test = test.drop(columns=[task.id_column])
    text_cols = X.select_dtypes(exclude="number").columns
    dropped = [c for c in text_cols if X[c].nunique() > MAX_CATEGORIES]
    keep = [c for c in text_cols if c not in dropped]
    X, X_test = X.drop(columns=dropped), X_test.drop(columns=dropped)
    # Missing values in text columns become their own "nan" category
    X[keep] = X[keep].astype(str)
    X_test[keep] = X_test[keep].astype(str)
    return X, X_test, dropped


def preprocessor() -> ColumnTransformer:
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include="number")),
        ("cat", categorical, make_column_selector(dtype_exclude="number")),
    ])


def run_direct(task: Task) -> list[dict]:
    train = pd.read_csv(task.agent_dir / "train.csv")
    test = pd.read_csv(task.agent_dir / "test.csv")
    X, X_test, dropped = build_features(task, train, test)
    y = train[task.target]
    print(f"Task: {task.name} | metric: {task.metric} | dropped columns: {dropped}")

    results = []
    for name, model in models_for(task).items():
        pipe = make_pipeline(preprocessor(), model)
        start = time.time()
        cv = cross_val_score(pipe, X, y, cv=5, scoring=CV_SCORING[task.metric])
        cv_mean = -cv.mean() if task.metric == "rmse" else cv.mean()
        pipe.fit(X, y)
        preds = pd.DataFrame({task.id_column: test[task.id_column],
                              task.target: pipe.predict(X_test)})
        holdout = score(task, preds)
        secs = time.time() - start
        print(f"  {name:20s} cv={cv_mean:.4f}  hidden_test={holdout:.4f}  ({secs:.1f}s)")
        results.append({"model": name, "cv_score": round(float(cv_mean), 4),
                        "holdout_score": round(holdout, 4),
                        "seconds": round(secs, 1)})

    out_dir = RESULTS_DIR / task.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"task": task.name, "system": "direct",
            "metric": task.metric, "dropped_columns": dropped, "results": results}
    (out_dir / "direct.json").write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_dir / 'direct.json'}")
    return results
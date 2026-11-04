"""Direct baseline: apply each ML family with default settings. No LLM, no tuning."""
from __future__ import annotations

import warnings
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

def clean_generic(train: pd.DataFrame, test: pd.DataFrame, skip: list[str]):
    """Dataset-agnostic fixes. Decisions are made on train, applied to both."""
    train, test = train.copy(), test.copy()
    for col in train.columns:
        if col in skip:
            continue
        s = train[col]
        present = s.notna().sum()

        # True/False columns -> 1/0
        if s.dtype == bool:
            train[col] = s.astype(int)
            test[col] = test[col].astype(int)
            continue
        if s.dtype != object or present == 0:
            continue

        # Numbers stored as text: "$1,200", "45%", " 3.5 "
        def to_num(x):
            return pd.to_numeric(x.astype(str).str.replace(r"[,$%\s]", "", regex=True),
                                 errors="coerce")
        if to_num(s).notna().sum() >= 0.9 * present:
            train[col], test[col] = to_num(s), to_num(test[col])
            continue

        # Date columns -> year, month, weekday
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        if parsed.notna().sum() >= 0.9 * present:
            for df in (train, test):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    d = pd.to_datetime(df[col], errors="coerce", format="mixed")
                df[f"{col}_year"] = d.dt.year
                df[f"{col}_month"] = d.dt.month
                df[f"{col}_weekday"] = d.dt.dayofweek
                df.drop(columns=[col], inplace=True)
    return train, test


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
    train, test = clean_generic(train, test, skip=[task.target, task.id_column])
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
    pick = min if task.metric == "rmse" else max
    best = pick(results, key=lambda r: r["cv_score"])   # choose by validation only
    print(f"  CHOSEN by CV: {best['model']}  hidden_test={best['holdout_score']:.4f}")
    out = {"task": task.name, "system": "direct", "metric": task.metric,
           "best_model": best["model"], "holdout_score": best["holdout_score"],
           "dropped_columns": dropped, "results": results}
    (out_dir / "direct.json").write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_dir / 'direct.json'}")
    return results
"""Dataset loading for CC-18 tasks (zadani 1.1), with an on-disk cache of encoded (X, y).

Fetched via OpenML once, then cached to runs/_datacache/ (git-ignored). Categorical features are
one-hot encoded and missing values filled, matching the P3 real-search preprocessing.
"""
from __future__ import annotations
from pathlib import Path
import joblib, numpy as np, pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "runs" / "_datacache"


def load_task(task_id: int):
    """Return (X: float ndarray, y: int ndarray, n_features:int) for an OpenML CC-18 task, cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"task_{task_id}.joblib"
    if f.exists():
        d = joblib.load(f)
        return d["X"], d["y"], d["X"].shape[1]
    import openml
    task = openml.tasks.get_task(task_id, download_splits=False)
    ds = openml.datasets.get_dataset(task.dataset_id)
    X, y, _, _ = ds.get_data(target=task.target_name)
    X = pd.get_dummies(X, dummy_na=True).fillna(0).to_numpy().astype(float)
    y = pd.Series(y).astype("category").cat.codes.to_numpy().astype(int)
    joblib.dump({"X": X, "y": y}, f)
    return X, y, X.shape[1]

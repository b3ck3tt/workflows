from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# Paths / cache
# ============================================================

DEFAULT_CACHE_DIR = Path("minimal_cache_cc18")
DEFAULT_RESULTS_DIR = Path("minimal_results_cc18")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_cache_key(prefix: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.md5(raw).hexdigest()
    return f"{prefix}_{digest}"


class DiskCache:
    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = ensure_dir(cache_dir)

    def path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.joblib"

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def load(self, key: str) -> Any:
        return joblib.load(self.path(key))

    def save(self, key: str, value: Any) -> None:
        joblib.dump(value, self.path(key))

    def get_or_compute(self, key: str, fn, *args, **kwargs):
        if self.exists(key):
            return self.load(key)
        value = fn(*args, **kwargs)
        self.save(key, value)
        return value


# ============================================================
# Text helpers
# ============================================================

def enrich_flow_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace(".", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("=", " ")
    text = text.replace(",", " ")
    text = text.replace("==", " ")
    text = text.replace(":", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def clean_text_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .map(enrich_flow_text)
    )


def text_series_hash(series: pd.Series) -> str:
    cleaned = clean_text_series(series)
    raw = "\n".join(cleaned.tolist()).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


# ============================================================
# OpenML-specific normalization
# ============================================================

def normalize_openml_flows_dict(
    flows: dict,
    flow_id_col: str = "flow_id",
    text_col: str = "flow_text",
) -> pd.DataFrame:
    rows = []

    for key, flow in flows.items():
        if not isinstance(flow, dict):
            continue

        flow_id = flow.get("id", key)
        name = flow.get("name")
        full_name = flow.get("full_name")
        version = flow.get("version")
        external_version = flow.get("external_version")
        uploader = flow.get("uploader")

        raw_parts = []
        if name:
            raw_parts.append(f"name: {name}")
        if full_name:
            raw_parts.append(f"full_name: {full_name}")
        if version:
            raw_parts.append(f"version: {version}")
        if external_version:
            raw_parts.append(f"external_version: {external_version}")

        raw_text = "\n".join(raw_parts)

        rows.append(
            {
                flow_id_col: int(flow_id),
                "flow_name": name,
                "flow_full_name": full_name,
                "flow_version": version,
                "flow_external_version": external_version,
                "flow_uploader": uploader,
                text_col: raw_text,
                f"{text_col}_clean": enrich_flow_text(raw_text),
            }
        )

    return pd.DataFrame(rows)


def normalize_cc18_tasks(tasks_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = tasks_df.copy()

    rename_map = {
        "tid": "task_id",
        "did": "dataset_id",
        "name": "task_name",
    }
    df = df.rename(columns=rename_map)

    numeric_meta_cols = [
        "MajorityClassSize",
        "MaxNominalAttDistinctValues",
        "MinorityClassSize",
        "NumberOfClasses",
        "NumberOfFeatures",
        "NumberOfInstances",
        "NumberOfInstancesWithMissingValues",
        "NumberOfMissingValues",
        "NumberOfNumericFeatures",
        "NumberOfSymbolicFeatures",
    ]

    keep_cols = ["task_id", "dataset_id", "task_name"] + [
        c for c in numeric_meta_cols if c in df.columns
    ]

    return df[keep_cols].copy(), [c for c in numeric_meta_cols if c in df.columns]


def aggregate_cc18_evaluations(
    evals_df: pd.DataFrame,
    value_col: str = "value",
    agg_mode: str = "mean",   # "mean" | "max"
) -> pd.DataFrame:
    df = evals_df.copy()

    required = ["task_id", "flow_id", value_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required evaluation columns: {missing}")

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])

    if agg_mode == "mean":
        agg = (
            df.groupby(["task_id", "flow_id"], as_index=False)
            .agg(
                target_value=(value_col, "mean"),
                value_std=(value_col, "std"),
                value_count=(value_col, "count"),
            )
        )
    elif agg_mode == "max":
        agg = (
            df.groupby(["task_id", "flow_id"], as_index=False)
            .agg(
                target_value=(value_col, "max"),
                value_std=(value_col, "std"),
                value_count=(value_col, "count"),
            )
        )
    else:
        raise ValueError(f"Unsupported agg_mode: {agg_mode}")

    agg["value_std"] = agg["value_std"].fillna(0.0)
    return agg


def filter_relevant_sklearn_flows(
    flows_df: pd.DataFrame,
    evals_df: pd.DataFrame,
    flow_id_col: str = "flow_id",
) -> pd.DataFrame:
    used_flow_ids = set(evals_df[flow_id_col].dropna().astype(int).unique())

    mask_used = flows_df[flow_id_col].isin(used_flow_ids)

    mask_sklearn = pd.Series(False, index=flows_df.index)
    if "flow_name" in flows_df.columns:
        mask_sklearn = mask_sklearn | flows_df["flow_name"].fillna("").str.startswith("sklearn.")
    if "flow_full_name" in flows_df.columns:
        mask_sklearn = mask_sklearn | flows_df["flow_full_name"].fillna("").str.startswith("sklearn.")

    return flows_df.loc[mask_used & mask_sklearn].reset_index(drop=True)


def build_cc18_supervised_dataset(
    flows_df: pd.DataFrame,
    tasks_df: pd.DataFrame,
    evals_agg_df: pd.DataFrame,
    flow_id_col: str = "flow_id",
    task_id_col: str = "task_id",
    text_col: str = "flow_text_clean",
    target_col: str = "target_value",
) -> pd.DataFrame:
    df = evals_agg_df.merge(tasks_df, on=task_id_col, how="left")
    df = df.merge(flows_df, on=flow_id_col, how="left")

    required = [flow_id_col, task_id_col, text_col, target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after merge: {missing}")

    df = df.dropna(subset=[text_col, target_col]).reset_index(drop=True)
    return df


# ============================================================
# Feature building
# ============================================================

def transform_numeric_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> sparse.csr_matrix:
    if not numeric_cols:
        raise ValueError("No numeric metafeatures columns provided.")

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    X = pipe.fit_transform(df[numeric_cols])
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    return X


def build_tfidf_features(
    texts: pd.Series,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
):
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=1,
        max_df=0.95,
    )
    X = vec.fit_transform(clean_text_series(texts))
    return X, vec, vec.get_feature_names_out().tolist()


def build_minilm_features(
    texts: pd.Series,
):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Package 'sentence-transformers' is required for MiniLM features. "
            "Install it with: pip install sentence-transformers"
        ) from e

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(clean_text_series(texts).tolist(), show_progress_bar=True)
    X = sparse.csr_matrix(np.asarray(emb))
    feature_names = [f"emb_{i}" for i in range(X.shape[1])]
    return X, model, feature_names


def build_feature_set(
    supervised_df: pd.DataFrame,
    text_mode: str = "tfidf",  # "tfidf" | "minilm" | "none"
    use_task_metafeatures: bool = True,
    text_col: str = "flow_text_clean",
    target_col: str = "target_value",
    cache: DiskCache | None = None,
):
    cache = cache or DiskCache()

    ignore_cols = {
        "flow_id",
        "task_id",
        "dataset_id",
        "flow_name",
        "flow_full_name",
        "flow_version",
        "flow_external_version",
        "flow_uploader",
        "flow_text",
        "flow_text_clean",
        "task_name",
        "target_value",
        "value_std",
        "value_count",
    }

    numeric_cols = [
        c for c in supervised_df.select_dtypes(include=["number", "bool"]).columns
        if c not in ignore_cols and c != target_col
    ]

    X_parts = []
    feature_names = []
    artifacts = {}

    if text_mode == "tfidf":
        key = make_cache_key(
            "tfidf_features",
            {
                "n": len(supervised_df),
                "text_col": text_col,
                "text_hash": text_series_hash(supervised_df[text_col]),
            },
        )

        def _compute():
            return build_tfidf_features(supervised_df[text_col])

        X_text, text_artifact, text_feature_names = cache.get_or_compute(key, _compute)
        X_parts.append(X_text)
        feature_names.extend([f"text::{x}" for x in text_feature_names])
        artifacts["text_artifact"] = text_artifact

    elif text_mode == "minilm":
        key = make_cache_key(
            "minilm_features",
            {
                "n": len(supervised_df),
                "text_col": text_col,
                "text_hash": text_series_hash(supervised_df[text_col]),
            },
        )

        def _compute():
            return build_minilm_features(supervised_df[text_col])

        X_text, text_artifact, text_feature_names = cache.get_or_compute(key, _compute)
        X_parts.append(X_text)
        feature_names.extend([f"text::{x}" for x in text_feature_names])
        artifacts["text_artifact"] = text_artifact

    elif text_mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported text_mode: {text_mode}")

    if use_task_metafeatures:
        if not numeric_cols:
            raise ValueError("No numeric task metafeatures detected.")
        X_meta = transform_numeric_matrix(supervised_df, numeric_cols)
        X_parts.append(X_meta)
        feature_names.extend([f"meta::{x}" for x in numeric_cols])

    if not X_parts:
        raise ValueError("No features selected. Use text_mode != 'none' or enable task metafeatures.")

    X = sparse.hstack(X_parts).tocsr()
    y = supervised_df[target_col].values

    return X, y, feature_names, artifacts


# ============================================================
# Models
# ============================================================


def get_models(random_state: int = 42) -> dict[str, Any]:
    return {
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gbrt": HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            early_stopping=True,
            random_state=random_state,
        ),
    }

# ============================================================
# CV
# ============================================================


@dataclass
class ShapFoldResult:
    fold: int
    expected_value: float
    feature_names: list[str]
    shap_values: np.ndarray
    X_eval: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    mean_abs_shap: np.ndarray
    row_ids: np.ndarray


@dataclass
class ShapInteractionFoldResult:
    fold: int
    feature_names: list[str]
    interaction_values: np.ndarray


def _to_dense_if_sparse(X):
    return X.toarray() if sparse.issparse(X) else X


def compute_tree_shap_for_fold(
    model,
    X_train,
    X_test,
    y_test,
    y_pred,
    feature_names: list[str],
    fold: int,
    background_size: int = 200,
    eval_size: int = 100,
    random_state: int = 42,
    row_ids_test: np.ndarray | None = None,
) -> ShapFoldResult:
    rng = np.random.default_rng(random_state)

    X_train_dense = np.asarray(_to_dense_if_sparse(X_train), dtype=np.float64)
    X_test_dense = np.asarray(_to_dense_if_sparse(X_test), dtype=np.float64)

    bg_n = min(background_size, X_train_dense.shape[0])
    bg_idx = rng.choice(X_train_dense.shape[0], size=bg_n, replace=False)
    X_bg = np.asarray(X_train_dense[bg_idx], dtype=np.float64)

    ev_n = min(eval_size, X_test_dense.shape[0])
    ev_idx = rng.choice(X_test_dense.shape[0], size=ev_n, replace=False)
    X_eval = np.asarray(X_test_dense[ev_idx], dtype=np.float64)
    y_eval = y_test[ev_idx]
    y_pred_eval = y_pred[ev_idx]
    if row_ids_test is None:
        row_ids_eval = np.asarray([f"row_{i}" for i in ev_idx], dtype=object)
    else:
        row_ids_eval = np.asarray(row_ids_test[ev_idx], dtype=object)

    explainer = shap.TreeExplainer(
        model,
        data=X_bg,
        feature_perturbation="interventional",
        model_output="raw",
        feature_names=feature_names,
    )

    shap_values = explainer.shap_values(X_eval, check_additivity=False)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray):
        expected_value = float(np.ravel(expected_value)[0])
    else:
        expected_value = float(expected_value)

    return ShapFoldResult(
        fold=fold,
        expected_value=expected_value,
        feature_names=feature_names,
        shap_values=shap_values,
        X_eval=X_eval,
        y_true=y_eval,
        y_pred=y_pred_eval,
        mean_abs_shap=mean_abs_shap,
        row_ids=row_ids_eval,
    )


def compute_tree_shap_interactions_for_fold(
    model,
    X_train,
    X_test,
    feature_names: list[str],
    fold: int,
    background_size: int = 200,
    eval_size: int = 50,
    random_state: int = 42,
) -> ShapInteractionFoldResult:
    rng = np.random.default_rng(random_state)

    X_train_dense = _to_dense_if_sparse(X_train)
    X_test_dense = _to_dense_if_sparse(X_test)

    bg_n = min(background_size, X_train_dense.shape[0])
    bg_idx = rng.choice(X_train_dense.shape[0], size=bg_n, replace=False)
    X_bg = X_train_dense[bg_idx]

    ev_n = min(eval_size, X_test_dense.shape[0])
    ev_idx = rng.choice(X_test_dense.shape[0], size=ev_n, replace=False)
    X_eval = X_test_dense[ev_idx]

    explainer = shap.TreeExplainer(
        model,
        data=X_bg,
        feature_perturbation="interventional",
        model_output="raw",
        feature_names=feature_names,
    )

    interaction_values = explainer.shap_interaction_values(X_eval)

    return ShapInteractionFoldResult(
        fold=fold,
        feature_names=feature_names,
        interaction_values=interaction_values,
    )


def summarize_global_shap(shap_artifacts: list[ShapFoldResult]) -> pd.DataFrame:
    if not shap_artifacts:
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "std_abs_shap"])

    feature_names = shap_artifacts[0].feature_names
    mat = np.vstack([s.mean_abs_shap for s in shap_artifacts])

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mat.mean(axis=0),
                "std_abs_shap": mat.std(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def _get_splitter(
    split_mode: str,
    n_splits: int = 5,
    random_state: int = 42,
):
    if split_mode == "row_kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if split_mode == "task_group_kfold":
        return GroupKFold(n_splits=n_splits)
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def evaluate_model_cv(
    X,
    y,
    model,
    model_name: str = "",
    feature_names: list[str] | None = None,
    split_mode: str = "row_kfold",
    groups: np.ndarray | None = None,
    row_ids: np.ndarray | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
    compute_shap: bool = False,
    shap_background_size: int = 200,
    shap_eval_size: int = 100,
    compute_shap_interactions: bool = False,
    shap_interaction_eval_size: int = 50,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[ShapFoldResult], list[ShapInteractionFoldResult]]:
    splitter = _get_splitter(split_mode, n_splits=cv_folds, random_state=random_state)
    rows = []
    fold_artifacts = []
    shap_artifacts = []
    shap_interaction_artifacts = []

    if split_mode == "task_group_kfold":
        if groups is None:
            raise ValueError("groups must be provided for task_group_kfold.")
        split_iter = splitter.split(X, y, groups=groups)
    else:
        split_iter = splitter.split(X, y)

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        row_ids_test = None if row_ids is None else np.asarray(row_ids[test_idx], dtype=object)

        if model_name == "hist_gbrt":
            X_train_fit = _to_dense_if_sparse(X_train)
            X_test_fit = _to_dense_if_sparse(X_test)
        else:
            X_train_fit = X_train
            X_test_fit = X_test

        model.fit(X_train_fit, y_train)
        y_pred = model.predict(X_test_fit)

        rows.append(
            {
                "fold": fold_idx,
                "model": model_name,
                "r2": r2_score(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
            }
        )
        fold_artifacts.append(
            {
                "fold": fold_idx,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "y_test": y_test,
                "y_pred": y_pred,
                "row_ids_test": row_ids_test,
            }
        )

        if compute_shap:
            if feature_names is None:
                raise ValueError("feature_names must be provided when compute_shap=True")

            shap_fold = compute_tree_shap_for_fold(
                model=model,
                X_train=X_train_fit,
                X_test=X_test_fit,
                y_test=y_test,
                y_pred=y_pred,
                feature_names=feature_names,
                fold=fold_idx,
                background_size=shap_background_size,
                eval_size=shap_eval_size,
                random_state=random_state + fold_idx,
                row_ids_test=row_ids_test,
            )
            shap_artifacts.append(shap_fold)

        if compute_shap_interactions:
            if feature_names is None:
                raise ValueError("feature_names must be provided when compute_shap_interactions=True")

            interaction_fold = compute_tree_shap_interactions_for_fold(
                model=model,
                X_train=X_train_fit,
                X_test=X_test_fit,
                feature_names=feature_names,
                fold=fold_idx,
                background_size=shap_background_size,
                eval_size=shap_interaction_eval_size,
                random_state=random_state + fold_idx,
            )
            shap_interaction_artifacts.append(interaction_fold)

    return pd.DataFrame(rows), fold_artifacts, shap_artifacts, shap_interaction_artifacts


def run_experiments(
    X,
    y,
    models: dict[str, Any],
    feature_names: list[str] | None = None,
    split_mode: str = "row_kfold",
    groups: np.ndarray | None = None,
    row_ids: np.ndarray | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
    compute_shap: bool = False,
    shap_background_size: int = 200,
    shap_eval_size: int = 100,
    compute_shap_interactions: bool = False,
    shap_interaction_eval_size: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    all_rows = []
    artifacts = {}

    for model_name, model in models.items():
        fold_df, fold_artifacts, shap_artifacts, shap_interaction_artifacts = evaluate_model_cv(
            X=X,
            y=y,
            model=model,
            model_name=model_name,
            feature_names=feature_names,
            split_mode=split_mode,
            groups=groups,
            row_ids=row_ids,
            cv_folds=cv_folds,
            random_state=random_state,
            compute_shap=compute_shap,
            shap_background_size=shap_background_size,
            shap_eval_size=shap_eval_size,
            compute_shap_interactions=compute_shap_interactions,
            shap_interaction_eval_size=shap_interaction_eval_size,
        )
        fold_df["split_mode"] = split_mode
        all_rows.append(fold_df)
        artifacts[model_name] = {
            "fold_artifacts": fold_artifacts,
            "shap_artifacts": shap_artifacts,
            "shap_interaction_artifacts": shap_interaction_artifacts,
        }

    result_df = pd.concat(all_rows, ignore_index=True)
    return result_df, artifacts


def summarize_results(result_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        result_df.groupby(["split_mode", "model"])[["r2", "mae", "mse"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary.sort_values(["split_mode", "r2_mean"], ascending=[True, False]).reset_index(drop=True)


def pairwise_wilcoxon_tests(
    result_df: pd.DataFrame,
    metric: str = "mse",
    split_mode: str | None = None,
) -> pd.DataFrame:
    df = result_df.copy()
    if split_mode is not None:
        df = df[df["split_mode"] == split_mode].copy()

    models = sorted(df["model"].unique())
    rows = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = models[i]
            b = models[j]

            x = df.loc[df["model"] == a, metric].values
            y = df.loc[df["model"] == b, metric].values

            if len(x) != len(y):
                continue

            stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")

            rows.append(
                {
                    "split_mode": split_mode if split_mode is not None else "all",
                    "model_a": a,
                    "model_b": b,
                    "metric": metric,
                    "statistic": float(stat),
                    "p_value": float(p),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty or "p_value" not in df.columns:
        return pd.DataFrame(
            columns=["split_mode", "model_a", "model_b", "metric", "statistic", "p_value"]
        )
    return df.sort_values("p_value").reset_index(drop=True)


# ============================================================
# Interpretability
# ============================================================

def compute_permutation_importance(
    trained_model,
    X,
    y,
    feature_names: list[str],
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "neg_mean_squared_error",
) -> pd.DataFrame:
    result = permutation_importance(
        trained_model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def topk_local_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    k: int = 10,
) -> list[list[tuple[str, float]]]:
    out = []
    for row in shap_values:
        idx = np.argsort(np.abs(row))[::-1][:k]
        out.append([(feature_names[i], float(row[i])) for i in idx])
    return out


def is_human_readable_feature(name: str) -> bool:
    if name.startswith("meta::"):
        return True
    if name.startswith("text::emb_"):
        return False
    if name.startswith("text::"):
        return True
    return False


def explanation_readability(topk: list[tuple[str, float]]) -> float:
    if not topk:
        return np.nan
    return float(np.mean([is_human_readable_feature(name) for name, _ in topk]))


def explanation_sparsity(shap_row: np.ndarray, mass: float = 0.8) -> int:
    abs_vals = np.abs(shap_row)
    total = abs_vals.sum()
    if total == 0:
        return 0
    order = np.sort(abs_vals)[::-1]
    csum = np.cumsum(order) / total
    return int(np.searchsorted(csum, mass) + 1)


def interaction_ratio(interactions: np.ndarray) -> float:
    abs_int = np.abs(interactions).mean(axis=0)
    diag = np.diag(abs_int).sum()
    total = abs_int.sum()
    offdiag = total - diag
    return float(offdiag / total) if total > 0 else np.nan


# ============================================================
# OpenML benchmark loading helpers
# ============================================================


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return np.nan
    return len(a & b) / len(union)


def summarize_local_shap_metrics(
    shap_artifacts: list[ShapFoldResult],
    top_k: int = 10,
    sparsity_mass: float = 0.8,
) -> pd.DataFrame:
    rows = []
    for fold_art in shap_artifacts:
        topk_rows = topk_local_features(fold_art.shap_values, fold_art.feature_names, k=top_k)
        for instance_idx, (topk_feats, shap_row) in enumerate(zip(topk_rows, fold_art.shap_values)):
            rows.append(
                {
                    "fold": fold_art.fold,
                    "instance_idx": instance_idx,
                    "readability": explanation_readability(topk_feats),
                    "sparsity": explanation_sparsity(shap_row, mass=sparsity_mass),
                    "top_k": top_k,
                    "sparsity_mass": sparsity_mass,
                }
            )
    return pd.DataFrame(rows)


def summarize_global_topk_stability(
    shap_artifacts: list[ShapFoldResult],
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not shap_artifacts:
        return pd.DataFrame(), pd.DataFrame()

    fold_topks = []
    for fold_art in shap_artifacts:
        idx = np.argsort(fold_art.mean_abs_shap)[::-1][:top_k]
        features = [fold_art.feature_names[i] for i in idx]
        fold_topks.append({"fold": fold_art.fold, "top_features": features})

    topk_df = pd.DataFrame(
        [{"fold": x["fold"], "rank": r + 1, "feature": feat} for x in fold_topks for r, feat in enumerate(x["top_features"])]
    )

    pair_rows = []
    for i in range(len(fold_topks)):
        for j in range(i + 1, len(fold_topks)):
            a = fold_topks[i]
            b = fold_topks[j]
            pair_rows.append(
                {
                    "fold_a": a["fold"],
                    "fold_b": b["fold"],
                    "top_k": top_k,
                    "jaccard": _jaccard(set(a["top_features"]), set(b["top_features"])),
                }
            )

    stability_df = pd.DataFrame(pair_rows)
    return topk_df, stability_df


def summarize_interaction_metrics(
    interaction_artifacts: list[ShapInteractionFoldResult],
) -> pd.DataFrame:
    rows = []
    for fold_art in interaction_artifacts:
        rows.append(
            {
                "fold": fold_art.fold,
                "interaction_ratio": interaction_ratio(fold_art.interaction_values),
            }
        )
    return pd.DataFrame(rows)


def _require_openml():
    try:
        import openml  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Package 'openml' is required. Install it with: pip install openml"
        ) from e


def configure_openml(api_key: str | None = None) -> None:
    _require_openml()
    import openml

    if api_key is None:
        api_key = os.getenv("OPENML_API_KEY")
    if api_key:
        openml.config.apikey = api_key


def load_cc18_from_openml(
    function: str = "predictive_accuracy",
    suite_name: str = "OpenML-CC18",
    cache: DiskCache | None = None,
    api_key: str | None = None,
):
    _require_openml()
    import openml

    cache = cache or DiskCache()
    key = make_cache_key(
        "cc18_bundle_v2_taskwise",
        {"suite_name": suite_name, "function": function},
    )

    def _download():
        configure_openml(api_key)

        suite = openml.study.get_suite(suite_name)
        task_ids = [int(t) for t in suite.tasks]
        tasks_df = openml.tasks.list_tasks(task_id=task_ids, output_format="dataframe")

        eval_frames = []
        print(f"Downloading evaluations for {len(task_ids)} tasks...")

        for i, tid in enumerate(task_ids, start=1):
            try:
                df = openml.evaluations.list_evaluations(
                    function=function,
                    tasks=[tid],
                    size=None,
                    output_format="dataframe",
                )
                if df is not None and len(df) > 0:
                    eval_frames.append(df)
            except Exception as e:
                print(f"Failed to fetch evaluations for task {tid}: {e}")

            if i % 10 == 0 or i == len(task_ids):
                print(f"Fetched evaluations for {i}/{len(task_ids)} tasks")

        if len(eval_frames) == 0:
            raise RuntimeError("No evaluations were downloaded from OpenML.")

        evals_df = pd.concat(eval_frames, ignore_index=True).drop_duplicates()

        return {"suite": suite, "tasks_df": tasks_df, "evals_df": evals_df}

    return cache.get_or_compute(key, _download)


# ============================================================
# High-level pipeline
# ============================================================

def run_cc18_pipeline(
    flows: dict,
    tasks_df: pd.DataFrame,
    evals_df: pd.DataFrame,
    agg_mode: str = "mean",                 # "mean" | "max"
    text_mode: str = "tfidf",              # "tfidf" | "minilm" | "none"
    use_task_metafeatures: bool = True,
    run_row_kfold: bool = True,
    run_task_group_kfold: bool = True,
    cv_folds: int = 5,
    random_state: int = 42,
    compute_shap: bool = False,
    shap_background_size: int = 200,
    shap_eval_size: int = 100,
    compute_shap_interactions: bool = False,
    shap_interaction_eval_size: int = 50,
    shap_top_k: int = 10,
    shap_sparsity_mass: float = 0.8,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> dict[str, Any]:
    cache = DiskCache(cache_dir)
    results_dir = ensure_dir(results_dir)

    # Normalize
    flows_df = normalize_openml_flows_dict(flows)
    tasks_norm_df, task_meta_cols = normalize_cc18_tasks(tasks_df)

    # Filter to relevant sklearn flows only
    flows_df = filter_relevant_sklearn_flows(flows_df, evals_df)

    # Aggregate evaluations
    evals_agg_df = aggregate_cc18_evaluations(
        evals_df=evals_df,
        value_col="value",
        agg_mode=agg_mode,
    )

    # Keep only rows with kept flows
    evals_agg_df = evals_agg_df[
        evals_agg_df["flow_id"].isin(flows_df["flow_id"])
    ].reset_index(drop=True)

    # Build supervised table
    supervised_df = build_cc18_supervised_dataset(
        flows_df=flows_df,
        tasks_df=tasks_norm_df,
        evals_agg_df=evals_agg_df,
        text_col="flow_text_clean",
        target_col="target_value",
    )

    # Build features
    X, y, feature_names, artifacts = build_feature_set(
        supervised_df=supervised_df,
        text_mode=text_mode,
        use_task_metafeatures=use_task_metafeatures,
        text_col="flow_text_clean",
        target_col="target_value",
        cache=cache,
    )

    models = get_models(random_state=random_state)

    if {"task_id", "flow_id"}.issubset(supervised_df.columns):
        row_ids = supervised_df[["task_id", "flow_id"]].astype(str).agg("::".join, axis=1).values
    else:
        row_ids = np.asarray([f"row_{i}" for i in range(len(supervised_df))], dtype=object)

    all_results = []
    all_artifacts = {}

    if run_row_kfold:
        row_results, row_artifacts = run_experiments(
            X=X,
            y=y,
            models=models,
            feature_names=feature_names,
            split_mode="row_kfold",
            groups=None,
            row_ids=row_ids,
            cv_folds=cv_folds,
            random_state=random_state,
            compute_shap=compute_shap,
            shap_background_size=shap_background_size,
            shap_eval_size=shap_eval_size,
            compute_shap_interactions=compute_shap_interactions,
            shap_interaction_eval_size=shap_interaction_eval_size,
        )
        all_results.append(row_results)
        all_artifacts["row_kfold"] = row_artifacts

    if run_task_group_kfold:
        groups = supervised_df["task_id"].values
        task_results, task_artifacts = run_experiments(
            X=X,
            y=y,
            models=models,
            feature_names=feature_names,
            split_mode="task_group_kfold",
            groups=groups,
            row_ids=row_ids,
            cv_folds=cv_folds,
            random_state=random_state,
            compute_shap=compute_shap,
            shap_background_size=shap_background_size,
            shap_eval_size=shap_eval_size,
            compute_shap_interactions=compute_shap_interactions,
            shap_interaction_eval_size=shap_interaction_eval_size,
        )
        all_results.append(task_results)
        all_artifacts["task_group_kfold"] = task_artifacts

    result_df = pd.concat(all_results, ignore_index=True)
    summary_df = summarize_results(result_df)

    wilcoxon_row_df = pairwise_wilcoxon_tests(result_df, metric="mse", split_mode="row_kfold")
    wilcoxon_task_df = pairwise_wilcoxon_tests(result_df, metric="mse", split_mode="task_group_kfold")

    run_name = f"acc_{agg_mode}_{text_mode}_{'meta' if use_task_metafeatures else 'nometa'}"
    run_dir = ensure_dir(results_dir / run_name)

    result_df.to_csv(run_dir / "cv_results.csv", index=False)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    if len(wilcoxon_row_df) > 0:
        wilcoxon_row_df.to_csv(run_dir / "wilcoxon_row_kfold_mse.csv", index=False)
    if len(wilcoxon_task_df) > 0:
        wilcoxon_task_df.to_csv(run_dir / "wilcoxon_task_group_kfold_mse.csv", index=False)

    if compute_shap:
        shap_summary_rows = []
        for split_name, split_artifacts in all_artifacts.items():
            for model_name, model_artifacts in split_artifacts.items():
                shap_df = summarize_global_shap(model_artifacts["shap_artifacts"])
                if len(shap_df) > 0:
                    shap_df.to_csv(run_dir / f"global_shap_{split_name}_{model_name}.csv", index=False)

                local_metrics_df = summarize_local_shap_metrics(
                    model_artifacts["shap_artifacts"],
                    top_k=shap_top_k,
                    sparsity_mass=shap_sparsity_mass,
                )
                if len(local_metrics_df) > 0:
                    local_metrics_df.to_csv(run_dir / f"local_shap_metrics_{split_name}_{model_name}.csv", index=False)
                    shap_summary_rows.append(
                        {
                            "split_mode": split_name,
                            "model": model_name,
                            "top_k": shap_top_k,
                            "sparsity_mass": shap_sparsity_mass,
                            "n_local_explanations": int(len(local_metrics_df)),
                            "readability_mean": float(local_metrics_df["readability"].mean()),
                            "readability_std": float(local_metrics_df["readability"].std(ddof=0)),
                            "sparsity_mean": float(local_metrics_df["sparsity"].mean()),
                            "sparsity_std": float(local_metrics_df["sparsity"].std(ddof=0)),
                        }
                    )

                global_topk_df, stability_df = summarize_global_topk_stability(
                    model_artifacts["shap_artifacts"],
                    top_k=shap_top_k,
                )
                if len(global_topk_df) > 0:
                    global_topk_df.to_csv(run_dir / f"global_shap_topk_{split_name}_{model_name}.csv", index=False)
                if len(stability_df) > 0:
                    stability_df.to_csv(run_dir / f"global_shap_stability_{split_name}_{model_name}.csv", index=False)
                    shap_summary_rows.append(
                        {
                            "split_mode": split_name,
                            "model": model_name,
                            "top_k": shap_top_k,
                            "sparsity_mass": shap_sparsity_mass,
                            "n_local_explanations": int(len(local_metrics_df)),
                            "readability_mean": float(local_metrics_df["readability"].mean()) if len(local_metrics_df) > 0 else np.nan,
                            "readability_std": float(local_metrics_df["readability"].std(ddof=0)) if len(local_metrics_df) > 0 else np.nan,
                            "sparsity_mean": float(local_metrics_df["sparsity"].mean()) if len(local_metrics_df) > 0 else np.nan,
                            "sparsity_std": float(local_metrics_df["sparsity"].std(ddof=0)) if len(local_metrics_df) > 0 else np.nan,
                            "global_topk_stability_mean": float(stability_df["jaccard"].mean()),
                            "global_topk_stability_std": float(stability_df["jaccard"].std(ddof=0)),
                        }
                    )

                if compute_shap_interactions:
                    interaction_df = summarize_interaction_metrics(model_artifacts["shap_interaction_artifacts"])
                    if len(interaction_df) > 0:
                        interaction_df.to_csv(run_dir / f"interaction_metrics_{split_name}_{model_name}.csv", index=False)

        if shap_summary_rows:
            shap_summary_df = pd.DataFrame(shap_summary_rows)
            shap_summary_df = (
                shap_summary_df.sort_values(["split_mode", "model"]).drop_duplicates(subset=["split_mode", "model", "top_k", "sparsity_mass"], keep="last")
            )
            shap_summary_df.to_csv(run_dir / "shap_summary_metrics.csv", index=False)

        with open(run_dir / "shap_artifacts.pkl", "wb") as f:
            pickle.dump(all_artifacts, f)

    joblib.dump(
        {
            "supervised_df": supervised_df,
            "feature_names": feature_names,
            "X": X,
            "y": y,
            "artifacts": artifacts,
            "all_artifacts": all_artifacts,
            "compute_shap": compute_shap,
            "task_meta_cols": task_meta_cols,
            "flows_df": flows_df,
            "tasks_norm_df": tasks_norm_df,
            "evals_agg_df": evals_agg_df,
        },
        run_dir / "pipeline_artifacts.joblib",
    )

    return {
        "run_dir": run_dir,
        "supervised_df": supervised_df,
        "feature_names": feature_names,
        "X": X,
        "y": y,
        "cv_results": result_df,
        "summary": summary_df,
        "wilcoxon_row": wilcoxon_row_df,
        "wilcoxon_task": wilcoxon_task_df,
        "all_artifacts": all_artifacts,
        "compute_shap": compute_shap,
        "shap_top_k": shap_top_k,
        "shap_sparsity_mass": shap_sparsity_mass,
        "flows_df": flows_df,
        "tasks_norm_df": tasks_norm_df,
        "evals_agg_df": evals_agg_df,
    }


# ============================================================
# Multi-configuration runner
# ============================================================

def run_standard_cc18_experiments(
    flows: dict,
    tasks_df: pd.DataFrame,
    evals_df: pd.DataFrame,
    agg_modes: list[str] | None = None,
    configs: list[dict[str, Any]] | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> dict[str, Any]:
    agg_modes = agg_modes or ["mean", "max"]
    configs = configs or [
        {"text_mode": "tfidf", "use_task_metafeatures": False},
        {"text_mode": "none", "use_task_metafeatures": True},
        {"text_mode": "tfidf", "use_task_metafeatures": True},
        # odkomentuj, až budeš chtít embeddings
        # {"text_mode": "minilm", "use_task_metafeatures": False},
        # {"text_mode": "minilm", "use_task_metafeatures": True},
    ]

    outputs = {}

    for agg_mode in agg_modes:
        for cfg in configs:
            out = run_cc18_pipeline(
                flows=flows,
                tasks_df=tasks_df,
                evals_df=evals_df,
                agg_mode=agg_mode,
                text_mode=cfg["text_mode"],
                use_task_metafeatures=cfg["use_task_metafeatures"],
                run_row_kfold=True,
                run_task_group_kfold=True,
                cv_folds=cv_folds,
                random_state=random_state,
                cache_dir=cache_dir,
                results_dir=results_dir,
            )
            key = f"{agg_mode}__{cfg['text_mode']}__{'meta' if cfg['use_task_metafeatures'] else 'nometa'}"
            outputs[key] = out

    return outputs

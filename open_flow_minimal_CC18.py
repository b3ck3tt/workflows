from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
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
            {"n": len(supervised_df), "text_col": text_col},
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
            {"n": len(supervised_df), "text_col": text_col},
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
    }


# ============================================================
# CV
# ============================================================

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
    split_mode: str = "row_kfold",
    groups: np.ndarray | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    splitter = _get_splitter(split_mode, n_splits=cv_folds, random_state=random_state)

    rows = []
    fold_artifacts = []

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

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rows.append(
            {
                "fold": fold_idx,
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
            }
        )

    return pd.DataFrame(rows), fold_artifacts


def run_experiments(
    X,
    y,
    models: dict[str, Any],
    split_mode: str = "row_kfold",
    groups: np.ndarray | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    all_rows = []
    artifacts = {}

    for model_name, model in models.items():
        fold_df, fold_artifacts = evaluate_model_cv(
            X=X,
            y=y,
            model=model,
            split_mode=split_mode,
            groups=groups,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        fold_df["model"] = model_name
        fold_df["split_mode"] = split_mode
        all_rows.append(fold_df)
        artifacts[model_name] = fold_artifacts

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

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


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


# ============================================================
# OpenML benchmark loading helpers
# ============================================================

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
        "cc18_bundle",
        {"suite_name": suite_name, "function": function},
    )

    def _download():
        configure_openml(api_key)

        suite = openml.study.get_suite(suite_name)
        tasks_df = openml.tasks.list_tasks(task_id=suite.tasks, output_format="dataframe")
        evals_df = openml.evaluations.list_evaluations(
            function=function,
            tasks=suite.tasks,
            output_format="dataframe",
        )
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

    all_results = []
    all_artifacts = {}

    if run_row_kfold:
        row_results, row_artifacts = run_experiments(
            X=X,
            y=y,
            models=models,
            split_mode="row_kfold",
            groups=None,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        all_results.append(row_results)
        all_artifacts["row_kfold"] = row_artifacts

    if run_task_group_kfold:
        groups = supervised_df["task_id"].values
        task_results, task_artifacts = run_experiments(
            X=X,
            y=y,
            models=models,
            split_mode="task_group_kfold",
            groups=groups,
            cv_folds=cv_folds,
            random_state=random_state,
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

    joblib.dump(
        {
            "supervised_df": supervised_df,
            "feature_names": feature_names,
            "X": X,
            "y": y,
            "artifacts": artifacts,
            "all_artifacts": all_artifacts,
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

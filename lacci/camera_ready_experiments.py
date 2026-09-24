"""Camera-ready experiments for the LACCI 2026 paper (reviewer 2's three asks).

R2-1 "generalizability beyond MiniLM"
    A dense control that is NOT MiniLM: project the paper's own TF-IDF matrix
    onto 384 latent dimensions with truncated SVD, keeping the lexical
    information but destroying the named coordinates. Run it through the same
    protocol as Tables I and II. If readability collapses while R^2 holds, the
    interpretability cost is a property of latent coordinates, not of MiniLM.

R2-2 "more complex AutoML workflows"
    Characterise how composite the studied workflows already are, and split the
    local explanation metrics by multi-step Pipeline vs. single estimator.

R2-3 "computational overhead"
    Wall-clock cost of each representation: building it, fitting the two
    meta-models on it, and explaining it with TreeSHAP.

Also computes a mass-based readability variant (share of |SHAP| mass carried by
human-readable features), which answers the structural objection that the
count-based readability of Table II is fixed by the representation.

Usage:  python3 camera_ready_experiments.py [--skip-cost] [--skip-svd]
Writes: camera_ready_results.json  (+ a printed summary)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                      # lacci/ -> repo root
# The paper sources live outside the repo tree (papers/ is gitignored); override
# with LACCI_PAPER_DIR if they are kept elsewhere.
PAPER = Path(os.environ.get("LACCI_PAPER_DIR", REPO / "papers" / "lacci2026"))
sys.path.insert(0, str(REPO))

AGG = "max"
RES = REPO / f"results_cc18_{AGG}"
OUT = PAPER / "camera_ready_results.json"

# Protocol constants, identical to _robust_full.py / the module defaults.
N_REPEATS, N_SPLITS = 6, 5
SHAP_BG, SHAP_EVAL, TOP_K = 200, 100, 10
SPARSITY_MASS = 0.8
SVD_DIMS = 384          # matches MiniLM's dimensionality exactly
RANDOM_STATE = 42

MODELS = ("hist_gbrt", "random_forest")


def models():
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "hist_gbrt": HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=300,
            max_leaf_nodes=31, min_samples_leaf=20, early_stopping=True,
            random_state=RANDOM_STATE),
    }


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def artifacts(cfg: str):
    return joblib.load(RES / cfg / "pipeline_artifacts.joblib")


def shap_artifacts(cfg: str):
    with open(RES / cfg / "shap_artifacts.pkl", "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Shared metric code -- deliberately the module's own definitions, imported
# rather than reimplemented, so new rows are comparable with Tables I and II.
# ---------------------------------------------------------------------------
from openml_flow import (  # noqa: E402
    explanation_readability, explanation_sparsity, is_human_readable_feature,
    topk_local_features,
)


def local_metrics(shap_values, feature_names):
    topk = topk_local_features(shap_values, feature_names, k=TOP_K)
    read = [explanation_readability(t) for t in topk]
    spars = [explanation_sparsity(r, mass=SPARSITY_MASS) for r in shap_values]
    return np.array(read, float), np.array(spars, float)


def readable_mass(shap_values, feature_names):
    """Share of total |SHAP| mass carried by human-readable features.

    Unlike the count-based readability of Table II, this is not fixed by the
    representation: it asks how much of the explanation actually lands on
    features a user can name.
    """
    readable = np.array([is_human_readable_feature(f) for f in feature_names])
    a = np.abs(np.asarray(shap_values))
    total = a.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, a[:, readable].sum(axis=1) / total, np.nan)


def mean_pairwise_jaccard(sets):
    js = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if (a | b)]
    return float(np.mean(js)) if js else float("nan")


def topk_via_shap(model, Xtr, Xte, names, rng):
    bg = Xtr[rng.choice(Xtr.shape[0], min(SHAP_BG, Xtr.shape[0]), replace=False)]
    ev = Xte[rng.choice(Xte.shape[0], min(SHAP_EVAL, Xte.shape[0]), replace=False)]
    ex = shap.TreeExplainer(model, data=np.asarray(bg, float),
                            feature_perturbation="interventional", model_output="raw")
    sv = ex.shap_values(np.asarray(ev, float), check_additivity=False)
    idx = np.argsort(np.abs(sv).mean(axis=0))[::-1][:TOP_K]
    return {names[i] for i in idx}


# ---------------------------------------------------------------------------
# R2-3: computational overhead
# ---------------------------------------------------------------------------
def measure_cost() -> dict:
    log("R2-3: representation build cost")
    a = artifacts("tfidf_meta")
    sup = a["supervised_df"]
    texts = sup["flow_text_clean"]
    uniq = texts.drop_duplicates()
    out = {"n_rows": int(len(texts)), "n_unique_texts": int(len(uniq))}

    t = time.perf_counter()
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1, max_df=0.95)
    Xt = vec.fit_transform(texts)
    out["tfidf_build_s"] = time.perf_counter() - t
    out["tfidf_dims"] = int(Xt.shape[1])
    out["tfidf_density"] = float(Xt.nnz / (Xt.shape[0] * Xt.shape[1]))
    out["tfidf_mem_mb"] = float((Xt.data.nbytes + Xt.indices.nbytes + Xt.indptr.nbytes) / 1e6)
    out["tfidf_mem_dense_mb"] = float(Xt.shape[0] * Xt.shape[1] * 8 / 1e6)
    log(f"  TF-IDF: {out['tfidf_build_s']:.1f}s, {out['tfidf_dims']} dims, "
        f"density {out['tfidf_density']:.4f}")

    try:
        from sentence_transformers import SentenceTransformer
        t = time.perf_counter()
        model = SentenceTransformer("all-MiniLM-L6-v2")
        out["minilm_load_s"] = time.perf_counter() - t
        t = time.perf_counter()
        emb = model.encode(texts.tolist(), show_progress_bar=False)
        out["minilm_build_s"] = time.perf_counter() - t
        t = time.perf_counter()
        model.encode(uniq.tolist(), show_progress_bar=False)
        out["minilm_build_unique_s"] = time.perf_counter() - t
        out["minilm_dims"] = int(np.asarray(emb).shape[1])
        out["minilm_mem_mb"] = float(np.asarray(emb).astype(np.float64).nbytes / 1e6)
        log(f"  MiniLM: load {out['minilm_load_s']:.1f}s, encode "
            f"{out['minilm_build_s']:.1f}s ({out['minilm_build_unique_s']:.1f}s deduplicated)")
    except Exception as exc:  # pragma: no cover - environment dependent
        log(f"  MiniLM timing skipped: {exc}")
        out["minilm_error"] = str(exc)

    # Per-fold fit and explanation cost, one deterministic fold per config.
    out["per_fold"] = {}
    for cfg in ("meta_only", "minilm_meta", "tfidf_meta"):
        A = artifacts(cfg)
        X, y = A["X"], A["y"]
        groups = A["supervised_df"]["task_id"].values
        names = A["feature_names"]
        tr, te = next(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))
        rec = {"n_features": int(X.shape[1])}
        for name, m in models().items():
            Xtr, Xte = X[tr], X[te]
            if name == "hist_gbrt" and sparse.issparse(Xtr):
                Xtr, Xte = Xtr.toarray(), Xte.toarray()
            t = time.perf_counter()
            m.fit(Xtr, y[tr])
            rec[f"{name}_fit_s"] = time.perf_counter() - t
            Xtr_d = Xtr.toarray() if sparse.issparse(Xtr) else Xtr
            Xte_d = Xte.toarray() if sparse.issparse(Xte) else Xte
            rng = np.random.default_rng(RANDOM_STATE)
            t = time.perf_counter()
            topk_via_shap(m, Xtr_d, Xte_d, names, rng)
            rec[f"{name}_shap_s"] = time.perf_counter() - t
            log(f"  {cfg}/{name}: fit {rec[f'{name}_fit_s']:.1f}s "
                f"shap {rec[f'{name}_shap_s']:.1f}s")
        out["per_fold"][cfg] = rec
    return out


# ---------------------------------------------------------------------------
# R2-1: SVD dense control
# ---------------------------------------------------------------------------
def build_svd_features():
    A = artifacts("tfidf_meta")
    X, y, names = A["X"], A["y"], A["feature_names"]
    groups = A["supervised_df"]["task_id"].values
    n_text = sum(n.startswith("text::") for n in names)
    assert names[n_text].startswith("meta::"), "text block is not the leading block"
    t = time.perf_counter()
    svd = TruncatedSVD(n_components=SVD_DIMS, random_state=RANDOM_STATE)
    Z = svd.fit_transform(X[:, :n_text])
    build_s = time.perf_counter() - t
    meta = np.asarray(X[:, n_text:].todense()) if sparse.issparse(X) else X[:, n_text:]
    Xs = np.hstack([Z, meta]).astype(np.float64)
    # "emb_" prefix marks a latent coordinate, so the module's readability rule
    # scores these exactly as it scores MiniLM dimensions.
    fnames = [f"text::emb_svd_{i}" for i in range(Z.shape[1])] + \
             [n for n in names if n.startswith("meta::")]
    return Xs, y, groups, fnames, {
        "svd_build_s": build_s,
        "explained_variance_ratio": float(svd.explained_variance_ratio_.sum()),
        "n_features": int(Xs.shape[1]),
    }


def svd_control() -> dict:
    log("R2-1: SVD dense control")
    X, y, groups, names, info = build_svd_features()
    log(f"  {SVD_DIMS} components retain "
        f"{info['explained_variance_ratio']:.1%} of TF-IDF variance "
        f"({info['svd_build_s']:.1f}s)")
    uniq = np.unique(groups)

    # (a) repeated shuffled protocol -> comparable with Table I and Fig. 4
    per = {m: {"r2": [], "mae": [], "mse": [], "stab": []} for m in MODELS}
    for rep in range(N_REPEATS):
        rng = np.random.default_rng(2000 + rep)
        folds = [set(c) for c in np.array_split(rng.permutation(uniq), N_SPLITS)]
        fold_m = {m: {"r2": [], "mae": [], "mse": [], "topk": []} for m in MODELS}
        for fi, test_tasks in enumerate(folds):
            te = np.isin(groups, list(test_tasks)); tr = ~te
            rng_fold = np.random.default_rng(2000 + rep * 100 + fi)
            for name, m in models().items():
                m.fit(X[tr], y[tr])
                pred = m.predict(X[te])
                fold_m[name]["r2"].append(r2_score(y[te], pred))
                fold_m[name]["mae"].append(mean_absolute_error(y[te], pred))
                fold_m[name]["mse"].append(mean_squared_error(y[te], pred))
                fold_m[name]["topk"].append(
                    topk_via_shap(m, X[tr], X[te], names, rng_fold))
        for name in MODELS:
            for k in ("r2", "mae", "mse"):
                per[name][k].append(float(np.mean(fold_m[name][k])))
            per[name]["stab"].append(mean_pairwise_jaccard(fold_m[name]["topk"]))
        log(f"  repeat {rep + 1}/{N_REPEATS}: "
            + " ".join(f"{n} R2={per[n]['r2'][-1]:.3f}" for n in MODELS))

    res = {"info": info, "predictive": {}, "stability": {}}
    for name in MODELS:
        r = per[name]
        res["predictive"][name] = {
            "r2_mean": float(np.mean(r["r2"])), "r2_std": float(np.std(r["r2"])),
            "mae_mean": float(np.mean(r["mae"])), "mae_std": float(np.std(r["mae"])),
            "mse_mean": float(np.mean(r["mse"])), "mse_std": float(np.std(r["mse"])),
        }
        res["stability"][name] = {
            "jaccard_mean": float(np.mean(r["stab"])), "jaccard_std": float(np.std(r["stab"])),
        }

    # Checkpoint: the repeated protocol above costs ~45 min, so persist it
    # before running anything else that could fail.
    _checkpoint("svd_control", res)

    # (b) deterministic split -> local metrics comparable with Table II
    log("  local explanation metrics on the deterministic split")
    res["local"] = {}
    for name, m in models().items():
        reads, spars, masses = [], [], []
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=N_SPLITS).split(X, y, groups)):
            # GroupKFold.split yields index arrays, not boolean masks
            m.fit(X[tr], y[tr])
            rng = np.random.default_rng(RANDOM_STATE + fold)
            bg = X[tr][rng.choice(len(tr), min(SHAP_BG, len(tr)), replace=False)]
            ev = X[te][rng.choice(len(te), min(SHAP_EVAL, len(te)), replace=False)]
            ex = shap.TreeExplainer(m, data=bg, feature_perturbation="interventional",
                                    model_output="raw")
            sv = ex.shap_values(ev, check_additivity=False)
            r, s = local_metrics(sv, names)
            reads.append(r); spars.append(s); masses.append(readable_mass(sv, names))
        reads = np.concatenate(reads); spars = np.concatenate(spars)
        masses = np.concatenate(masses)
        res["local"][name] = {
            "n": int(len(reads)),
            "readability_mean": float(reads.mean()), "readability_std": float(reads.std()),
            "sparsity_mean": float(spars.mean()), "sparsity_std": float(spars.std()),
            "readable_mass_mean": float(np.nanmean(masses)),
            "readable_mass_std": float(np.nanstd(masses)),
        }
        log(f"    {name}: readability {reads.mean():.3f}, sparsity {spars.mean():.2f}, "
            f"readable mass {np.nanmean(masses):.3f}")
        _checkpoint("svd_control", res)
    return res


# ---------------------------------------------------------------------------
# R2-2: workflow complexity, and the mass-based readability of the two
# representations already in the paper (both from cached artifacts).
# ---------------------------------------------------------------------------
PIPELINE_RE = re.compile(r"pipeline\.Pipeline|make_pipeline", re.I)


def workflow_complexity() -> dict:
    log("R2-2: workflow complexity")
    A = artifacts("tfidf_meta")
    sup, flows = A["supervised_df"], A["flows_df"]
    used = flows[flows["flow_id"].isin(sup["flow_id"].unique())].copy()
    used["is_pipeline"] = used["flow_name"].astype(str).str.contains(PIPELINE_RE)
    used["n_components"] = used["flow_name"].astype(str).str.count(r"sklearn\.")
    pipe = used[used["is_pipeline"]]
    out = {
        "n_flows": int(len(used)),
        "n_pipelines": int(used["is_pipeline"].sum()),
        "pipeline_share": float(used["is_pipeline"].mean()),
        "components_median": float(pipe["n_components"].median()),
        "components_max": int(pipe["n_components"].max()),
        "components_p90": float(pipe["n_components"].quantile(0.9)),
    }
    log(f"  {out['n_pipelines']}/{out['n_flows']} flows are multi-step pipelines "
        f"(median {out['components_median']:.0f} components, max {out['components_max']})")

    # Split the cached local explanations by workflow complexity.
    is_pipe = dict(zip(used["flow_id"], used["is_pipeline"]))
    out["by_complexity"] = {}
    for cfg in ("tfidf_meta", "minilm_meta"):
        art = shap_artifacts(cfg)["task_group_kfold"]
        out["by_complexity"][cfg] = {}
        for model in MODELS:
            rows = []
            for fold in art[model]["shap_artifacts"]:
                r, s = local_metrics(fold.shap_values, fold.feature_names)
                mass = readable_mass(fold.shap_values, fold.feature_names)
                flow_ids = [int(str(rid).split("::")[1]) for rid in fold.row_ids]
                rows.append(pd.DataFrame({
                    "readability": r, "sparsity": s, "readable_mass": mass,
                    "is_pipeline": [bool(is_pipe.get(f, False)) for f in flow_ids]}))
            df = pd.concat(rows, ignore_index=True)
            grp = df.groupby("is_pipeline").agg(["mean", "std", "count"])
            rec = {}
            for flag, label in [(True, "pipeline"), (False, "single_estimator")]:
                if flag in grp.index:
                    rec[label] = {
                        "n": int(grp.loc[flag, ("readability", "count")]),
                        "readability": float(grp.loc[flag, ("readability", "mean")]),
                        "sparsity": float(grp.loc[flag, ("sparsity", "mean")]),
                        "readable_mass": float(grp.loc[flag, ("readable_mass", "mean")]),
                    }
            rec["all_readable_mass"] = float(df["readable_mass"].mean())
            rec["all_readable_mass_std"] = float(df["readable_mass"].std(ddof=0))
            out["by_complexity"][cfg][model] = rec
            log(f"  {cfg}/{model}: readable mass {rec['all_readable_mass']:.3f}; "
                + "; ".join(f"{k} sparsity {v['sparsity']:.1f}"
                            for k, v in rec.items() if isinstance(v, dict)))
    return out


def _checkpoint(key: str, value) -> None:
    """Merge one section into the results file immediately."""
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[key] = value
    OUT.write_text(json.dumps(data, indent=2))
    log(f"  checkpointed '{key}'")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cost", action="store_true")
    ap.add_argument("--skip-svd", action="store_true")
    args = ap.parse_args()

    results = {}
    if OUT.exists():
        results = json.loads(OUT.read_text())

    t0 = time.perf_counter()
    results["workflow_complexity"] = workflow_complexity()
    OUT.write_text(json.dumps(results, indent=2))
    if not args.skip_cost:
        results["cost"] = measure_cost()
        OUT.write_text(json.dumps(results, indent=2))
    if not args.skip_svd:
        results["svd_control"] = svd_control()
        OUT.write_text(json.dumps(results, indent=2))
    log(f"done in {(time.perf_counter() - t0) / 60:.1f} min -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

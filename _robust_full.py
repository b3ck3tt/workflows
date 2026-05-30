"""Combined robust run: R2 + MAE + MSE + global top-k stability over many shuffled group folds.

For each (agg, config) using cached features:
  for repeat r:
    - shuffled task->fold assignment, 5 folds
    - per fold: fit RF + HGB, record R2/MAE/MSE on the held-out fold
    - per fold: compute SHAP global top-k feature set (interventional, paper settings)
  per repeat -> 5-fold mean of R2/MAE/MSE per model, and mean pairwise Jaccard of top-k across folds

Writes per agg:
  results_cc18_<agg>/robust_predictive_anchor_v2.csv  -> R2/MAE/MSE mean+/-std (Table I/III)
  results_cc18_<agg>/robust_stability.csv             -> Jaccard mean+/-std (Fig 4)

meta_only is skipped for stability (10 features -> top-10 == all -> Jaccard trivially 1.0).
"""
import gc, time
from itertools import combinations

import numpy as np
import pandas as pd
import joblib
import shap
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

N_REPEATS, N_SPLITS = 6, 5
SHAP_BG, SHAP_EVAL, TOP_K = 200, 100, 10        # match paper SHAP settings
CONFIGS = ["tfidf_meta", "minilm_meta", "meta_only"]
SKIP_STAB = {"meta_only"}


def models():
    return {
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "hist_gbrt": HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05,
            max_iter=300, max_leaf_nodes=31, min_samples_leaf=20, early_stopping=True, random_state=42),
    }


def random_group_folds(uniq, n_splits, rng):
    return [set(c) for c in np.array_split(rng.permutation(uniq), n_splits)]


def mean_pairwise_jaccard(sets):
    js = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if (a | b)]
    return float(np.mean(js)) if js else float("nan")


def topk_features_via_shap(model, Xtr_fit, Xte_fit, feature_names, rng):
    bg_n = min(SHAP_BG, Xtr_fit.shape[0])
    ev_n = min(SHAP_EVAL, Xte_fit.shape[0])
    bg_idx = rng.choice(Xtr_fit.shape[0], bg_n, replace=False)
    ev_idx = rng.choice(Xte_fit.shape[0], ev_n, replace=False)
    X_bg = Xtr_fit[bg_idx]; X_ev = Xte_fit[ev_idx]
    if sparse.issparse(X_bg): X_bg = X_bg.toarray()
    if sparse.issparse(X_ev): X_ev = X_ev.toarray()
    X_bg = np.asarray(X_bg, dtype=np.float64); X_ev = np.asarray(X_ev, dtype=np.float64)
    expl = shap.TreeExplainer(model, data=X_bg, feature_perturbation="interventional", model_output="raw")
    sv = expl.shap_values(X_ev, check_additivity=False)
    mas = np.abs(sv).mean(axis=0)
    idx = np.argsort(mas)[::-1][:TOP_K]
    return set(feature_names[i] for i in idx)


t0 = time.time()
for agg in ["max", "mean"]:
    rows_pred, rows_stab = [], []
    for cfg in CONFIGS:
        print(f"\n[{(time.time()-t0)/60:5.1f}m] {agg} :: {cfg}", flush=True)
        a = joblib.load(f"results_cc18_{agg}/{cfg}/pipeline_artifacts.joblib")
        X, y, fnames = a["X"], a["y"], a["feature_names"]
        groups = a["supervised_df"]["task_id"].values
        uniq = np.unique(groups)
        do_stab = cfg not in SKIP_STAB

        per = {n: {"r2": [], "mae": [], "mse": [], "stab": []} for n in ("random_forest", "hist_gbrt")}

        for rep in range(N_REPEATS):
            rng = np.random.default_rng(2000 + rep)
            folds = random_group_folds(uniq, N_SPLITS, rng)
            fold_metrics = {n: {"r2": [], "mae": [], "mse": [], "topk": []} for n in per}
            for fi, test_tasks in enumerate(folds):
                te = np.isin(groups, list(test_tasks)); tr = ~te
                rng_fold = np.random.default_rng(2000 + rep * 100 + fi)
                for name, m in models().items():
                    Xtr, Xte = X[tr], X[te]
                    if name == "hist_gbrt":
                        Xtr, Xte = Xtr.toarray(), Xte.toarray()
                    m.fit(Xtr, y[tr])
                    yhat = m.predict(Xte)
                    fold_metrics[name]["r2"].append(r2_score(y[te], yhat))
                    fold_metrics[name]["mae"].append(mean_absolute_error(y[te], yhat))
                    fold_metrics[name]["mse"].append(mean_squared_error(y[te], yhat))
                    if do_stab:
                        fold_metrics[name]["topk"].append(
                            topk_features_via_shap(m, Xtr, Xte, fnames, rng_fold)
                        )
            for name in per:
                per[name]["r2"].append(float(np.mean(fold_metrics[name]["r2"])))
                per[name]["mae"].append(float(np.mean(fold_metrics[name]["mae"])))
                per[name]["mse"].append(float(np.mean(fold_metrics[name]["mse"])))
                if do_stab:
                    per[name]["stab"].append(mean_pairwise_jaccard(fold_metrics[name]["topk"]))
            r_rf, r_hgb = per["random_forest"]["r2"][-1], per["hist_gbrt"]["r2"][-1]
            stab_str = ""
            if do_stab:
                stab_str = f" | stab RF {per['random_forest']['stab'][-1]:.3f} HGB {per['hist_gbrt']['stab'][-1]:.3f}"
            print(f"  [{(time.time()-t0)/60:5.1f}m] rep {rep+1}/{N_REPEATS} "
                  f"R2 RF {r_rf:.3f} HGB {r_hgb:.3f}{stab_str}", flush=True)

        for name in per:
            r = per[name]
            rows_pred.append({
                "experiment": cfg, "model": name, "agg": agg, "n_repeats": N_REPEATS,
                "r2_mean": round(np.mean(r["r2"]), 3), "r2_std": round(np.std(r["r2"]), 3),
                "mae_mean": round(np.mean(r["mae"]), 4), "mae_std": round(np.std(r["mae"]), 4),
                "mse_mean": round(np.mean(r["mse"]), 4), "mse_std": round(np.std(r["mse"]), 4),
            })
            if r["stab"]:
                rows_stab.append({
                    "experiment": cfg, "model": name, "agg": agg, "n_repeats": N_REPEATS,
                    "jaccard_mean": round(float(np.mean(r["stab"])), 3),
                    "jaccard_std": round(float(np.std(r["stab"])), 3),
                })
        del a, X, y; gc.collect()

    pd.DataFrame(rows_pred).to_csv(f"results_cc18_{agg}/robust_predictive_anchor_v2.csv", index=False)
    if rows_stab:
        pd.DataFrame(rows_stab).to_csv(f"results_cc18_{agg}/robust_stability.csv", index=False)
    print(f"--- wrote {agg} CSVs ---", flush=True)

print(f"\nROBUST FULL DONE in {(time.time()-t0)/60:.0f}m", flush=True)

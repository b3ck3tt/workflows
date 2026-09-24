"""Stage 1 of #3: cross-toolkit on CC18 (zero new downloads).
Sweep toolkit unions and measure heterogeneity (task_dependence) vs adaptation value
(model / kNN / global_default), to test whether more algorithm diversity activates
data-dependence. LambdaMART ranker; openml_full metafeatures for meta/kNN."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import openml_flow as wf

b = wf.load_cc18_from_openml(cache=wf.DiskCache("minimal_cache_cc18"))
flows = wf.load_or_cache_flows(b["evals_df"], "minimal_cache_cc18/flows.joblib")
qual = wf.load_dataset_qualities()
cache = wf.DiskCache("minimal_cache_cc18")

UNIONS = [
    ("sklearn", ("sklearn",)),
    ("+weka", ("sklearn", "weka")),
    ("+mlr", ("sklearn", "weka", "mlr", "classif")),
    ("all", None),
]
CONFIGS = [
    {"name": "tfidf_meta", "text_mode": "tfidf", "use_task_metafeatures": True},
    {"name": "tfidf_only", "text_mode": "tfidf", "use_task_metafeatures": False},
]
rows = []
for label, tk in UNIONS:
    print(f"\n### toolkits = {label} ###", flush=True)
    ds = wf.build_cc18_dataset(flows, b["tasks_df"], b["evals_df"], agg_mode="max",
            text_mode="none", use_task_metafeatures=True, toolkits=tk,
            metafeature_set="openml_full", qualities=qual, min_coverage=0.7, cache=cache)
    sup = ds["supervised_df"]
    td = wf.task_dependence_report(sup)
    # best-flow toolkit spread: which toolkit owns each task's best flow
    tk_of = sup.drop_duplicates("flow_id").set_index("flow_id")["toolkit"]
    best_tk = td["per_task_best"]["best_flow_id"].map(tk_of).value_counts()

    out = wf.run_recommender_evaluation(
        flows=flows, tasks_df=b["tasks_df"], evals_df=b["evals_df"],
        configs=CONFIGS, agg_mode="max", model_names=["lambdamart"],
        cv_folds=5, n_random_shuffles=10, include_knn=True, knn_k=5,
        toolkits=tk, metafeature_set="openml_full", qualities=qual, min_coverage=0.7,
        cache_dir="minimal_cache_cc18", results_dir=f"results_recommender_tk_{label.replace('+','plus').replace(' ','')}",
    )
    s = out["summary"]
    def nreg(exp, meth):
        r = s[(s.experiment == exp) & (s.method == meth)]["nregret@1_mean"]
        return float(r.iloc[0]) if len(r) else np.nan
    def hit(exp, meth):
        r = s[(s.experiment == exp) & (s.method == meth)]["hit@1_mean"]
        return float(r.iloc[0]) if len(r) else np.nan
    w = out["wilcoxon"]
    knn_p = w[(w.method == "knn") & (w.baseline == "global_default") & (w.metric == "nregret@1")]
    knn_p = float(knn_p.sort_values("n_tasks").iloc[-1]["p_value"]) if len(knn_p) else np.nan

    rows.append({
        "toolkits": label,
        "n_flows": int(sup["flow_id"].nunique()),
        "n_pairs": len(sup),
        "cross_task_spearman": round(td["cross_task_spearman_mean"], 3),
        "n_distinct_best": td["n_distinct_best_flows"],
        "top_best_share": round(td["top_best_flow_share"], 3),
        "best_flow_toolkits": dict(best_tk.head(4)),
        "model(lmart,tfidf_meta)": round(nreg("tfidf_meta", "lambdamart"), 4),
        "text_only(lmart)": round(nreg("tfidf_only", "lambdamart"), 4),
        "kNN": round(nreg("tfidf_meta", "knn"), 4),
        "global_default": round(nreg("tfidf_meta", "global_default"), 4),
        "kNN_vs_default_p": round(knn_p, 4),
        "model_hit@1": round(hit("tfidf_meta", "lambdamart"), 3),
    })

res = pd.DataFrame(rows)
res.to_csv("results_recommender_toolkit_sweep.csv", index=False)
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
print("\n================ CROSS-TOOLKIT SWEEP (CC18, LambdaMART, openml_full meta) ================")
print(res.drop(columns=["best_flow_toolkits"]).to_string(index=False))
print("\nbest-flow toolkit spread per union:")
for r in rows:
    print(f"  {r['toolkits']:8s}: {r['best_flow_toolkits']}")
print("\nDONE")

"""Powered Paper 2 headline: full ~1,236-task long-tail. task_dependence + recommender
comparison (does kNN reach significance vs the portfolio at full power?). Basic metafeatures
(available for all tasks); rich-metadata refinement is a separate step."""
import warnings; warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
import openml_flow as wf

state = joblib.load("minimal_cache_cc18/longtail_collected.joblib")
tasks_all = joblib.load("minimal_cache_cc18/longtail_tasks.joblib")

evals = pd.concat(state["evals"].values(), ignore_index=True)
evals["flow_id"] = pd.to_numeric(evals["flow_id"], errors="coerce")
evals["value"] = pd.to_numeric(evals["value"], errors="coerce")
evals = evals.dropna(subset=["flow_id", "value"]); evals["flow_id"] = evals["flow_id"].astype(int)
uf = evals.drop_duplicates("flow_id")[["flow_id", "flow_name"]]
flows = {int(r.flow_id): {"id": int(r.flow_id), "name": (r.flow_name or ""),
                          "full_name": (r.flow_name or ""), "version": ""} for r in uf.itertuples()}
tids = set(int(t) for t in state["evals"].keys())
tasks_df = tasks_all[tasks_all["tid"].astype(int).isin(tids)].copy()
print(f"FULL long-tail: {len(tids)} tasks | {evals['flow_id'].nunique()} flows | "
      f"{evals.drop_duplicates(['task_id','flow_id']).shape[0]} pairs", flush=True)

# heterogeneity of the full set
ds = wf.build_cc18_dataset(flows, tasks_df, evals, agg_mode="max", text_mode="none",
        use_task_metafeatures=True, toolkits=None, cache=wf.DiskCache("minimal_cache_cc18"))
td = wf.task_dependence_report(ds["supervised_df"])
print(f"\n=== HETEROGENEITY (full long-tail vs CC18) ===")
print(f"  full long-tail: n={td['n_tasks']} Spearman={td['cross_task_spearman_mean']:.3f} "
      f"distinct_best={td['n_distinct_best_flows']} top_share={td['top_best_flow_share']:.3f}")
print(f"  CC18 ref      : n=72  Spearman=0.642 distinct_best=25 top_share=0.250")

out = wf.run_recommender_evaluation(
    flows=flows, tasks_df=tasks_df, evals_df=evals,
    configs=[{"name": "tfidf_meta", "text_mode": "tfidf", "use_task_metafeatures": True},
             {"name": "tfidf_only", "text_mode": "tfidf", "use_task_metafeatures": False}],
    agg_mode="max", model_names=["lambdamart"], cv_folds=5, n_random_shuffles=10,
    include_knn=True, knn_k=5, toolkits=None,
    cache_dir="minimal_cache_cc18", results_dir="results_recommender_longtail_full",
)
s = out["summary"]; w = out["wilcoxon"]
def g(exp, m, col="nregret@1_mean"):
    r = s[(s.experiment == exp) & (s.method == m)][col]; return float(r.iloc[0]) if len(r) else np.nan
print("\n=== POWERED COMPARISON (full long-tail, LambdaMART, all toolkits, agg=max) ===")
for m in ["lambdamart", "knn", "global_default", "random"]:
    print(f"  {m:16s}: nregret@1={g('tfidf_meta',m):.4f}  hit@1={g('tfidf_meta',m,'hit@1_mean'):.3f}")
print(f"  {'lambdamart(text)':16s}: nregret@1={g('tfidf_only','lambdamart'):.4f}")
print("\nWilcoxon vs global_default (nregret@1) -- KEY: kNN significance at full power:")
print(w[(w.baseline=='global_default')&(w.metric=='nregret@1')]
      [['method','median_diff','win_rate','p_value']].round(4).to_string(index=False))
print("\nn_tasks in comparison:", int(s['n_tasks'].max()))
print("DONE")

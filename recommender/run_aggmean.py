"""Robustness: does the headline recommendation result hold under agg=mean (typical
accuracy) as well as agg=max (best-case)? Same protocol, sklearn CC18, task_group_kfold."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import openml_flow as wf

b = wf.load_cc18_from_openml(cache=wf.DiskCache("minimal_cache_cc18"))
flows = wf.load_or_cache_flows(b["evals_df"], "minimal_cache_cc18/flows.joblib")

CONFIGS = [
    {"name": "tfidf_meta", "text_mode": "tfidf", "use_task_metafeatures": True},
    {"name": "tfidf_only", "text_mode": "tfidf", "use_task_metafeatures": False},
]
order = {"lambdamart":0,"extra_trees":1,"knn":2,"global_default":3,"random":4}

for agg in ["max", "mean"]:
    print(f"\n############### agg_mode = {agg} ###############", flush=True)
    ds = wf.build_cc18_dataset(flows, b["tasks_df"], b["evals_df"], agg_mode=agg,
            text_mode="none", use_task_metafeatures=True, cache=wf.DiskCache("minimal_cache_cc18"))
    td = wf.task_dependence_report(ds["supervised_df"])
    print(f"task-dependence: cross-task Spearman={td['cross_task_spearman_mean']:.3f} "
          f"distinct_best={td['n_distinct_best_flows']} top_share={td['top_best_flow_share']:.3f}")

    out = wf.run_recommender_evaluation(
        flows=flows, tasks_df=b["tasks_df"], evals_df=b["evals_df"],
        configs=CONFIGS, agg_mode=agg, model_names=["extra_trees","lambdamart"],
        cv_folds=5, n_random_shuffles=20, include_knn=True, knn_k=5,
        cache_dir="minimal_cache_cc18", results_dir=f"results_recommender_agg_{agg}",
    )
    s = out["summary"]
    # headline: tfidf_meta for model methods + kNN + default + random (kNN/default/random from tfidf_meta rows)
    sub = s[s.experiment=="tfidf_meta"][["method","nregret@1_mean","hit@1_mean","spearman_mean"]].copy()
    # add tfidf_only lambdamart (best config) as reference
    to = s[(s.experiment=="tfidf_only")&(s.method=="lambdamart")][["method","nregret@1_mean","hit@1_mean","spearman_mean"]].copy()
    to["method"] = "lambdamart(text_only)"
    sub = pd.concat([sub, to], ignore_index=True)
    sub["o"] = sub["method"].map(lambda m: order.get(m, 9))
    sub = sub.sort_values("o").drop(columns="o")
    print(sub.round(4).to_string(index=False))
    w = out["wilcoxon"]
    ww = w[(w.baseline=="global_default")&(w.metric=="nregret@1")][["method","median_diff","win_rate","p_value"]]
    print("Wilcoxon vs global_default (nregret@1):")
    print(ww.round(4).to_string(index=False))
print("\nDONE")

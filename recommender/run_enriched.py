"""Option A: does LLM-authored semantic enrichment of flow text beat plain TF-IDF?
Compares text_mode tfidf vs tfidf_enriched (LambdaMART) on sklearn-only and all-toolkits.
Enrichment aligns algorithm families across toolkits, so the biggest gain (if any) is
expected in the all-toolkits pool."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from scipy.stats import wilcoxon
import openml_flow as wf

b = wf.load_cc18_from_openml(cache=wf.DiskCache("minimal_cache_cc18"))
flows = wf.load_or_cache_flows(b["evals_df"], "minimal_cache_cc18/flows.joblib")
cache = wf.DiskCache("minimal_cache_cc18")

CONFIGS = [
    {"name": "tfidf_only",    "text_mode": "tfidf",          "use_task_metafeatures": False},
    {"name": "enriched_only", "text_mode": "tfidf_enriched", "use_task_metafeatures": False},
    {"name": "tfidf_meta",    "text_mode": "tfidf",          "use_task_metafeatures": True},
    {"name": "enriched_meta", "text_mode": "tfidf_enriched", "use_task_metafeatures": True},
]
POOLS = [("sklearn", ("sklearn",)), ("all", None)]

rows, pertask = [], {}
for plabel, tk in POOLS:
    print(f"\n### pool={plabel} ###", flush=True)
    out = wf.run_recommender_evaluation(
        flows=flows, tasks_df=b["tasks_df"], evals_df=b["evals_df"],
        configs=CONFIGS, agg_mode="max", model_names=["lambdamart"],
        cv_folds=5, n_random_shuffles=10, include_knn=False,
        toolkits=tk, cache_dir="minimal_cache_cc18",
        results_dir=f"results_recommender_enriched_{plabel}",
    )
    s = out["summary"]
    for exp in ["tfidf_only","enriched_only","tfidf_meta","enriched_meta"]:
        r = s[(s.experiment==exp)&(s.method=="lambdamart")]
        rows.append({"pool":plabel,"config":exp,
                     "nregret@1":round(float(r["nregret@1_mean"].iloc[0]),4),
                     "hit@1":round(float(r["hit@1_mean"].iloc[0]),3),
                     "ndcg@5":round(float(r["ndcg@5_mean"].iloc[0]),4),
                     "spearman":round(float(r["spearman_mean"].iloc[0]),3)})
    pertask[plabel] = out["per_task"]

res = pd.DataFrame(rows)
pd.set_option("display.width",200)
print("\n================ ENRICHED vs PLAIN TF-IDF (LambdaMART) ================")
print(res.to_string(index=False))

print("\n--- paired Wilcoxon: enriched vs plain (nregret@1), per pool ---")
for plabel in ["sklearn","all"]:
    pt = pertask[plabel]
    for base_only, enr in [("tfidf_only","enriched_only"),("tfidf_meta","enriched_meta")]:
        a = pt[(pt.method=="lambdamart")&(pt.experiment==enr)][["task_id","nregret@1"]].rename(columns={"nregret@1":"enr"})
        c = pt[(pt.method=="lambdamart")&(pt.experiment==base_only)][["task_id","nregret@1"]].rename(columns={"nregret@1":"base"})
        m = a.merge(c,on="task_id"); d = m["enr"].to_numpy()-m["base"].to_numpy()
        p = wilcoxon(m["enr"],m["base"]).pvalue if np.any(d!=0) else float("nan")
        print(f"  {plabel:8s} {enr:14s} vs {base_only:12s}: enr={m['enr'].mean():.4f} base={m['base'].mean():.4f} "
              f"diff={d.mean():+.4f} (enr better if<0) win={ (d<0).mean():.2f} p={p:.3f}")
print("\nDONE")

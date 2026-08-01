"""Paper 4 significance: paired Wilcoxon on nregret@1 for each frozen LLM vs baselines (portfolio, kNN,
trained text-LTR) on the SAME tasks, plus the Opus-vs-Sonnet model-tier test. Bootstrap 95% CIs."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
RNG = np.random.default_rng(0)
OUT = "results_recommender_paper4"

L = pd.read_csv(f"{OUT}/llm_metrics_per_task.csv")   # model, tid, nregret@1, hit@1, spearman
PAPER = ["claude-sonnet-5","claude-haiku-4-5-20251001","claude-opus-4-8"]
L = L[L.model.isin(PAPER)]
# per-task baselines (long-tail full, tfidf_meta) for the LLM tasks
pt = pd.read_csv("results_recommender_longtail_full/ranking_metrics_per_task.csv")
pt = pt[pt.experiment=="tfidf_meta"]
base = {m: pt[pt.method==m].set_index("task_id")["nregret@1"] for m in ["global_default","knn","lambdamart"]}

def ci(x):
    x=np.asarray(x); bs=[np.mean(RNG.choice(x,len(x),True)) for _ in range(5000)]
    return x.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)

print("=== nregret@1: frozen LLM vs baselines (paired Wilcoxon, matched tasks) ===")
short={"claude-sonnet-5":"Sonnet","claude-haiku-4-5-20251001":"Haiku","claude-opus-4-8":"Opus"}
for m in PAPER:
    lm = L[L.model==m].set_index("tid")["nregret@1"]
    mean,lo,hi = ci(lm.values)
    print(f"\n{short[m]} (n={len(lm)}): nregret@1 {mean:.4f}  95%CI[{lo:.4f},{hi:.4f}]")
    for bname,blab in [("global_default","portfolio"),("knn","kNN"),("lambdamart","trained text-LTR")]:
        b = base[bname]; common = sorted(set(lm.index)&set(b.index))
        a = lm.reindex(common).values; c = b.reindex(common).values
        w = wilcoxon(a,c)
        sign = "worse than" if a.mean()>c.mean() else "better than"
        print(f"    vs {blab:>16} ({c.mean():.4f}): LLM {sign} baseline, p={w.pvalue:.2e}, win={np.mean(a<c):.2f}")

# model-tier tests (paired across the same tasks)
print("\n=== model-tier tests (paired Wilcoxon on nregret@1) ===")
piv = L.pivot(index="tid", columns="model", values="nregret@1").dropna()
for a,b in [("claude-opus-4-8","claude-sonnet-5"),("claude-opus-4-8","claude-haiku-4-5-20251001"),
            ("claude-sonnet-5","claude-haiku-4-5-20251001")]:
    w = wilcoxon(piv[a],piv[b])
    print(f"  {short[a]} ({piv[a].mean():.4f}) vs {short[b]} ({piv[b].mean():.4f}): p={w.pvalue:.3f} "
          f"{'(n.s.)' if w.pvalue>0.05 else '(sig)'}")
# spearman by model (ordering quality)
print("\n=== ordering quality (mean Spearman) ===")
for m in PAPER: print(f"  {short[m]:>7}: {L[L.model==m]['spearman'].mean():.3f}")
print("\nDONE")

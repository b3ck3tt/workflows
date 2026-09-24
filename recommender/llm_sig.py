"""Paper 4 significance (cross-vendor): paired Wilcoxon on nregret@1 for EACH frozen LLM vs baselines
(portfolio, kNN, trained text-LTR) on the SAME tasks; BH correction across the 'vs portfolio' family
(the central 'no frozen LLM beats the portfolio' claim); tier/family pairwise tests; bootstrap 95% CIs."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
RNG = np.random.default_rng(0)
OUT = "results_recommender_paper4"

LABEL = {
    "claude-opus-4-8": "Opus", "claude-sonnet-5": "Sonnet", "claude-haiku-4-5-20251001": "Haiku",
    "claude-fable-5": "Fable", "claude-sonnet-5-think": "Sonnet-think",
    "gpt-5": "GPT-5", "gpt-5-mini": "GPT-5-mini",
    "deepinfra_deepseek-ai_DeepSeek-V3": "DeepSeek-V3",
    "deepinfra_Qwen_Qwen2.5-72B-Instruct": "Qwen-72B",
    "deepinfra_meta-llama_Llama-3.3-70B-Instruct": "Llama-70B",
    "deepinfra_meta-llama_Meta-Llama-3.1-8B-Instruct": "Llama-8B",
}
def lab(m): return LABEL.get(m, m)

L = pd.read_csv(f"{OUT}/llm_metrics_per_task.csv")     # model, tid, nregret@1, hit@1, spearman
models = [m for m in L.model.unique()]
# order by mean nregret@1 (best first)
models = sorted(models, key=lambda m: L[L.model == m]["nregret@1"].mean())

pt = pd.read_csv("results_recommender_longtail_full/ranking_metrics_per_task.csv")
pt = pt[pt.experiment == "tfidf_meta"]
base = {m: pt[pt.method == m].set_index("task_id")["nregret@1"] for m in ["global_default", "knn", "lambdamart"]}

def ci(x):
    x = np.asarray(x); bs = [np.mean(RNG.choice(x, len(x), True)) for _ in range(5000)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def bh(pvals):  # Benjamini-Hochberg adjusted p-values
    p = np.asarray(pvals); n = len(p); order = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        r = n - rank
        prev = min(prev, p[idx] * n / r); adj[idx] = prev
    return adj

print("=== nregret@1: each frozen LLM vs baselines (paired Wilcoxon, matched tasks) ===")
port_p = []      # collect 'vs portfolio' p-values for BH correction
for m in models:
    lm = L[L.model == m].set_index("tid")["nregret@1"]
    mean, lo, hi = ci(lm.values)
    print(f"\n{lab(m)} (n={len(lm)}): nregret@1 {mean:.4f}  95%CI[{lo:.4f},{hi:.4f}]")
    for bname, blab in [("global_default", "portfolio"), ("knn", "kNN"), ("lambdamart", "trained text-LTR")]:
        b = base[bname]; common = sorted(set(lm.index) & set(b.index))
        a = lm.reindex(common).values; c = b.reindex(common).values
        w = wilcoxon(a, c)
        sign = "worse than" if a.mean() > c.mean() else "better than"
        print(f"    vs {blab:>16} ({c.mean():.4f}): LLM {sign} baseline, p={w.pvalue:.2e}, win={np.mean(a < c):.2f}")
        if bname == "global_default": port_p.append((m, w.pvalue, a.mean() > c.mean()))

# BH correction across the 'vs portfolio' family: is ANY frozen LLM significantly BETTER than portfolio?
print("\n=== 'no frozen LLM beats the portfolio' — BH-corrected across all models (vs portfolio) ===")
adj = bh([p for _, p, _ in port_p])
any_better = False
for (m, p, worse), pa in sorted(zip(port_p, adj), key=lambda z: z[1]):
    tag = "worse" if worse else "BETTER"
    if not worse and pa < 0.05: any_better = True
    print(f"  {lab(m):>13}: raw p={p:.2e}  BH p={pa:.2e}  ({tag} than portfolio)")
print(f"  => {'SOME LLM beats portfolio' if any_better else 'NO frozen LLM significantly beats the portfolio'} (BH<0.05)")

# tier / family pairwise tests on common tasks
print("\n=== tier / family pairwise tests (paired Wilcoxon on nregret@1, common tasks) ===")
def pair(a, b, note=""):
    if a not in set(L.model) or b not in set(L.model): return
    x = L[L.model == a].set_index("tid")["nregret@1"]; y = L[L.model == b].set_index("tid")["nregret@1"]
    common = sorted(set(x.index) & set(y.index))
    xa = x.reindex(common).values; yb = y.reindex(common).values
    w = wilcoxon(xa, yb)
    print(f"  {lab(a):>12} ({xa.mean():.4f}) vs {lab(b):<12} ({yb.mean():.4f}) [n={len(common)}]: "
          f"p={w.pvalue:.3f} {'(n.s.)' if w.pvalue > 0.05 else '(sig)'}  {note}")
pair("claude-opus-4-8", "claude-sonnet-5", "Claude tier")
pair("claude-opus-4-8", "claude-haiku-4-5-20251001", "Claude tier")
pair("claude-sonnet-5", "claude-haiku-4-5-20251001", "Claude tier")
pair("gpt-5", "gpt-5-mini", "OpenAI tier")
pair("deepinfra_meta-llama_Llama-3.3-70B-Instruct", "deepinfra_meta-llama_Meta-Llama-3.1-8B-Instruct", "Llama size")
pair("claude-sonnet-5", "claude-sonnet-5-think", "thinking ablation")
pair("gpt-5", "deepinfra_deepseek-ai_DeepSeek-V3", "best proprietary vs best open")

print("\n=== ordering quality (mean Spearman, best first) ===")
for m in sorted(models, key=lambda m: -L[L.model == m]["spearman"].mean()):
    print(f"  {lab(m):>13}: {L[L.model == m]['spearman'].mean():.3f}")
print("\nDONE")

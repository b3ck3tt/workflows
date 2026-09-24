"""Paper 4 thinking ablation + per-task effort analysis.
(1) Sonnet thinking vs no-thinking, paired on 200 tasks, vs baselines + Fable.
(2) Effort (tok_out) vs task difficulty (n_candidates, n_classes, atypicality) and vs accuracy."""
import warnings; warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
OUT="results_recommender_paper4"
d=joblib.load(f"{OUT}/pack.joblib"); truth=d["truth"]

def per_task(model):
    r=joblib.load(f"{OUT}/llm_{model}.joblib"); rows={}
    for tid,v in r.items():
        tid=int(tid)
        if "ranking" not in v or not v["ranking"] or tid not in truth: continue
        acc={x["cat"]:x["acc"] for x in truth[tid]}
        best=max(acc.values()); worst=min(acc.values()); rng=best-worst
        if rng<=0: continue
        rank=v["ranking"]; pos={c:i for i,c in enumerate(rank)}
        common=[c for c in acc if c in pos]
        rho=spearmanr([pos[c] for c in common],[acc[c] for c in common]).correlation if len(common)>2 else np.nan
        rows[tid]={"nregret":(best-acc.get(rank[0],worst))/rng,"spearman":-rho if rho==rho else np.nan,
                   "tok_out":v.get("tok_out",np.nan),"n_cand":v.get("n_cand",len(acc))}
    return rows

nt=per_task("claude-sonnet-5"); th=per_task("claude-sonnet-5-think")
common=sorted(set(nt)&set(th))
a=np.array([th[t]["nregret"] for t in common]); b=np.array([nt[t]["nregret"] for t in common])
print("=== (1) THINKING ABLATION: Sonnet think vs no-think (paired, n=%d) ==="%len(common))
print(f"  no-think nregret@1 {b.mean():.4f} | think {a.mean():.4f} | "
      f"paired Wilcoxon p={wilcoxon(a,b).pvalue:.4g} | think-better win={np.mean(a<b):.2f}")
print(f"  spearman: no-think {np.nanmean([nt[t]['spearman'] for t in common]):.3f} | "
      f"think {np.nanmean([th[t]['spearman'] for t in common]):.3f}")

# baselines + fable for context
pt=pd.read_csv("results_recommender_longtail_full/ranking_metrics_per_task.csv")
pt=pt[(pt.experiment=="tfidf_meta")&(pt.task_id.isin(common))]
print("\n  baselines on same tasks:")
print("   ",pt.groupby("method")["nregret@1"].mean().round(4).to_dict())
fb=per_task("claude-fable-5"); fbn=[fb[t]["nregret"] for t in fb]
print(f"    fable(n={len(fbn)}) nregret {np.mean(fbn):.4f}")

# === (2) EFFORT ANALYSIS (sonnet-think) ===
print("\n=== (2) EFFORT (tok_out) analysis, Sonnet-think 200 tasks ===")
tid_list=[t for t in th if not np.isnan(th[t]["tok_out"])]
eff=np.array([th[t]["tok_out"] for t in tid_list])
ncand=np.array([th[t]["n_cand"] for t in tid_list])
nreg=np.array([th[t]["nregret"] for t in tid_list])
# n_classes + atypicality
b2=joblib.load("minimal_cache_cc18/cc18_bundle_v2_taskwise_a27a0c13de3118edc992e8a1afa833de.joblib")
# use long-tail tasks_df for classes
ltt=joblib.load("minimal_cache_cc18/longtail_tasks.joblib").set_index("tid")
ncls=np.array([ltt.loc[t,"NumberOfClasses"] if t in ltt.index else np.nan for t in tid_list])
# atypicality from long-tail agg
coll=joblib.load("minimal_cache_cc18/longtail_collected.joblib")
ev=pd.concat(coll["evals"].values(),ignore_index=True)
ev["flow_id"]=pd.to_numeric(ev["flow_id"],errors="coerce"); ev["value"]=pd.to_numeric(ev["value"],errors="coerce")
ev=ev.dropna(subset=["flow_id","value"])
agg=ev.groupby(["task_id","flow_id"],as_index=False)["value"].max().rename(columns={"value":"t"})
gm=agg.groupby("flow_id")["t"].mean(); atypd={}
for tid,g in agg.groupby("task_id"):
    if len(g)<5: continue
    x=gm.loc[g.flow_id].to_numpy(); y=g.t.to_numpy()
    if np.std(x)>0 and np.std(y)>0:
        rr=spearmanr(y,x).correlation; atypd[tid]=1-(rr if rr==rr else 0)
atyp=np.array([atypd.get(t,np.nan) for t in tid_list])
print(f"  effort tok_out: mean {eff.mean():.0f}, median {np.median(eff):.0f}, range {eff.min():.0f}-{eff.max():.0f}")
def corr(x,y,lab):
    m=~(np.isnan(x)|np.isnan(y))
    if m.sum()>5: print(f"    Spearman(effort, {lab}) = {spearmanr(x[m],y[m]).correlation:+.3f}  (n={m.sum()})")
corr(eff,ncand,"n_candidates"); corr(eff,ncls,"n_classes"); corr(eff,atyp,"atypicality")
corr(eff,nreg,"nregret@1 (does more thinking help?)")
# does thinking help more where it thought harder? split by effort tercile: think-vs-nothink gain
gain=np.array([nt[t]["nregret"]-th[t]["nregret"] for t in tid_list])  # >0 = thinking better
q1,q2=np.percentile(eff,[33,67])
for lab,msk in [("low-effort",eff<=q1),("high-effort",eff>=q2)]:
    print(f"    {lab} tasks: think-vs-nothink gain {gain[msk].mean():+.4f} (n={msk.sum()})")
print("DONE")

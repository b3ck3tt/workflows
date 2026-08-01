"""Fix E, done consistently on the CANONICAL harness outputs (results_recommender_ltfull_*), the same
runs that produced Paper 2's tables. Per-task nregret@1 for kNN and the portfolio under basic /
landmarking / openml_full metadata, tfidf_meta experiment, all 1236 long-tail tasks.

(1) metadata comparison on the COMMON task set: does richer metadata lower kNN regret / beat portfolio?
(2) factorial gain ~ H(atypicality tercile) * M(metadata), interaction test + task-clustered SE + CIs.
"""
import warnings; warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
import statsmodels.formula.api as smf

RNG=np.random.default_rng(0)
def load(ms):
    pt=pd.read_csv(f"results_recommender_ltfull_{ms}/ranking_metrics_per_task.csv")
    pt=pt[pt.experiment=="tfidf_meta"]
    knn=pt[pt.method=="knn"].set_index("task_id")["nregret@1"]
    port=pt[pt.method=="global_default"].set_index("task_id")["nregret@1"]
    return knn,port
K={}; P={}
for ms in ["basic","landmarking","openml_full"]:
    K[ms],P[ms]=load(ms)
port=P["basic"]  # metadata-independent

# atypicality per task (method-independent heterogeneity)
coll=joblib.load("minimal_cache_cc18/longtail_collected.joblib")
ev=pd.concat(coll["evals"].values(),ignore_index=True)
ev["flow_id"]=pd.to_numeric(ev["flow_id"],errors="coerce"); ev["value"]=pd.to_numeric(ev["value"],errors="coerce")
ev=ev.dropna(subset=["flow_id","value"]); ev["flow_id"]=ev["flow_id"].astype(int)
agg=ev.groupby(["task_id","flow_id"],as_index=False)["value"].max().rename(columns={"value":"t"})
gm=agg.groupby("flow_id")["t"].mean(); atyp={}
for tid,g in agg.groupby("task_id"):
    if len(g)<5: continue
    a=gm.loc[g.flow_id].to_numpy(); b=g.t.to_numpy()
    if np.std(a)>0 and np.std(b)>0:
        r=spearmanr(b,a).correlation; atyp[tid]=1.0-(r if r==r else 0.0)
atyp=pd.Series(atyp,name="atyp")

# sanity: reproduce tab:meta means (portfolio should be metadata-independent across runs)
common=set(K["basic"].index)&set(K["landmarking"].index)&set(K["openml_full"].index)&set(port.index)&set(atyp.index)
common=sorted(common); print(f"common tasks: {len(common)}")
print("\n(1) metadata comparison on the COMMON task set (canonical harness):")
print(f"{'metadata':>12} {'kNN':>8} {'portfolio':>10} {'kNN win%':>9} {'kNN vs port p':>14} {'vs basic p':>11}")
bk=K["basic"].reindex(common).to_numpy()
for ms in ["basic","landmarking","openml_full"]:
    k=K[ms].reindex(common).to_numpy(); pt=port.reindex(common).to_numpy()
    win=float(np.mean(k<pt)); pvp=wilcoxon(k,pt).pvalue
    vsb="--" if ms=="basic" else f"{wilcoxon(k,bk).pvalue:.4f}"
    # portfolio value per run (should match)
    pv_ms=P[ms].reindex(common).to_numpy()
    print(f"{ms:>12} {k.mean():>8.4f} {pt.mean():>10.4f} {win:>9.2f} {pvp:>14.4f} {vsb:>11}   (port_run={pv_ms.mean():.4f})")

# (2) factorial on basic vs landmarking (real 45-feature contrast), atypicality terciles
tv=atyp.reindex(common); q1,q2=tv.quantile([1/3,2/3])
rows=[]
for t in common:
    a=atyp[t]; h="low" if a<=q1 else ("high" if a>=q2 else "mid")
    if h=="mid": continue
    rows.append((t,a,h,"basic",port[t]-K["basic"][t]))
    rows.append((t,a,h,"landmarking",port[t]-K["landmarking"][t]))
df=pd.DataFrame(rows,columns=["task_id","atyp","hetero","meta","gain"])
print(f"\n(2) factorial n={df.task_id.nunique()} tasks; terciles @ {q1:.3f}/{q2:.3f}")
def boot(x,B=2000):
    x=np.asarray(x); return x.mean(),*np.percentile([np.mean(RNG.choice(x,len(x),True)) for _ in range(B)],[2.5,97.5])
print(f"{'hetero':>6} {'meta':>12} {'n':>5} {'gain':>8}  95% CI              Wilcoxon p")
for h in ["low","high"]:
    for m in ["basic","landmarking"]:
        c=df[(df.hetero==h)&(df.meta==m)]["gain"].to_numpy(); mn,lo,hi=boot(c)
        nz=c[c!=0]; wp=wilcoxon(nz).pvalue if len(nz)>10 else float("nan")
        print(f"{h:>6} {m:>12} {len(c):>5} {mn:>+8.4f}  [{lo:+.4f},{hi:+.4f}]  p={wp:.4f}")
df["H"]=(df.hetero=="high").astype(int); df["M"]=(df.meta=="landmarking").astype(int)
mod=smf.ols("gain ~ H*M",data=df).fit(cov_type="cluster",cov_kwds={"groups":df.task_id})
print("\nfactorial OLS gain ~ H*M (task-clustered SE):")
ci=mod.conf_int()
for t in mod.params.index:
    print(f"  {t:>10}: {mod.params[t]:>+8.4f} [{ci.loc[t,0]:+.4f},{ci.loc[t,1]:+.4f}] p={mod.pvalues[t]:.4f}")
print("DONE")

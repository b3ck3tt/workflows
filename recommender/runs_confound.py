"""Fix G: is agg=max a #runs artifact? Quantify how much the number of uploaded runs per (task,flow)
inflates the max target, and whether the recommender just ranks high-#run flows.

Tests:
  1. within-task Spearman(n_runs, target) under agg=max vs agg=mean — max should correlate with n_runs
     (more draws -> higher max) if the confound is real; mean should not.
  2. a shrinkage/bias-corrected 'best' = mean + z*sd/sqrt(n) is n-robust; how far does the flow RANKING
     move (Spearman between max-ranking and shrinkage-ranking within task)? If high, agg choice is benign.
  3. does the LTR recommender's top-1 pick just track n_runs? corr(recommended rank, n_runs).
"""
import warnings; warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from scipy.stats import spearmanr
import openml_flow as wf

b = joblib.load("minimal_cache_cc18/cc18_bundle_v2_taskwise_a27a0c13de3118edc992e8a1afa833de.joblib")
ev = b["evals_df"].copy()
ev["flow_id"]=pd.to_numeric(ev["flow_id"],errors="coerce"); ev["value"]=pd.to_numeric(ev["value"],errors="coerce")
ev=ev.dropna(subset=["flow_id","value","task_id"]); ev["flow_id"]=ev["flow_id"].astype(int)
# keep sklearn flows only (P1 setting)
flows=joblib.load("minimal_cache_cc18/flows.joblib")
skl={int(fid) for fid,f in flows.items() if str(f.get("name","")).startswith("sklearn.")}
ev=ev[ev.flow_id.isin(skl)]
g=ev.groupby(["task_id","flow_id"])["value"]
agg=pd.DataFrame({"n":g.size(),"mx":g.max(),"mean":g.mean(),"sd":g.std().fillna(0.0)}).reset_index()
print(f"{agg.task_id.nunique()} tasks, {len(agg)} (task,flow) cells; median n_runs={agg.n.median():.0f}, "
      f"mean={agg.n.mean():.1f}, p90={agg.n.quantile(.9):.0f}",flush=True)

# 1. within-task Spearman(n, target)
def within_task_corr(col):
    rs=[]
    for t,gr in agg.groupby("task_id"):
        if gr.n.nunique()>2 and len(gr)>5:
            r=spearmanr(gr.n,gr[col]).correlation
            if r==r: rs.append(r)
    return np.mean(rs),np.std(rs),len(rs)
for col in ["mx","mean"]:
    m,s,k=within_task_corr(col)
    print(f"within-task Spearman(n_runs, {col:>4}): mean={m:+.3f}  sd={s:.3f}  (n={k} tasks)")

# 2. shrinkage-corrected best vs max: does the RANKING move?
z=1.0
agg["shrink"]=agg["mean"]+z*agg["sd"]/np.sqrt(agg["n"])
rank_moves=[]
for t,gr in agg.groupby("task_id"):
    if len(gr)>5:
        r=spearmanr(gr.mx,gr.shrink).correlation
        if r==r: rank_moves.append(r)
print(f"\nSpearman(max-ranking, shrinkage-ranking) within task: mean={np.mean(rank_moves):.3f} "
      f"(1.0 = agg choice does not change the order)")
# how often does the argmax flow change?
chg=0; tot=0
for t,gr in agg.groupby("task_id"):
    if len(gr)>5:
        tot+=1
        if gr.loc[gr.mx.idxmax(),"flow_id"]!=gr.loc[gr.shrink.idxmax(),"flow_id"]: chg+=1
print(f"top-1 flow changes under shrinkage in {chg}/{tot} tasks ({100*chg/tot:.0f}%)")

# 3. does the LTR recommender track n_runs? build dataset + rank, corr(rank, n)
ds=wf.build_cc18_dataset(flows,b["tasks_df"],b["evals_df"],agg_mode="max",text_mode="tfidf",
                         use_task_metafeatures=True,toolkits=("sklearn",),cache=wf.DiskCache("minimal_cache_cc18"))
sup=ds["supervised_df"].reset_index(drop=True); X=ds["X"].tocsr()
from lightgbm import LGBMRanker
RNG=np.random.default_rng(0); tasks=sup.task_id.unique(); RNG.shuffle(tasks)
folds=np.array_split(tasks,5)
def graded(y):
    r=y.max()-y.min(); return np.round(31*(y-y.min())/r).astype(int) if r>0 else np.zeros(len(y),int)
ncorr=[]
nkey=agg.set_index(["task_id","flow_id"])["n"]
for fi in range(5):
    test=set(folds[fi]); tr=~sup.task_id.isin(test).to_numpy()
    tri=np.where(tr)[0]; o=np.argsort(sup.loc[tr,"task_id"].to_numpy(),kind="stable")
    yt=sup.loc[tr,"target_value"].to_numpy()[o]; gt=sup.loc[tr,"task_id"].to_numpy()[o]
    _,sz=np.unique(gt,return_counts=True)
    rel=np.concatenate([graded(yt[s:s+n]) for s,n in zip(np.cumsum([0]+list(sz[:-1])),sz)])
    rk=LGBMRanker(objective="lambdarank",n_estimators=300,learning_rate=0.05,num_leaves=31,
                  min_child_samples=20,random_state=0,n_jobs=-1,label_gain=[float(i) for i in range(32)],verbose=-1)
    rk.fit(X[tri][o],rel,group=list(sz))
    for t in test:
        gg=sup[sup.task_id==t]
        if len(gg)<6: continue
        sc=rk.predict(X[gg.index.to_numpy()])
        ns=np.array([nkey.get((t,f),1) for f in gg.flow_id])
        if len(set(ns))>2:
            r=spearmanr(sc,ns).correlation
            if r==r: ncorr.append(r)
print(f"\nSpearman(recommender score, n_runs) within held-out task: mean={np.mean(ncorr):+.3f} "
      f"(near 0 = recommender is NOT just ranking by popularity)")
print("DONE")

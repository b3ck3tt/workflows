"""Fix C on the long-tail (P3 deployment setting). Same full-pool budgeted protocol as fullpool.py,
on the heterogeneous long-tail pool. Subsampled test tasks for tractability (full pool is large)."""
import warnings; warnings.filterwarnings("ignore")
import sys, joblib, numpy as np, pandas as pd
from lightgbm import LGBMRanker
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
import openml_flow as wf

RNG = np.random.default_rng(0)
MIN_CAND=12; BUDGETS=[5,10,25]; K=5; TOL=0.01
coll = joblib.load("minimal_cache_cc18/longtail_collected.joblib")
tasks_all = joblib.load("minimal_cache_cc18/longtail_tasks.joblib")
ev = pd.concat(coll["evals"].values(), ignore_index=True)
ev["flow_id"]=pd.to_numeric(ev["flow_id"],errors="coerce"); ev["value"]=pd.to_numeric(ev["value"],errors="coerce")
ev=ev.dropna(subset=["flow_id","value"]); ev["flow_id"]=ev["flow_id"].astype(int)
flows={int(f):{"id":int(f),"name":n or "","full_name":n or "","version":""}
       for f,n in ev.drop_duplicates("flow_id")[["flow_id","flow_name"]].itertuples(index=False)}
tids=set(coll["evals"].keys()); tasks_df=tasks_all[tasks_all["tid"].astype(int).isin(tids)].copy()
ds=wf.build_cc18_dataset(flows,tasks_df,ev,agg_mode="max",text_mode="tfidf",use_task_metafeatures=True,
                         toolkits=None,cache=wf.DiskCache("minimal_cache_cc18"))
sup=ds["supervised_df"].reset_index(drop=True); X=ds["X"].tocsr(); meta_cols=ds["task_meta_cols"]
cc=sup.groupby("task_id").size(); keep=cc[cc>=MIN_CAND].index
mask=sup.task_id.isin(set(keep)).to_numpy(); sup=sup[mask].reset_index(drop=True); X=X[mask]
n_text=X.shape[1]-len(meta_cols) if len(meta_cols) else X.shape[1]
tasks=sup.task_id.unique(); RNG.shuffle(tasks)
nte=int(len(tasks)*0.25); test_all=list(tasks[:nte]); train=set(tasks[nte:])
test=test_all[:150]  # subsample test tasks for tractability
print(f"{len(tasks)} tasks pool, {sup.flow_id.nunique()} flows; {len(train)} train, {len(test)} test(sub)",flush=True)

tr=sup.task_id.isin(train).to_numpy()
flow_mean=sup.loc[tr].groupby("flow_id")["target_value"].mean(); gmean=float(sup.loc[tr,"target_value"].mean())
pool_flows=flow_mean.index.to_numpy()
tm=sup.drop_duplicates("task_id").set_index("task_id")[meta_cols].astype(float)
imp=SimpleImputer(strategy="median"); scl=StandardScaler(); train_tasks=sorted(train)
Mtr=scl.fit_transform(imp.fit_transform(tm.loc[train_tasks]))
def graded(y):
    r=y.max()-y.min(); return np.round(31*(y-y.min())/r).astype(int) if r>0 else np.zeros(len(y),int)
tri=np.where(tr)[0]; o=np.argsort(sup.loc[tr,"task_id"].to_numpy(),kind="stable")
Xtr=X[tri][o]; yt=sup.loc[tr,"target_value"].to_numpy()[o]; gt=sup.loc[tr,"task_id"].to_numpy()[o]
_,sz=np.unique(gt,return_counts=True)
rel=np.concatenate([graded(yt[s:s+n]) for s,n in zip(np.cumsum([0]+list(sz[:-1])),sz)])
rk=LGBMRanker(objective="lambdarank",n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,
              random_state=0,n_jobs=-1,label_gain=[float(i) for i in range(32)],verbose=-1)
rk.fit(Xtr,rel,group=list(sz))
posmap=dict(zip(sup.loc[tr].drop_duplicates("flow_id")["flow_id"],
                sup.loc[tr].drop_duplicates("flow_id").index.to_numpy()))
pool=[f for f in pool_flows if f in posmap]
Xpool_text=X[[posmap[f] for f in pool]]  # text rows for whole pool (meta overwritten per task)

MODELS=["text_ltr","portfolio","kNN","random"]
acc={"cand":{m:[] for m in MODELS}}; acc.update({f"full@{B}":{m:[] for m in MODELS} for B in BUDGETS})
depth={m:[] for m in MODELS}; hitk={m:[] for m in MODELS}
for ii,tid in enumerate(test):
    g=sup[sup.task_id==tid]
    if g.target_value.max()-g.target_value.min()<=0 or len(g)<MIN_CAND: continue
    rng=g.target_value.max()-g.target_value.min(); best=g.target_value.max()
    eval_flows=set(g.flow_id); truth=dict(zip(g.flow_id,g.target_value))
    def reg(f): return (best-truth[f])/rng
    tsc_c=dict(zip(g.flow_id, rk.predict(X[g.index.to_numpy()])))
    psc_c={f:float(flow_mean.get(f,gmean)) for f in g.flow_id}
    v=scl.transform(imp.transform(tm.loc[[tid]]))
    d=np.linalg.norm(Mtr-v,axis=1); nn=[train_tasks[i] for i in np.argsort(d)[:5]]
    neigh=sup[sup.task_id.isin(nn)].groupby("flow_id")["target_value"].mean()
    ksc_c={f:neigh.get(f,gmean) for f in g.flow_id}
    for m,sc in [("text_ltr",tsc_c),("portfolio",psc_c),("kNN",ksc_c),("random",{f:RNG.random() for f in g.flow_id})]:
        acc["cand"][m].append(reg(max(sc,key=sc.get)))
    # full pool: paste raw meta of tid
    Xp=Xpool_text.tolil()
    if len(meta_cols):
        Xp[:,n_text:]=np.tile(X[g.index[0]].toarray().ravel()[n_text:],(len(pool),1))
    Xp=Xp.tocsr()
    tsc_f=dict(zip(pool,rk.predict(Xp)))
    psc_f={f:float(flow_mean.get(f,gmean)) for f in pool}
    ksc_f={f:neigh.get(f,gmean) for f in pool}; rsc_f={f:RNG.random() for f in pool}
    for m,sc in [("text_ltr",tsc_f),("portfolio",psc_f),("kNN",ksc_f),("random",rsc_f)]:
        order=sorted(pool,key=lambda f:sc[f],reverse=True)
        w=0
        for f in order:
            if f in eval_flows: break
            w+=1
        depth[m].append(w)
        hitk[m].append(1.0 if any(f in eval_flows and reg(f)<=TOL for f in order[:K]) else 0.0)
        for B in BUDGETS:
            evb=[f for f in order[:B] if f in eval_flows]
            acc[f"full@{B}"][m].append((best-max(truth[f] for f in evb))/rng if evb else 1.0)
    if (ii+1)%30==0: print(f"  {ii+1}/{len(test)}",flush=True)

print("\n=== full-pool (deployment) vs candidate-only — LONG-TAIL, nregret ===")
print(f"{'model':>10} {'cand@1':>8} "+" ".join(f"{'full@'+str(B):>8}" for B in BUDGETS)+f" {'walk-dep':>9} {'hit@5':>7}")
for m in MODELS:
    print(f"{m:>10} {np.mean(acc['cand'][m]):>8.4f} "+" ".join(f"{np.mean(acc[f'full@{B}'][m]):>8.4f}" for B in BUDGETS)
          +f" {np.mean(depth[m]):>9.1f} {np.mean(hitk[m]):>7.2f}")
for B in BUDGETS:
    tf=np.array(acc[f"full@{B}"]["text_ltr"]); pf=np.array(acc[f"full@{B}"]["portfolio"])
    print(f"full@{B}: text_ltr {tf.mean():.4f} vs portfolio {pf.mean():.4f}  Wilcoxon p={wilcoxon(tf,pf).pvalue:.4g}")
print("DONE")

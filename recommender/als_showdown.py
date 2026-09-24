"""Phase-0 fix B: FAITHFUL, TUNED collaborative-filtering baseline vs text-LTR.
Biased ALS probabilistic matrix factorization (Salakhutdinov-Mnih MAP): global+flow+task biases +
low-rank U,V via alternating least squares; rank d and reg lambda TUNED on held-out train entries.
Cold-start for held-out tasks via (a) metafeature-regressed factors and (b) fold-in. Same k-revealed
sweep as before, so results are directly comparable to the earlier weak-SGD showdown."""
import warnings; warnings.filterwarnings("ignore")
import sys, joblib, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from lightgbm import LGBMRanker
import openml_flow as wf

RNG = np.random.default_rng(0)
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
KS = [0, 1, 2, 3, 5, 10]; REVEAL = 5; MIN_CAND = 12

coll = joblib.load("minimal_cache_cc18/longtail_collected.joblib")
tasks_all = joblib.load("minimal_cache_cc18/longtail_tasks.joblib")
ev = pd.concat(coll["evals"].values(), ignore_index=True)
ev["flow_id"] = pd.to_numeric(ev["flow_id"], errors="coerce"); ev["value"] = pd.to_numeric(ev["value"], errors="coerce")
ev = ev.dropna(subset=["flow_id","value"]); ev["flow_id"] = ev["flow_id"].astype(int)
flows = {int(f): {"id":int(f),"name":n or "","full_name":n or "","version":""}
         for f,n in ev.drop_duplicates("flow_id")[["flow_id","flow_name"]].itertuples(index=False)}
tids = set(coll["evals"].keys()); tasks_df = tasks_all[tasks_all["tid"].astype(int).isin(tids)].copy()
ds = wf.build_cc18_dataset(flows, tasks_df, ev, agg_mode="max", text_mode="tfidf",
                           use_task_metafeatures=True, toolkits=None, cache=wf.DiskCache("minimal_cache_cc18"))
sup = ds["supervised_df"].reset_index(drop=True); X = ds["X"].tocsr(); meta_cols = ds["task_meta_cols"]
cc = sup.groupby("task_id").size(); keep = cc[cc>=MIN_CAND].index
if LIMIT: keep = list(keep)[:LIMIT]
mask = sup.task_id.isin(set(keep)).to_numpy(); sup = sup[mask].reset_index(drop=True); X = X[mask]
tasks = sup.task_id.unique(); RNG.shuffle(tasks)
nte = int(len(tasks)*0.25); test = set(tasks[:nte]); train = set(tasks[nte:])
tr = sup.task_id.isin(train).to_numpy(); te = sup.task_id.isin(test).to_numpy()
print(f"{len(tasks)} tasks ({len(train)}tr/{len(test)}te)", flush=True)

train_flows = sorted(sup.loc[tr,"flow_id"].unique()); fidx={f:i for i,f in enumerate(train_flows)}; nF=len(train_flows)
train_tasks = sorted(train); tkx={t:i for i,t in enumerate(train_tasks)}; nT=len(train_tasks)

def biased_residual(rows):
    mu = rows["target_value"].mean()
    bf = ((rows.assign(r=rows.target_value-mu).groupby("flow_id")["r"].sum()) /
          (rows.groupby("flow_id").size()+5)).to_dict()
    rows2 = rows.assign(r2=rows.target_value-mu-rows.flow_id.map(lambda f: bf.get(f,0.0)))
    bt = ((rows2.groupby("task_id")["r2"].sum())/(rows2.groupby("task_id").size()+5)).to_dict()
    return mu, bf, bt

def als(rows, d, lam, iters=15):
    mu,bf,bt = biased_residual(rows)
    fi = rows.flow_id.map(fidx).to_numpy(); ti = rows.task_id.map(tkx).to_numpy()
    e = (rows.target_value - mu - rows.flow_id.map(lambda f: bf.get(f,0.0))
         - rows.task_id.map(lambda t: bt.get(t,0.0))).to_numpy()
    U = 0.1*RNG.standard_normal((nF,d)); V = 0.1*RNG.standard_normal((nT,d))
    # group indices
    byf = [[] for _ in range(nF)]; byt=[[] for _ in range(nT)]
    for k,(f,t) in enumerate(zip(fi,ti)): byf[f].append(k); byt[t].append(k)
    for _ in range(iters):
        for t in range(nT):
            idx=byt[t]
            if not idx: continue
            A=U[fi[idx]]; y=e[idx]
            V[t]=np.linalg.solve(A.T@A+lam*np.eye(d), A.T@y)
        for f in range(nF):
            idx=byf[f]
            if not idx: continue
            A=V[ti[idx]]; y=e[idx]
            U[f]=np.linalg.solve(A.T@A+lam*np.eye(d), A.T@y)
    return mu,bf,bt,U,V

# tune (d,lam) on held-out train entries
trrows = sup.loc[tr & sup.flow_id.isin(set(train_flows)).to_numpy()].reset_index(drop=True)
val_mask = RNG.random(len(trrows))<0.1
fitrows=trrows[~val_mask]; valrows=trrows[val_mask]
best=None
for d in [10,20,40]:
    for lam in [0.05,0.2,1.0]:
        mu,bf,bt,U,V=als(fitrows,d,lam)
        pv=[mu+bf.get(f,0)+bt.get(t,0)+U[fidx[f]]@V[tkx[t]] for f,t in zip(valrows.flow_id,valrows.task_id) if t in tkx]
        yv=[v for f,t,v in zip(valrows.flow_id,valrows.task_id,valrows.target_value) if t in tkx]
        rmse=float(np.sqrt(np.mean((np.array(pv)-np.array(yv))**2)))
        if best is None or rmse<best[0]: best=(rmse,d,lam)
_,D,LAM=best; print(f"tuned CF: rank={D} lambda={LAM} valRMSE={best[0]:.4f}", flush=True)
mu,bf,bt,U,V=als(trrows,D,LAM,iters=20)

# metareg factors from metafeatures
tm = sup.drop_duplicates("task_id").set_index("task_id")[meta_cols].astype(float)
from sklearn.impute import SimpleImputer; from sklearn.preprocessing import StandardScaler
imp=SimpleImputer(strategy="median"); scl=StandardScaler()
Mtr=scl.fit_transform(imp.fit_transform(tm.loc[train_tasks]))
regV=Ridge(alpha=1.0).fit(Mtr,V); regbt=Ridge(alpha=1.0).fit(Mtr,np.array([bt.get(t,0.0) for t in train_tasks]))

# text-LTR
def graded(y):
    r=y.max()-y.min(); return np.round(31*(y-y.min())/r).astype(int) if r>0 else np.zeros(len(y),int)
tri=np.where(tr)[0]; o=np.argsort(sup.loc[tr,"task_id"].to_numpy(),kind="stable")
Xtr=X[tri][o]; yt=sup.loc[tr,"target_value"].to_numpy()[o]; gt=sup.loc[tr,"task_id"].to_numpy()[o]
_,sz=np.unique(gt,return_counts=True)
rel=np.concatenate([graded(yt[s:s+n]) for s,n in zip(np.cumsum([0]+list(sz[:-1])),sz)])
rk=LGBMRanker(objective="lambdarank",n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,
              random_state=0,n_jobs=-1,label_gain=[float(i) for i in range(32)],verbose=-1)
rk.fit(Xtr,rel,group=list(sz))
text=np.full(len(sup),np.nan); text[te]=rk.predict(X[np.where(te)[0]])
flow_mean=sup.loc[tr].groupby("flow_id")["target_value"].mean(); gmean=float(sup.loc[tr,"target_value"].mean())

def foldin(rf,rv):
    A=np.array([np.concatenate([U[fidx[f]],[1.0]]) for f in rf]); y=np.array([rv[j]-mu-bf.get(f,0) for j,f in enumerate(rf)])
    sol=np.linalg.solve(A.T@A+LAM*np.eye(A.shape[1]),A.T@y); return sol[:D],sol[D]

res={m:{k:[] for k in KS} for m in ["text_ltr","cf_metareg","cf_foldin","kNN","portfolio","random"]}
for tid in sorted(test):
    g=sup[sup.task_id==tid]; cand=g[g.flow_id.isin(set(train_flows))]
    if len(cand)<MIN_CAND: continue
    fids=cand.flow_id.to_numpy(); truth=dict(zip(fids,cand.target_value.to_numpy())); rng=cand.target_value.max()-cand.target_value.min()
    if rng<=0: continue
    best=cand.target_value.max()
    tsc=dict(zip(fids,text[cand.index.to_numpy()]))
    v=scl.transform(imp.transform(tm.loc[[tid]])); vmr=regV.predict(v)[0]; bmr=float(regbt.predict(v)[0])
    msc={f: mu+bf.get(f,0)+bmr+U[fidx[f]]@vmr for f in fids}
    # kNN
    d=np.linalg.norm(Mtr-v,axis=1); nn=[train_tasks[i] for i in np.argsort(d)[:5]]
    neigh=sup[sup.task_id.isin(nn)].groupby("flow_id")["target_value"].mean(); ksc={f:neigh.get(f,gmean) for f in fids}
    psc={f:float(flow_mean.get(f,gmean)) for f in fids}
    def reg_on(rem,sc):
        return (max(truth[f] for f in rem)-truth[max(rem,key=lambda f:sc[f])])/rng if rem else np.nan
    for k in KS:
        for _ in range(REVEAL if k>0 else 1):
            if k==0: revealed=set()
            elif k>=len(fids): continue
            else: revealed=set(RNG.choice(fids,size=k,replace=False))
            rem=[f for f in fids if f not in revealed]
            if len(rem)<2: continue
            res["text_ltr"][k].append(reg_on(rem,tsc)); res["cf_metareg"][k].append(reg_on(rem,msc))
            res["kNN"][k].append(reg_on(rem,ksc)); res["portfolio"][k].append(reg_on(rem,psc))
            res["random"][k].append(reg_on(rem,{f:RNG.random() for f in rem}))
            if k>0:
                vf,bfi=foldin(list(revealed),[truth[f] for f in revealed])
                res["cf_foldin"][k].append(reg_on(rem,{f:mu+bf.get(f,0)+bfi+U[fidx[f]]@vf for f in rem}))
print("\n=== TUNED-CF (ALS-PMF) showdown vs text-LTR (nregret@1 on remaining) ===")
print(f"{'k':>3} "+" ".join(f"{m:>11}" for m in res))
for k in KS:
    print(f"{k:>3} "+" ".join(f"{np.mean([x for x in res[m][k] if x==x]):>11.4f}" if [x for x in res[m][k] if x==x] else f"{'n/a':>11}" for m in res))
print("DONE")

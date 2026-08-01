"""Phase-0 fix C: full-pool (deployment-setting) ranking vs candidate-only (leaky) ranking.

Critique (P1/P3): the recommender is only ever asked to rank the pre-filtered set of flows already
*evaluated* on the held-out task. In deployment you don't know that set; you must rank the ENTIRE
training-flow pool and pay for ranking un-evaluable flows highly.

Protocol.
  candidate-only (leaky, current):  rank only flows evaluated on the held-out task; nregret@1 of top pick.
  full-pool (deployment):           score EVERY training-pool flow; walk the ranking top-down; the pick
                                    you can realize is the highest-ranked flow that happens to be
                                    evaluated on the held-out task. nregret@1 of that realized pick vs
                                    the best evaluable flow. Also report the rank-depth you had to walk
                                    (how many un-evaluable flows sit above the first evaluable one) and
                                    hit@k = P(a top-k full-pool pick is evaluable AND within regret 0.01).

Compares text-LTR / portfolio (global flow mean) / kNN(meta) / random. Leave-tasks-out CV on CC18.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, joblib, numpy as np, pandas as pd
from lightgbm import LGBMRanker
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import openml_flow as wf

RNG = np.random.default_rng(0)
b = joblib.load("minimal_cache_cc18/cc18_bundle_v2_taskwise_a27a0c13de3118edc992e8a1afa833de.joblib")
flows = joblib.load("minimal_cache_cc18/flows.joblib")
ds = wf.build_cc18_dataset(flows, b["tasks_df"], b["evals_df"], agg_mode="max", text_mode="tfidf",
                           use_task_metafeatures=True, toolkits=("sklearn",),
                           cache=wf.DiskCache("minimal_cache_cc18"))
sup = ds["supervised_df"].reset_index(drop=True); X = ds["X"].tocsr(); meta_cols = ds["task_meta_cols"]
print(f"{sup.task_id.nunique()} tasks, {sup.flow_id.nunique()} flows, {len(sup)} (task,flow) rows", flush=True)

def graded(y):
    r = y.max()-y.min(); return np.round(31*(y-y.min())/r).astype(int) if r>0 else np.zeros(len(y),int)

NF = 5; tasks = sup.task_id.unique(); RNG.shuffle(tasks); folds = np.array_split(tasks, NF)
TOL = 0.01; BUDGETS = [5, 10, 25]
MODELS = ["text_ltr","portfolio","kNN","random"]
acc = {"cand":{m:[] for m in MODELS}}
acc.update({f"full@{B}":{m:[] for m in MODELS} for B in BUDGETS})
depth = {m:[] for m in MODELS}
hitk = {m:[] for m in MODELS}; K=5

for fi in range(NF):
    test = set(folds[fi]); train = set(t for t in tasks if t not in test)
    tr = sup.task_id.isin(train).to_numpy(); te = sup.task_id.isin(test).to_numpy()
    # ---- portfolio + kNN meta model on train ----
    flow_mean = sup.loc[tr].groupby("flow_id")["target_value"].mean(); gmean=float(sup.loc[tr,"target_value"].mean())
    pool_flows = flow_mean.index.to_numpy()               # the FULL training pool of flows
    tm = sup.drop_duplicates("task_id").set_index("task_id")[meta_cols].astype(float)
    imp=SimpleImputer(strategy="median"); scl=StandardScaler()
    train_tasks=sorted(train); Mtr=scl.fit_transform(imp.fit_transform(tm.loc[train_tasks]))
    # per-flow mean by neighbour tasks precomputed lazily per test task
    # ---- text-LTR ----
    tri=np.where(tr)[0]; o=np.argsort(sup.loc[tr,"task_id"].to_numpy(),kind="stable")
    Xtr=X[tri][o]; yt=sup.loc[tr,"target_value"].to_numpy()[o]; gt=sup.loc[tr,"task_id"].to_numpy()[o]
    _,sz=np.unique(gt,return_counts=True)
    rel=np.concatenate([graded(yt[s:s+n]) for s,n in zip(np.cumsum([0]+list(sz[:-1])),sz)])
    rk=LGBMRanker(objective="lambdarank",n_estimators=300,learning_rate=0.05,num_leaves=31,
                  min_child_samples=20,random_state=0,n_jobs=-1,
                  label_gain=[float(i) for i in range(32)],verbose=-1)
    rk.fit(Xtr,rel,group=list(sz))
    # one representative feature row per (pool flow) for full-pool scoring: use the flow's train text row.
    # text features are identical across tasks for a given flow, so grab first occurrence per flow.
    first_row = sup.loc[tr].drop_duplicates("flow_id").set_index("flow_id")
    # text part is columns [0:n_text); meta part depends on task. We score full pool per test task by
    # taking each pool flow's text row and pasting the test task's meta row.
    n_text = X.shape[1]-len(meta_cols) if len(meta_cols) else X.shape[1]
    for tid in test:
        g=sup[sup.task_id==tid]
        if g.target_value.max()-g.target_value.min()<=0 or len(g)<8: continue
        rng=g.target_value.max()-g.target_value.min(); best=g.target_value.max()
        eval_flows=set(g.flow_id); truth=dict(zip(g.flow_id,g.target_value))
        # ---- candidate-only scores (leaky) ----
        tsc_c = dict(zip(g.flow_id, rk.predict(X[g.index.to_numpy()])))
        psc_c = {f:float(flow_mean.get(f,gmean)) for f in g.flow_id}
        v=scl.transform(imp.transform(tm.loc[[tid]]))
        d=np.linalg.norm(Mtr-v,axis=1); nn=[train_tasks[i] for i in np.argsort(d)[:5]]
        neigh=sup[sup.task_id.isin(nn)].groupby("flow_id")["target_value"].mean()
        ksc_c={f:neigh.get(f,gmean) for f in g.flow_id}
        def reg(pick): return (best-truth[pick])/rng
        for m,sc in [("text_ltr",tsc_c),("portfolio",psc_c),("kNN",ksc_c),
                     ("random",{f:RNG.random() for f in g.flow_id})]:
            acc["cand"][m].append(reg(max(sc,key=sc.get)))
        # ---- full-pool scores (deployment): rank ALL pool flows ----
        pf=[f for f in pool_flows if f in first_row.index]
        # matrix rows for each pool flow (text block); paste this task's RAW meta block (from any of
        # tid's own rows in X, which already carries the correct un-standardized metafeatures).
        posmap=dict(zip(sup.loc[tr].drop_duplicates("flow_id")["flow_id"],
                        sup.loc[tr].drop_duplicates("flow_id").index.to_numpy()))
        Xp=X[[posmap[f] for f in pf]].tolil()
        if len(meta_cols):
            meta_row=X[g.index[0]].toarray().ravel()[n_text:]        # raw meta for tid
            Xp[:, n_text:] = np.tile(meta_row, (len(pf),1))
        Xp=Xp.tocsr()
        tsc_f=dict(zip(pf, rk.predict(Xp)))
        psc_f={f:float(flow_mean.get(f,gmean)) for f in pf}
        ksc_f={f:neigh.get(f,gmean) for f in pf}
        rsc_f={f:RNG.random() for f in pf}
        for m,sc in [("text_ltr",tsc_f),("portfolio",psc_f),("kNN",ksc_f),("random",rsc_f)]:
            order=sorted(pf,key=lambda f:sc[f],reverse=True)
            # walk-depth: un-evaluable flows above the first evaluable one
            walked=0
            for f in order:
                if f in eval_flows: break
                walked+=1
            depth[m].append(walked)
            topk=order[:K]; hitk[m].append(1.0 if any(f in eval_flows and reg(f)<=TOL for f in topk) else 0.0)
            # budgeted deployment nregret@B: best evaluable flow among the top-B full-pool picks
            for B in BUDGETS:
                ev=[f for f in order[:B] if f in eval_flows]
                acc[f"full@{B}"][m].append((best-max(truth[f] for f in ev))/rng if ev else 1.0)
    print(f"fold {fi+1}/{NF} done", flush=True)

print("\n=== full-pool (deployment) vs candidate-only (leaky) — CC18, nregret ===")
hdr=f"{'model':>10} {'cand@1':>8} "+" ".join(f"{'full@'+str(B):>8}" for B in BUDGETS)+f" {'walk-dep':>9} {'hit@5':>7}"
print(hdr)
for m in MODELS:
    row=f"{m:>10} {np.mean(acc['cand'][m]):>8.4f} "+" ".join(f"{np.mean(acc[f'full@{B}'][m]):>8.4f}" for B in BUDGETS)
    print(row+f" {np.mean(depth[m]):>9.1f} {np.mean(hitk[m]):>7.2f}")
from scipy.stats import wilcoxon
for B in BUDGETS:
    tf=np.array(acc[f"full@{B}"]["text_ltr"]); pf=np.array(acc[f"full@{B}"]["portfolio"])
    print(f"full@{B}: text_ltr {tf.mean():.4f} vs portfolio {pf.mean():.4f}  Wilcoxon p={wilcoxon(tf,pf).pvalue:.4g}")
print("DONE")

"""Paper 3 cost-aware search: measure regret vs cumulative WALL-CLOCK (usercpu_time), not #evals.
A search incurs each evaluated flow's actual runtime. Orders:
  random | portfolio(acc) | text-LTR(acc) | cost-aware text-LTR(predicted acc / predicted runtime).
Runtime is predicted (you don't know it before running) from flow text + task metafeatures; cost is
charged at the ACTUAL runtime. Reports cumulative-time-to-within-1%-of-best and regret at time budgets.
"""
import warnings; warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from lightgbm import LGBMRanker, LGBMRegressor
import openml_flow as wf

RNG = np.random.default_rng(0)
MIN_CAND = 15
TIME_BUDGETS_MS = [100, 300, 1000, 3000, 10000, 30000, 100000]

coll = joblib.load("minimal_cache_cc18/longtail_collected.joblib")
rt_state = joblib.load("minimal_cache_cc18/longtail_runtime.joblib")
tasks_all = joblib.load("minimal_cache_cc18/longtail_tasks.joblib")

# accuracy: max per (task,flow)
ev = pd.concat(coll["evals"].values(), ignore_index=True)
ev["flow_id"] = pd.to_numeric(ev["flow_id"], errors="coerce"); ev["value"] = pd.to_numeric(ev["value"], errors="coerce")
ev = ev.dropna(subset=["flow_id", "value"]); ev["flow_id"] = ev["flow_id"].astype(int)
acc = ev.groupby(["task_id", "flow_id"], as_index=False)["value"].max().rename(columns={"value": "acc"})

# runtime: median per (task,flow)
rt = pd.concat(rt_state["rt"].values(), ignore_index=True) if rt_state["rt"] else pd.DataFrame(columns=["task_id","flow_id","value"])
rt["flow_id"] = pd.to_numeric(rt["flow_id"], errors="coerce"); rt["value"] = pd.to_numeric(rt["value"], errors="coerce")
rt = rt.dropna(subset=["flow_id", "value"]); rt["flow_id"] = rt["flow_id"].astype(int)
rt = rt[rt["value"] > 0]
rtm = rt.groupby(["task_id", "flow_id"], as_index=False)["value"].median().rename(columns={"value": "rt_ms"})

df = acc.merge(rtm, on=["task_id", "flow_id"], how="left")
cov = df["rt_ms"].notna().mean()
print(f"pairs {len(df)} | runtime coverage {cov:.1%} | tasks with runtime {rtm.task_id.nunique()}", flush=True)
# impute missing runtime by flow-median then global-median
flow_med = rtm.groupby("flow_id")["rt_ms"].median(); gmed = rtm["rt_ms"].median()
df["rt_ms"] = df.apply(lambda r: r["rt_ms"] if r["rt_ms"] == r["rt_ms"] else flow_med.get(r["flow_id"], gmed), axis=1)

# features via build_cc18_dataset (align to df's (task,flow) via supervised_df)
flows = {int(f): {"id": int(f), "name": n or "", "full_name": n or "", "version": ""}
         for f, n in ev.drop_duplicates("flow_id")[["flow_id", "flow_name"]].itertuples(index=False)}
tids = set(df.task_id.unique())
tasks_df = tasks_all[tasks_all["tid"].astype(int).isin(tids)].copy()
ds = wf.build_cc18_dataset(flows, tasks_df, ev, agg_mode="max", text_mode="tfidf",
                           use_task_metafeatures=True, toolkits=None, cache=wf.DiskCache("minimal_cache_cc18"))
sup = ds["supervised_df"].reset_index(drop=True); X = ds["X"].tocsr()
key = sup["task_id"].astype(str) + "_" + sup["flow_id"].astype(str)
dfk = (df["task_id"].astype(str) + "_" + df["flow_id"].astype(str))
rtmap = dict(zip(dfk, df["rt_ms"]))
sup["rt_ms"] = key.map(rtmap)
sup = sup.dropna(subset=["rt_ms"]).reset_index(drop=True)
X = X[sup.index.to_numpy()] if len(sup) == X.shape[0] else ds["X"].tocsr()[sup.index.to_numpy()]

cc = sup.groupby("task_id").size(); keep = cc[cc >= MIN_CAND].index
m = sup.task_id.isin(set(keep)).to_numpy(); sup = sup[m].reset_index(drop=True); X = X[m]
tasks = sup.task_id.unique(); RNG.shuffle(tasks)
nte = int(len(tasks) * 0.25); test = set(tasks[:nte]); train = set(tasks[nte:])
tr = sup.task_id.isin(train).to_numpy(); te = sup.task_id.isin(test).to_numpy()
print(f"{len(tasks)} tasks ({len(train)}tr/{len(test)}te) | rows {len(sup)}", flush=True)

# accuracy ranker (text-LTR)
def graded(y):
    r = y.max()-y.min(); return np.round(31*(y-y.min())/r).astype(int) if r>0 else np.zeros(len(y),int)
tri = np.where(tr)[0]; o = np.argsort(sup.loc[tr,"task_id"].to_numpy(), kind="stable")
Xtr = X[tri][o]; yt = sup.loc[tr,"acc"].to_numpy()[o] if "acc" in sup else sup.loc[tr,"target_value"].to_numpy()[o]
gt = sup.loc[tr,"task_id"].to_numpy()[o]; _, sz = np.unique(gt, return_counts=True)
rel = np.concatenate([graded(yt[s:s+n]) for s,n in zip(np.cumsum([0]+list(sz[:-1])), sz)])
rk = LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, random_state=0, n_jobs=-1, label_gain=[float(i) for i in range(32)], verbose=-1)
rk.fit(Xtr, rel, group=list(sz))
acc_pred = np.full(len(sup), np.nan); acc_pred[te] = rk.predict(X[np.where(te)[0]])

# runtime predictor (log ms) from same features
rtr = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                    random_state=0, n_jobs=-1, verbose=-1)
rtr.fit(X[tri], np.log1p(sup.loc[tr,"rt_ms"].to_numpy()))
rt_pred = np.full(len(sup), np.nan); rt_pred[te] = np.expm1(rtr.predict(X[np.where(te)[0]]))
rt_pred = np.clip(rt_pred, 1.0, None)

flow_mean = sup.loc[tr].groupby("flow_id")["target_value"].mean() if "target_value" in sup else sup.loc[tr].groupby("flow_id")["acc"].mean()
gmean = float((sup.loc[tr,"target_value"] if "target_value" in sup else sup.loc[tr,"acc"]).mean())
acc_col = "target_value" if "target_value" in sup.columns else "acc"

def time_to_target(order, truth, rt_true, best, rng_acc, eps=0.01):
    t = 0.0; bs = -1
    for loc in order:
        t += rt_true[loc]; bs = max(bs, truth[loc])
        if (best - bs) <= eps * rng_acc: return t
    return t  # never reached within budget -> total time

rows = []
tta = {m: [] for m in ["portfolio","text_acc","text_costaware","random"]}
regret_at = {m: {b: [] for b in TIME_BUDGETS_MS} for m in tta}
for tid in sorted(test):
    g = sup[sup.task_id == tid]; idx = g.index.to_numpy()
    truth = g[acc_col].to_numpy(); rt_true = g["rt_ms"].to_numpy(); fids = g.flow_id.to_numpy()
    best = truth.max(); rng_acc = truth.max()-truth.min()
    if rng_acc <= 0: continue
    ap = acc_pred[idx]; rp = rt_pred[idx]; port = np.array([flow_mean.get(f,gmean) for f in fids])
    orders = {
        "portfolio": np.argsort(-port),
        "text_acc": np.argsort(-ap),
        "text_costaware": np.argsort(-(ap - ap.min() + 1e-6) / rp),  # predicted acc-gain per predicted ms
        "random": RNG.permutation(len(idx)),
    }
    for mname, order in orders.items():
        tta[mname].append(time_to_target(order, truth, rt_true, best, rng_acc))
        # regret at cumulative-time budgets
        t = 0.0; bs = -1; bi = 0
        cum = []
        for loc in order:
            t += rt_true[loc]; bs = max(bs, truth[loc]); cum.append((t, (best-bs)/rng_acc))
        for b in TIME_BUDGETS_MS:
            r = next((rr for (tt, rr) in cum if tt >= b), cum[-1][1])
            # regret after spending up to b ms = regret at last eval with cum time <= b
            below = [rr for (tt, rr) in cum if tt <= b]
            regret_at[mname][b].append(below[-1] if below else 1.0)

print("\n=== cumulative runtime (ms) to within 1% of best ===")
for m in tta:
    v = np.array(tta[m]); print(f"  {m:16s}: median={np.median(v):.0f}ms  mean={v.mean():.0f}ms")
print("\n=== mean nregret@best-so-far at cumulative-time budgets ===")
print("budget_ms " + " ".join(f"{m:>14}" for m in tta))
for b in TIME_BUDGETS_MS:
    print(f"{b:>9} " + " ".join(f"{np.mean(regret_at[m][b]):>14.4f}" for m in tta))
print("DONE")

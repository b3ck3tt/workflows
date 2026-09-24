"""Paper 3 online-updating recommender: during warm-started search, re-rank the remaining flows using
the text prior blended with a text-space kernel estimate from the accuracies observed so far
(flows textually similar to good observed flows get boosted). Compare anytime regret of:
  portfolio | text-static | text-online | random.
Run: python online_search.py [N_TASK_LIMIT]
"""
import warnings; warnings.filterwarnings("ignore")
import sys, joblib, numpy as np, pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
from lightgbm import LGBMRanker
import openml_flow as wf

RNG = np.random.default_rng(0)
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
BUDGET = 30
MIN_CAND = 15

state = joblib.load("minimal_cache_cc18/longtail_collected.joblib")
tasks_all = joblib.load("minimal_cache_cc18/longtail_tasks.joblib")
ev = pd.concat(state["evals"].values(), ignore_index=True)
ev["flow_id"] = pd.to_numeric(ev["flow_id"], errors="coerce"); ev["value"] = pd.to_numeric(ev["value"], errors="coerce")
ev = ev.dropna(subset=["flow_id", "value"]); ev["flow_id"] = ev["flow_id"].astype(int)
uf = ev.drop_duplicates("flow_id")[["flow_id", "flow_name"]]
flows = {int(r.flow_id): {"id": int(r.flow_id), "name": (r.flow_name or ""), "full_name": (r.flow_name or ""), "version": ""} for r in uf.itertuples()}
tids = set(int(t) for t in state["evals"].keys())
tasks_df = tasks_all[tasks_all["tid"].astype(int).isin(tids)].copy()

ds = wf.build_cc18_dataset(flows, tasks_df, ev, agg_mode="max", text_mode="tfidf",
                           use_task_metafeatures=True, toolkits=None, cache=wf.DiskCache("minimal_cache_cc18"))
sup = ds["supervised_df"].reset_index(drop=True); X = ds["X"].tocsr()
# text-only columns for flow-flow similarity
text_cols = np.array([i for i, n in enumerate(ds["feature_names"]) if n.startswith("text::")])
Xtext = normalize(X[:, text_cols])   # L2-normalized -> dot = cosine

cand_counts = sup.groupby("task_id").size(); keep = cand_counts[cand_counts >= MIN_CAND].index
if LIMIT: keep = list(keep)[:LIMIT]
mask = sup.task_id.isin(set(keep)).to_numpy(); sup = sup[mask].reset_index(drop=True)
X = X[mask]; Xtext = Xtext[mask]
tasks = sup.task_id.unique(); RNG.shuffle(tasks)
ntest = int(len(tasks) * 0.25); test_tasks = set(tasks[:ntest]); train_tasks = set(tasks[ntest:])
tr = sup.task_id.isin(train_tasks).to_numpy(); te = sup.task_id.isin(test_tasks).to_numpy()
print(f"{len(tasks)} tasks ({len(train_tasks)}tr/{len(test_tasks)}te)", flush=True)

# text-LTR prior
def graded(y):
    r = y.max()-y.min(); return np.round(31*(y-y.min())/r).astype(int) if r>0 else np.zeros(len(y),int)
tri = np.where(tr)[0]; o = np.argsort(sup.loc[tr,"task_id"].to_numpy(), kind="stable")
Xtr = X[tri][o]; yt = sup.loc[tr,"target_value"].to_numpy()[o]; gt = sup.loc[tr,"task_id"].to_numpy()[o]
_, sz = np.unique(gt, return_counts=True)
rel = np.concatenate([graded(yt[s:s+n]) for s,n in zip(np.cumsum([0]+list(sz[:-1])), sz)])
rk = LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, random_state=0, n_jobs=-1, label_gain=[float(i) for i in range(32)], verbose=-1)
rk.fit(Xtr, rel, group=list(sz))
prior_all = np.full(len(sup), np.nan); prior_all[te] = rk.predict(X[np.where(te)[0]])
flow_mean = sup.loc[tr].groupby("flow_id")["target_value"].mean(); gmean = float(sup.loc[tr,"target_value"].mean())

def z(a):
    a = np.asarray(a, float); s = a.std(); return (a-a.mean())/s if s>0 else a*0

def simulate(order_fn, task_rows, truth, best, rng_acc):
    """order_fn(observed_idx, observed_acc, remaining_idx)->next remaining local index; returns anytime regret list"""
    n = len(task_rows); evaluated = []; obs_acc = []; remaining = list(range(n)); reg = []
    bestsofar = -1
    for step in range(min(BUDGET, n)):
        j = order_fn(evaluated, obs_acc, remaining)
        loc = remaining.pop(j)
        evaluated.append(loc); a = truth[loc]; obs_acc.append(a); bestsofar = max(bestsofar, a)
        reg.append((best - bestsofar)/rng_acc)
    return reg

curves = {m: np.zeros(BUDGET) for m in ["portfolio","text_static","text_online","random"]}
counts = np.zeros(BUDGET)
for tid in sorted(test_tasks):
    g = sup[sup.task_id==tid]; rows = g.index.to_numpy()
    truth = g.target_value.to_numpy(); fids = g.flow_id.to_numpy()
    best = truth.max(); rng_acc = truth.max()-truth.min()
    if rng_acc<=0 or len(rows)<MIN_CAND: continue
    prior = prior_all[rows]; port = np.array([flow_mean.get(f,gmean) for f in fids])
    Xt = Xtext[rows]  # (n, d) normalized
    prior_z = z(prior)

    def f_static(ev_,oa,rem): return int(np.argmax(prior[rem]))
    def f_port(ev_,oa,rem): return int(np.argmax(port[rem]))
    def f_rand(ev_,oa,rem): return int(RNG.integers(len(rem)))
    def f_online(ev_,oa,rem):
        if len(ev_) < 3:  # not enough obs -> follow prior
            return int(np.argmax(prior[rem]))
        obs = np.array(ev_); oav = np.array(oa)
        sims = (Xt[rem] @ Xt[obs].T)  # cosine sims (n_rem, n_obs)
        sims = sims.toarray() if sparse.issparse(sims) else np.asarray(sims)
        w = np.clip(sims, 0, None)
        denom = w.sum(1); denom[denom==0] = 1
        online = (w @ oav) / denom
        blend = prior_z[rem] + min(1.0, len(ev_)/8) * z(online)
        return int(np.argmax(blend))

    for m, fn in [("portfolio",f_port),("text_static",f_static),("text_online",f_online),("random",f_rand)]:
        reg = simulate(fn, rows, truth, best, rng_acc)
        for t in range(len(reg)): curves[m][t]+=reg[t]
    for t in range(min(BUDGET,len(rows))): counts[t]+=1

for m in curves: curves[m] = curves[m]/np.where(counts>0,counts,1)
print("\nanytime nregret by budget:")
print(f"{'evals':>5} " + " ".join(f"{m:>12}" for m in curves))
for b in [1,2,3,5,8,12,20,30]:
    if b<=BUDGET: print(f"{b:>5} " + " ".join(f"{curves[m][b-1]:>12.4f}" for m in curves))
joblib.dump(curves, "results_recommender_online_curves.joblib")
print("DONE")

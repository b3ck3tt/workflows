"""Analyze the real warm-started AutoML search (Phase 1-A). For each task, compute a common reference
best (max CV acc across all conditions/seeds), then per (task,condition,seed): evals-to-target and
wall-time-to-target (within 1% of ref best) and the anytime normalized-regret curve. Aggregate across
tasks (seed-averaged), paired Wilcoxon recommender vs portfolio vs random, and emit summary + curve."""
import warnings; warnings.filterwarnings("ignore")
import glob, joblib, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from pathlib import Path
OUT=Path("results_recommender_realsearch"); EPS=0.01; KMAX=40
CONDS=["recommender","portfolio","random"]

recs=[]
for fp in glob.glob(str(OUT/"task_*.joblib")):
    for r in joblib.load(fp):
        if "best" in r and r.get("log"): recs.append(r)
tasks=sorted({r["tid"] for r in recs})
print(f"{len(recs)} runs over {len(tasks)} tasks")

def cumbest(log):
    acc=[l["acc"] for l in log]; return np.maximum.accumulate(acc)
# per-task reference best/worst
refbest={}; refworst={}
for t in tasks:
    a=[l["acc"] for r in recs if r["tid"]==t for l in r["log"]]
    refbest[t]=max(a); refworst[t]=min(a)

rows=[]; curve={c:np.zeros(KMAX) for c in CONDS}; curve_n={c:np.zeros(KMAX) for c in CONDS}
for r in recs:
    t=r["tid"]; cb=cumbest(r["log"]); tt=[l["t"] for l in r["log"]]
    tgt=refbest[t]-EPS
    hit=np.where(cb>=tgt)[0]
    e2t=int(hit[0]) if len(hit) else len(cb)
    t2t=float(tt[hit[0]]) if len(hit) else float(tt[-1])
    rng=max(1e-9,refbest[t]-refworst[t])
    rows.append({"tid":t,"cond":r["cond"],"seed":r["seed"],"evals2tgt":e2t,"time2tgt":t2t,
                 "best":r["best"],"final_regret":(refbest[t]-cb[-1])/rng})
    # anytime normalized-regret curve (pad with last value to KMAX)
    nreg=(refbest[t]-cb)/rng
    padded=np.concatenate([nreg,np.full(max(0,KMAX-len(nreg)),nreg[-1])])[:KMAX]
    curve[r["cond"]]+=padded; curve_n[r["cond"]]+=1
df=pd.DataFrame(rows)
# seed-average within (task,cond)
g=df.groupby(["tid","cond"]).agg(evals2tgt=("evals2tgt","mean"),time2tgt=("time2tgt","mean"),
                                 final_regret=("final_regret","mean")).reset_index()
print("\n=== mean over tasks (seed-averaged) ===")
summ=g.groupby("cond").agg(evals2tgt=("evals2tgt","mean"),time2tgt=("time2tgt","mean"),
                           final_regret=("final_regret","mean"),n=("tid","nunique"))
print(summ.round(3).to_string())

# paired Wilcoxon recommender vs others on per-task seed-mean
piv=g.pivot(index="tid",columns="cond",values="evals2tgt").dropna()
pivt=g.pivot(index="tid",columns="cond",values="time2tgt").dropna()
print(f"\npaired tasks n={len(piv)}")
for base in ["portfolio","random"]:
    for metric,P in [("evals2tgt",piv),("time2tgt",pivt)]:
        w=wilcoxon(P["recommender"],P[base])
        print(f"  {metric}: recommender {P['recommender'].mean():.2f} vs {base} {P[base].mean():.2f}  "
              f"Wilcoxon p={w.pvalue:.4g}  win={float((P['recommender']<P[base]).mean()):.2f}")

# save
df.to_csv(OUT/"realsearch_per_run.csv",index=False)
g.to_csv(OUT/"realsearch_per_task.csv",index=False)
cur=pd.DataFrame({c:curve[c]/np.maximum(1,curve_n[c]) for c in CONDS}); cur.index.name="evals"
cur.to_csv(OUT/"realsearch_anytime_curve.csv")
print("\nanytime nregret @1/@3/@5/@10:")
for c in CONDS:
    v=cur[c].values; print(f"  {c:>11}: {v[0]:.3f} {v[2]:.3f} {v[4]:.3f} {v[9]:.3f}")
print("saved per_run/per_task/anytime_curve CSVs")

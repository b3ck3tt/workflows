"""Analyze Exp 1 (our recommender-warm-started search vs auto-sklearn 2.0, budget-matched test accuracy)
and Exp 2 (recommender order vs auto-sklearn 2.0 portfolio order vs our-portfolio vs random, isolated
warm-start). Merges the sharded result files. isolet (tid 3481, timeout on our side) excluded from
paired comparisons and noted."""
import warnings; warnings.filterwarnings("ignore")
import joblib, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
R="scratch/results_asklearn"

def merge(files):
    d={}
    for f in files:
        p=f"{R}/{f}.joblib"
        if os.path.exists(p):
            for k,v in joblib.load(p).items():
                if k not in d or ("acc" in v and "acc" not in d[k]): d[k]=v
    return d

recp=merge(["oursearch_base_b60","oursearch_sh0_b60","oursearch_sh1_b60","oursearch_sh2_b60"])
ak=merge(["oursearch_askl2_base_b60","oursearch_askl2_sh0_b60","oursearch_askl2_sh1_b60"])
rnd=merge(["oursearch_random_b60"]); asl=merge(["asklearn2_b60"])

# per (tid): mean test acc over seeds, per method
def by_task(d, cond=None):
    from collections import defaultdict
    acc=defaultdict(list)
    for k,v in d.items():
        if "acc" not in v: continue
        if cond and k[1]!=cond: continue
        acc[k[0]].append(v["acc"])
    return {t:np.mean(a) for t,a in acc.items()}

rec=by_task(recp,"recommender"); port=by_task(recp,"portfolio")
a2=by_task(ak,"askl2"); rd=by_task(rnd,"random")
ask={}  # auto-sklearn keyed by (tid,seed)
from collections import defaultdict
tmp=defaultdict(list)
for k,v in asl.items():
    if "acc" in v: tmp[k[0]].append(v["acc"])
ask={t:np.mean(a) for t,a in tmp.items()}

ISOLET=3481
methods={"auto-sklearn 2.0":ask,"our-rec search":rec,"our-portfolio search":port,
         "askl2-portfolio order":a2,"random order":rd}
# common tasks with valid results across all methods, excluding isolet
common=set.intersection(*[set(m) for m in methods.values()]) - {ISOLET}
common=sorted(common)
print(f"=== Exp 1+2: budget-matched (60s) held-out test accuracy, {len(common)} CC18 datasets ===")
print(f"(isolet excluded: our search times out on its 26-class problem at 60s; auto-sklearn completes it)\n")
print(f"{'method':>24} {'mean test acc':>14}")
for name,m in methods.items():
    print(f"{name:>24} {np.mean([m[t] for t in common]):>14.4f}")

print("\n=== Exp 1: our-rec search vs auto-sklearn 2.0 (paired) ===")
a=np.array([rec[t] for t in common]); b=np.array([ask[t] for t in common])
print(f"our-rec {a.mean():.4f} vs auto-sklearn {b.mean():.4f} | diff {a.mean()-b.mean():+.4f} | "
      f"our-rec wins {np.mean(a>b):.2f} | Wilcoxon p={wilcoxon(a,b).pvalue:.4g}")

print("\n=== Exp 2: warm-start order comparison in OUR harness (paired vs recommender) ===")
for name,m in [("askl2-portfolio",a2),("our-portfolio",port),("random",rd)]:
    x=np.array([rec[t] for t in common]); y=np.array([m[t] for t in common])
    print(f"  recommender {x.mean():.4f} vs {name} {y.mean():.4f} | diff {x.mean()-y.mean():+.4f} | "
          f"rec wins {np.mean(x>y):.2f} | p={wilcoxon(x,y).pvalue:.4g}")
pd.DataFrame({n:[m.get(t,np.nan) for t in common] for n,m in methods.items()},index=common).to_csv(f"{R}/exp12_per_task.csv")
print(f"\nsaved exp12_per_task.csv ({len(common)} tasks). isolet handling noted.")

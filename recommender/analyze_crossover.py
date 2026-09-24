"""Matched 60s-vs-300s crossover: our recommender-warm-started search vs auto-sklearn 2.0, on the SAME
CC18 datasets (isolet excluded: our search times out on its 26-class problem). Held-out test accuracy."""
import warnings; warnings.filterwarnings("ignore")
import joblib, os, numpy as np
from collections import defaultdict
from scipy.stats import wilcoxon
R="scratch/results_asklearn"
def load(files):
    d={}
    for f in files:
        p=f"{R}/{f}.joblib"
        if os.path.exists(p):
            for k,v in joblib.load(p).items():
                if "acc" in v: d.setdefault(k,v)
    return d
def bytask(d,cond=None):
    a=defaultdict(list)
    for k,v in d.items():
        if cond and k[1]!=cond: continue
        a[k[0]].append(v["acc"])
    return {t:np.mean(x) for t,x in a.items()}
# 60s
ask60=bytask(load(["asklearn2_b60"]))
our60=bytask(load(["oursearch_base_b60","oursearch_sh0_b60","oursearch_sh1_b60","oursearch_sh2_b60"]),"recommender")
# 300s
ask300=bytask(load(["asklearn2_b300_sh0","asklearn2_b300_sh1","asklearn2_b300_sh2"]))
our300=bytask(load(["oursearch_rec_b300_sh0","oursearch_rec_b300_sh1","oursearch_rec_b300_sh2"]),"recommender")
ISOLET=3481
common=sorted((set(ask60)&set(our60)&set(ask300)&set(our300))-{ISOLET})
print(f"=== Crossover: our-rec search vs auto-sklearn 2.0, same {len(common)} CC18 datasets (isolet excluded) ===\n")
def row(lbl,m): return f"{lbl:>16}: {np.mean([m[t] for t in common]):.4f}"
print("mean held-out test accuracy:")
print(f"  60s :  our-rec {np.mean([our60[t] for t in common]):.4f}  |  auto-sklearn {np.mean([ask60[t] for t in common]):.4f}")
print(f"  300s:  our-rec {np.mean([our300[t] for t in common]):.4f}  |  auto-sklearn {np.mean([ask300[t] for t in common]):.4f}")
def paired(o,a,lbl):
    o=np.array([o[t] for t in common]); a=np.array([a[t] for t in common])
    p=wilcoxon(o,a).pvalue
    print(f"  {lbl}: our-rec {o.mean():.4f} vs auto-sklearn {a.mean():.4f} | diff {o.mean()-a.mean():+.4f} | our wins {np.mean(o>a):.2f} | p={p:.4g}")
print("\npaired (our-rec vs auto-sklearn):")
paired(our60,ask60,"60s ")
paired(our300,ask300,"300s")
# within-method budget effect
print("\nbudget effect (300s vs 60s, same datasets):")
for lbl,m6,m3 in [("our-rec",our60,our300),("auto-sklearn",ask60,ask300)]:
    d6=np.array([m6[t] for t in common]); d3=np.array([m3[t] for t in common])
    print(f"  {lbl:>12}: 60s {d6.mean():.4f} -> 300s {d3.mean():.4f} ({d3.mean()-d6.mean():+.4f})")
print("DONE")

"""Exp 1 (auto-sklearn 2.0 side): run AutoSklearn2Classifier at a fixed wall-clock budget, 3 seeds,
per dataset; record held-out TEST accuracy. Runs in the asklearn64 (osx-64/Rosetta) conda env."""
import warnings; warnings.filterwarnings("ignore")
import time, joblib, glob, os
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
SCR="scratch"
D=f"{SCR}/asklearn_data"; OUT=f"{SCR}/results_asklearn"; os.makedirs(OUT,exist_ok=True)
BUDGET=int(os.environ.get("BUDGET","60")); SEEDS=[1,2,3]
res_path=os.environ.get("RESPATH",f"{OUT}/asklearn2_b{BUDGET}.joblib")
done=joblib.load(res_path) if os.path.exists(res_path) else {}
bl=os.environ.get("BASELINE")
if bl and os.path.exists(bl):
    for k,v in joblib.load(bl).items(): done.setdefault(k,v)
files=sorted(glob.glob(f"{D}/task_*.joblib"))
sh=os.environ.get("SHARD")
if sh:
    i,n=map(int,sh.split("/")); files=[f for j,f in enumerate(files) if j%n==i]
for f in files:
    tid=int(os.path.basename(f).split("_")[1].split(".")[0])
    Xtr,Xte,ytr,yte,name=joblib.load(f)
    for seed in SEEDS:
        key=(tid,seed)
        if key in done and "acc" in done[key]: continue
        t=time.time()
        tmp=f"/tmp/ask_{tid}_{seed}_{int(t)}"
        try:
            clf=AutoSklearn2Classifier(time_left_for_this_task=BUDGET, per_run_time_limit=max(10,BUDGET//4),
                                       seed=seed, memory_limit=8192, n_jobs=1, tmp_folder=tmp)
            clf.fit(Xtr,ytr); acc=float(clf.score(Xte,yte))
            done[key]={"acc":acc,"wall":time.time()-t,"name":name}
        except Exception as e:
            done[key]={"error":str(e)[:300],"name":name}
        joblib.dump(done,res_path)
        print(f"{name[:24]:>24} seed{seed}: acc={done[key].get('acc','ERR')} wall={done[key].get('wall',0):.0f}s",flush=True)
print("ASKLEARN EXP1 DONE")

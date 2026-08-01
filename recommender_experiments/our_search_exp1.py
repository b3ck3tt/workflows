"""Exp 1 (our side): recommender-warm-started AutoML search, wall-clock budgeted, held-out TEST accuracy
— matched to auto-sklearn's protocol. Optuna TPE over our 11-family sklearn config space, first trials
seeded by a family order (recommender / portfolio / random). Runs until the wall-clock budget, picks the
best config by an internal validation split, refits it on full train, scores test. Base (arm64) env."""
import warnings; warnings.filterwarnings("ignore")
import sys, time, os, glob, joblib, numpy as np, signal
class _RunTimeout(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f: (_ for _ in ()).throw(_RunTimeout()))
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.trial import FixedTrial
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

SCR="scratch"
D=f"{SCR}/asklearn_data"; OUT=f"{SCR}/results_asklearn"; os.makedirs(OUT,exist_ok=True)
BUDGET=int(os.environ.get("BUDGET","60")); SEEDS=[1,2,3]; SVC_MAX_N=3000; MAX_FEAT=256
ORD=joblib.load(f"{SCR}/realsearch_orderings.joblib")
FAMILIES=["hist_gbrt","random_forest","extra_trees","gradient_boosting","adaboost","decision_tree",
          "logistic","svc","sgd","knn","gaussian_nb"]

def make_clf(trial,n_used,seed):
    fam=trial.suggest_categorical("family",FAMILIES)
    if fam=="hist_gbrt": c=HistGradientBoostingClassifier(learning_rate=trial.suggest_float("lr",.01,.3,log=True),max_leaf_nodes=trial.suggest_int("leaves",15,255,log=True),max_iter=trial.suggest_int("iter",50,300),l2_regularization=trial.suggest_float("l2",1e-6,1.,log=True),random_state=seed)
    elif fam=="gradient_boosting": c=GradientBoostingClassifier(learning_rate=trial.suggest_float("glr",.01,.3,log=True),n_estimators=trial.suggest_int("gn",50,200),max_depth=trial.suggest_int("gd",2,5),random_state=seed)
    elif fam in("random_forest","extra_trees"):
        C=RandomForestClassifier if fam=="random_forest" else ExtraTreesClassifier
        c=C(n_estimators=trial.suggest_int("trees",50,400),max_depth=trial.suggest_categorical("depth",[None,8,16,32]),min_samples_leaf=trial.suggest_int("leaf",1,8),max_features=trial.suggest_categorical("mf",["sqrt","log2",0.5]),n_jobs=1,random_state=seed)
    elif fam=="adaboost": c=AdaBoostClassifier(n_estimators=trial.suggest_int("an",30,200),learning_rate=trial.suggest_float("alr",.1,2.,log=True),random_state=seed)
    elif fam=="decision_tree": c=DecisionTreeClassifier(max_depth=trial.suggest_categorical("ddepth",[None,4,8,16]),min_samples_leaf=trial.suggest_int("dleaf",1,16),random_state=seed)
    elif fam=="logistic": c=LogisticRegression(C=trial.suggest_float("C",1e-3,1e2,log=True),max_iter=500)
    elif fam=="svc":
        if n_used>SVC_MAX_N: raise optuna.TrialPruned()
        c=SVC(C=trial.suggest_float("svC",1e-2,1e2,log=True),gamma=trial.suggest_categorical("gamma",["scale","auto"]))
    elif fam=="sgd": c=SGDClassifier(alpha=trial.suggest_float("sa",1e-6,1e-1,log=True),loss=trial.suggest_categorical("sl",["hinge","log_loss","modified_huber"]),random_state=seed)
    elif fam=="knn": c=KNeighborsClassifier(n_neighbors=trial.suggest_int("k",3,30),weights=trial.suggest_categorical("w",["uniform","distance"]))
    else: c=GaussianNB(var_smoothing=trial.suggest_float("vs",1e-11,1e-6,log=True))
    return fam,c

BASE={"hist_gbrt":{"lr":.1,"leaves":31,"iter":100,"l2":1e-6},"gradient_boosting":{"glr":.1,"gn":100,"gd":3},
      "random_forest":{"trees":200,"depth":None,"leaf":1,"mf":"sqrt"},"extra_trees":{"trees":200,"depth":None,"leaf":1,"mf":"sqrt"},
      "adaboost":{"an":50,"alr":1.},"decision_tree":{"ddepth":None,"dleaf":1},"logistic":{"C":1.},
      "svc":{"svC":1.,"gamma":"scale"},"sgd":{"sa":1e-4,"sl":"hinge"},"knn":{"k":5,"w":"uniform"},"gaussian_nb":{"vs":1e-9}}
def warm(order,n=8): return [{"family":f,**BASE[f]} for f in order[:n]]

def pre_for(p,seed):
    steps=[("sc",StandardScaler())]
    if p>500: steps.append(("svd",TruncatedSVD(n_components=min(MAX_FEAT,p-1),random_state=seed)))
    return Pipeline(steps)

ASKL2_ORDER=["hist_gbrt","extra_trees","sgd","random_forest","gradient_boosting","adaboost",
             "decision_tree","logistic","svc","knn","gaussian_nb"]  # auto-sklearn 2.0 portfolio family order
def run(tid,cond,seed,Xtr,Xte,ytr,yte):
    rng=np.random.default_rng(seed)
    if cond=="askl2": order=list(ASKL2_ORDER)
    elif cond=="recommender": order=list(ORD[tid]["rec"])
    else: order=list(ORD[tid]["port"])
    if cond=="random": rng.shuffle(order)
    Xa,Xv,ya,yv=train_test_split(Xtr,ytr,test_size=0.25,random_state=seed,stratify=ytr)
    n_used=len(Xa); p=Xtr.shape[1]; pre=pre_for(p,seed)
    Xap=pre.fit_transform(Xa); Xvp=pre.transform(Xv)
    def obj(trial):
        fam,clf=make_clf(trial,n_used,seed)
        clf.fit(Xap,ya); return clf.score(Xvp,yv)
    st=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=seed))
    for cfg in warm(order): st.enqueue_trial(cfg)
    t0=time.time(); st.optimize(obj,timeout=BUDGET); wall=time.time()-t0
    # refit best config on FULL train, score test (matches auto-sklearn refit)
    pre2=pre_for(p,seed); Xtrp=pre2.fit_transform(Xtr); Xtep=pre2.transform(Xte)
    _,bclf=make_clf(FixedTrial(st.best_params),0,seed); bclf.fit(Xtrp,ytr)
    return {"acc":float(bclf.score(Xtep,yte)),"n_trials":len(st.trials),"wall":wall,"best_fam":st.best_params.get("family")}

if __name__=="__main__":
    res_path=os.environ.get("RESPATH",f"{OUT}/oursearch_b{BUDGET}.joblib")
    done=joblib.load(res_path) if os.path.exists(res_path) else {}
    bl=os.environ.get("BASELINE")   # preload prior results so shards skip already-done keys
    if bl and os.path.exists(bl):
        for k,v in joblib.load(bl).items(): done.setdefault(k,v)
    files=sorted(glob.glob(f"{D}/task_*.joblib"))
    sh=os.environ.get("SHARD")
    if sh:
        i,n=map(int,sh.split("/")); files=[f for j,f in enumerate(files) if j%n==i]
    for f in files:
        tid=int(os.path.basename(f).split("_")[1].split(".")[0])
        if tid not in ORD: continue
        Xtr,Xte,ytr,yte,name=joblib.load(f)
        for cond in os.environ.get("CONDS","recommender,portfolio").split(","):
            for seed in SEEDS:
                key=(tid,cond,seed)
                if key in done and "acc" in done[key]: continue
                signal.alarm(BUDGET+300)   # hard per-run timeout so no single run can hang the process
                try: done[key]={**run(tid,cond,seed,Xtr,Xte,ytr,yte),"name":name}
                except _RunTimeout: done[key]={"error":"run timeout>300s","name":name}
                except Exception as e: done[key]={"error":str(e)[:200],"name":name}
                finally: signal.alarm(0)
                joblib.dump(done,res_path)
        print(f"{name[:24]:>24} done ({len([k for k in done if k[0]==tid])} runs)",flush=True)
    print("OUR SEARCH EXP1 DONE")

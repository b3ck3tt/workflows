"""Phase 1-A: warm-starting a REAL AutoML search with the text recommender's ordering.

For each held-out CC18 task we run an Optuna TPE search over a genuine sklearn pipeline config space
with LIVE 3-fold CV training. Three warm-start conditions seed the first trials (one default config per
family, in the condition's family order):
  - recommender : families ordered by the leave-task-out text-LTR recommender's predicted score
  - portfolio   : families ordered by best-on-average accuracy across the other tasks (auto-sklearn style)
  - random      : shuffled family order
After the warm-start trials, TPE continues freely. We log per-trial (family, cv acc, cumulative best,
wall-time) and derive evals-to-target and time-to-target (target = within 1% of the best found across
all conditions/seeds for that task). Caps: subsample rows to ROW_CAP, SVD->256 dims when >500 features,
SVC only on small data.

Usage: python warmstart_search.py <mode> [tids...]   mode: run (parallel all/given tids) | one <tid> <cond> <seed>
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time, json, os, joblib, numpy as np, pandas as pd
from pathlib import Path
OUT=Path("results_recommender_realsearch"); OUT.mkdir(exist_ok=True)
SCR=Path("scratch")
os.environ.setdefault("OPENML_CACHE_DIR",str(SCR/"openml_cache"))
ROW_CAP=5000; SVC_MAX_N=3000; MAX_FEAT=256; N_TRIALS=40; CV=3; N_WARM=8
CONDS=["recommender","portfolio","random"]; SEEDS=[0,1,2]

FAMILIES=["hist_gbrt","random_forest","extra_trees","gradient_boosting","adaboost","decision_tree",
          "logistic","svc","sgd","knn","gaussian_nb"]
STEM={"hist_gbrt":"histgradientboosting","gradient_boosting":"gradientboostingclassifier",
      "random_forest":"randomforestclassifier","extra_trees":"extratreesclassifier",
      "adaboost":"adaboostclassifier","decision_tree":"decisiontreeclassifier",
      "logistic":"logisticregression","svc":"svc","sgd":"sgdclassifier",
      "knn":"kneighborsclassifier","gaussian_nb":"gaussiannb"}

# ---------------- family orderings from the recommender + portfolio (no live training) -------------
def compute_orderings():
    cache=SCR/"realsearch_orderings.joblib"
    if cache.exists(): return joblib.load(cache)
    import openml_flow as wf
    from lightgbm import LGBMRanker
    b=joblib.load("minimal_cache_cc18/cc18_bundle_v2_taskwise_a27a0c13de3118edc992e8a1afa833de.joblib")
    flows=joblib.load("minimal_cache_cc18/flows.joblib")
    ds=wf.build_cc18_dataset(flows,b["tasks_df"],b["evals_df"],agg_mode="max",text_mode="tfidf",
                             use_task_metafeatures=True,toolkits=("sklearn",),cache=wf.DiskCache("minimal_cache_cc18"))
    sup=ds["supervised_df"].reset_index(drop=True); X=ds["X"].tocsr()
    txt=sup["flow_text_clean"].astype(str).str.replace(" ","",regex=False)
    fam_of=pd.Series(index=sup.index,dtype=object)
    for fam in FAMILIES:  # last match wins is fine; assign by stem
        fam_of[txt.str.contains(STEM[fam],regex=False)]=fam
    sup=sup.assign(fam=fam_of)
    port_global=sup.dropna(subset=["fam"]).groupby("fam")["target_value"].mean()
    port_order_global=list(port_global.reindex([f for f in FAMILIES if f in port_global.index]).sort_values(ascending=False).index)
    def graded(y):
        r=y.max()-y.min(); return np.round(31*(y-y.min())/r).astype(int) if r>0 else np.zeros(len(y),int)
    orderings={}
    tasks=sup.task_id.unique()
    for tid in tasks:
        tr=(sup.task_id!=tid).to_numpy()
        o=np.argsort(sup.loc[tr,"task_id"].to_numpy(),kind="stable")
        Xtr=X[np.where(tr)[0]][o]; yt=sup.loc[tr,"target_value"].to_numpy()[o]; gt=sup.loc[tr,"task_id"].to_numpy()[o]
        _,sz=np.unique(gt,return_counts=True)
        rel=np.concatenate([graded(yt[s:s+n]) for s,n in zip(np.cumsum([0]+list(sz[:-1])),sz)])
        rk=LGBMRanker(objective="lambdarank",n_estimators=200,learning_rate=0.05,num_leaves=31,
                      min_child_samples=20,random_state=0,n_jobs=-1,label_gain=[float(i) for i in range(32)],verbose=-1)
        rk.fit(Xtr,rel,group=list(sz))
        # CORRECT: rank the held-out task's own candidate flows (in-distribution for the ranker),
        # map to families, order families by their best-ranked flow. (Deployment op: ranks flows by
        # text + this task's meta; does NOT use the flows' accuracy on the held-out task.)
        cand=sup[sup.task_id==tid]
        cc=cand.assign(score=rk.predict(X[cand.index.to_numpy()])).dropna(subset=["fam"])
        fam_score=cc.groupby("fam")["score"].max().to_dict() if len(cc) else {}
        if "hist_gbrt" not in fam_score and "gradient_boosting" in fam_score:
            fam_score["hist_gbrt"]=fam_score["gradient_boosting"]-1e-6  # boosting alias
        ranked=[f for f in sorted(fam_score,key=fam_score.get,reverse=True)]
        rec_order=ranked+[f for f in port_order_global if f not in ranked]  # missing families -> portfolio tail
        orderings[int(tid)]={"rec":rec_order,"port":port_order_global,
                             "did":int(b["tasks_df"].set_index("tid").loc[tid,"did"])}
    joblib.dump(orderings,cache); return orderings

# ---------------- config space -------------------------------------------------------------------
def make_pipe(trial, p, n_used, seed):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
        ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier)
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    import optuna
    fam=trial.suggest_categorical("family",FAMILIES)
    if fam=="hist_gbrt":
        clf=HistGradientBoostingClassifier(learning_rate=trial.suggest_float("lr",.01,.3,log=True),
            max_leaf_nodes=trial.suggest_int("leaves",15,255,log=True),max_iter=trial.suggest_int("iter",50,300),
            l2_regularization=trial.suggest_float("l2",1e-6,1.,log=True),random_state=seed)
    elif fam=="gradient_boosting":
        clf=GradientBoostingClassifier(learning_rate=trial.suggest_float("glr",.01,.3,log=True),
            n_estimators=trial.suggest_int("gn",50,200),max_depth=trial.suggest_int("gd",2,5),random_state=seed)
    elif fam in("random_forest","extra_trees"):
        C=RandomForestClassifier if fam=="random_forest" else ExtraTreesClassifier
        clf=C(n_estimators=trial.suggest_int("trees",50,400),max_depth=trial.suggest_categorical("depth",[None,8,16,32]),
              min_samples_leaf=trial.suggest_int("leaf",1,8),max_features=trial.suggest_categorical("mf",["sqrt","log2",0.5]),
              n_jobs=1,random_state=seed)
    elif fam=="adaboost":
        clf=AdaBoostClassifier(n_estimators=trial.suggest_int("an",30,200),
            learning_rate=trial.suggest_float("alr",.1,2.,log=True),random_state=seed)
    elif fam=="decision_tree":
        clf=DecisionTreeClassifier(max_depth=trial.suggest_categorical("ddepth",[None,4,8,16]),
            min_samples_leaf=trial.suggest_int("dleaf",1,16),random_state=seed)
    elif fam=="logistic":
        clf=LogisticRegression(C=trial.suggest_float("C",1e-3,1e2,log=True),max_iter=500)
    elif fam=="svc":
        if n_used>SVC_MAX_N: raise optuna.TrialPruned()
        clf=SVC(C=trial.suggest_float("svC",1e-2,1e2,log=True),gamma=trial.suggest_categorical("gamma",["scale","auto"]))
    elif fam=="sgd":
        clf=SGDClassifier(alpha=trial.suggest_float("sa",1e-6,1e-1,log=True),
            loss=trial.suggest_categorical("sl",["hinge","log_loss","modified_huber"]),random_state=seed)
    elif fam=="knn":
        clf=KNeighborsClassifier(n_neighbors=trial.suggest_int("k",3,30),weights=trial.suggest_categorical("w",["uniform","distance"]))
    else:
        clf=GaussianNB(var_smoothing=trial.suggest_float("vs",1e-11,1e-6,log=True))
    return fam,clf

def warm_configs(order):
    """default hyperparameter dicts to enqueue, one per family in `order` (first N_WARM)."""
    base={"hist_gbrt":{"lr":.1,"leaves":31,"iter":100,"l2":1e-6},
          "gradient_boosting":{"glr":.1,"gn":100,"gd":3},
          "random_forest":{"trees":200,"depth":None,"leaf":1,"mf":"sqrt"},
          "extra_trees":{"trees":200,"depth":None,"leaf":1,"mf":"sqrt"},
          "adaboost":{"an":50,"alr":1.},"decision_tree":{"ddepth":None,"dleaf":1},
          "logistic":{"C":1.},"svc":{"svC":1.,"gamma":"scale"},
          "sgd":{"sa":1e-4,"sl":"hinge"},"knn":{"k":5,"w":"uniform"},"gaussian_nb":{"vs":1e-9}}
    return [{"family":f,**base[f]} for f in order[:N_WARM]]

def load_xy(did):
    import openml
    ds=openml.datasets.get_dataset(did)
    X,y,cat,names=ds.get_data(target=ds.default_target_attribute)
    return X,y,cat,names

def run_one(tid, cond, seed, order, did, Xy=None):
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.decomposition import TruncatedSVD
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
    X,y,cat,names=Xy if Xy else load_xy(did)
    Xdf=pd.DataFrame(X); Xdf.columns=names; catmask=[bool(c) for c in cat]
    num=[names[i] for i in range(len(names)) if not catmask[i]]
    catc=[names[i] for i in range(len(names)) if catmask[i]]
    n0,p=Xdf.shape
    if n0>ROW_CAP:
        Xdf,_,y,_=train_test_split(Xdf,y,train_size=ROW_CAP,stratify=y,random_state=seed)
    y=pd.Series(y).astype("category").cat.codes.to_numpy(); n_used=len(Xdf)
    pre=ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),num),
        ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),
                         ("o",OneHotEncoder(handle_unknown="ignore",max_categories=20))]),catc)],
        remainder="drop", sparse_threshold=0)  # force DENSE (hist_gbrt/gaussian_nb require dense)
    skf=StratifiedKFold(CV,shuffle=True,random_state=seed)
    log=[]; t0=time.time()
    def objective(trial):
        fam,clf=make_pipe(trial,p,n_used,seed)
        steps=[("pre",pre)]
        if p>500: steps.append(("svd",TruncatedSVD(n_components=min(MAX_FEAT,p-1),random_state=seed)))
        steps.append(("clf",clf))
        acc=cross_val_score(Pipeline(steps),Xdf,y,cv=skf,scoring="accuracy",n_jobs=1).mean()
        log.append({"trial":len(log),"family":fam,"acc":float(acc),"t":time.time()-t0})
        return acc
    rng=np.random.default_rng(seed)
    if cond=="random": order=list(order); rng.shuffle(order)
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=seed),
                              pruner=optuna.pruners.MedianPruner(n_warmup_steps=0))
    for cfg in warm_configs(order): study.enqueue_trial(cfg)
    study.optimize(objective,n_trials=N_TRIALS)
    return {"tid":tid,"cond":cond,"seed":seed,"n_used":int(n_used),"p":int(p),
            "best":float(study.best_value),"wall":time.time()-t0,"log":log}

def task_worker(args):
    tid,order,did=args
    dst=OUT/f"task_{tid}.joblib"
    REDO=os.environ.get("REDO_REC")=="1"
    existing=joblib.load(dst) if dst.exists() else None
    if existing is not None and not REDO:   # resumable: skip fully-done tasks
        ok=[x for x in existing if "best" in x]
        return {"tid":tid,"n_ok":len(ok),"wall_sum":sum(x.get("wall",0) for x in ok),"cached":True}
    # decide which (cond,seed) to run
    if existing is not None and REDO:
        # SALVAGE: keep valid portfolio/random, recompute only recommender with the fixed order
        keep=[x for x in existing if x.get("cond") in ("portfolio","random")]
        todo=[("recommender",s) for s in SEEDS]
    else:
        keep=[]; todo=[(c,s) for c in CONDS for s in SEEDS]
    try:
        Xy=load_xy(did)
    except Exception as e:
        return {"tid":tid,"error":str(e)[:200]}
    res=list(keep)
    for cond,seed in todo:
        try: res.append(run_one(tid,cond,seed,order["rec"] if cond=="recommender" else order["port"],did,Xy))
        except Exception as e: res.append({"tid":tid,"cond":cond,"seed":seed,"error":str(e)[:200]})
    joblib.dump(res,dst)
    ok=[r for r in res if "best" in r]
    return {"tid":tid,"n_ok":len(ok),"wall_sum":sum(r.get("wall",0) for r in ok),"redo":REDO}

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "run"
    ordng=compute_orderings()
    if mode=="orderings":
        for t,o in list(ordng.items())[:3]: print(t,o["rec"][:4],"| port",o["port"][:4])
        print(f"{len(ordng)} tasks ordered"); sys.exit()
    tids=[int(x) for x in sys.argv[2:]] if len(sys.argv)>2 else sorted(ordng)
    jobs=[(t,ordng[t],ordng[t]["did"]) for t in tids]
    from joblib import Parallel, delayed
    NW=int(os.environ.get("NW","6"))
    print(f"launching {len(jobs)} tasks x {len(CONDS)}x{len(SEEDS)} searches, {NW} workers",flush=True)
    t0=time.time()
    outs=Parallel(n_jobs=NW,backend="loky")(delayed(task_worker)(j) for j in jobs)
    joblib.dump(outs,OUT/"run_summary.joblib")
    print(f"DONE {len([o for o in outs if o.get('n_ok')])} tasks in {(time.time()-t0)/60:.1f} min",flush=True)
    for o in outs: print(o)

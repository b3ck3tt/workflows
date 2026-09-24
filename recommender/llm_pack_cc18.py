import warnings; warnings.filterwarnings("ignore")
import json, joblib, numpy as np, pandas as pd
import openml_flow as wf
RNG=np.random.default_rng(3); N=100; MIN_CAND=8
FAMILY=[("boosting",("gradientboost","xgboost","adaboost","histgradient","gbm")),
 ("tree_ensemble",("randomforest","extratrees","baggedtrees")),
 ("neural_net",("mlp","multilayerperceptron","keras","neural")),
 ("svm",("svc","svm","supportvector","linearsvc")),
 ("naive_bayes",("naivebayes","nbayes","gaussiannb","bernoullinb","multinomialnb")),
 ("knn",("kneighbors","knn","nearestneighbor")),
 ("discriminant",("discriminant","lda","qda")),
 ("decision_tree",("decisiontree","extratreeclassifier")),
 ("linear",("logistic","sgd","ridge","perceptron","passiveaggressive")),
 ("rule_based",("dummy",))]
def fam(name):
    n=(name or "").lower().replace(".","")
    for lab,subs in FAMILY:
        if any(s in n for s in subs): return lab
    return "other"
b=wf.load_cc18_from_openml(cache=wf.DiskCache("minimal_cache_cc18"))
flows=wf.load_or_cache_flows(b["evals_df"],"minimal_cache_cc18/flows.joblib")
ds=wf.build_cc18_dataset(flows,b["tasks_df"],b["evals_df"],agg_mode="max",text_mode="none",
    use_task_metafeatures=True,toolkits=("sklearn",),cache=wf.DiskCache("minimal_cache_cc18"))
sup=ds["supervised_df"]; fmap=dict(zip(sup.flow_id,sup.flow_name))
sup=sup.copy(); sup["family"]=sup.flow_id.map(lambda f: fam(fmap.get(f,"")))
tn,_=wf.normalize_cc18_tasks(b["tasks_df"]); tn=tn.set_index("task_id")
cc=sup.groupby("task_id").size(); tasks=cc[cc>=MIN_CAND].index.to_numpy(); RNG.shuffle(tasks); tasks=sorted(tasks[:N])
pack=[]; truth={}
for tid in tasks:
    g=sup[sup.task_id==tid]; mf=tn.loc[tid]
    mino=mf.get("MinorityClassSize"); maj=mf.get("MajorityClassSize")
    imb=round(float(maj)/float(mino),2) if mino and mino==mino and mino>0 else None
    pack.append({"task":int(tid),"metafeatures":{"n_instances":int(mf["NumberOfInstances"]),
      "n_features":int(mf["NumberOfFeatures"]),"n_classes":int(mf["NumberOfClasses"]),
      "n_numeric":int(mf["NumberOfNumericFeatures"]),"n_categorical":int(mf["NumberOfSymbolicFeatures"]),
      "class_imbalance_ratio":imb,"has_missing":bool(mf["NumberOfInstancesWithMissingValues"]>0)},
      "candidate_families":sorted(g.family.unique())})
    truth[int(tid)]=g[["flow_id","family","target_value"]].rename(columns={"target_value":"acc"}).to_dict("records")
json.dump(pack,open("scratchpad_llm_cc18_pack.json","w"),indent=1)
joblib.dump({"truth":truth},"scratchpad_llm_cc18_truth.joblib")
print(f"CC18 pack: {len(pack)} tasks")

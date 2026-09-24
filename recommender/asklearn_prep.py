import sys, os, joblib, numpy as np, pandas as pd, openml
from sklearn.model_selection import train_test_split
os.environ.setdefault("OPENML_CACHE_DIR","scratch/openml_cache")
OUT="scratch/asklearn_data"
b=joblib.load("minimal_cache_cc18/cc18_bundle_v2_taskwise_a27a0c13de3118edc992e8a1afa833de.joblib")
t=b["tasks_df"].set_index("tid")
tids=[int(x) for x in sys.argv[1:]]
for tid in tids:
    did=int(t.loc[tid,"did"])
    ds=openml.datasets.get_dataset(did)
    X,y,cat,names=ds.get_data(target=ds.default_target_attribute)
    X=pd.get_dummies(X,dummy_na=True).fillna(0).to_numpy().astype(float)
    y=pd.Series(y).astype("category").cat.codes.to_numpy()
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.33,random_state=0,stratify=y)
    joblib.dump((Xtr,Xte,ytr,yte,t.loc[tid,"name"]), f"{OUT}/task_{tid}.joblib")
    print(f"tid {tid} (did {did}, {t.loc[tid,'name']}): {X.shape} -> saved", flush=True)
print("PREP DONE")

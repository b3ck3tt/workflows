"""Fix I: a QUANTITATIVE SHAP-alignment metric replacing 'SHAP agrees with priors' (qualitative).

For each algorithm family we compute:
  - SHAP attribution: mean over all rows of the summed TreeSHAP value of that family's text tokens
    (how much the model credits/debits the family, on average, toward predicted accuracy).
  - empirical direction: mean target of flows whose text contains the family, minus the global mean
    (does that family actually do better/worse than average, in the data?).
Alignment = Spearman(family SHAP, family empirical delta) + sign-agreement fraction. High alignment
means the explanation is faithful to the data, not just a plausible-sounding story.
"""
import warnings; warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
import shap, openml_flow as wf

b=joblib.load("minimal_cache_cc18/cc18_bundle_v2_taskwise_a27a0c13de3118edc992e8a1afa833de.joblib")
flows=joblib.load("minimal_cache_cc18/flows.joblib")
ds=wf.build_cc18_dataset(flows,b["tasks_df"],b["evals_df"],agg_mode="max",text_mode="tfidf",
                         use_task_metafeatures=True,toolkits=("sklearn",),cache=wf.DiskCache("minimal_cache_cc18"))
sup=ds["supervised_df"].reset_index(drop=True); X=ds["X"].tocsr(); y=sup["target_value"].to_numpy()
fn=ds["feature_names"]; tok=np.array([f.split("::",1)[1] if f.startswith("text::") else None for f in fn],dtype=object)
gmean=float(y.mean())

FAMILIES={
 "gradient_boosting":"gradientboostingclassifier","random_forest":"randomforestclassifier",
 "extra_trees":"extratreesclassifier","adaboost":"adaboostclassifier","bagging":"baggingclassifier",
 "decision_tree":"decisiontreeclassifier","svc":"svc","logistic_reg":"logisticregression",
 "sgd":"sgdclassifier","gaussian_nb":"gaussiannb","bernoulli_nb":"bernoullinb",
 "knn":"kneighborsclassifier","mlp":"mlpclassifier","xgboost":"xgboost"}
# column indices per family (tokens containing the family stem)
fam_cols={}; text=sup["flow_text_clean"].astype(str)
for fam,stem in FAMILIES.items():
    cols=[j for j,t in enumerate(tok) if t is not None and stem in t.replace(" ","")]
    if cols: fam_cols[fam]=cols

# train regressor + TreeSHAP. Shallow/small forest keeps SHAP over 5k feats tractable while remaining
# a valid accuracy predictor (the metric is about explanation alignment, not squeezing max R^2).
Xd=X.toarray()
reg=ExtraTreesRegressor(n_estimators=60,max_depth=10,random_state=42,n_jobs=-1).fit(Xd,y)
rng=np.random.default_rng(0); idx=rng.choice(len(y),size=min(500,len(y)),replace=False)
expl=shap.TreeExplainer(reg,feature_perturbation="tree_path_dependent")
sv=expl.shap_values(Xd[idx],check_additivity=False)   # (n_sample, n_feat)
print(f"SHAP on {len(idx)} rows x {sv.shape[1]} feats",flush=True)
is_text=np.array([f.startswith("text::") for f in fn])
text_total=sv[:,is_text].sum(axis=1)   # per-flow TOTAL text-SHAP push on predicted accuracy

# family membership of each SAMPLED flow (by its text); use the classifier stem
samp_text=text.iloc[idx].str.replace(" ","",regex=False).reset_index(drop=True)
rows=[]
for fam,stem in FAMILIES.items():
    stem2=stem.replace(" ","")
    present_all=text.str.replace(" ","",regex=False).str.contains(stem2,regex=False)
    n=int(present_all.sum())
    if n<5: continue
    emp_delta=float(y[present_all.to_numpy()].mean()-gmean)          # empirical over/under-performance
    samp_mask=samp_text.str.contains(stem2,regex=False).to_numpy()
    # (a) family-name-token SHAP (concentrated); (b) TOTAL text-SHAP for the family's flows (distributed)
    name_shap=float(sv[:,fam_cols[fam]].sum(axis=1).mean()) if fam in fam_cols else 0.0
    dist_shap=float(text_total[samp_mask].mean()) if samp_mask.sum()>=3 else np.nan
    rows.append((fam,n,int(samp_mask.sum()),name_shap,dist_shap,emp_delta))
R=pd.DataFrame(rows,columns=["family","n_flows","n_samp","name_shap","dist_shap","emp_delta"]).sort_values("emp_delta")
pd.set_option("display.float_format",lambda v:f"{v:+.4f}")
print(R.to_string(index=False))

Rd=R.dropna(subset=["dist_shap"])
def stats(col,frame):
    x=frame[col].to_numpy(); e=frame["emp_delta"].to_numpy()
    return spearmanr(x,e).correlation, float(np.mean(np.sign(x)==np.sign(e)))
rho_n,sg_n=stats("name_shap",R); rho_d,sg_d=stats("dist_shap",Rd)
line=(f"SHAP-alignment vs empirical family performance:\n"
      f"  family-NAME-token SHAP (concentrated): Spearman={rho_n:+.3f} sign-agree={sg_n:.2f} (n={len(R)})\n"
      f"  TOTAL text-SHAP per family (distributed): Spearman={rho_d:+.3f} sign-agree={sg_d:.2f} (n={len(Rd)})")
print("\n"+line)
with open("papers/recommender/results_phase0/fix_I_shap_alignment.txt","w") as f:
    f.write("Fix I — quantitative SHAP-alignment metric (CC18, extra_trees, TreeSHAP; shallow 60x depth10 "
            "recompute for per-flow values — full model's namespace split 51/49 matches paper).\n"
            "name_shap = mean summed SHAP of the family's own name tokens; dist_shap = mean TOTAL text-SHAP\n"
            "over the family's flows (distributed); emp_delta = family mean target - global mean.\n\n")
    f.write(R.to_string(index=False)+"\n\n"+line+"\n")
print("DONE")

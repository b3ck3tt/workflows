"""Fix K: bootstrap 95% CIs + multiple-comparison correction on the headline tables (all papers).
Paired bootstrap over tasks for each method's nregret@1; paired Wilcoxon vs the portfolio with
Holm correction across the per-table family of comparisons."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
RNG=np.random.default_rng(0)

def boot_ci(x,B=5000):
    x=np.asarray(x); return x.mean(),*np.percentile([np.mean(RNG.choice(x,len(x),True)) for _ in range(B)],[2.5,97.5])

def load(d,exp,method):
    pt=pd.read_csv(f"results_recommender_{d}/ranking_metrics_per_task.csv")
    s=pt[(pt.experiment==exp)&(pt.method==method)].set_index("task_id")["nregret@1"]
    return s

def table(name,rows,default_key):
    """rows: list of (label, series). Bootstrap CI each; Holm-correct Wilcoxon vs default across rows."""
    idx=set.intersection(*[set(s.index) for _,s in rows])
    idx=sorted(idx); dflt=dict(rows)[default_key].reindex(idx).to_numpy()
    print(f"\n=== {name} (n={len(idx)} tasks; paired bootstrap 95% CI; Holm-corrected Wilcoxon vs {default_key}) ===")
    pvals=[]; lab_for_p=[]
    for label,s in rows:
        v=s.reindex(idx).to_numpy()
        if label!=default_key and not np.allclose(v,dflt):
            pvals.append(wilcoxon(v,dflt).pvalue); lab_for_p.append(label)
    corr=dict(zip(lab_for_p,multipletests(pvals,method="holm")[1])) if pvals else {}
    out=[]
    for label,s in rows:
        v=s.reindex(idx).to_numpy(); m,lo,hi=boot_ci(v)
        praw="";
        if label in corr:
            raw=wilcoxon(v,dflt).pvalue; praw=f"p={raw:.4f} -> Holm {corr[label]:.4f}"
        out.append((label,m,lo,hi,praw))
    for label,m,lo,hi,p in out:
        print(f"  {label:>28} {m:>7.4f}  [{lo:.4f},{hi:.4f}]   {p}")
    return out

alltxt=["Fix K — bootstrap 95% CIs + Holm multiple-comparison correction on headline tables\n"]
import io,contextlib
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    # ---- Paper 1 Table 1 (CC18, agg=max) ----
    table("Paper 1 — Table 1 (CC18 recommendation quality)",[
        ("LambdaMART + tfidf",       load("agg_max","tfidf_only","lambdamart")),
        ("extra_trees + tfidf+meta", load("max_allmodels","tfidf_meta","extra_trees")),
        ("random_forest + tfidf+meta",load("max_allmodels","tfidf_meta","random_forest")),
        ("hist_gbrt + tfidf+meta",   load("max_allmodels","tfidf_meta","hist_gbrt")),
        ("ridge + tfidf+meta",       load("max_allmodels","tfidf_meta","ridge")),
        ("global-default (portfolio)",load("max_allmodels","tfidf_meta","global_default")),
        ("kNN (similar tasks)",      load("agg_max","tfidf_meta","knn")),
        ("random",                   load("max_allmodels","tfidf_meta","random")),
    ],"global-default (portfolio)")

    # ---- Paper 2/3 long-tail headline (ltfull_basic) ----
    table("Paper 2/3 — long-tail headline (basic metadata)",[
        ("LambdaMART (text)",        load("ltfull_basic","tfidf_meta","lambdamart")),
        ("kNN (similar tasks)",      load("ltfull_basic","tfidf_meta","knn")),
        ("global-default (portfolio)",load("ltfull_basic","tfidf_meta","global_default")),
        ("random",                   load("ltfull_basic","tfidf_meta","random")),
    ],"global-default (portfolio)")

    # ---- Paper 2 metadata scaling (BH across the 3 kNN-vs-default tests) ----
    print("\n=== Paper 2 — metadata scaling: kNN vs portfolio, BH-corrected across metadata levels ===")
    port=load("ltfull_basic","tfidf_meta","global_default")
    ps=[]; labs=[]
    for ms in ["basic","landmarking","openml_full"]:
        k=load(f"ltfull_{ms}","tfidf_meta","knn"); idx=sorted(set(k.index)&set(port.index))
        kk=k.reindex(idx).to_numpy(); pp=port.reindex(idx).to_numpy()
        m,lo,hi=boot_ci(kk); ps.append(wilcoxon(kk,pp).pvalue); labs.append((ms,m,lo,hi))
    bh=multipletests(ps,method="fdr_bh")[1]
    for (ms,m,lo,hi),praw,padj in zip(labs,ps,bh):
        print(f"  kNN {ms:>12} {m:>7.4f}  [{lo:.4f},{hi:.4f}]   vs port p={praw:.4f} -> BH {padj:.4f}")
alltxt.append(buf.getvalue())
print(buf.getvalue())
open("papers/recommender/results_phase0/fix_K_cis_correction.txt","w").write("".join(alltxt))
print("DONE")

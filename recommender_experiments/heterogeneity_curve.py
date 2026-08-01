"""Paper 2 signature figure: does adaptation pay MORE on more heterogeneous (atypical) tasks?
Per-task atypicality = 1 - Spearman(task's true flow ranking, global mean flow ranking) -- a
method-independent heterogeneity measure. Per-task adaptation value = portfolio regret minus
method regret (model, kNN). Bin tasks by atypicality and plot mean adaptation value."""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import openml_flow as wf

# assemble full long-tail supervised table (for per-task atypicality)
state = joblib.load("minimal_cache_cc18/longtail_collected.joblib")
tasks_all = joblib.load("minimal_cache_cc18/longtail_tasks.joblib")
ev = pd.concat(state["evals"].values(), ignore_index=True)
ev["flow_id"] = pd.to_numeric(ev["flow_id"], errors="coerce")
ev["value"] = pd.to_numeric(ev["value"], errors="coerce")
ev = ev.dropna(subset=["flow_id", "value"]); ev["flow_id"] = ev["flow_id"].astype(int)
agg = ev.groupby(["task_id", "flow_id"], as_index=False)["value"].max().rename(columns={"value": "t"})

global_mean = agg.groupby("flow_id")["t"].mean()   # global flow ranking
atyp = {}
for tid, g in agg.groupby("task_id"):
    if len(g) < 5: continue
    gm = global_mean.loc[g["flow_id"]].to_numpy()
    tv = g["t"].to_numpy()
    if np.std(gm) > 0 and np.std(tv) > 0:
        rho = spearmanr(tv, gm).correlation
        atyp[tid] = 1.0 - (rho if rho == rho else 0.0)
atyp = pd.Series(atyp, name="atypicality")

# per-task method regrets from the powered run
pt = pd.read_csv("results_recommender_longtail_full/ranking_metrics_per_task.csv")
pt = pt[pt.experiment == "tfidf_meta"]
def col(method):
    return pt[pt.method == method].set_index("task_id")["nregret@1"]
model, knn, deft = col("lambdamart"), col("knn"), col("global_default")

df = pd.DataFrame({"atyp": atyp, "model": model, "knn": knn, "default": deft}).dropna()
df["model_gain"] = df["default"] - df["model"]   # improvement over portfolio
df["knn_gain"] = df["default"] - df["knn"]
print(f"tasks with atypicality + regrets: {len(df)}")
print("corr(atypicality, model_gain):", round(spearmanr(df.atyp, df.model_gain).correlation, 3))
print("corr(atypicality, knn_gain):  ", round(spearmanr(df.atyp, df.knn_gain).correlation, 3))

# bin by atypicality deciles
df["bin"] = pd.qcut(df["atyp"], 8, labels=False, duplicates="drop")
b = df.groupby("bin").agg(atyp=("atyp", "mean"), model_gain=("model_gain", "mean"),
                          knn_gain=("knn_gain", "mean"), default=("default", "mean"),
                          n=("atyp", "size")).reset_index()

FIG = Path("paper_recommender/figures"); FIG.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(b["atyp"], b["model_gain"], marker="o", color="#2c7fb8", label="text ranker (LambdaMART)")
ax.plot(b["atyp"], b["knn_gain"], marker="s", color="#d95f02", label="kNN (similar tasks)")
ax.axhline(0, color="#999", lw=0.8, ls=":")
ax.set_xlabel("task atypicality  (1 - rank agreement with global flow ordering)")
ax.set_ylabel("adaptation value\n(portfolio regret - method regret)")
ax.set_title("Adaptation pays more on more heterogeneous tasks (long-tail, 1,236 tasks)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "heterogeneity_curve.png", dpi=150)
fig.savefig(FIG / "heterogeneity_curve.pdf")
print("\nper-bin means:")
print(b.round(4).to_string(index=False))
print("saved", FIG / "heterogeneity_curve.png")

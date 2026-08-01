"""Long-tail census: how many OpenML classification tasks are dense enough (many flow
evaluations) to support the recommender? Sample tasks, count distinct flows/task, extrapolate
to the full 5,576-task population."""
import warnings; warnings.filterwarnings("ignore")
import time
import numpy as np, pandas as pd
import openml, openml_flow as wf
from openml.tasks import TaskType
wf.configure_openml()

def retry(fn, t=4):
    for a in range(t):
        try: return fn()
        except Exception: time.sleep(2 + 2 * a)
    return "ERR"

allt = retry(lambda: openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_CLASSIFICATION, output_format="dataframe"))
task_ids = sorted(int(t) for t in allt["tid"].unique())
N = len(task_ids)
rng = np.random.default_rng(0)
SAMPLE = 150
sample = sorted(rng.choice(task_ids, size=min(SAMPLE, N), replace=False).tolist())
print(f"population: {N} classification tasks; sampling {len(sample)}", flush=True)

rows = []
for i, tid in enumerate(sample, 1):
    df = retry(lambda: openml.evaluations.list_evaluations(
        function="predictive_accuracy", tasks=[tid], size=1000, output_format="dataframe"))
    if isinstance(df, str):
        rows.append((tid, -1)); continue
    if df is None or len(df) == 0 or "flow_id" not in df.columns:
        rows.append((tid, 0)); continue
    rows.append((tid, df["flow_id"].nunique()))
    if i % 25 == 0:
        print(f"  {i}/{len(sample)} done", flush=True)

r = pd.DataFrame(rows, columns=["task", "distinct_flows"])
ok = r[r.distinct_flows >= 0]  # drop errors
err = (r.distinct_flows == -1).sum()

def frac(th): return (ok.distinct_flows >= th).mean()
print(f"\n=== LONG-TAIL CENSUS (n_sampled_ok={len(ok)}, errors={err}) ===")
print(f"  distinct flows/task: median={ok.distinct_flows.median():.0f} "
      f"mean={ok.distinct_flows.mean():.0f} max={ok.distinct_flows.max()}")
for th in [1, 20, 50, 100]:
    f = frac(th)
    print(f"  tasks with >= {th:>3} flows: {int((ok.distinct_flows>=th).sum())}/{len(ok)} "
          f"({f*100:.0f}%)  -> extrapolated ~{int(f*N)} of {N}")
print(f"  tasks with 0 evals: {int((ok.distinct_flows==0).sum())}/{len(ok)}")
r.to_csv("longtail_census_sample.csv", index=False)
# save the dense task ids we found (>=50) as a starter list
dense = ok[ok.distinct_flows >= 50]["task"].tolist()
print(f"\n dense (>=50) sampled task ids ({len(dense)}): {dense}")
print("DONE")

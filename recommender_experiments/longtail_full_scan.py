"""Full scan (Paper 2 phase 1): scan ALL 5,576 OpenML classification tasks, record each
task's distinct-flow count, and collect first-<=1000 evals for the dense ones (>=MIN_FLOWS).
Extends the existing checkpoint. Robust for a multi-hour run through flaky-endpoint outages:
  - on a failed call (all retries -> ERR), do NOT mark the task scanned (re-run retries it);
  - record flow counts for every scanned task (full census);
  - auto-stop after too many consecutive errors (endpoint likely down) -- re-run to resume."""
import warnings; warnings.filterwarnings("ignore")
import time, os, joblib
import numpy as np, pandas as pd
import openml, openml_flow as wf
from openml.tasks import TaskType
wf.configure_openml()

CKPT = "minimal_cache_cc18/longtail_collected.joblib"
TASKS_CACHE = "minimal_cache_cc18/longtail_tasks.joblib"
MIN_FLOWS = 50
SIZE = 1000
MAX_CONSEC_ERR = 25   # stop if the endpoint appears down

def retry(fn, t=5):
    for a in range(t):
        try: return fn()
        except Exception: time.sleep(2 + 2 * a)
    return "ERR"

state = joblib.load(CKPT) if os.path.exists(CKPT) else {"scanned": set(), "evals": {}}
state.setdefault("counts", {})   # tid -> distinct flow count (census over all scanned tasks)

allt = joblib.load(TASKS_CACHE) if os.path.exists(TASKS_CACHE) else \
    retry(lambda: openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_CLASSIFICATION, output_format="dataframe"))
task_ids = sorted(int(t) for t in allt["tid"].unique())
todo = [t for t in task_ids if t not in state["scanned"]]
print(f"population {len(task_ids)} | already scanned {len(state['scanned'])} | kept {len(state['evals'])} "
      f"| todo {len(todo)}", flush=True)

consec_err = 0
for i, tid in enumerate(todo, 1):
    df = retry(lambda: openml.evaluations.list_evaluations(
        function="predictive_accuracy", tasks=[tid], size=SIZE, output_format="dataframe"))
    if isinstance(df, str):                      # all retries failed -> likely outage
        consec_err += 1
        if consec_err >= MAX_CONSEC_ERR:
            joblib.dump(state, CKPT)
            print(f"STOP: {consec_err} consecutive errors (endpoint down). "
                  f"Progress saved: {len(state['scanned'])} scanned. Re-run to resume.", flush=True)
            break
        continue
    consec_err = 0
    n = df["flow_id"].nunique() if (df is not None and len(df) and "flow_id" in df.columns) else 0
    state["counts"][tid] = int(n)
    state["scanned"].add(tid)
    if n >= MIN_FLOWS:
        keep = [c for c in ["task_id", "flow_id", "flow_name", "value"] if c in df.columns]
        state["evals"][tid] = df[keep].copy()
    if i % 50 == 0:
        joblib.dump(state, CKPT)
        print(f"  {i}/{len(todo)} this run | scanned {len(state['scanned'])} | "
              f"dense(>= {MIN_FLOWS}) kept {len(state['evals'])}", flush=True)
else:
    print("scan complete (all tasks processed)", flush=True)

joblib.dump(state, CKPT)
counts = pd.Series(state["counts"])
print(f"\nSCAN STATE: scanned {len(state['scanned'])}/{len(task_ids)} | dense kept {len(state['evals'])}")
if len(counts):
    for th in [1, 20, 50, 100]:
        print(f"  tasks with >= {th:>3} flows: {int((counts>=th).sum())}")
print("DONE" if len(todo) and consec_err < MAX_CONSEC_ERR else "PARTIAL (re-run to continue)")

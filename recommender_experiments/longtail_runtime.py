"""Collect per-(task,flow) runtime (usercpu_time_millis) for the 1,236 long-tail tasks, for
cost-aware AutoML search (Paper 3). Resumable + checkpointed + self-stop on outage."""
import warnings; warnings.filterwarnings("ignore")
import time, os, joblib
import numpy as np, pandas as pd
import openml, openml_flow as wf
wf.configure_openml()

CKPT = "minimal_cache_cc18/longtail_runtime.joblib"
SIZE = 1000
MAX_CONSEC_ERR = 25

def retry(fn, t=5):
    for a in range(t):
        try: return fn()
        except Exception: time.sleep(2 + 2 * a)
    return "ERR"

state = joblib.load(CKPT) if os.path.exists(CKPT) else {"scanned": set(), "rt": {}}
coll = joblib.load("minimal_cache_cc18/longtail_collected.joblib")
task_ids = sorted(int(t) for t in coll["evals"].keys())   # the 1,236 dense tasks
todo = [t for t in task_ids if t not in state["scanned"]]
print(f"{len(task_ids)} dense tasks | scanned {len(state['scanned'])} | with runtime {len(state['rt'])} | todo {len(todo)}", flush=True)

consec = 0
for i, tid in enumerate(todo, 1):
    df = retry(lambda: openml.evaluations.list_evaluations(
        function="usercpu_time_millis", tasks=[tid], size=SIZE, output_format="dataframe"))
    if isinstance(df, str):
        consec += 1
        if consec >= MAX_CONSEC_ERR:
            joblib.dump(state, CKPT); print(f"STOP: outage. {len(state['rt'])} tasks have runtime. Re-run to resume.", flush=True); break
        continue
    consec = 0; state["scanned"].add(tid)
    if df is not None and len(df) and "flow_id" in df.columns:
        state["rt"][tid] = df[["task_id", "flow_id", "value"]].copy()
    if i % 50 == 0:
        joblib.dump(state, CKPT); print(f"  {i}/{len(todo)} | runtime for {len(state['rt'])} tasks", flush=True)
else:
    print("runtime scan complete", flush=True)
joblib.dump(state, CKPT)
print(f"\nDONE: {len(state['scanned'])}/{len(task_ids)} scanned, {len(state['rt'])} with runtime")

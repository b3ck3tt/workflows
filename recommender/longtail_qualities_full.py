"""Fetch OpenML qualities for ALL 1,236 long-tail datasets (resumable; extends the 235 already
fetched). Self-stops on sustained outage; re-run to resume."""
import warnings; warnings.filterwarnings("ignore")
import time, os, joblib
import pandas as pd, openml, openml_flow as wf
wf.configure_openml()
state = joblib.load("minimal_cache_cc18/longtail_collected.joblib")
tasks_all = joblib.load("minimal_cache_cc18/longtail_tasks.joblib")
tids = set(int(t) for t in state["evals"].keys())
tmap = tasks_all[tasks_all["tid"].astype(int).isin(tids)]
dids = sorted(set(int(d) for d in tmap["did"].dropna().unique()))
CKPT = "minimal_cache_cc18/longtail_qualities_partial.joblib"
FINAL = "minimal_cache_cc18/longtail_qualities.joblib"
rows = joblib.load(CKPT) if os.path.exists(CKPT) else {}
print(f"{len(tids)} tasks -> {len(dids)} datasets | already have {len(rows)} | todo {len([d for d in dids if d not in rows])}", flush=True)
def getq(did):
    for a in range(4):
        try:
            d = openml.datasets.get_dataset(did, download_data=False, download_qualities=True, download_features_meta_data=False)
            return dict(d.qualities or {})
        except Exception: time.sleep(2+2*a)
    return None
todo = [d for d in dids if d not in rows]; consec=0
for i, did in enumerate(todo, 1):
    q = getq(did)
    if q is None:
        consec += 1
        if consec >= 25:
            joblib.dump(rows, CKPT); print(f"STOP: outage after {len(rows)} done. Re-run to resume.", flush=True); break
        continue
    consec = 0; rows[did] = q
    if i % 25 == 0:
        joblib.dump(rows, CKPT); print(f"  {i}/{len(todo)} | have {len(rows)}/{len(dids)}", flush=True)
joblib.dump(rows, CKPT)
qdf = pd.DataFrame.from_dict(rows, orient="index"); qdf.index.name="dataset_id"; joblib.dump(qdf, FINAL)
print(f"\nDONE/PARTIAL: {qdf.shape[0]}/{len(dids)} datasets, {qdf.shape[1]} cols")

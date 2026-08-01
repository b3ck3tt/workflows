"""Paper 4 (full): LLM-as-recommender at flow level, with prompt caching + cost dry-run.

Design for cheap caching: every prompt = [ cached prefix: instructions + FULL flow catalog ] +
[ per-task suffix: anonymized metafeatures + candidate flow IDs + "rank them" ]. The catalog is
byte-identical across tasks -> written once, read at ~0.1x thereafter.

Modes:
  build            build catalog + per-task packs (+ hidden ground truth/baselines), save to disk
  dryrun [model]   estimate tokens + project cost per model (NO API, NO key needed)
  run <model> [N]  call Anthropic with caching on N tasks; save rankings + real token usage
  eval             score LLM rankings vs trained baselines (text-LTR/kNN/portfolio/random)
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, re, joblib, numpy as np, pandas as pd
from pathlib import Path
OUT = Path("results_recommender_paper4"); OUT.mkdir(exist_ok=True)
SCR = Path("scratch")

RNG = np.random.default_rng(7)
N_TASKS = int(os.environ.get("N_TASKS", "200"))
MIN_CAND = 15
META = ["NumberOfInstances","NumberOfFeatures","NumberOfClasses","NumberOfNumericFeatures",
        "NumberOfSymbolicFeatures","MajorityClassSize","MinorityClassSize","NumberOfInstancesWithMissingValues"]

# approx Anthropic pricing $/Mtok (input, output, cache-write-1h, cache-read). VERIFY before spend.
PRICING = {
    "claude-opus-4-8":            dict(inp=15.0, out=75.0, cw=18.75, cr=1.50),
    "claude-sonnet-5":            dict(inp=3.0,  out=15.0, cw=3.75,  cr=0.30),
    "claude-haiku-4-5-20251001":  dict(inp=0.80, out=4.0,  cw=1.00,  cr=0.08),
    "claude-fable-5":             dict(inp=10.0, out=50.0, cw=12.50, cr=1.00),  # premium-priced
}
def est_tokens(s): return int(len(s) / 3.5)  # rough; real usage comes from API response

def clean_flow(name):
    n = str(name or "")
    n = re.sub(r"\(.*?\)", "", n)                       # drop param blobs
    n = n.replace("sklearn.", "").replace("weka.classifiers.", "weka.").replace("mlr.", "mlr.")
    return n.strip()[:80]

def load_longtail():
    coll = joblib.load("minimal_cache_cc18/longtail_collected.joblib")
    tasks_all = joblib.load("minimal_cache_cc18/longtail_tasks.joblib")
    ev = pd.concat(coll["evals"].values(), ignore_index=True)
    ev["flow_id"] = pd.to_numeric(ev["flow_id"], errors="coerce"); ev["value"] = pd.to_numeric(ev["value"], errors="coerce")
    ev = ev.dropna(subset=["flow_id","value"]); ev["flow_id"] = ev["flow_id"].astype(int)
    agg = ev.groupby(["task_id","flow_id"], as_index=False)["value"].max().rename(columns={"value":"acc"})
    fname = ev.drop_duplicates("flow_id").set_index("flow_id")["flow_name"].to_dict()
    return agg, fname, tasks_all

INSTRUCTIONS = (
 "You are an expert AutoML assistant. Given a dataset described ONLY by anonymized numeric "
 "meta-features (no name), and a list of candidate machine-learning pipelines (referenced by ID from "
 "the catalog below), rank the candidates from most to least likely to achieve the highest predictive "
 "accuracy on that dataset. Use your knowledge of how algorithm families behave on datasets of "
 "different size, dimensionality, and class balance. Output ONLY the candidate IDs in ranked order, "
 "best first, comma-separated (e.g. F12,F3,F45). No explanation.\n\nPIPELINE CATALOG:\n")

def build():
    agg, fname, tasks_all = load_longtail()
    tmeta = tasks_all.set_index("tid")[META]
    cand = agg.groupby("task_id").size(); tasks = cand[cand >= MIN_CAND].index.to_numpy()
    RNG.shuffle(tasks); tasks = sorted(tasks[:N_TASKS])
    # catalog = union of candidate flows across selected tasks
    used = sorted(agg[agg.task_id.isin(tasks)]["flow_id"].unique())
    fid2cat = {f: f"F{i}" for i, f in enumerate(used)}
    catalog = "\n".join(f"{fid2cat[f]}: {clean_flow(fname.get(f,''))}" for f in used)
    prefix = INSTRUCTIONS + catalog
    packs = []; truth = {}
    for tid in tasks:
        g = agg[agg.task_id == tid]
        if tid not in tmeta.index: continue
        mv = tmeta.loc[tid]
        meta_str = "; ".join(f"{c}={int(mv[c]) if pd.notna(mv[c]) else 'NA'}" for c in META)
        cids = [fid2cat[f] for f in g["flow_id"]]
        suffix = (f"\n\nDATASET (anonymized) meta-features: {meta_str}\n"
                  f"Candidate pipeline IDs ({len(cids)}): {','.join(cids)}\n"
                  f"Rank ALL of these candidate IDs best-to-worst, comma-separated:")
        packs.append({"tid": int(tid), "suffix": suffix, "cids": cids})
        truth[int(tid)] = g[["flow_id","acc"]].assign(cat=g["flow_id"].map(fid2cat)).to_dict("records")
    joblib.dump({"prefix": prefix, "packs": packs, "truth": truth, "fid2cat": fid2cat},
                OUT / "pack.joblib")
    ntok_prefix = est_tokens(prefix)
    ntok_suffix = int(np.mean([est_tokens(p["suffix"]) for p in packs]))
    ncand = int(np.mean([len(p["cids"]) for p in packs]))
    print(f"built: {len(packs)} tasks, catalog {len(used)} flows")
    print(f"prefix ~{ntok_prefix} tok (cached), mean suffix ~{ntok_suffix} tok, mean {ncand} candidates/task")
    return ntok_prefix, ntok_suffix, ncand, len(packs)

def dryrun():
    d = joblib.load(OUT / "pack.joblib")
    prefix = d["prefix"]; packs = d["packs"]
    ntok_prefix = est_tokens(prefix)
    tok_suffix = [est_tokens(p["suffix"]) for p in packs]
    est_out = [max(60, len(p["cids"]) * 4) for p in packs]   # ~4 tok per "F123," in output
    N = len(packs)
    print(f"\n{N} tasks | prefix(cached) ~{ntok_prefix} tok | suffix sum ~{sum(tok_suffix)} | output sum ~{sum(est_out)}\n")
    print(f"{'model':>28} {'naive $':>9} {'cached $':>9} {'cached+batch $':>15}")
    for m, pr in PRICING.items():
        # naive: full prefix every call
        naive = (N*ntok_prefix + sum(tok_suffix))/1e6*pr["inp"] + sum(est_out)/1e6*pr["out"]
        # cached: write prefix once (cw), read prefix N-1 times (cr), suffixes full, output full
        cached = (ntok_prefix/1e6*pr["cw"] + (N-1)*ntok_prefix/1e6*pr["cr"]
                  + sum(tok_suffix)/1e6*pr["inp"] + sum(est_out)/1e6*pr["out"])
        print(f"{m:>28} {naive:>9.2f} {cached:>9.2f} {cached*0.5:>15.2f}")
    print("\n(estimates: tokens ~chars/3.5, pricing approximate — verify; real usage logged on first live call)")

def _load_key():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k: return k
    p = Path.home() / ".anthropic_key"
    if p.exists(): return p.read_text().strip()
    raise SystemExit("No API key: set ANTHROPIC_API_KEY or write ~/.anthropic_key")

def _parse_ranking(text, valid):
    ids = re.findall(r"F\d+", text or "")
    seen = set(); order = []
    for i in ids:
        if i in valid and i not in seen:
            seen.add(i); order.append(i)
    return order

def run(model, N=None):
    import anthropic
    d = joblib.load(OUT / "pack.joblib")
    prefix, packs = d["prefix"], d["packs"]
    if N: packs = packs[:N]
    client = anthropic.Anthropic(api_key=_load_key(), timeout=600.0, max_retries=1)
    resfile = OUT / f"llm_{model}.joblib"
    done = joblib.load(resfile) if resfile.exists() else {}
    usage_tot = dict(inp=0, out=0, cw=0, cr=0)
    for i, p in enumerate(packs):
        tid = p["tid"]
        prev = done.get(tid) or done.get(str(tid))
        if prev and "ranking" in prev:   # skip only successful tasks; retry errors
            continue
        valid = set(p["cids"])
        try:
            think = model.endswith("-think")          # e.g. "claude-sonnet-5-think" -> thinking ablation
            api_model = model[:-6] if think else model
            kw = dict(model=api_model,
                      system=[{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}],
                      messages=[{"role": "user", "content": p["suffix"]}])
            if think:              # Claude 5 extended thinking: adaptive, medium effort, GENEROUS ceiling
                kw["thinking"] = {"type": "adaptive"}
                kw["output_config"] = {"effort": "medium"}
                kw["max_tokens"] = min(20000, len(p["cids"]) * 8 + 10000)
            elif "fable" in model: # Fable forces adaptive thinking -> leave room for thinking + ranking
                kw["max_tokens"] = min(16000, len(p["cids"]) * 8 + 6000)
            else:                  # others: disable thinking, tight budget for the ranking
                kw["max_tokens"] = min(8000, len(p["cids"]) * 8 + 500)
                kw["thinking"] = {"type": "disabled"}
            msg = client.messages.create(**kw)
            txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            ranking = _parse_ranking(txt, valid)
            u = msg.usage
            cw = getattr(u, "cache_creation_input_tokens", 0) or 0
            cr = getattr(u, "cache_read_input_tokens", 0) or 0
            usage_tot["inp"] += u.input_tokens; usage_tot["out"] += u.output_tokens
            usage_tot["cw"] += cw; usage_tot["cr"] += cr
            done[tid] = {"ranking": ranking, "n_parsed": len(ranking), "n_cand": len(valid),
                         # per-task usage = a "reasoning effort / complexity" signal (esp. tok_out for thinking)
                         "tok_in": u.input_tokens, "tok_out": u.output_tokens,
                         "tok_cache_read": cr, "tok_cache_write": cw}
        except Exception as e:
            done[tid] = {"error": str(e)[:200]}
        if (i + 1) % 3 == 0 or i + 1 == len(packs):
            joblib.dump(done, resfile)
            print(f"  {i+1}/{len(packs)} done", flush=True)
    joblib.dump(done, resfile)
    pr = PRICING.get(model, PRICING["claude-sonnet-5"])
    cost = (usage_tot["inp"]/1e6*pr["inp"] + usage_tot["out"]/1e6*pr["out"]
            + usage_tot["cw"]/1e6*pr["cw"] + usage_tot["cr"]/1e6*pr["cr"])
    ok = sum(1 for v in done.values() if "ranking" in v)
    print(f"\n{model}: {ok}/{len(done)} tasks ranked | REAL usage {usage_tot} | est cost ${cost:.2f}")
    print(f"  (cache_read {usage_tot['cr']} vs input {usage_tot['inp']} -> caching {'WORKING' if usage_tot['cr']>usage_tot['inp'] else 'check'})")

def evaluate():
    from scipy.stats import spearmanr
    d = joblib.load(OUT / "pack.joblib"); truth = d["truth"]
    models = [f.stem.replace("llm_", "") for f in OUT.glob("llm_*.joblib")]
    if not models: raise SystemExit("no llm_<model>.joblib results yet")
    rows = []
    for model in models:
        res = joblib.load(OUT / f"llm_{model}.joblib")
        for tid, r in res.items():
            tid = int(tid)
            if "ranking" not in r or not r["ranking"] or tid not in truth: continue
            recs = truth[tid]; acc = {x["cat"]: x["acc"] for x in recs}
            best = max(acc.values()); worst = min(acc.values()); rng = best - worst
            if rng <= 0: continue
            rank = r["ranking"]
            top = rank[0]
            nreg1 = (best - acc.get(top, worst)) / rng
            hit1 = int(acc.get(top, -1) == best)
            # spearman: predicted rank position (0=best) vs accuracy -> expect negative; report -rho
            pos = {c: i for i, c in enumerate(rank)}
            common = [c for c in acc if c in pos]
            rho = np.nan
            if len(common) > 2:
                rho = spearmanr([pos[c] for c in common], [acc[c] for c in common]).correlation
            rows.append({"model": model, "tid": tid, "nregret@1": nreg1, "hit@1": hit1, "spearman": -rho if rho==rho else np.nan})
    L = pd.DataFrame(rows)
    print("\n=== LLM-as-recommender (flow-level) vs trained baselines ===")
    print(L.groupby("model").agg(n=("tid","nunique"), nregret1=("nregret@1","mean"),
                                 hit1=("hit@1","mean"), spearman=("spearman","mean")).round(4).to_string())
    tasks = set(L["tid"])
    pt = pd.read_csv("results_recommender_longtail_full/ranking_metrics_per_task.csv")
    pt = pt[(pt.experiment=="tfidf_meta") & (pt.task_id.isin(tasks))]
    print(f"\nbaselines on the same {len(tasks)} tasks:")
    print(pt.groupby("method").agg(nregret1=("nregret@1","mean"), hit1=("hit@1","mean"),
                                   spearman=("spearman","mean")).round(4).to_string())
    L.to_csv(OUT / "llm_metrics_per_task.csv", index=False)
    print("\nsaved llm_metrics_per_task.csv")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    if mode == "build": build()
    elif mode == "dryrun":
        if not (OUT/"pack.joblib").exists(): build()
        dryrun()
    elif mode == "run":
        model = sys.argv[2] if len(sys.argv) > 2 else "claude-sonnet-5"
        N = int(sys.argv[3]) if len(sys.argv) > 3 else None
        run(model, N)
    elif mode == "eval":
        evaluate()
    else:
        print("modes: build | dryrun | run <model> [N] | eval")

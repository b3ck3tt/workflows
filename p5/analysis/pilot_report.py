"""Decision-gate report from the pilot (zadani v2, section 2.3 metrics). Reads runs/<run>/summary.json,
aggregates per operator, and prints the numbers the gate decision is made on. No new spend."""
import sys, json, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import Counter, defaultdict
from metrics import diversity as D

run = sys.argv[1] if len(sys.argv) > 1 else "runs/pilot"
res = json.loads((Path(run) / "summary.json").read_text())
by = defaultdict(list)
for r in res:
    by[r["operator"]].append(r)


def mean(xs): return sum(xs) / len(xs) if xs else float("nan")


def best_at(r, k):
    """Best-so-far at <= k genuine evals, from the anytime trace (GA/BatchGA: 'anytime'; random_search:
    'trace'). Falls back to final best_fitness if no trace (old pilot data). This is what makes the
    LLM-vs-baseline comparison fair: all methods compared at the SAME evaluation budget, not their
    (unequal) final eval counts."""
    tr = r.get("anytime") or r.get("trace")
    if not tr:
        return r["best_fitness"]
    b = tr[0]["best"]
    for p in tr:
        if p["evals"] <= k:
            b = p["best"]
        else:
            break
    return b

print(f"=== PILOT DECISION-GATE REPORT ({run}) ===  runs: {len(res)}\n")

# --- search quality: per-operator mean best, and head-to-head vs LLM ---
print("Search quality (mean best accuracy over 9 runs):")
for op in ("llm", "random", "random_search"):
    if by[op]:
        print(f"  {op:14s} {mean([r['best_fitness'] for r in by[op]]):.4f}")
# paired head-to-head LLM vs each, per (task,seed)
def keyed(op): return {(r['task_id'], r['seed']): r['best_fitness'] for r in by[op]}
llm = keyed("llm")
for opp in ("random", "random_search"):
    o = keyed(opp); common = sorted(set(llm) & set(o))
    wins = sum(llm[k] > o[k] + 1e-9 for k in common)
    ties = sum(abs(llm[k] - o[k]) <= 1e-9 for k in common)
    losses = len(common) - wins - ties
    print(f"  LLM vs {opp:14s} over {len(common)} runs: {wins}W / {ties}T / {losses}L  "
          f"(mean Δ {mean([llm[k]-o[k] for k in common]):+.4f})")

# --- FAIR comparison at a COMMON evaluation budget (fixes unequal final eval counts) ---
resmaps = {op: {(r["task_id"], r["seed"]): r for r in by[op]} for op in by}
if any(r.get("anytime") or r.get("trace") for r in res):
    print("\nEqual-budget head-to-head (best-so-far at the COMMON min eval budget per run):")
    keys = sorted(resmaps["llm"]) if "llm" in resmaps else []
    from metrics.stats import bootstrap_ci, tost, DEFAULT_MARGIN
    from collections import defaultdict as _dd
    for opp in ("random", "random_search"):
        w = t = l = 0; per_task = _dd(list)
        for k in keys:
            if k not in resmaps.get(opp, {}):
                continue
            rl, ro = resmaps["llm"][k], resmaps[opp][k]
            budget = min(rl["total_evals"], ro["total_evals"])   # compare at the smaller budget
            a, b = best_at(rl, budget), best_at(ro, budget)
            d = a - b
            per_task[k[0]].append(d)               # k=(task,seed); average seeds within a task
            w += d > 1e-9; t += abs(d) <= 1e-9; l += d < -1e-9
        task_deltas = [mean(v) for v in per_task.values()]       # one delta per task (review §3.1)
        ci = bootstrap_ci(task_deltas); eq = tost(task_deltas, DEFAULT_MARGIN)
        print(f"  LLM vs {opp:14s}: {w}W/{t}T/{l}L per-run | per-task Δ {ci['mean']:+.4f} "
              f"[95% CI {ci['lo']:+.4f},{ci['hi']:+.4f}] | TOST±{DEFAULT_MARGIN}: "
              f"{'EQUIVALENT' if eq['equivalent'] else 'not-equiv'} (p={eq['p_equiv']:.3f}, n={ci['n']})")
else:
    print("\n[note] no anytime traces in this run -> equal-budget comparison unavailable "
          "(final-best above is confounded by unequal eval counts; Phase-3 runs carry traces).")

# --- operator quality metrics (LLM vs random), zadani 2.3 ---
print("\nOperator metrics (mean over runs):")
hdr = f"  {'metric':24s} {'llm':>8} {'random':>8}"
print(hdr)
for m in ("invalid_rate", "dup_pop_rate", "dup_hist_rate", "hp_substitution_rate", "new_structure_rate"):
    print(f"  {m:24s} {mean([r[m] for r in by['llm']]):8.3f} {mean([r[m] for r in by['random']]):8.3f}")

# violation breakdown (LLM)
vb = Counter()
for r in by["llm"]:
    for k, v in r.get("violation_breakdown", {}).items():
        vb[k] += v
print(f"\nLLM invalid-violation breakdown (counts): {dict(vb)}")

# --- diversity of final populations, THREE levels (review §3.6), LLM vs random ---
print("\nFinal-population diversity (mean over runs) — structural | genotypic | embedding:")
for op in ("llm", "random"):
    pops = [[json.loads(s) for s in r["final_population"]] for r in by[op]]
    st = mean([D.structure_entropy(p) for p in pops])          # structure-only (entropy of pre|fs|est)
    ge = mean([D.structural_diversity(p) for p in pops])       # genotypic (edit distance incl HP)
    em = mean([D.embedding_diversity(p) for p in pops])        # SNCS-D0 embedding
    print(f"  {op:8s} structural {st:.3f}   genotypic {ge:.3f}   embedding {em:.3f}")

# --- H4 (reformulated): are the LLM's duplicates concentrated (narrow) vs random's? ---
from metrics.stats import duplicate_concentration, bootstrap_ci, tost
print("\nDuplicate concentration (0=all repeats hit one genotype, 1=spread):")
for op in ("llm", "random"):
    dk = [k for r in by[op] for k in r.get("dup_keys", [])]
    print(f"  {op:8s} n_dups {len(dk):4d}  concentration {duplicate_concentration(dk):.3f}")

# --- extrapolated Phase-3 cost ---
spent = json.loads((Path(run) / "accounting.jsonl").read_text().splitlines()[-1])["cum_usd"]
pilot_calls = sum(1 for _ in open(Path(run) / "accounting.jsonl"))
# pilot: 3 tasks x 3 seeds; phase3: 15 x 5, population 50 gens 30 (vs pilot pop 12 gens 6)
scale = (15*5)/(3*3) * (50-2)*30 / ((12-2)*6)
print(f"\nCost: pilot spent ${spent:.3f} over {pilot_calls} calls. "
      f"Phase-3 extrapolation x{scale:.0f} -> ~${spent*scale:.1f} (cap check needed before Phase 3).")

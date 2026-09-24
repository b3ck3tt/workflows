"""Phase-3 driver (zadani sec.3): LLM-operator GA via the Batch API, plus offline random-GA and
random-search baselines, all at an equal evaluation budget. RUN ONLY AFTER the decision-gate consultation.

The LLM arm advances all (task, seed) runs in lockstep: one Batch-API call per generation carries every
run's offspring proposals together (~ (population-elitism) * n_runs requests), for the 50% discount and
robustness to flaky networks. The random-GA and random-search arms make no API calls and run offline.

  P5_PREFLIGHT_ONLY=1 python run_phase3.py config/phase3_base.yaml   # estimate only, no spend
  P5_ASSUME_YES=1     python run_phase3.py config/phase3_base.yaml   # skip the confirm prompt

Outputs mirror run_pilot: runs/<run_id>/result_{llm,random,random_search}_<tid>_<seed>.json + summary.json.
Idempotent-friendly: existing valid result files are skipped, so a killed run can be relaunched.
"""
import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, yaml
from llm.accounting import Ledger, BudgetExceeded
from llm.batch_client import BatchClient
from llm.prompt import build_suffix
from operators.llm_operator import LLMOperator
from operators.random_operator import RandomOperator
from search import representation as R
from search.ga import GA
from search.ga_batch import BatchGA
from search.random_search import random_search
from eval.evaluate import Evaluator


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    grammar = R.load_grammar(cfg["grammar"])
    run_dir = Path("runs") / cfg["run_id"]; run_dir.mkdir(parents=True, exist_ok=True)
    tasks, seeds = cfg["datasets"], cfg["seeds"]
    pop, gens, elit = cfg["population"], cfg["generations"], cfg["elitism"]
    max_evals = cfg.get("eval_budget", pop + gens * (pop - elit))   # genuine-eval budget (C1: primary stop)
    gen_ceiling = cfg.get("max_generations", 3 * gens)             # safety ceiling; eval_budget binds first

    ledger = Ledger(run_dir, pricing_path=cfg["pricing"], cap_usd=cfg["budget_usd"])
    op = LLMOperator(None, grammar, include_fitness=cfg.get("fitness_in_prompt", True),
                     include_history=cfg.get("history_in_prompt", True),
                     p_crossover=cfg.get("p_crossover", 0.5))
    # Executor by vendor: Anthropic -> server-side Batch API (50% off); otherwise (open-weight via an
    # OpenAI-compatible endpoint) -> concurrent thread-pool with the same run_batch() interface (C2).
    if cfg["model"].startswith("claude"):
        bc = BatchClient(cfg["model"], ledger, max_tokens=1200, poll_s=float(cfg.get("poll_s", 20)))
    else:
        from llm.openai_client import ConcurrentBatchClient
        bc = ConcurrentBatchClient(cfg["model"], ledger, base_url=cfg.get("base_url"), max_tokens=1200,
                                   temperature=float(cfg.get("temperature", 0.7)),
                                   max_workers=int(cfg.get("concurrency", 8)))
    fallback = RandomOperator(grammar)
    prefix = op.prefix

    # ---- preflight (zadani 0.4), batch-priced ----
    rng = np.random.default_rng(0)
    sample_suffix = op.build([R.random_genotype(grammar, rng), R.random_genotype(grammar, rng)],
                             {"rng": rng, "note": {}, "fitness": [0.8, 0.7], "history": ["{...}"] * 5})
    # C1: with eval-budget stopping, a run keeps proposing until it spends max_evals GENUINE evals, so it
    # issues more proposals than max_evals when it duplicates. Inflate the estimate by a headroom factor
    # (~1/(1-dup)); the hard cap is the real guard.
    dup_headroom = float(cfg.get("dup_headroom", 1.3))
    n_llm_calls = int(len(tasks) * len(seeds) * (max_evals - pop) * dup_headroom)
    est = bc.estimate_usd(prefix, sample_suffix, n_llm_calls)
    print(f"[preflight] Batch LLM-operator: ~{n_llm_calls} calls (eval_budget {max_evals}/run x{dup_headroom} "
          f"headroom) -> est ${est:.2f} (batch-priced, cap ${cfg['budget_usd']})")
    print(f"[preflight] {len(tasks)} tasks x {len(seeds)} seeds; equal genuine-eval budget {max_evals}/run; "
          f"gen ceiling {gen_ceiling}; lockstep batches up to {len(tasks)*len(seeds)*(pop-elit)} req/round")
    if os.environ.get("P5_PREFLIGHT_ONLY") == "1":
        print("[preflight-only] no spend."); return
    if ledger.would_exceed(est):
        raise SystemExit(f"[abort] estimate ${est:.2f} would exceed cap ${cfg['budget_usd']}")
    if os.environ.get("P5_ASSUME_YES") != "1" and input("[preflight] proceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("aborted at preflight")

    def rfile(kind, tid, seed):
        return run_dir / f"result_{kind}_{tid}_{seed}.json"

    def llm_valid(p):   # skip re-running a clean LLM result (resume after a kill)
        return p.exists() and "api_error" not in json.loads(p.read_text()).get("violation_breakdown", {})

    # ---- LLM arm: lockstep Batch-API GA over all (task, seed) that still need it ----
    gas = {}
    for tid in tasks:
        for seed in seeds:
            if llm_valid(rfile("llm", tid, seed)):
                continue
            # cache_dir=None -> in-memory eval cache (budget-correct even on resume, see Evaluator)
            ev = Evaluator(cv_folds=cfg["cv_folds"], eval_seed=cfg["eval_seed"], row_cap=cfg["row_cap"],
                           cache_dir=None, eval_timeout=cfg.get("eval_timeout"))
            ga = BatchGA(ev, op, grammar, population=pop, generations=gens, tournament_k=cfg["tournament_k"],
                         elitism=elit, fallback=fallback, max_evals=max_evals,
                         max_generations=gen_ceiling, run_id=cfg["run_id"])
            ga.start(tid, seed)
            gas[(tid, seed)] = ga

    rnd = 0
    while any(not ga.finished for ga in gas.values()):
        rnd += 1
        items, pendings = [], {}
        for key, ga in gas.items():
            if ga.finished:
                continue
            ctxs = ga.gen_contexts()
            if not ctxs:
                continue
            pendings[key] = ctxs
            for j, (parents, ctx) in enumerate(ctxs):
                suffix = op.build(parents, ctx)   # sets ctx note mode; prefix is shared/static
                # custom_id must match ^[a-zA-Z0-9_-]{1,64}$ (no ':')
                items.append((f"{key[0]}_{key[1]}-{j}", suffix,
                              {"run": f"{key[0]}_{key[1]}", "generation": ctx["generation"]}))
        if not items:
            break
        results = bc.run_batch(prefix, items, run_id=cfg["run_id"])
        for key, ctxs in pendings.items():
            raws = [op.interpret(results.get(f"{key[0]}_{key[1]}-{j}"), ctx)
                    for j, (parents, ctx) in enumerate(ctxs)]
            gas[key].apply(raws)
        active = sum(1 for g in gas.values() if not g.finished)
        print(f"  round {rnd}: {len(items)} proposals | {active} runs active | spent ${ledger.spent:.3f}",
              flush=True)

    for key, ga in gas.items():
        r = ga.result(); r["run_id"] = cfg["run_id"]
        rfile("llm", key[0], key[1]).write_text(json.dumps(r))

    # ---- offline baselines (no API): random-GA and random-search ----
    for tid in tasks:
        for seed in seeds:
            if not rfile("random", tid, seed).exists():
                ev = Evaluator(cv_folds=cfg["cv_folds"], eval_seed=cfg["eval_seed"], row_cap=cfg["row_cap"],
                               cache_dir=None, eval_timeout=cfg.get("eval_timeout"))
                r = GA(ev, fallback, grammar, population=pop, generations=gens, tournament_k=cfg["tournament_k"],
                       elitism=elit, fallback=fallback, max_evals=max_evals,
                       max_generations=gen_ceiling).run(tid, seed)
                r["run_id"] = cfg["run_id"]; rfile("random", tid, seed).write_text(json.dumps(r))
            if not rfile("random_search", tid, seed).exists():
                ev = Evaluator(cv_folds=cfg["cv_folds"], eval_seed=cfg["eval_seed"], row_cap=cfg["row_cap"],
                               cache_dir=None, eval_timeout=cfg.get("eval_timeout"))
                r = random_search(ev, grammar, tid, seed, max_evals)
                rfile("random_search", tid, seed).write_text(json.dumps(r))
            print(f"  baselines {tid}/{seed} done | spent ${ledger.spent:.3f}", flush=True)

    results = [json.loads(p.read_text()) for p in sorted(run_dir.glob("result_*.json"))]
    (run_dir / "summary.json").write_text(json.dumps(results))
    print(f"\n[done] {len(results)} runs | {ledger.summary()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config/phase3_base.yaml")

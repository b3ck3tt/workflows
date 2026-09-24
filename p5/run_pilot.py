"""Pilot runner (zadani 2.4): LLM-operator GA vs random-operator GA vs random search, on the pilot
tasks x seeds, at an equal evaluation budget. Preflights the LLM cost and refuses to start if the
estimate would exceed the cap; set P5_PREFLIGHT_ONLY=1 to print the estimate and exit without spending,
or P5_ASSUME_YES=1 to skip the interactive confirm.

Usage:
  P5_PREFLIGHT_ONLY=1 python run_pilot.py config/pilot.yaml     # estimate only, no spend
  python run_pilot.py config/pilot.yaml                         # runs after preflight confirm
"""
from __future__ import annotations
import sys, os, json, time
from pathlib import Path
import numpy as np, yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search import representation as R
from search.ga import GA
from search.random_search import random_search
from operators.random_operator import RandomOperator
from operators.llm_operator import LLMOperator
from eval.evaluate import Evaluator
from llm.accounting import Ledger
from llm.client import LLMClient
from llm.prompt import build_prefix, build_suffix


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    grammar = R.load_grammar(cfg["grammar"])
    run_dir = Path("runs") / cfg["run_id"]; run_dir.mkdir(parents=True, exist_ok=True)
    tasks, seeds = cfg["datasets"], cfg["seeds"]
    pop, gens, elit = cfg["population"], cfg["generations"], cfg["elitism"]
    max_evals = pop + gens * (pop - elit)             # equal budget for every method (uncached upper bound)

    ledger = Ledger(run_dir, pricing_path=cfg["pricing"], cap_usd=cfg["budget_usd"])
    client = LLMClient(cfg["model"], ledger)

    # ---- preflight (zadani 0.4) ----
    rng = np.random.default_rng(0)
    prefix = build_prefix(R.grammar_text(grammar))
    sample_suffix = build_suffix([R.random_genotype(grammar, rng), R.random_genotype(grammar, rng)],
                                 fitness=[0.8, 0.7], history=["{...}"] * 5)
    n_llm_calls = len(tasks) * len(seeds) * gens * (pop - elit)   # one propose per offspring
    est = client.estimate_run_cost(prefix, sample_suffix, n_llm_calls, est_out_tokens=250)
    print(f"[preflight] LLM-operator: ~{est['in_tokens']} in-tok x {n_llm_calls} calls "
          f"(+ up to 1 retry each) -> est ${est['est_usd']}  (cap ${cfg['budget_usd']})")
    print(f"[preflight] budget/run = {max_evals} evals; {len(tasks)} tasks x {len(seeds)} seeds "
          f"x 3 methods (LLM-GA, random-GA, random-search)")
    if os.environ.get("P5_PREFLIGHT_ONLY") == "1":
        print("[preflight-only] no spend; rerun without P5_PREFLIGHT_ONLY to execute.")
        return
    if ledger.would_exceed(est["est_usd"]):
        raise SystemExit(f"[abort] estimate ${est['est_usd']} would exceed cap ${cfg['budget_usd']}")
    if os.environ.get("P5_ASSUME_YES") != "1" and input("[preflight] proceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("aborted at preflight")

    # ---- run ----
    def evaluator(tag):    # fresh cache per (method,task,seed) -> fair budget (README fairness rule)
        return Evaluator(cv_folds=cfg["cv_folds"], eval_seed=cfg["eval_seed"],
                         row_cap=cfg["row_cap"], cache_dir=run_dir / f"_cache_{tag}")
    results = []
    rnd = RandomOperator(grammar)
    for tid in tasks:
        for seed in seeds:
            # LLM-operator GA (random fallback)
            llm_op = LLMOperator(client, grammar,
                                 include_fitness=cfg.get("fitness_in_prompt", True),
                                 include_history=cfg.get("history_in_prompt", True),
                                 p_crossover=cfg.get("p_crossover", 0.5))   # 0.0 = mutation-only ablation
            r_llm = GA(evaluator(f"llm_{tid}_{seed}"), llm_op, grammar, population=pop, generations=gens,
                       tournament_k=cfg["tournament_k"], elitism=elit, fallback=rnd,
                       max_evals=max_evals).run(tid, seed)
            r_llm["run_id"] = cfg["run_id"]
            # random-operator GA
            r_rnd = GA(evaluator(f"rnd_{tid}_{seed}"), rnd, grammar, population=pop, generations=gens,
                       tournament_k=cfg["tournament_k"], elitism=elit, fallback=rnd,
                       max_evals=max_evals).run(tid, seed)
            # random search
            r_rs = random_search(evaluator(f"rs_{tid}_{seed}"), grammar, tid, seed, max_evals)
            for r in (r_llm, r_rnd, r_rs):
                results.append(r)
                (run_dir / f"result_{r['operator']}_{tid}_{seed}.json").write_text(json.dumps(r))
            print(f"  task {tid} seed {seed}: LLM {r_llm['best_fitness']:.4f} "
                  f"(inv {r_llm['invalid_rate']:.0%}, dup {r_llm['dup_hist_rate']:.0%}) | "
                  f"randGA {r_rnd['best_fitness']:.4f} | randSearch {r_rs['best_fitness']:.4f} | "
                  f"spent ${ledger.spent:.3f}", flush=True)
    (run_dir / "summary.json").write_text(json.dumps(results))
    print(f"\n[done] {ledger.summary()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config/pilot.yaml")

"""Idempotent targeted rerun for a pilot run dir after a partial failure (e.g. a network outage that
made LLM calls raise api_error). For every (task, seed) in the config it reruns only the method-arms
that are missing or corrupted, then rebuilds summary.json from all result_*.json on disk.

  A run needs rerun if:  llm  -> file missing OR violation_breakdown contains 'api_error'
                         random/random_search -> file missing (offline, cannot be net-corrupted)

Reuses the exact construction from run_pilot.py (fresh evaluator per run = fairness rule) and the same
Ledger (resume-safe: continues cumulative spend under the cap). Usage: python analysis/rerun_missing.py config/pilot_hard.yaml
"""
import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from llm.accounting import Ledger
from llm.client import LLMClient
from search import representation as R
from search.ga import GA
from search.random_search import random_search
from operators.random_operator import RandomOperator
from operators.llm_operator import LLMOperator
from eval.evaluate import Evaluator


def llm_corrupt(path: Path) -> bool:
    if not path.exists():
        return True
    d = json.loads(path.read_text())
    return "api_error" in d.get("violation_breakdown", {})


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    grammar = R.load_grammar(cfg["grammar"])
    run_dir = Path("runs") / cfg["run_id"]
    tasks, seeds = cfg["datasets"], cfg["seeds"]
    pop, gens, elit = cfg["population"], cfg["generations"], cfg["elitism"]
    max_evals = pop + gens * (pop - elit)
    ledger = Ledger(run_dir, pricing_path=cfg["pricing"], cap_usd=cfg["budget_usd"])
    client = LLMClient(cfg["model"], ledger)
    rnd = RandomOperator(grammar)

    def evaluator(tag):
        return Evaluator(cv_folds=cfg["cv_folds"], eval_seed=cfg["eval_seed"],
                         row_cap=cfg["row_cap"], cache_dir=run_dir / f"_cache_{tag}")

    def rfile(op, tid, seed):
        return run_dir / f"result_{op}_{tid}_{seed}.json"

    for tid in tasks:
        for seed in seeds:
            did = False
            if llm_corrupt(rfile("llm", tid, seed)):
                op = LLMOperator(client, grammar, include_fitness=cfg.get("fitness_in_prompt", True),
                                 include_history=cfg.get("history_in_prompt", True),
                                 p_crossover=cfg.get("p_crossover", 0.5))
                r = GA(evaluator(f"llm_{tid}_{seed}"), op, grammar, population=pop, generations=gens,
                       tournament_k=cfg["tournament_k"], elitism=elit, fallback=rnd,
                       max_evals=max_evals).run(tid, seed)
                r["run_id"] = cfg["run_id"]
                rfile("llm", tid, seed).write_text(json.dumps(r))
                print(f"  reran LLM {tid}/{seed}: best {r['best_fitness']:.4f} inv {r['invalid_rate']:.0%} "
                      f"vb {r['violation_breakdown']} | spent ${ledger.spent:.3f}", flush=True); did = True
            if not rfile("random", tid, seed).exists():
                r = GA(evaluator(f"rnd_{tid}_{seed}"), rnd, grammar, population=pop, generations=gens,
                       tournament_k=cfg["tournament_k"], elitism=elit, fallback=rnd,
                       max_evals=max_evals).run(tid, seed)
                r["run_id"] = cfg["run_id"]; rfile("random", tid, seed).write_text(json.dumps(r))
                print(f"  reran randGA {tid}/{seed}: best {r['best_fitness']:.4f}", flush=True); did = True
            if not rfile("random_search", tid, seed).exists():
                r = random_search(evaluator(f"rs_{tid}_{seed}"), grammar, tid, seed, max_evals)
                rfile("random_search", tid, seed).write_text(json.dumps(r))
                print(f"  reran randSearch {tid}/{seed}: best {r['best_fitness']:.4f}", flush=True); did = True
            if not did:
                print(f"  ok {tid}/{seed} (all arms valid)", flush=True)

    # rebuild summary.json from every result file on disk
    results = [json.loads(p.read_text()) for p in sorted(run_dir.glob("result_*.json"))]
    (run_dir / "summary.json").write_text(json.dumps(results))
    print(f"\n[done] rebuilt summary.json with {len(results)} runs | {ledger.summary()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config/pilot_hard.yaml")

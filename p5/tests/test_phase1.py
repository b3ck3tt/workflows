"""Phase-1 offline tests: GA control flow with a mock evaluator (no network/data), operator fallback
accounting, and representation round-trips. Live CV eval is smoke-tested separately on CC-18 task 11.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from search import representation as R
from search.ga import GA
from operators.random_operator import RandomOperator
from operators.base import Operator


class MockEvaluator:
    """Deterministic fake fitness from the genotype hash; counts genuine evals with a cache."""
    def __init__(self):
        self.n_evals = 0; self._c = {}
    def evaluate(self, g, task_id):
        k = R.serialize(g)
        if k not in self._c:
            self.n_evals += 1
            self._c[k] = {"accuracy": (hash(k) % 1000) / 1000.0, "error": None}
        return self._c[k]


class AlwaysNoneOperator(Operator):
    name = "none"
    def propose(self, parents, context): return None   # forces fallback every time


def test_ga_runs_and_improves():
    g = R.load_grammar("config/grammar.yaml")
    op = RandomOperator(g)
    res = GA(MockEvaluator(), op, g, population=10, generations=6, elitism=2, fallback=op).run(0, seed=0)
    assert len(res["generations"]) == 7
    bests = [gg["best"] for gg in res["generations"]]
    assert bests == sorted(bests), "elitism should make best monotonic non-decreasing"
    assert res["total_evals"] <= 10 + 6 * 8  # bounded by population + offspring
    print(f"OK GA runs, best monotone {bests[0]:.3f}->{bests[-1]:.3f}, evals {res['total_evals']}")


def test_fallback_accounting():
    g = R.load_grammar("config/grammar.yaml")
    res = GA(MockEvaluator(), AlwaysNoneOperator(), g, population=8, generations=3,
             elitism=2, fallback=RandomOperator(g)).run(0, seed=1)
    # every offspring should have triggered a fallback (operator always returns None)
    assert res["fallbacks"] > 0 and res["invalids"] == res["fallbacks"]
    assert "hp_substitution_rate" in res and "dup_hist_rate" in res
    print(f"OK fallback accounting: {res['fallbacks']} fallbacks logged (LLM-operator failure path)")


def test_max_evals_budget():
    g = R.load_grammar("config/grammar.yaml")
    res = GA(MockEvaluator(), RandomOperator(g), g, population=10, generations=100,
             elitism=2, fallback=RandomOperator(g), max_evals=30).run(0, seed=2)
    assert res["total_evals"] <= 31, "max_evals budget not respected"
    print(f"OK max_evals budget respected ({res['total_evals']} <= 30ish)")


if __name__ == "__main__":
    test_ga_runs_and_improves()
    test_fallback_accounting()
    test_max_evals_budget()
    print("\nall phase-1 offline tests passed")

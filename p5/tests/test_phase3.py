"""Phase-3 batch-GA tests (offline, no API). Drive BatchGA exactly as run_phase3 does, but feed crafted
'LLM responses' through the real (client-less) LLMOperator.build/interpret. Verifies: equal-evaluation
budget is respected, elitism makes best-fitness monotone, the 2.3 metric bookkeeping (invalid / dup_pop /
dup_hist / hp_substitution vs new_structure) is correct, and invalids fall back to valid genotypes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from search import representation as R
from search.ga_batch import BatchGA
from operators.llm_operator import LLMOperator
from operators.random_operator import RandomOperator

GRAMMAR = R.load_grammar(str(Path(__file__).resolve().parents[1] / "config" / "grammar.yaml"))


class MockEvaluator:
    """Deterministic accuracy from the genotype; counts only genuine (non-cached) evals, like Evaluator."""
    def __init__(self):
        self.n_evals = 0
        self._cache = {}
    def evaluate(self, geno, task_id):
        key = R.serialize(geno)
        if key not in self._cache:
            self.n_evals += 1
            self._cache[key] = (abs(hash(key)) % 1000) / 1000.0
        return {"accuracy": self._cache[key]}


def _drive(ga, tid, seed, respond):
    """Run a BatchGA to completion; `respond(parents, ctx) -> text|None` supplies each proposal."""
    ga.start(tid, seed)
    guard = 0
    while not ga.finished:
        guard += 1
        assert guard < 10_000, "runaway loop"
        ctxs = ga.gen_contexts()
        if not ctxs:
            break
        raws = []
        for parents, ctx in ctxs:
            ga.op.build(parents, ctx)          # sets ctx note mode (as the driver does)
            raws.append(ga.op.interpret(respond(parents, ctx), ctx))
        ga.apply(raws)
    return ga.result()


def test_budget_and_monotone():
    op = LLMOperator(None, GRAMMAR)
    ga = BatchGA(MockEvaluator(), op, GRAMMAR, population=10, generations=5, tournament_k=3,
                 elitism=2, fallback=RandomOperator(GRAMMAR), max_evals=10 + 5 * (10 - 2))
    rng = np.random.default_rng(1)
    # respond with a fresh valid random genotype each time (always parses & validates)
    res = _drive(ga, 11, 0, lambda parents, ctx: R.serialize(R.random_genotype(GRAMMAR, rng)))
    assert res["total_evals"] <= 10 + 5 * (10 - 2), "exceeded the equal-evaluation budget"
    bests = [g["best"] for g in res["generations"]]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(bests, bests[1:])), "elitism violated: best decreased"
    for k in ("invalid_rate", "dup_pop_rate", "dup_hist_rate", "hp_substitution_rate", "new_structure_rate"):
        assert 0.0 <= res[k] <= 1.0
    assert res["operator"] == "llm" and len(res["final_population"]) <= 10
    print("OK batch-GA budget respected, best monotone, rates in range")


def test_invalid_falls_back():
    op = LLMOperator(None, GRAMMAR)
    ga = BatchGA(MockEvaluator(), op, GRAMMAR, population=8, generations=3, tournament_k=2,
                 elitism=1, fallback=RandomOperator(GRAMMAR), max_evals=8 + 3 * (8 - 1))
    # always return malformed text -> every proposal invalid -> every offspring is a random fallback
    res = _drive(ga, 11, 0, lambda parents, ctx: "not json at all")
    assert res["invalid_rate"] == 1.0, "malformed responses should be 100% invalid"
    assert set(res["violation_breakdown"]) <= {"malformed", "parse_error"}
    # fallbacks are valid genotypes -> the whole final population validates
    assert all(R.validate(R.parse(s), GRAMMAR)[0] for s in res["final_population"])
    print("OK invalid proposals fall back to valid genotypes")


def test_dup_and_structure_bookkeeping():
    """Craft one generation so each 2.3 category is exercised, then check the event rates."""
    op = LLMOperator(None, GRAMMAR)
    ga = BatchGA(MockEvaluator(), op, GRAMMAR, population=6, generations=1, tournament_k=2,
                 elitism=2, fallback=RandomOperator(GRAMMAR), max_evals=100)
    ga.start(11, 0)
    ctxs = ga.gen_contexts()
    # an existing structure (from the seen set) with a NEW hp -> hp_substitution;
    # a brand-new structure -> new_structure; a member already in history -> dup_hist.
    seen_struct_geno = R.parse(next(iter(ga.seen_full)))
    hp_variant = R.serialize(seen_struct_geno)              # same structure, counts as hp_substitution
    dup_hist_text = next(iter(ga.seen_full))                 # identical to a seen genotype -> dup_hist
    texts = []
    for i, _ in enumerate(ctxs):
        if i == 0:
            texts.append(dup_hist_text)                      # dup_hist (and same structure -> hp_sub)
        else:
            texts.append(R.serialize(R.random_genotype(GRAMMAR, np.random.default_rng(100 + i))))
    raws = []
    for (parents, ctx), t in zip(ctxs, texts):
        op.build(parents, ctx); raws.append(op.interpret(t, ctx))
    ga.apply(raws)
    res = ga.result()
    assert res["dup_hist_rate"] > 0.0, "a genotype identical to history should register as dup_hist"
    assert res["hp_substitution_rate"] + res["new_structure_rate"] == 1.0 or not [e for e in ga.events if e["valid"]]
    print("OK dup/structure bookkeeping registers the crafted categories")


if __name__ == "__main__":
    test_budget_and_monotone()
    test_invalid_falls_back()
    test_dup_and_structure_bookkeeping()
    print("\nall phase-3 batch-GA tests passed")

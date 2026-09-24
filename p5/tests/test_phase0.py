"""Phase-0 tests: prefix cache byte-identity (zadani 0.3) and the budget cap (zadani 0.4). No API calls."""
import sys, tempfile, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # p5/ on path

from llm.prompt import build_prefix, build_suffix, system_block
from llm.accounting import Ledger, call_cost, BudgetExceeded


def test_prefix_byte_identical():
    """The static prefix must be byte-identical across calls given the same grammar (zadani 0.3)."""
    grammar = "estimator: [rf, svc, gb]\npreprocessing: [none, standardize]\n"
    a = build_prefix(grammar)
    b = build_prefix(grammar)
    assert a == b and a.encode() == b.encode(), "prefix not byte-identical -> cache would silently miss"
    # per-call VARIABLE content must NOT leak into the static prefix. Guard the actual markers that
    # build_suffix emits (the PARENTS block and the rendered "(fitness=<value>)"), not the mere word
    # "fitness" -- the instructions may legitimately describe fitness in static prose.
    assert "PARENTS" not in a and "(fitness=" not in a
    # and the suffix (variable) is where parents/fitness live
    suf = build_suffix([{"estimator": "rf"}], fitness=[0.9])
    assert "PARENTS" in suf and "(fitness=" in suf
    print("OK prefix byte-identical; variable content isolated in suffix")


def test_prefix_changes_only_with_grammar():
    assert build_prefix("g1") != build_prefix("g2")
    print("OK prefix changes when the grammar changes")


def test_cost_math():
    price = {"inp": 0.80, "out": 4.0, "cw": 1.00, "cr": 0.08}
    # 1000 cache-read in + 400 out = (1000*0.08 + 400*4.0)/1e6
    c = call_cost({"cr": 1000, "out": 400}, price)
    assert abs(c - (1000*0.08 + 400*4.0)/1e6) < 1e-12
    print(f"OK cost math ({c:.6f})")


def test_budget_cap_halts():
    with tempfile.TemporaryDirectory() as d:
        pricing = Path(d) / "pricing.yaml"
        pricing.write_text("m: {inp: 1000.0, out: 1000.0, cw: 1000.0, cr: 1000.0}\n")  # absurd to hit cap fast
        led = Ledger(d, pricing_path=pricing, cap_usd=0.01)
        # first call: 5000 in-tok at $1000/Mtok = $5.0 -> over $0.01 cap -> must raise, but AFTER logging
        raised = False
        try:
            led.record("m", {"inp": 5000})
        except BudgetExceeded:
            raised = True
        assert raised, "budget cap did not halt the run"
        # the call is still logged (audit trail) even though it tripped the cap
        rows = [json.loads(l) for l in (Path(d) / "accounting.jsonl").read_text().splitlines() if l.strip()]
        assert len(rows) == 1 and rows[0]["cost_usd"] == 5.0
        print("OK budget cap halts and logs the tripping call")


def test_ledger_resume():
    """Ledger sums an existing log on init (resume-safe)."""
    with tempfile.TemporaryDirectory() as d:
        pricing = Path(d) / "pricing.yaml"; pricing.write_text("m: {inp: 1.0, out: 1.0, cw: 1.0, cr: 1.0}\n")
        Ledger(d, pricing, cap_usd=100.0).record("m", {"inp": 1_000_000})   # $1.0
        led2 = Ledger(d, pricing, cap_usd=100.0)
        assert abs(led2.spent - 1.0) < 1e-9, "ledger did not resume cumulative spend"
        print("OK ledger resumes cumulative spend")


if __name__ == "__main__":
    test_prefix_byte_identical()
    test_prefix_changes_only_with_grammar()
    test_cost_math()
    test_budget_cap_halts()
    test_ledger_resume()
    print("\nall phase-0 tests passed")

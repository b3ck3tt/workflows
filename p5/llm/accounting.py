"""Cost accounting and hard budget cap for P5 LLM calls (zadani 0.2, 0.4).

Every LLM call is logged to an append-only jsonl with token counts (incl. cache), latency, and the
dollar cost computed from the configured price list. A cumulative hard cap aborts the run when exceeded --
the run stops, it does not continue (zadani 0.4).
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import yaml


class BudgetExceeded(RuntimeError):
    """Raised when a logged call pushes cumulative spend past the configured cap."""


def load_pricing(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def call_cost(usage: dict, price: dict) -> float:
    """usage keys: inp, out, cw (cache write), cr (cache read). price in $/Mtok. Returns USD."""
    return (usage.get("inp", 0) * price["inp"]
            + usage.get("out", 0) * price["out"]
            + usage.get("cw", 0) * price["cw"]
            + usage.get("cr", 0) * price["cr"]) / 1e6


class Ledger:
    """Append-only spend ledger with a hard cumulative cap.

    Usage:
        led = Ledger("runs/pilot", pricing_path="config/pricing.yaml", cap_usd=15.0)
        led.record(model, usage, run_id=..., generation=..., latency_s=...)   # raises BudgetExceeded if over cap
    """
    def __init__(self, run_dir: str | Path, pricing_path: str | Path, cap_usd: float):
        self.dir = Path(run_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "accounting.jsonl"
        self.pricing = load_pricing(pricing_path)
        self.cap = float(cap_usd)
        self.spent = self._replay()          # resume-safe: sum any existing log

    def _replay(self) -> float:
        if not self.path.exists():
            return 0.0
        tot = 0.0
        for line in self.path.read_text().splitlines():
            if line.strip():
                try: tot += json.loads(line)["cost_usd"]
                except Exception: pass
        return tot

    def price_for(self, model: str) -> dict:
        if model not in self.pricing:
            raise KeyError(f"no pricing for model {model!r}; add it to the price list")
        return self.pricing[model]

    def would_exceed(self, projected_usd: float) -> bool:
        return self.spent + projected_usd > self.cap

    def record(self, model: str, usage: dict, *, batch: bool = False, **meta) -> float:
        """Log one call, update cumulative spend, and enforce the cap. Returns this call's cost.
        batch=True applies the Message Batches API 50% discount to this call's cost."""
        cost = call_cost(usage, self.price_for(model))
        if batch:
            cost *= 0.5
        self.spent += cost
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model, "batch": batch,
               "inp": usage.get("inp", 0), "out": usage.get("out", 0),
               "cw": usage.get("cw", 0), "cr": usage.get("cr", 0),
               "cost_usd": round(cost, 6), "cum_usd": round(self.spent, 6), **meta}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if self.spent > self.cap:
            raise BudgetExceeded(
                f"cumulative spend ${self.spent:.4f} exceeded cap ${self.cap:.2f} "
                f"(run_dir={self.dir}); run halted per zadani 0.4")
        return cost

    def summary(self) -> dict:
        return {"spent_usd": round(self.spent, 4), "cap_usd": self.cap,
                "remaining_usd": round(self.cap - self.spent, 4), "log": str(self.path)}

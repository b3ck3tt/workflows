"""Anthropic Message Batches client for the Phase-3 driver: submit many independent proposal requests
at once for the 50% batch discount (zadani sec.3 "Batch API na vse co nemusi byt synchronni").

One batch = all offspring proposals of one generation across all runs (~pop*runs requests). We submit,
poll until the batch ends, retrieve results, log each call's usage to the ledger at the batch (halved)
price, and return {custom_id: text-or-None}. Failed/errored requests map to None so the GA falls back to
a random proposal (same as an api_error in the synchronous path). Robust to flaky networks: submit-once
then poll, so a dropped poll is just retried, not a multi-hour hang.
"""
from __future__ import annotations
import time
from .accounting import BudgetExceeded
from .client import load_key, _usage
from .prompt import system_block


class BatchClient:
    def __init__(self, model: str, ledger, *, max_tokens: int = 1200, timeout: float = 60.0,
                 poll_s: float = 20.0):
        import anthropic
        self.model = model
        self.ledger = ledger
        self.max_tokens = max_tokens
        self.poll_s = poll_s
        self.client = anthropic.Anthropic(api_key=load_key(), timeout=timeout, max_retries=4)

    def estimate_usd(self, prefix: str, sample_suffix: str, n_calls: int, est_out: int = 200) -> float:
        """Cheap pre-submit estimate (one count_tokens), at the batch (halved) price."""
        from .client import _cache_min_tokens
        from .accounting import call_cost
        price = self.ledger.price_for(self.model)
        n_in = self.client.messages.count_tokens(
            model=self.model, system=system_block(prefix),
            messages=[{"role": "user", "content": sample_suffix}]).input_tokens
        prefix_tok = self.client.messages.count_tokens(
            model=self.model, system=system_block(prefix),
            messages=[{"role": "user", "content": "x"}]).input_tokens
        if prefix_tok >= _cache_min_tokens(self.model):
            suffix_tok = max(0, n_in - prefix_tok)
            first = call_cost({"cw": prefix_tok, "inp": suffix_tok, "out": est_out}, price)
            rest = call_cost({"cr": prefix_tok, "inp": suffix_tok, "out": est_out}, price) * max(0, n_calls - 1)
            per = (first + rest)
        else:
            per = call_cost({"inp": n_in, "out": est_out}, price) * n_calls
        return per * 0.5   # batch discount

    def run_batch(self, prefix: str, items: list[tuple], **meta) -> dict:
        """items: list of (custom_id, suffix, per_item_meta_dict). Returns {custom_id: text|None}.
        Raises BudgetExceeded (before submitting) if the pre-estimate would trip the cap."""
        if not items:
            return {}
        import re
        bad = [cid for cid, _, _ in items if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", cid)]
        if bad:
            raise ValueError(f"invalid custom_id(s) (must match ^[a-zA-Z0-9_-]{{1,64}}$): {bad[:3]}")
        est = self.estimate_usd(prefix, items[0][1], len(items))
        if self.ledger.would_exceed(est):
            raise BudgetExceeded(
                f"batch pre-estimate ${est:.2f} + spent ${self.ledger.spent:.2f} would exceed "
                f"cap ${self.ledger.cap:.2f}; not submitting (zadani 0.4)")
        sys_block = system_block(prefix)
        requests = [{"custom_id": cid,
                     "params": {"model": self.model, "max_tokens": self.max_tokens,
                                "system": sys_block,
                                "messages": [{"role": "user", "content": suffix}],
                                "thinking": {"type": "disabled"}}}
                    for cid, suffix, _ in items]
        batch = self.client.messages.batches.create(requests=requests)
        bid = batch.id
        while True:
            time.sleep(self.poll_s)
            b = self.client.messages.batches.retrieve(bid)
            if b.processing_status == "ended":
                break
        per_meta = {cid: m for cid, _, m in items}
        out: dict = {}
        for entry in self.client.messages.batches.results(bid):
            cid = entry.custom_id
            if entry.result.type == "succeeded":
                msg = entry.result.message
                text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
                out[cid] = text
                self.ledger.record(self.model, _usage(msg), batch=True,
                                   **{**meta, **per_meta.get(cid, {})})
            else:
                out[cid] = None   # errored / canceled / expired -> GA falls back
        return out

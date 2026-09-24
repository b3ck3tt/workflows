"""Operator interface (zadani 1.3). The GA is agnostic to the operator: an LLM operator must be a
drop-in replacement for the random one, so both implement the same method.

    propose(parents: list[genotype], context: dict) -> genotype | None

`context` carries {grammar, rng, generation, history, fitness}. Returning None means "no valid proposal"
(the GA / a fallback then handles it). The random operator never returns None.
"""
from __future__ import annotations


class Operator:
    name = "base"

    def propose(self, parents: list[dict], context: dict) -> dict | None:
        raise NotImplementedError

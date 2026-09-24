# BIBLIOGRAPHY (annotated, for related work) — zadani phase 4

Structured as the future related-work section. Each entry: one line on how **we** differ.

**Verification rule (zadani):** nothing here is cited until it has been opened and checked. Author list,
title, venue, year, and identifier must all match the source. Missing fields stay blank and marked
`TODO-verify`; a fabricated arXiv id in an LNCS submission is an error a reviewer finds immediately. The
bibliographic fields below are placeholders pending a web-verification pass; the "differ" lines come from
the project spec/positioning, not from the sources.

## Conceptual basis
- **LMX — Language Model Crossover.** authors: TODO-verify; venue: TODO-verify; year: TODO-verify; id: TODO-verify.
  Differ: we constrain to a grammar-defined sklearn pipeline space, not free text/code, and evaluate on CC-18.
- **ELM — Evolution through Large Models.** TODO-verify (authors/venue/year/id).
  Differ: population EA over a bounded pipeline grammar, not open-ended code evolution.
- **FunSearch.** TODO-verify.
  Differ: we quantify the "unreliable proposer + selection" effect domain-specifically; not claimed as novel.
- **AlphaEvolve.** TODO-verify.
  Differ: frozen model, no fine-tuning, classical pipeline space, cost-bounded academic budget.

## LLM operators in EC
- **EoH — Evolution of Heuristics.** TODO-verify.
  Differ: pipelines under a grammar, not heuristic code.
- **LLaMEA.** TODO-verify.
  Differ (near work): free-form code, single-lineage, no crossover, no CC-18; we do a population with LLM
  crossover on a pipeline grammar.
- **AVO.** TODO-verify.  Differ: TODO after reading.
- **ReEvo.** TODO-verify.  Differ: TODO after reading.

## Tabular & AutoML
- **LLM-FE.** TODO-verify.
  Differ (near work): feature engineering only; we evolve pipeline structure + estimator, not features.
- **SemPipes.** TODO-verify.
  Differ (near work): tree search over semantic operator code; we use a population with crossover and a
  different pipeline layer.
- **TPOT.** TODO-verify.  Role: classical GP-AutoML baseline for community calibration.
- **GAMA.** TODO-verify.  Role: classical AutoML baseline for community calibration.
- **AutoML-DSGE.** TODO-verify.  Differ: grammar-based AutoML without an LLM operator.

## Agentic AutoML (no population)
- **SELA.** TODO-verify.  Differ: agentic/tree search, not a population EA.
- **AutoML-Agent.** TODO-verify.  Differ: agentic pipeline, no evolutionary population.
- **MLE-STAR.** TODO-verify.  Differ: agentic, not population-based.

## Critical / negative results
- **Sim, Renau & Hart — EvoApplications 2025.** TODO-verify (exact title/venue/id).
  Use: precedent that a critical evaluation of LLM-generated artifacts is publishable even when the LLM
  loses to a simple classical method.
- **Gurkan et al. — GECCO 2026.** TODO-verify (exact title/authors/id).
  Use: over a DSL, LLM mutations mostly repeat seen structural forms (variation is within-template
  substitution) while classical GP does not; we transfer this analysis to the pipeline grammar and design
  our structural-novelty metric to be comparable.

## Pipeline embeddings
- **LACCI / SNCS (companion, third-person).** SNCS: DOI 10.1007/s42979-026-05295-9 (verify vol/page).
  Use: the learned embedding space for the cosine-distance diversity metric.

# POSITIONING — what P5 claims (reference document)

Reference for the whole project (zadani v2, phase 0). Implementation must not drift from what the paper
can claim. Not a draft; no prose for the paper is written yet.

## The contribution is the conjunction of four axes
None is novel alone; the novelty is their intersection.

1. A **classical, grammar-constrained pipeline space** (preprocessing → feature selection → estimator →
   hyperparameters) — not free-form code, not feature engineering.
2. A **population-based evolutionary algorithm** with an LLM as the operator for **both mutation and
   crossover** — not single-lineage, not tree search.
3. A **frozen** model, no fine-tuning.
4. Evaluation on **OpenML CC-18** against classical AutoML baselines, at an **equal evaluation budget**.

## The measurement program (the most defensible part; near-uncovered in the literature)
- fraction of grammar-invalid proposals, broken down by violation type;
- fraction of proposals duplicating the current population, and the whole run history;
- population diversity collapse measured by structural edit distance and by cosine distance in a learned
  (LACCI/SNCS) embedding space;
- fraction of proposals that only substitute hyperparameters inside an already-seen structure vs proposals
  with genuinely new structure (transfers Gurkan et al., GECCO 2026, from a DSL to the pipeline grammar);
- proposal accuracy in isolation (the P4 metric) vs search efficiency.

## What the paper does NOT claim
- Not "first LLM as an operator" — that is Language Model Crossover and FunSearch.
- Not "first LLM for tabular AutoML" — that is LLM-FE.
- Not "an unreliable proposer works because of selection" as a conceptual novelty — that is FunSearch;
  our version is quantified and domain-specific.

## Three nearest works and how we differ (must be explicit and early in the paper)
- **LLM-FE** — LLM as an evolutionary optimizer on tabular data, but feature engineering only; does not
  touch pipeline structure or estimator choice.
- **LLaMEA** — LLM in an evolutionary loop (incl. an ML-pipeline example), but over free-form code,
  single-lineage without crossover, no CC-18.
- **SemPipes** — LLM mutation over the semantic code of operators in an sklearn pipeline, but tree search
  instead of a population with crossover, and a different pipeline layer.

## Decision-gate stance (zadani v2)
A well-measured diversity/validity collapse is a publishable result even if the LLM operator does not win
(precedent: Sim, Renau & Hart, EvoApplications 2025). The project stops only if the infrastructure gives
inconsistent numbers, the extrapolated full-run cost exceeds budget, or the diversity/validity metrics show
no signal in any direction.

# Paper plan for external review — Frozen LLM as a variation operator in evolutionary AutoML pipeline search

Working document for a pre-run external review (double-blind LNCS submission, target 2026-11-01). It
describes the intended paper, the experimental infrastructure (built and validated), the theses, and the
expected results grounded in a completed pilot. Author-identifying information is omitted deliberately.
References to a companion pipeline-embedding paper are third-person. Nothing here is final prose.

---

## 1. One-paragraph summary (TL;DR)
We place a **frozen** large language model as the variation operator — doing both **mutation and
crossover** — inside a **population** evolutionary algorithm that searches a **grammar-constrained
scikit-learn pipeline space**, and we evaluate it on OpenML CC-18 against random-operator evolution and
random search **at an equal evaluation budget**. The contribution is not "an LLM can do AutoML" but a
**careful, quantified characterisation** of *how* the LLM proposes: validity, duplication, population
diversity collapse, and — transferring a recent DSL result to the pipeline grammar — the extent to which
its variation is confined to hyperparameter substitution within already-seen structures. A completed
pilot indicates the frozen LLM operator gives **no search-quality advantage at equal budget** while
exhibiting a distinct **structural-narrowness** signature; the full study confirms and characterises this
across 15 tasks with statistical power and ablations.

---

## 2. Motivation and research question
LLMs are increasingly used as search operators (code, heuristics, prompts) and for tabular AutoML. Most
reports are positive and celebratory; few isolate *what the operator actually does* or compare it fairly
against trivial baselines at matched cost. Evolutionary-computation reviewers routinely reject LLM-operator
claims that compare on wall-clock or a fixed number of iterations rather than a **matched number of
expensive evaluations**. Our question:

> Inside a population EA over a bounded, grammar-defined pipeline space, does a frozen LLM variation
> operator search better than random variation at an equal evaluation budget — and if not (or not much),
> *why*, measured concretely?

---

## 3. Positioning and contribution
The contribution is the **conjunction** of four axes; none is novel alone.
1. A classical, **grammar-constrained pipeline space** (preprocessing → feature selection → estimator →
   hyperparameters) — not free-form code, not feature engineering only.
2. A **population** EA with the LLM as operator for **both mutation and crossover** — not single-lineage,
   not tree search.
3. A **frozen** model (no fine-tuning), cost-bounded to an academic budget.
4. **OpenML CC-18** evaluation against classical baselines at an **equal evaluation budget**.

**What the paper does NOT claim** (stated early and explicitly):
- Not "first LLM as an operator" — that is Language Model Crossover (LMX) / FunSearch. **Axis 2 (LLM for
  mutation *and* crossover in a population) overlaps LMX directly**; our differentiation is the
  grammar-constrained pipeline space + the operator-behaviour measurement program, **not crossover itself**.
  This is stated explicitly and early.
- Not "first LLM for tabular AutoML" — that is LLM-based feature engineering (LLM-FE).
- Not "an unreliable proposer works because of selection" as a concept — that is FunSearch; our version is
  domain-specific and quantified.

**Three nearest works and the differentiation** (must appear early):
- *LLM-FE* — LLM as an evolutionary optimiser on tabular data, but feature engineering only; does not
  touch pipeline structure or estimator choice.
- *LLaMEA* — LLM in an evolutionary loop (incl. an ML-pipeline example), but over free-form code,
  single-lineage, no crossover, no CC-18.
- *SemPipes* — LLM mutation over the semantic code of sklearn operators, but tree search rather than a
  population with crossover, and a different pipeline layer.

The **most defensible part**, near-uncovered in the literature, is the **measurement program** (§4.7): a
quantitative anatomy of the operator's proposals. A well-measured diversity/validity collapse is
publishable **even if the LLM operator does not win** (precedent: a critical EvoApplications 2025 study of
LLM-generated artefacts that lose to a simple classical method).

---

## 4. Technical design of the experiments

### 4.1 Pipeline search space (grammar)
An external YAML grammar defines three ordered stages:
- **preprocessing**: {none, standardize, minmax};
- **feature_selection**: {none, selectk(k), pca(n)};
- **estimator**: {random_forest, extra_trees, hist_gbrt, logistic, svc, knn, gaussian_nb} with their
  hyperparameter specs (type ∈ int|float|cat, range, log-scale, choices, optional).
A genotype is `{stage: {name, params}}`. The grammar is in config, not code, so the space is auditable and
swappable. Feature names are namespaced; the space is small enough to be exhaustively describable in the
prompt yet large enough (continuous hyperparameters) to make search non-trivial.

### 4.2 Evolutionary algorithm
Generational population GA: tournament selection, elitism, fixed population and generation count from
config. The operator interface is minimal — `propose(parents, context) -> candidate | None` — so the LLM
operator is a drop-in replacement for the random operator with **no change to the GA**. Invalid or
None proposals fall back to a random proposal (and are logged as such), so a run always completes.

### 4.3 The LLM operator (frozen)
The prompt is split into a **static prefix** (task instructions + output schema + the grammar + a small
set of worked examples) and a **variable suffix** (the parents, their fitness, a short population
history). The operator performs **mutation** (one parent) or **crossover** (recombine two parents),
chosen by a configurable probability; crossover is a deliberate positioning axis against single-lineage
LLM-EA work. Output is a single JSON genotype (no prose). **Model is a first-class axis: two co-primary
frozen models** — a closed frontier-small model (Haiku, via the Anthropic Batch API) and a large
open-weight model (Qwen2.5-72B, via an OpenAI-compatible endpoint) — each at full scale, so the claim is
made only where it holds on both (a closed model that drifts AND a permanently reproducible open one). A
reduced Sonnet run is a secondary closed-size cross-check; the single-axis ablations run on the open model.

### 4.4 Baselines
- **Random-operator GA**: identical GA, operator = uniform mutation/crossover over the grammar (also the
  fallback). Isolates the *operator*, holding the evolutionary scaffold fixed.
- **Random search**: i.i.d. sampling from the grammar at the same budget — the null any search must beat.
- **Classical AutoML** (TPOT / GAMA) for community calibration — to wire before submission; not required
  for the operator comparison, which is the core.
All methods start from the **same random initial population / random draws** (the LLM influences only
offspring), so no method gets a head start.

### 4.5 Equal evaluation budget (the primary axis)
The budget unit is a **genuine (non-cached) pipeline evaluation**; duplicates are cached and free. All
methods run to **exactly the same number of genuine evaluations** (GA methods run generations up to a
safety ceiling until the budget is met; random search draws until the budget is met), so final values are
directly comparable. We also report the **secondary best-vs-proposals axis** (API calls / offspring),
which exposes the true cost of an operator that duplicates or emits invalids — the LLM issues more
proposals than genuine evaluations. Every headline figure is best-vs-genuine-evaluations; anytime curves
accompany it. (An earlier design compared at a common budget from unequal totals; per external review we
switched to exactly-equal genuine evaluations as primary — see §7 and PREREGISTRATION.md.)

### 4.6 Evaluation harness
Each genotype is compiled to an sklearn `Pipeline` and scored by **StratifiedKFold** accuracy (5 folds,
fixed seed), with large datasets row-capped for a defensible per-evaluation cost. Failed pipelines score
0 with an error string (treated as bad candidates, never crash). Results are cached **in-memory per run**
by a content hash so duplicates are free but a killed-and-resumed run cannot silently reuse warm results
and mis-count the budget.

### 4.7 Measurement program (first-class metrics)
For every operator proposal we record and aggregate:
- **grammar-invalid rate**, broken down by **violation type**;
- **duplicate rate** vs the current population **and** vs the whole run history;
- **population diversity**, two ways: structural edit distance over the pipeline, and cosine distance in a
  learned pipeline-embedding space (the companion SNCS D0 space: an off-the-shelf sentence encoder applied
  to the cleaned sklearn-flow text, so our numbers are directly comparable to that work);
- **structural-novelty**: the fraction of valid proposals that only substitute hyperparameters inside an
  already-seen structure vs those introducing a genuinely new (pre|fs|est) structure — transferring a
  GECCO 2026 DSL result to the pipeline grammar;
- **proposal accuracy in isolation vs search efficiency** (the anytime curve).

### 4.8 Datasets (pre-registered; see PREREGISTRATION.md)
15 CC-18 tasks selected by **task-intrinsic learnability only** (historical best-achievable accuracy and
headroom over the majority baseline), replacing the "smallest 15" whose near-ceiling accuracy made method
differences invisible in the pilot. The criterion is a function of the task alone, **independent of any
method's outcome**. Each task is tagged, from a variance decomposition, as **structure-decisive** (9) or
**hp-decisive** (6). The 6 hp-decisive tasks are a **control subset**: the LLM's structural narrowness is
not a handicap there, so if the LLM still shows no advantage on them, the negative result is not a
task-selection artefact. (hp-decisive tasks with headroom are genuinely rare in CC-18 — structure usually
dominates — a reportable finding in itself.) 5 seeds → 75 runs per method.

### 4.9 Infrastructure, cost, and budget safety
- **Anthropic Batch API** for the 50% discount: all runs advance in **lockstep**, one batch per generation
  carrying every run's offspring (~3600 requests × ~30 rounds). Robust to flaky networks (submit-once +
  poll, no synchronous hang).
- **Cost accounting** ledger (append-only, per-call cost at the configured price list, cumulative **hard
  cap** that aborts before a batch that would breach it). Every run is **preflight-estimated** from a
  token count before spending.
- **Prompt caching**: the static prefix is byte-identical for cache hits; on the Haiku model the prefix is
  below the (empirically 4096-token) cache minimum, so caching is off there; the Sonnet ablation caches.
- **Reproducibility**: config-per-experiment; fixed seeds; grammar and dataset selection in files; all
  metrics and per-run artefacts persisted; anonymised repository.

### 4.10 Ablations (gated after the base run, one axis at a time)
Fitness feedback in the prompt on/off; population history vs parents-only; mutation-only vs
mutation+crossover; model size (Haiku vs a reduced-scale Sonnet). Each isolates one design choice.

---

## 5. Theses (hypotheses under test)
- **H1 (main).** At an equal evaluation budget, the frozen LLM operator does **not** outperform random
  variation (random-operator GA and random search) in final/anytime best accuracy on CC-18.
- **H2 (mechanism).** LLM variation is **structurally narrow**: it overwhelmingly substitutes
  hyperparameters within already-seen structures and rarely proposes new structures, more so than random
  variation; correspondingly its populations collapse in embedding-diversity while remaining
  hyperparameter-diverse.
- **H3 (reliability).** Under a correct schema prompt, a frozen small model proposes **grammar-valid**
  pipelines almost always; residual invalids are format, not value-range, violations.
- **H4 (self-repetition, reformulated).** The LLM revisits a **narrow, concentrated** region of the space
  (low duplicate-entropy); random variation duplicates *more* overall but *diffusely*. (The naive "LLM
  wastes more budget on duplicates" is **false** — the pilot shows the opposite, so it is dropped.)
- **H5 (control, descriptive).** H1 holds **even on the 6 hp-decisive tasks**, where structural narrowness
  is not a handicap — a plausibility check against task cherry-picking (n=6, no statistical power claimed).
- **H6 (ablations).** Fitness/history in prompt, crossover, sampling temperature, and model size/family
  (incl. an open-weight model) do not overturn H1. **H2 must survive a prompt ablation** (no worked
  examples / explicit new-structure instruction) to be claimed as a model rather than few-shot property.
- **H7 (space size, secondary).** The LLM-vs-random gap does not move in the LLM's favour beyond the ±0.005
  margin when the space grows from 63 structures (G1) to ~1,560 (G2). Turns "bounded space" into a result.

---

## 6. Expected results (grounded in the pilot; to be confirmed at scale)
A completed pilot (3 low-ceiling tasks × 3 seeds, reduced population, corrected operator prompt) gives:
- **Validity (H3): ~0.4% invalid**, essentially all "dropped/mangled stage" format errors, **zero**
  value-range violations. (An earlier ~19% invalid rate was traced to a prompt/schema mismatch, since
  fixed — worth noting as a methodological caution.)
- **Search quality (H1): no advantage at equal budget.** LLM-GA vs random-search ≈ tie (Δ ≈ −0.003 at a
  common budget); LLM-GA vs random-operator GA slightly negative (≈ 4 wins / 5 losses of 9 runs). The
  effect is small and does not favour the LLM.
- **Mechanism (H2): strong and stable.** LLM proposals are ~92% hyperparameter-substitution within seen
  structures vs ~8% new structures (random ≈ 79%/21%). Embedding (SNCS-D0) diversity of the final
  population ≈ 0.02 for the LLM vs ≈ 0.12 for random, while structural (edit-distance) diversity is
  *higher* for the LLM — high surface variation, low structural novelty.
- **Self-repetition (H4):** duplicate-vs-history ≈ 0.13 for the LLM vs ≈ 0.37 for random (random repeats
  more overall, but the LLM's repeats concentrate a narrow region).
- **Cross-model replication (Qwen pilot, open-weight):** the mechanism replicates and is STRONGER on the
  larger open model — hp-substitution 0.95 (vs 0.92), embedding diversity 0.013 (vs 0.019), duplicate
  concentration 0.92 — and the no-advantage result is MORE decisive (LLM vs random-GA Δ −0.014, 1W/8L, vs
  Haiku −0.006, 4W/5L). A bigger open model does not rescue the operator. (Qwen's ~4% raw "invalid" is
  network-straggler fallback, not model format error.)
- **Cost:** co-primary bases ≈ \$171 (Haiku, batch) + \$42 (Qwen); recommended program (both bases + Qwen
  ablation suite + Sonnet size-check + G2 secondary) ≈ **\$530**.

Expected at scale (Phase 3): H1–H4 confirmed with confidence intervals across 15 tasks; H5 tested on the
6-task hp-decisive control; H6 mapped across the four ablation axes. The anticipated paper is therefore a
**critical, mechanism-level** contribution: *a frozen LLM operator does not beat random variation at
equal budget in this space, and we can say precisely why.* The result that would overturn H1 is stated in
advance (§7).

---

## 7. Decision gate, pre-registration, and what would change the conclusion
- The project is staged with a **decision gate** after the pilot: proceed only if the infrastructure gives
  consistent numbers, the extrapolated cost is within budget, and the metrics show signal in some
  direction. All three are satisfied.
- Dataset selection and the analysis plan are **pre-registered before any Phase-3 run** (PREREGISTRATION.md):
  primary axis = equal evaluation budget (anytime curves); per-task reporting; structure/hp subsets
  analysed separately; Wilcoxon signed-rank across tasks with bootstrap CIs.
- **What would overturn H1** (declared in advance): across the 15 tasks with CIs, LLM-GA's mean
  best-accuracy exceeds both baselines by a margin whose CI excludes zero on the equal-budget axis —
  including on the hp-decisive control subset.
- Two infrastructure confounds were found and fixed during an internal audit **before** spending: (i) the
  three methods reach different total evaluation counts, so the comparison must be anytime-at-common-budget,
  not final-value (fixed by recording fine-grained anytime traces); (ii) a disk evaluation cache surviving
  a resumed run mis-counted the budget (fixed by an in-memory per-run cache). A live end-to-end smoke run
  also caught a batch request-id format bug that would otherwise have failed the first paid run.

---

## 8. Threats to validity / limitations
- **Model coverage.** The claim rests on two co-primary models — a closed frontier-small (Haiku) and a
  large open-weight (Qwen2.5-72B) — plus a Sonnet size cross-check; not a full sweep across many families.
  The pilot shows the result is stronger, not weaker, on the larger open model, so "it's just a weak/small
  model" is directly answered. The hp-decisive control mitigates the related "you rigged the tasks" concern.
  Open-weight serving (DeepInfra quantization/stack) can shift; we record the exact model revision, and full
  reproduction is possible from the open weights.
- **Two space sizes, not open-ended.** The primary space has 63 structures; a secondary experiment (G2,
  ~1,560 structures, §H7) tests whether the picture changes with size. Results may still not transfer to
  open-ended code spaces where LLM priors might help more. Scoped explicitly. (A prompt-describable space
  is deliberately where the LLM *should* have an edge — that it does not is the stronger claim.)
- **Fitness = accuracy.** Some tasks are imbalanced; balanced accuracy is a secondary reported metric, not
  the search target.
- **15 of 72 CC-18 tasks, 5 seeds.** Task-level generalisation is bounded; the selection is pre-registered
  and outcome-independent, with near-ceiling tasks retained for range.
- **Prompt dependence.** Operator behaviour is a property of the (fixed, reported) prompt; a materially
  different prompt could shift the mechanism metrics. The prompt is frozen and published.
- **Batch/non-determinism.** LLM sampling is non-deterministic; seeds fix the GA scaffold and evaluation,
  not the model outputs, so runs are statistically (not bit-) reproducible.

---

## 9. Reproducibility and cost
Config-per-experiment (population, generations, seeds, datasets, operator flags, model, budget cap);
external grammar and dataset files; fixed evaluation seed and CV; all per-run artefacts and the full cost
ledger persisted; anonymised repository. Full program cost ~\$585 (batch), each step preflight-estimated
and hard-capped.

---

## 10. Questions for the external reviewer
1. Is **anytime-at-common-budget** the comparison you would expect, or would you also want all methods
   forced to exactly equal genuine evaluations (a design option we considered and can enable)?
2. Is the **structure-decisive / hp-decisive control** convincing as a defence against task cherry-picking,
   given hp-decisive tasks with headroom are scarce in CC-18 (9:6 split)?
3. Is a **reduced-scale Sonnet** ablation sufficient for the model-strength objection, or is a full
   second-model run required?
4. Is **accuracy** an acceptable fitness/target given the imbalance on some tasks, with balanced accuracy
   reported secondarily?
5. Are the **classical AutoML baselines** (TPOT/GAMA) necessary for the core operator claim, or expected
   only as calibration?
6. Is the framing as a **critical/mechanism** contribution (rather than a "we beat AutoML" result)
   appropriately scoped for the venue?

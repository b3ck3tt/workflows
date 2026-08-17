# LACCI 2026: Comparing Lexical and Dense Representations for Interpretable Workflow Recommendation

Code for the LACCI 2026 paper. The study asks how the *workflow representation*
(sparse lexical TF-IDF vs dense MiniLM embeddings) affects the quality of SHAP
explanations in recommendation-oriented meta-models, under a task-grouped
protocol where every test fold is a set of entirely unseen tasks.

## Contents

| File | What it produces |
|---|---|
| `_robust_full.py` | the repeated shuffled task-group protocol: $R^2$/MAE/MSE and global top-$k$ stability over 6 random task-to-fold assignments → Tables I and III, Figs. 1 and 4 |
| `camera_ready_experiments.py` | the camera-ready additions: SVD dense control, workflow-complexity split, readable attribution mass, representation cost → new rows in Tables I/II and the cost paragraph |
| `make_figures.py` | renders all figures **and** verifies every table cell in `main.tex` against the result files; refuses to render on a mismatch |

The core pipeline these build on lives at the repository root:
`openml_flow.py` (data → features → CV → SHAP), driven by `experiments.ipynb`,
with `figures.ipynb` / `figures_max.ipynb` / `figures_mean.ipynb` for the
per-run plots.

## Order of operations

```bash
# 1. per-run results (edit the Configuration cell first): writes results_cc18_max/, results_cc18_mean/
jupyter notebook experiments.ipynb

# 2. the repeated-shuffled protocol -> robust_predictive_anchor_v2.csv, robust_stability.csv
python3 lacci_experiments/_robust_full.py

# 3. camera-ready additions -> camera_ready_results.json
python3 lacci_experiments/camera_ready_experiments.py

# 4. figures + table/figure consistency check
python3 lacci_experiments/make_figures.py            # renders
python3 lacci_experiments/make_figures.py --verify-only
```

Steps 2 and 3 read the artifacts written by step 1 and take roughly 45 min and
55 min respectively on a laptop; step 3 checkpoints after each stage.

The paper sources are not in this repository. Steps 3 and 4 look for them in
`papers/lacci2026/` by default; set `LACCI_PAPER_DIR` to point elsewhere. Only
`make_figures.py` needs `main.tex` (for the verification pass) — everything else
runs from the result files alone.

## Why the consistency check exists

In the submitted version, Figure 1 was plotted from a different run than the
Table I it accompanied, and one category was labelled `nan`. Neither error is
visible in the LaTeX source. `make_figures.py` therefore parses `main.tex`,
compares every table cell against the CSV the figure is drawn from, and exits
non-zero on any disagreement (76 checks at the time of writing).

## Relation to `sncs_experiments/`

`sncs_experiments/` is a different study on the same OpenML data: it
cross-validates over shuffled splits of individual workflow–dataset evaluations
and reports test $R^2 \approx 0.80$. The protocol here assigns all rows of a
task to one fold and aggregates repeated evaluations of a pair into a single
target, which is stricter and yields substantially lower $R^2$. The difference
is interpolating within known datasets versus generalizing to new ones — not a
contradiction.

"""Regenerate every figure in the LACCI 2026 paper from the result CSVs, and
verify that what is plotted equals what the tables in main.tex report.

Why this exists: in the submitted PDF, Fig. 1 was plotted from a different
(much more optimistic) run than Table I, and the meta-only group was labelled
"nan". Both are invisible in the .tex source. This script removes the class of
error by (a) plotting only from the CSVs the tables are built from and (b)
parsing main.tex and asserting cell-by-cell agreement.

Usage
    python3 make_figures.py                 # verify + write figures for agg=max
    python3 make_figures.py --verify-only   # just the consistency report
    python3 make_figures.py --agg both      # also write the mean-agg variants

Outputs (into this directory, both .pdf for LaTeX and .png for previewing):
    fig1_predictive_anchor      Table I   <- robust_predictive_anchor_v2.csv
    fig2_global_shap            Sec. VI   <- global_shap_task_group_kfold_*.csv
    fig2_global_shap_1col       (compact single-column alternative to fig2)
    fig3_local_quality          Table II  <- local_shap_metrics_summary.csv
    fig4_global_stability       Sec. VIII <- robust_stability.csv
    fig5a_case_study_tfidf      Sec. VII  <- shap_artifacts.pkl
    fig5b_case_study_minilm     Sec. VII  <- shap_artifacts.pkl

Exit code is 1 if any figure source disagrees with main.tex.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import joblib
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                      # lacci/ -> repo root
# The paper sources live outside the repo tree (papers/ is gitignored); override
# with LACCI_PAPER_DIR if they are kept elsewhere.
PAPER = Path(os.environ.get("LACCI_PAPER_DIR", REPO / "papers" / "lacci2026"))
TEX = PAPER / "main.tex"

# ---------------------------------------------------------------------------
# Display vocabulary. Everything user-facing goes through these maps; nothing
# in a figure is ever a raw column name or a raw config key (that is how the
# submitted Fig. 1 ended up with a category labelled "nan").
# ---------------------------------------------------------------------------
REPRESENTATIONS = ["meta_only", "minilm_meta", "tfidf_meta"]
REPR_LABEL = {
    "meta_only": "Meta only",
    "minilm_meta": "MiniLM + meta",
    "tfidf_meta": "TF-IDF + meta",
}
MODELS = ["hist_gbrt", "random_forest"]
MODEL_LABEL = {"hist_gbrt": "HistGBRT", "random_forest": "Random Forest"}
MODEL_SHORT = {"hist_gbrt": "HGB", "random_forest": "RF"}
# Camera-ready dense control (truncated SVD of the TF-IDF matrix).
SVD_LABEL = "SVD + meta"

# Okabe-Ito steps, validated for CVD separation over all pairs (protan/deutan
# OKLab dE >= 8, normal-vision dE >= 15) against a white page surface.
#
# Encoding policy -- each figure's hue encodes exactly one dimension, and the
# same dimension always gets the same hues:
#   hue = predictive model        -> blue / vermillion      (Figs. 1, 3)
#   hue = feature type            -> blue / green / vermillion (Figs. 2, 5)
#   hue = workflow representation -> green / vermillion     (Fig. 4)
# The third row deliberately reuses the feature-type hues: TF-IDF features are
# the green named tokens and MiniLM features are the vermillion latent dims, so
# Fig. 4's colours carry the same meaning as Figs. 2 and 5.
C_MODEL = {"hist_gbrt": "#0072B2", "random_forest": "#D55E00"}
C_NAMESPACE = {
    "meta": "#0072B2",        # dataset meta-feature
    "text_token": "#009E73",  # named workflow token (readable)
    "text_emb": "#D55E00",    # latent embedding dimension (not readable)
}
NAMESPACE_LABEL = {
    "meta": "Dataset meta-feature",
    "text_token": "Workflow token (readable)",
    "text_emb": "Embedding dimension (opaque)",
}
GRID = "#d9d9d9"

IEEE_COL = 3.45   # single column width, inches
IEEE_FULL = 7.16  # full text width, inches


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(PAPER / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


def results_dir(agg: str) -> Path:
    return REPO / f"results_cc18_{agg}"


def load_shap_artifacts(agg: str, cfg: str):
    """shap_artifacts.pkl pickles openml_flow.ShapFoldResult -- the module has
    to be importable from the repo root for unpickling to resolve the class."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    with open(results_dir(agg) / cfg / "shap_artifacts.pkl", "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Data loading. One loader per table, so a figure and its table can never drift.
# ---------------------------------------------------------------------------
def load_predictive(agg: str) -> pd.DataFrame:
    df = pd.read_csv(results_dir(agg) / "robust_predictive_anchor_v2.csv")
    return df.set_index(["experiment", "model"])


def load_stability(agg: str) -> pd.DataFrame:
    df = pd.read_csv(results_dir(agg) / "robust_stability.csv")
    return df.set_index(["experiment", "model"])


def load_local(agg: str) -> pd.DataFrame:
    df = pd.read_csv(results_dir(agg) / "local_shap_metrics_summary.csv")
    return df.set_index(["experiment", "model"])


def load_global_shap(agg: str, cfg: str, model: str) -> pd.DataFrame:
    p = results_dir(agg) / cfg / f"global_shap_task_group_kfold_{model}.csv"
    return pd.read_csv(p)


def namespace_of(feature: str) -> str:
    """Classify a namespaced feature name for colouring and readability."""
    if feature.startswith("meta::"):
        return "meta"
    short = feature.split("::", 1)[-1]
    return "text_emb" if re.fullmatch(r"emb[_ ]?\d+", short) else "text_token"


def short_name(feature: str, limit: int = 26) -> str:
    short = feature.split("::", 1)[-1].replace("_", " ")
    return short if len(short) <= limit else short[: limit - 1] + "\u2026"


# ---------------------------------------------------------------------------
# Verification against main.tex
# ---------------------------------------------------------------------------
def tex_source() -> str:
    return TEX.read_text(encoding="utf-8")


def parse_tabular(tex: str, label: str) -> list[list[str]]:
    """Return the data rows of the tabular carrying \\label{label}."""
    block = re.search(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", tex, re.S)
    blocks = re.findall(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", tex, re.S)
    for block in blocks:
        if f"\\label{{{label}}}" not in block:
            continue
        body = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", block, re.S)
        rows = []
        for line in body.group(1).split("\\\\"):
            line = line.replace("\\hline", "").strip()
            if not line:
                continue
            cells = [c.strip() for c in line.split("&")]
            if len(cells) > 1:
                rows.append(cells)
        return rows
    raise KeyError(f"no table with label {label} in main.tex")


def fmt(x: float, digits: int) -> str:
    """Round half away from zero, on the decimal the CSV actually stores.

    Plain f-strings round the binary float, so a stored 0.0055 prints as
    '0.005' while the table (rounding the decimal) says 0.006. Going through
    Decimal(str(x)) reproduces the table's convention exactly.
    """
    q = Decimal(1).scaleb(-digits)
    return str(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def cell(mean: float, std: float, dm: int, ds: int) -> str:
    return f"{fmt(mean, dm)} $\\pm$ {fmt(std, ds)}"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, bool]] = []

    def check(self, what: str, expected: str, found: str) -> None:
        norm = lambda s: re.sub(r"\s+", " ", s).replace("$\\pm$", "±").strip()
        self.rows.append((what, expected, found, norm(expected) == norm(found)))

    def report(self) -> bool:
        ok = all(r[-1] for r in self.rows)
        width = max(len(r[0]) for r in self.rows)
        for what, exp, found, good in self.rows:
            flag = "ok  " if good else "FAIL"
            line = f"  [{flag}] {what:<{width}}  csv: {exp}"
            if not good:
                line += f"   |   main.tex: {found}"
            print(line)
        print(f"\n  {sum(r[-1] for r in self.rows)}/{len(self.rows)} checks passed")
        return ok


def readable_mass_for(repr_label: str, model: str, svd: dict | None) -> float | None:
    """Share of |SHAP| mass on human-readable features, by representation.

    meta-only and TF-IDF are 1.0 by definition (every feature is nameable);
    MiniLM comes from the cached artifacts and SVD from the control run, both
    computed by camera_ready_experiments.py.
    """
    if repr_label in ("Meta only", "TF-IDF + meta"):
        return 1.0
    if repr_label == SVD_LABEL:
        return svd["local"][model]["readable_mass_mean"] if svd else None
    p = PAPER / "camera_ready_results.json"
    if not p.exists():
        return None
    wc = json.loads(p.read_text()).get("workflow_complexity", {}).get("by_complexity", {})
    cfg = next(k for k, v in REPR_LABEL.items() if v == repr_label)
    return wc.get(cfg, {}).get(model, {}).get("all_readable_mass")


def load_svd_control() -> dict | None:
    """Camera-ready dense control; produced by camera_ready_experiments.py."""
    p = PAPER / "camera_ready_results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("svd_control")


def verify(tex: str) -> bool:
    """Assert every number a figure plots also appears in main.tex."""
    rep = Report()
    svd = load_svd_control()

    # Table I  (agg=max predictive anchor) and Table III (agg=mean)
    for label, agg, cols in [
        ("tab:predictive-anchor", "max", ("r2", "mae", "mse")),
        ("tab:mean-robustness", "mean", ("r2", "mae", "mse")),
    ]:
        pred = load_predictive(agg)
        loc = load_local(agg)
        rows = parse_tabular(tex, label)
        header, data = rows[0], rows[1:]
        for cells in data:
            repr_label, model_label = cells[0], cells[1]
            model = next(k for k, v in MODEL_SHORT.items() if v == model_label)
            if repr_label == SVD_LABEL:
                if svd is None or agg != "max":
                    rep.check(f"{label} {repr_label}/{model_label}",
                              "camera_ready_results.json", "MISSING (run camera_ready_experiments.py)")
                    continue
                r = svd["predictive"][model]
            else:
                cfg = next(k for k, v in REPR_LABEL.items() if v == repr_label)
                r = pred.loc[(cfg, model)]
            expect = [cell(r["r2_mean"], r["r2_std"], 3, 3),
                      cell(r["mae_mean"], r["mae_std"], 3, 3),
                      cell(r["mse_mean"], r["mse_std"], 3, 3)]
            if "Readability" in header:  # Table III also carries local metrics
                lr = loc.loc[(cfg, model)]
                expect += [cell(lr["readability_mean"], lr["readability_std"], 3, 3),
                           cell(lr["sparsity_mean"], lr["sparsity_std"], 2, 2)]
            for name, exp, found in zip(header[2:], expect, cells[2:]):
                rep.check(f"{label} {repr_label}/{model_label} {name}", exp, found)

    # Table II (local explanation quality, agg=max)
    loc = load_local("max")
    rows = parse_tabular(tex, "tab:local-quality")
    for cells in rows[1:]:
        model = next(k for k, v in MODEL_SHORT.items() if v == cells[1])
        if cells[0] == SVD_LABEL:
            if svd is None:
                rep.check(f"tab:local-quality {cells[0]}/{cells[1]}",
                          "camera_ready_results.json", "MISSING")
                continue
            r = svd["local"][model]
        else:
            cfg = next(k for k, v in REPR_LABEL.items() if v == cells[0])
            r = loc.loc[(cfg, model)]
        rep.check(f"tab:local-quality {cells[0]}/{cells[1]} Readability",
                  cell(r["readability_mean"], r["readability_std"], 3, 3), cells[2])
        rep.check(f"tab:local-quality {cells[0]}/{cells[1]} Sparsity",
                  cell(r["sparsity_mean"], r["sparsity_std"], 2, 2), cells[3])
        if len(cells) > 4:  # readable-mass column added for the camera-ready
            mass = readable_mass_for(cells[0], model, svd)
            rep.check(f"tab:local-quality {cells[0]}/{cells[1]} Read. mass",
                      fmt(mass, 3) if mass is not None else "unavailable", cells[4])

    # Stability values live in prose (Sec. VIII and Sec. IX), not in a table.
    flat = re.sub(r"\s+", " ", tex)
    for agg in ("max", "mean"):
        stab = load_stability(agg)
        for (cfg, model), r in stab.iterrows():
            want = f"{r['jaccard_mean']:.3f} \\pm {r['jaccard_std']:.3f}"
            rep.check(f"stability({agg}) {REPR_LABEL[cfg]}/{MODEL_SHORT[model]}",
                      want, want if want in flat else "not found in main.tex")

    return rep.report()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def grouped_bars(ax, groups, series, value, err, colors, labels):
    """Grouped bar helper: one cluster per group, one bar per series."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        off = (i - (n - 1) / 2) * width
        vals = [value(g, s) for g in groups]
        errs = [err(g, s) for g in groups] if err else None
        ax.bar(x + off, vals, width * 0.92, yerr=errs, label=labels[s],
               color=colors[s], edgecolor="none",
               error_kw=dict(ecolor="#333333", lw=0.7, capsize=2, capthick=0.7))
    ax.set_xticks(x)
    return x


def plot_groups(agg: str, svd: dict | None) -> tuple[list[str], dict[str, str]]:
    """Figure groups, including the dense control when it has been computed.

    Tables I and II carry the control, so the figures must too -- a figure that
    shows fewer conditions than its own table is how the submitted Fig. 1 went
    wrong in the first place.
    """
    groups = list(REPRESENTATIONS)
    labels = dict(REPR_LABEL)
    if svd is not None and agg == "max":
        groups.insert(2, "svd_control")
        labels["svd_control"] = SVD_LABEL
    return groups, labels


def fig1_predictive(agg: str) -> None:
    pred = load_predictive(agg)
    svd = load_svd_control()
    groups, labels = plot_groups(agg, svd)

    def val(g, s, key):
        if g == "svd_control":
            return svd["predictive"][s][key]
        return pred.loc[(g, s), key]

    fig, ax = plt.subplots(figsize=(IEEE_COL, 2.15))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    grouped_bars(
        ax, groups, MODELS,
        value=lambda g, s: val(g, s, "r2_mean"),
        err=lambda g, s: val(g, s, "r2_std"),
        colors=C_MODEL, labels=MODEL_LABEL,
    )
    ax.axhline(0, color="#555555", lw=0.6)
    ax.set_xticklabels([labels[r] for r in groups],
                       rotation=20 if len(groups) > 3 else 0,
                       ha="right" if len(groups) > 3 else "center")
    ax.set_ylabel("$R^2$ (mean $\\pm$ s.d. over 6 assignments)")
    ax.legend(frameon=False, loc="lower right", handlelength=1.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "fig1_predictive_anchor" if agg == "max" else f"fig1_predictive_anchor_{agg}")


def _shap_panel(ax, df, top_n, show_ylabel=True):
    d = df.nlargest(top_n, "mean_abs_shap").iloc[::-1]
    ns = [namespace_of(f) for f in d["feature"]]
    ax.barh(np.arange(len(d)), d["mean_abs_shap"],
            color=[C_NAMESPACE[n] for n in ns], height=0.72, edgecolor="none")
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels([short_name(f) for f in d["feature"]] if show_ylabel else [])
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def fig2_global_shap(agg: str) -> None:
    # Wide variant: all six panels, sized for a two-column figure* environment.
    fig, axes = plt.subplots(2, 3, figsize=(IEEE_FULL, 3.6), constrained_layout=True)
    for r, model in enumerate(MODELS):
        for c, cfg in enumerate(REPRESENTATIONS):
            ax = axes[r, c]
            _shap_panel(ax, load_global_shap(agg, cfg, model), 10)
            ax.set_title(f"{REPR_LABEL[cfg]} | {MODEL_LABEL[model]}", pad=3)
            if r == 1:
                ax.set_xlabel("Mean $|$SHAP$|$")
    handles = [mpl.patches.Patch(color=C_NAMESPACE[k], label=NAMESPACE_LABEL[k])
               for k in ("meta", "text_token", "text_emb")]
    fig.legend(handles=handles, ncol=3, frameon=False,
               loc="outside lower center", handlelength=1.2)
    save(fig, "fig2_global_shap" if agg == "max" else f"fig2_global_shap_{agg}")

    # Compact variant: the contrast the text actually argues (TF-IDF vs MiniLM
    # under HistGBRT), legible at single-column width.
    fig, axes = plt.subplots(2, 1, figsize=(IEEE_COL, 3.3), constrained_layout=True)
    for ax, cfg in zip(axes, ["tfidf_meta", "minilm_meta"]):
        _shap_panel(ax, load_global_shap(agg, cfg, "hist_gbrt"), 8)
        ax.set_title(f"{REPR_LABEL[cfg]} | HistGBRT", pad=3)
    axes[-1].set_xlabel("Mean $|$SHAP$|$")
    handles = [mpl.patches.Patch(color=C_NAMESPACE[k], label=NAMESPACE_LABEL[k])
               for k in ("meta", "text_token", "text_emb")]
    fig.legend(handles=handles, ncol=1, frameon=False,
               loc="outside lower center", handlelength=1.2)
    save(fig, "fig2_global_shap_1col" if agg == "max" else f"fig2_global_shap_1col_{agg}")


def fig3_local_quality(agg: str) -> None:
    loc = load_local(agg)
    svd = load_svd_control()
    groups, labels = plot_groups(agg, svd)

    def val(g, s, key):
        if g == "svd_control":
            return svd["local"][s][key]
        return loc.loc[(g, s), key]

    fig, axes = plt.subplots(1, 2, figsize=(IEEE_COL, 2.4), constrained_layout=True)
    specs = [("readability", "Readability (top-10 share)", axes[0]),
             ("sparsity", "Sparsity (features for 80% mass)", axes[1])]
    for key, ylabel, ax in specs:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        grouped_bars(
            ax, groups, MODELS,
            value=lambda g, s, k=key: val(g, s, f"{k}_mean"),
            err=lambda g, s, k=key: val(g, s, f"{k}_std"),
            colors=C_MODEL, labels=MODEL_LABEL,
        )
        ax.set_xticklabels([labels[r] for r in groups],
                           rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False,
               loc="outside lower center", handlelength=1.2)
    save(fig, "fig3_local_quality" if agg == "max" else f"fig3_local_quality_{agg}")


def fig4_stability(agg: str) -> None:
    stab = load_stability(agg)
    reps = ["tfidf_meta", "minilm_meta"]  # meta_only is degenerate (top-10 == all 10)
    fig, ax = plt.subplots(figsize=(IEEE_COL, 2.0))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    n = len(reps)
    width = 0.8 / n
    x = np.arange(len(MODELS))
    for i, cfg in enumerate(reps):
        off = (i - (n - 1) / 2) * width
        vals = [stab.loc[(cfg, m), "jaccard_mean"] for m in MODELS]
        errs = [stab.loc[(cfg, m), "jaccard_std"] for m in MODELS]
        ax.bar(x + off, vals, width * 0.92, yerr=errs, label=REPR_LABEL[cfg],
               color=(C_NAMESPACE["text_token"] if cfg == "tfidf_meta"
                      else C_NAMESPACE["text_emb"]), edgecolor="none",
               error_kw=dict(ecolor="#333333", lw=0.7, capsize=2, capthick=0.7))
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean top-10 Jaccard overlap")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "fig4_global_stability" if agg == "max" else f"fig4_global_stability_{agg}")


def pick_case_row(agg: str, model: str = "hist_gbrt") -> tuple[str, dict]:
    """Deterministically choose the held-out instance for the case study.

    Both runs evaluate SHAP on exactly the same held-out rows, so the same
    (task, workflow) pair can be explained under either representation. Rule:
    among fold-0 evaluation rows, take the one with the highest observed
    accuracy (ties -> lowest row id) -- i.e. explaining a genuinely strong
    recommendation, which is the scenario the paper describes.
    """
    art = load_shap_artifacts(agg, "tfidf_meta")
    fold = art["task_group_kfold"][model]["shap_artifacts"][0]
    ids = np.asarray(fold.row_ids)          # "<task_id>::<flow_id>"
    best = str(ids[np.lexsort((ids, -np.asarray(fold.y_true)))[0]])
    task_id, flow_id = (int(v) for v in best.split("::"))
    sup = joblib.load(results_dir(agg) / "tfidf_meta" / "pipeline_artifacts.joblib")
    df, flows = sup["supervised_df"], sup["flows_df"]
    row = df[(df["task_id"] == task_id) & (df["flow_id"] == flow_id)].iloc[0]
    name = flows.loc[flows["flow_id"] == flow_id, "flow_name"]
    return best, {
        "task_id": task_id,
        "flow_id": flow_id,
        "flow_name": str(name.iloc[0]) if len(name) else "(unknown)",
        "accuracy": float(row["target_value"]),
    }


def _case_panel(ax, agg, cfg, row_id, model):
    art = load_shap_artifacts(agg, cfg)
    fold = art["task_group_kfold"][model]["shap_artifacts"][0]
    pos = int(np.where(np.asarray(fold.row_ids) == row_id)[0][0])
    sv = np.asarray(fold.shap_values)[pos]
    names = np.asarray(fold.feature_names)
    idx = np.argsort(np.abs(sv))[::-1][:10][::-1]
    ns = [namespace_of(f) for f in names[idx]]
    ax.barh(np.arange(len(idx)), sv[idx], color=[C_NAMESPACE[n] for n in ns],
            height=0.72, edgecolor="none")
    ax.set_yticks(np.arange(len(idx)))
    ax.set_yticklabels([short_name(f) for f in names[idx]])
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color="#555555", lw=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    ax.set_title(f"{REPR_LABEL[cfg]} | {MODEL_LABEL[model]}", pad=3)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    return ns


def _case_legend(fig, namespaces, ncol=3):
    used = [k for k in ("meta", "text_token", "text_emb") if k in namespaces]
    fig.legend(handles=[mpl.patches.Patch(color=C_NAMESPACE[k], label=NAMESPACE_LABEL[k])
                        for k in used],
               ncol=ncol, frameon=False, loc="outside lower center", handlelength=1.2)


def fig5_case_study(agg: str, row_id: str, model: str = "hist_gbrt") -> None:
    # Separate panels, as main.tex currently includes them.
    for cfg, stem in [("tfidf_meta", "fig5a_case_study_tfidf"),
                      ("minilm_meta", "fig5b_case_study_minilm")]:
        fig, ax = plt.subplots(figsize=(IEEE_COL, 2.2), constrained_layout=True)
        ns = _case_panel(ax, agg, cfg, row_id, model)
        ax.set_xlabel("SHAP value (impact on predicted accuracy)")
        if len(set(ns)) > 1:
            # below the axes: any in-axes corner collides with the bars, which
            # span the full width at both ends of the ranking
            _case_legend(fig, ns, ncol=2)
        save(fig, stem if agg == "max" else f"{stem}_{agg}")

    # Combined two-panel version: same content in one float, which costs one
    # caption instead of two and puts the two explanations side by side.
    fig, axes = plt.subplots(2, 1, figsize=(IEEE_COL, 3.9), constrained_layout=True)
    seen = []
    for ax, cfg in zip(axes, ["tfidf_meta", "minilm_meta"]):
        seen += _case_panel(ax, agg, cfg, row_id, model)
    axes[-1].set_xlabel("SHAP value (impact on predicted accuracy)")
    _case_legend(fig, seen, ncol=2)
    save(fig, "fig5_case_study_combined" if agg == "max"
         else f"fig5_case_study_combined_{agg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agg", choices=["max", "mean", "both"], default="max")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--case-row", default=None,
                    help="row id for Figs. 5a/5b (default: deterministic rule)")
    args = ap.parse_args()

    print("Verifying figure sources against main.tex")
    ok = verify(tex_source())
    if args.verify_only:
        return 0 if ok else 1
    if not ok:
        print("\nFigures NOT regenerated: main.tex and the CSVs disagree (see above).")
        return 1

    style()
    aggs = ["max", "mean"] if args.agg == "both" else [args.agg]
    for agg in aggs:
        print(f"\nRendering figures for agg={agg}")
        fig1_predictive(agg)
        fig2_global_shap(agg)
        fig3_local_quality(agg)
        fig4_stability(agg)
        row_id = args.case_row if args.case_row is not None else pick_case_row(agg)[0]
        info = pick_case_row(agg)[1] if args.case_row is None else {}
        fig5_case_study(agg, row_id)
        if info:
            print(f"  case study row {row_id}: task {info['task_id']}, "
                  f"flow {info['flow_id']} ({info['flow_name']}), "
                  f"observed accuracy {info['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

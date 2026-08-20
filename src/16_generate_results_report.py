#!/usr/bin/env python3
"""Generate publication-oriented plots and a compact HTML run report."""
import argparse
import html
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "results" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROJECT_ROOT, check_version, get_logger, load_config, save_json

log = get_logger("results_report")
COLORS = {"blue": "#2667A8", "green": "#2A9D70", "orange": "#E78A32", "red": "#C94C4C"}


def savefig(path):
    plt.tight_layout(); plt.savefig(path, dpi=220, bbox_inches="tight"); plt.close()


def plot_mutation_metrics(results, out):
    p = results / "model_metrics.json"
    if not p.exists(): return None
    m = json.loads(p.read_text()); names = ["Accuracy", "ROC AUC"]
    vals = [m.get("accuracy"), m.get("roc_auc")]
    plt.figure(figsize=(5.5, 4)); plt.bar(names, vals, color=[COLORS["blue"], COLORS["green"]])
    plt.ylim(0, 1); plt.ylabel("Cross-validated score"); plt.title("Secondary mutation–product response model")
    for i, v in enumerate(vals):
        if v is not None: plt.text(i, v + .025, f"{v:.3f}", ha="center")
    savefig(out); return out.name


def plot_integrated_cv(results, out):
    p = results / "integrated_qsar_cv_predictions.csv"
    if not p.exists(): return None
    d = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))
    ax[0].scatter(d.observed_pX, d.cv_predicted_pX, c=d.observed_active,
                  cmap="coolwarm", alpha=.75, edgecolor="white")
    lo = min(d.observed_pX.min(), d.cv_predicted_pX.min()); hi = max(d.observed_pX.max(), d.cv_predicted_pX.max())
    ax[0].plot([lo, hi], [lo, hi], "--", color="gray"); ax[0].set(xlabel="Observed pX", ylabel="CV predicted pX", title="Potency regression")
    active = d[d.observed_active == 1].cv_active_probability
    inactive = d[d.observed_active == 0].cv_active_probability
    ax[1].boxplot([inactive, active], tick_labels=["Inactive", "Active"], patch_artist=True,
                  boxprops={"facecolor": COLORS["blue"], "alpha": .65})
    ax[1].set(ylabel="CV activity probability", title="Activity classification", ylim=(0, 1))
    savefig(out); return out.name


def plot_model_comparison(results, out):
    p = results / "bioactivity_model_comparison.csv"
    if not p.exists(): return None
    d = pd.read_csv(p).sort_values("r2", ascending=True).tail(15)
    labels = d["model"].astype(str) + " | " + d["feature_set"].astype(str)
    plt.figure(figsize=(9, max(4, .35 * len(d)))); plt.barh(labels, d.r2, color=np.where(d.r2 >= 0, COLORS["green"], COLORS["red"]))
    plt.axvline(0, color="black", lw=.8); plt.xlabel("Cross-validated R²"); plt.title("CFTR bioactivity model comparison")
    savefig(out); return out.name


def plot_docking(results, out):
    p = results / "docking_scores.csv"
    if not p.exists(): return None
    d = pd.read_csv(p).dropna(subset=["best_affinity_kcal_mol"])
    if d.empty: return None
    pivot = d.pivot_table(index="molecule_id", columns="binding_site", values="best_affinity_kcal_mol", aggfunc="min")
    # Rank compounds by their strongest predicted interaction.  Vina uses a
    # lower-is-better convention, so the most negative per-row minimum comes
    # first and unfavorable/positive scores naturally fall to the bottom.
    pivot = pivot.loc[pivot.min(axis=1).sort_values(ascending=True).index]
    plt.figure(figsize=(max(7, .7 * len(pivot.columns)), max(4, .28 * len(pivot))))
    plt.imshow(pivot.values, aspect="auto", cmap="viridis_r", vmin=pivot.min().min(), vmax=0)
    plt.colorbar(label="Vina affinity (kcal/mol; more negative = stronger)")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right", fontsize=8)
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=7)
    plt.title("CFTR pocket docking landscape — strongest binders first")
    savefig(out); return out.name


def plot_leads(results, out):
    p = results / "virtual_screening_hits.csv"
    if not p.exists(): p = results / "ranked_cftr_leads.csv"
    if not p.exists(): return None
    d = pd.read_csv(p).head(20).sort_values("lead_score")
    label_col = "molecule_chembl_id" if "molecule_chembl_id" in d else "canonical_smiles"
    plt.figure(figsize=(8, max(4, .3 * len(d)))); plt.barh(d[label_col].astype(str), d.lead_score, color=COLORS["orange"])
    plt.xlim(0, 1); plt.xlabel("Composite lead-priority score"); plt.title("Top virtual-screening CFTR leads")
    savefig(out); return out.name


def main():
    check_version(); ap = argparse.ArgumentParser(); ap.add_argument("--run-id", default="manual")
    ap.add_argument("--log", default=""); args = ap.parse_args()
    cfg = load_config(); results = PROJECT_ROOT / cfg["paths"]["results_dir"]
    report = results / "report"; plots = report / "plots"; plots.mkdir(parents=True, exist_ok=True)
    generated = []
    jobs = [(plot_mutation_metrics, "mutation_response_metrics.png"),
            (plot_integrated_cv, "integrated_qsar_validation.png"),
            (plot_model_comparison, "bioactivity_model_comparison.png"),
            (plot_docking, "docking_heatmap.png"), (plot_leads, "top_ranked_leads.png")]
    for fn, name in jobs:
        try:
            result = fn(results, plots / name)
            if result: generated.append(result)
        except Exception as exc: log.warning(f"Could not generate {name}: {exc}")
    tables = [p.name for p in sorted(results.glob("*.csv"))]
    metrics = [p.name for p in sorted(results.glob("*.json"))]
    cards = "\n".join(f'<figure><img src="plots/{html.escape(x)}"><figcaption>{html.escape(x)}</figcaption></figure>' for x in generated)
    links = "\n".join(f'<li><a href="../{html.escape(x)}">{html.escape(x)}</a></li>' for x in tables + metrics)
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>CFTR pipeline report</title>
    <style>body{{font:15px system-ui;max-width:1200px;margin:35px auto;padding:0 20px;color:#21313c}}h1{{color:#174d76}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:18px}}figure{{margin:0;padding:15px;border:1px solid #d9e2e8;border-radius:10px}}img{{width:100%}}code{{background:#eef3f6;padding:2px 5px}}</style></head>
    <body><h1>Novel CFTR small-molecule discovery report</h1>
    <p>Run <code>{html.escape(args.run_id)}</code>. The primary outputs are candidate-compound activity, potency, docking support, developability, and lead rank. Approved modulators are reference controls; marketed combination products are not the discovery targets.</p>
    <p><strong>Research prioritization only:</strong> experimental CFTR assays are required for confirmation.</p>
    <div class='grid'>{cards}</div><h2>Result files</h2><ul>{links}</ul></body></html>"""
    (report / "index.html").write_text(page)
    save_json({"run_id": args.run_id, "log": args.log, "plots": generated,
               "tables": tables, "metrics": metrics}, report / "run_manifest.json")
    log.info(f"Generated {len(generated)} plots and report -> {report / 'index.html'}")


if __name__ == "__main__": main()

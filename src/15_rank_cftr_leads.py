#!/usr/bin/env python3
"""Rank CFTR compounds using activity, potency, docking and drug-likeness.

Scores are reported separately as well as in a transparent composite. This
avoids presenting a docking score as proof of CFTR modulation.
"""
import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROJECT_ROOT, check_version, get_logger, load_config
from importlib.util import spec_from_file_location, module_from_spec

log = get_logger("rank_leads")


def scale01(s, higher=True):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 2 or s.max() == s.min(): return pd.Series(0.5, index=s.index)
    z = (s - s.min()) / (s.max() - s.min())
    return z if higher else 1 - z


def main():
    check_version(); ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="CSV with molecule_chembl_id, canonical_smiles and step-09 features")
    ap.add_argument("--docking", default=None, help="Docking CSV (default: results/docking_scores.csv)")
    ap.add_argument("--output", default="ranked_cftr_leads.csv", help="Output filename under results/")
    ap.add_argument("--top", type=int, default=50); args = ap.parse_args()
    cfg = load_config(); proc = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    results = PROJECT_ROOT / cfg["paths"]["results_dir"]; models = PROJECT_ROOT / cfg["paths"]["models_dir"]
    inp = Path(args.input) if args.input else proc / "chembl_bioactivity_features.csv"
    if not inp.exists(): log.error(f"{inp} missing; run step 09 first."); sys.exit(1)
    df = pd.read_csv(inp)
    cb = joblib.load(models / "cftr_activity_classifier.joblib")
    rb = joblib.load(models / "cftr_potency_regressor.joblib")
    dock_path = Path(args.docking).expanduser().resolve() if args.docking else results / "docking_scores.csv"
    if dock_path.exists():
        d = pd.read_csv(dock_path); wide = d.pivot_table(index="molecule_id", columns="binding_site",
            values="best_affinity_kcal_mol", aggfunc="min").reset_index()
        wide.columns = ["molecule_chembl_id" if c == "molecule_id" else f"dock_{c}" for c in wide.columns]
        df = df.merge(wide, on="molecule_chembl_id", how="left")
    for c in set(cb["feature_cols"]) | set(rb["feature_cols"]):
        if c not in df: df[c] = np.nan
    df["predicted_active_probability"] = cb["pipeline"].predict_proba(df[cb["feature_cols"]])[:, 1]
    df["predicted_pX"] = rb["pipeline"].predict(df[rb["feature_cols"]])
    dock_cols = [c for c in df if c.startswith("dock_")]
    df["best_docking_affinity_kcal_mol"] = df[dock_cols].min(axis=1) if dock_cols else np.nan
    if dock_cols:
        # pandas raises on idxmin when a row contains no docking result.  Those
        # compounds remain valid QSAR candidates, but their site is unknown.
        dock_values = df[dock_cols].apply(pd.to_numeric, errors="coerce")
        has_docking = dock_values.notna().any(axis=1)
        df["predicted_binding_site"] = "not_docked"
        df.loc[has_docking, "predicted_binding_site"] = (
            dock_values.loc[has_docking].idxmin(axis=1)
            .str.replace("dock_", "", regex=False)
        )
    else:
        df["predicted_binding_site"] = "not_docked"
    qed = df.get("qed_druglikeness", pd.Series(np.nan, index=df.index)).fillna(0.5).clip(0, 1)
    lip = df.get("lipinski_violations", pd.Series(np.nan, index=df.index)).fillna(2)
    df["developability_score"] = (0.75 * qed + 0.25 * (1 - lip.clip(0, 4) / 4)).clip(0, 1)
    dock_support = scale01(df["best_docking_affinity_kcal_mol"], higher=False) if dock_cols else 0.5
    df["lead_score"] = (0.40 * df["predicted_active_probability"] +
                        0.30 * scale01(df["predicted_pX"]) +
                        0.20 * df["developability_score"] + 0.10 * dock_support)
    df["evidence_warning"] = "computational priority; requires orthogonal CFTR assay confirmation"
    out = df.sort_values("lead_score", ascending=False).head(args.top)
    path = results / args.output; out.to_csv(path, index=False)
    log.info(f"Ranked {len(df)} compounds; wrote top {len(out)} -> {path}")


if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Train paired CFTR activity-classification and potency-regression models.

Input: data/processed/chembl_bioactivity_features.csv (step 09)
Optional: results/docking_scores.csv (step 13)

The classifier and regressor use the same RDKit descriptor/Morgan feature
space. Validation is grouped by Bemis-Murcko scaffold when possible so close
analogues cannot be split across train and validation folds.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import (GroupKFold, KFold, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_predict)
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROJECT_ROOT, check_version, get_logger, load_config, save_json

log = get_logger("integrated_qsar")
DESCRIPTORS = [
    "mol_weight", "logp", "tpsa", "h_bond_donors", "h_bond_acceptors",
    "rotatable_bonds", "aromatic_rings", "ring_count", "heavy_atom_count",
    "heteroatom_count", "fraction_csp3", "molar_refractivity",
    "qed_druglikeness", "lipinski_violations",
]


def scaffold(smiles):
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(str(smiles))
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or str(smiles)
    except Exception:
        return str(smiles)


def merge_docking(df, results_dir):
    path = results_dir / "docking_scores.csv"
    if not path.exists():
        return df, []
    dock = pd.read_csv(path)
    if not {"molecule_id", "binding_site", "best_affinity_kcal_mol"}.issubset(dock.columns):
        return df, []
    wide = dock.pivot_table(index="molecule_id", columns="binding_site",
                            values="best_affinity_kcal_mol", aggfunc="min").reset_index()
    wide.columns = ["molecule_chembl_id" if c == "molecule_id" else f" dock_{c}".strip()
                    for c in wide.columns]
    cols = [c for c in wide.columns if c.startswith("dock_")]
    return df.merge(wide, on="molecule_chembl_id", how="left"), cols


def model(kind):
    estimator = (RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                        class_weight="balanced", random_state=42)
                 if kind == "classification" else
                 RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                       random_state=42))
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", estimator)])


def main():
    check_version()
    ap = argparse.ArgumentParser()
    ap.add_argument("--activity-threshold", type=float, default=6.0,
                    help="pX threshold for active/inactive labels (default: 6 = 1 uM)")
    args = ap.parse_args()
    cfg = load_config()
    proc = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    results = PROJECT_ROOT / cfg["paths"]["results_dir"]
    models = PROJECT_ROOT / cfg["paths"]["models_dir"]
    results.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    path = proc / "chembl_bioactivity_features.csv"
    if not path.exists():
        log.error(f"{path} missing; run steps 08 and 09 first."); sys.exit(1)
    df = pd.read_csv(path).dropna(subset=["pX_median", "canonical_smiles"]).copy()
    if len(df) < 20:
        log.error(f"Need at least 20 measured compounds; found {len(df)}."); sys.exit(1)
    df, dock_cols = merge_docking(df, results)
    fp_cols = [c for c in df if c.startswith("fp_")]
    feature_cols = [c for c in DESCRIPTORS if c in df] + fp_cols + dock_cols
    X, y_reg = df[feature_cols], df["pX_median"].astype(float)
    y_cls = (y_reg >= args.activity_threshold).astype(int)
    if y_cls.nunique() != 2 or y_cls.value_counts().min() < 3:
        log.error("Activity threshold does not yield at least 3 compounds in each class."); sys.exit(1)

    groups = df["canonical_smiles"].map(scaffold)
    n_groups = groups.nunique()
    if n_groups >= 5:
        cv = StratifiedGroupKFold(n_splits=min(5, n_groups), shuffle=True, random_state=42)
        split_args = {"groups": groups}
        cv_name = f"scaffold_group_{cv.n_splits}_fold"
    else:
        cv = StratifiedKFold(n_splits=min(5, int(y_cls.value_counts().min())), shuffle=True, random_state=42)
        split_args = {}; cv_name = "stratified_fallback"

    clf = model("classification")
    cls_prob = cross_val_predict(clf, X, y_cls, cv=cv, method="predict_proba", **split_args)[:, 1]
    cls_pred = (cls_prob >= 0.5).astype(int)
    # Regression uses the same held-out scaffold groups.
    reg_cv = GroupKFold(n_splits=min(5, n_groups)) if n_groups >= 5 else KFold(n_splits=5, shuffle=True, random_state=42)
    reg_args = split_args if n_groups >= 5 else {}
    reg = model("regression")
    reg_pred = cross_val_predict(reg, X, y_reg, cv=reg_cv, **reg_args)

    metrics = {
        "n_compounds": int(len(df)), "n_scaffolds": int(n_groups),
        "activity_threshold_pX": args.activity_threshold, "cv_scheme": cv_name,
        "classification": {"accuracy": float(accuracy_score(y_cls, cls_pred)),
                           "balanced_accuracy": float(balanced_accuracy_score(y_cls, cls_pred)),
                           "roc_auc": float(roc_auc_score(y_cls, cls_prob))},
        "regression": {"r2": float(r2_score(y_reg, reg_pred)),
                       "rmse": float(mean_squared_error(y_reg, reg_pred) ** 0.5),
                       "mae": float(mean_absolute_error(y_reg, reg_pred))},
        "docking_features": dock_cols,
        "warning": "Research prioritization only; docking is supporting evidence, not experimental confirmation."
    }
    clf.fit(X, y_cls); reg.fit(X, y_reg)
    common = {"feature_cols": feature_cols, "activity_threshold_pX": args.activity_threshold,
              "cv_scheme": cv_name}
    joblib.dump({**common, "pipeline": clf, "task": "classification"}, models / "cftr_activity_classifier.joblib")
    joblib.dump({**common, "pipeline": reg, "task": "regression"}, models / "cftr_potency_regressor.joblib")
    save_json(metrics, results / "integrated_qsar_metrics.json")
    pd.DataFrame({"molecule_chembl_id": df["molecule_chembl_id"], "observed_pX": y_reg,
                  "observed_active": y_cls, "cv_active_probability": cls_prob,
                  "cv_predicted_pX": reg_pred, "scaffold": groups}).to_csv(
                      results / "integrated_qsar_cv_predictions.csv", index=False)
    log.info(f"Saved paired models; classifier AUC={metrics['classification']['roc_auc']:.3f}, "
             f"regressor R2={metrics['regression']['r2']:.3f} ({cv_name})")


if __name__ == "__main__":
    main()

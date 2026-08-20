#!/usr/bin/env python3
"""
06_train_model.py

Trains a gradient-boosted classifier to predict whether a given
(mutation, modulator product) pair is expected to be clinically responsive,
using the rows in mutation_modulator_dataset.csv that carry a known seed
label as training data (leave-one-out CV given the tiny seed set size --
this is a scaffold/demo model; real training requires a much larger labeled
set, e.g. curated from CFTR2.org genotype-phenotype data + drug labels).

Output:
  models/cftr_modulator_response_model.joblib
  results/model_metrics.json
  results/feature_importances.csv
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, check_version, save_json, PROJECT_ROOT

log = get_logger("train_model")

NUMERIC_FEATURES = [
    "position", "dist_to_F508", "dist_to_nearest_ATP_pocket",
    "in_ABP1_degenerate_pocket", "in_ABP2_hydrolytic_pocket", "delta_hydrophobicity_kd",
    "delta_volume_A3", "delta_charge", "polarity_class_changed",
    "plddt", "dist3d_to_F508_A", "dist3d_to_ATP_pocket_A",
    "product_mean_mol_weight", "product_mean_logp", "product_mean_tpsa",
    "product_mean_h_bond_donors", "product_mean_h_bond_acceptors",
    "product_mean_rotatable_bonds", "product_mean_aromatic_rings",
    "product_mean_fraction_csp3", "product_mean_lipinski_violations",
    "n_correctors_in_product", "n_potentiators_in_product",
]
CATEGORICAL_FEATURES = ["domain", "mutation_type", "functional_class", "product"]


def build_pipeline(numeric_cols, categorical_cols):
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])
    clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=2, learning_rate=0.08, random_state=42
    )
    return Pipeline([("prep", preprocess), ("clf", clf)])


def main():
    check_version()
    cfg = load_config()
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = proc_dir / "mutation_modulator_dataset.csv"
    if not data_path.exists():
        log.error(f"{data_path} missing. Run 05_build_mutation_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    labeled = df[df["known_response_label"].notna()].copy()
    if len(labeled) < 8:
        log.error(
            f"Only {len(labeled)} labeled rows available -- too few to train "
            "or evaluate meaningfully. Expand data/raw/cf_mutations.csv and "
            "the KNOWN_LABELS table in 05_build_mutation_dataset.py with "
            "more curated genotype/product outcomes (e.g. from CFTR2.org), "
            "then re-run steps 02-06."
        )
        sys.exit(1)

    numeric_cols = [c for c in NUMERIC_FEATURES if c in labeled.columns]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in labeled.columns]
    X = labeled[numeric_cols + categorical_cols]
    y = labeled["known_response_label"].astype(int)

    log.info(f"Training on {len(labeled)} labeled rows, {len(numeric_cols)} numeric + "
              f"{len(categorical_cols)} categorical features")

    # Leave-one-out CV, appropriate given the tiny seed dataset
    loo = LeaveOneOut()
    y_true, y_pred, y_prob = [], [], []
    for train_idx, test_idx in loo.split(X):
        pipe = build_pipeline(numeric_cols, categorical_cols)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])[0]
        prob = pipe.predict_proba(X.iloc[test_idx])[0, 1]
        y_true.append(y.iloc[test_idx].values[0])
        y_pred.append(pred)
        y_prob.append(prob)

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = None  # e.g. if only one class present in y_true

    metrics = {
        "n_labeled_samples": int(len(labeled)),
        "cv_scheme": "leave_one_out",
        "accuracy": round(float(acc), 3),
        "roc_auc": round(float(auc), 3) if auc is not None else None,
        "note": (
            "Metrics computed on a small hand-curated seed label set for "
            "pipeline validation only -- NOT a validated clinical predictor. "
            "Expand the labeled set substantially (CFTR2.org genotype-"
            "phenotype data, drug label eligibility tables) before drawing "
            "any biological or clinical conclusions."
        ),
    }
    save_json(metrics, results_dir / "model_metrics.json")
    log.info(f"LOO-CV accuracy={acc:.3f}, roc_auc={auc}")

    # Final model trained on all labeled data, for use by 07_predict_*.py
    final_pipe = build_pipeline(numeric_cols, categorical_cols)
    final_pipe.fit(X, y)
    joblib.dump(
        {"pipeline": final_pipe, "numeric_cols": numeric_cols,
         "categorical_cols": categorical_cols},
        models_dir / "cftr_modulator_response_model.joblib",
    )
    log.info(f"Saved trained pipeline -> {models_dir / 'cftr_modulator_response_model.joblib'}")

    # Feature importances (from the underlying GradientBoostingClassifier)
    try:
        feature_names = final_pipe.named_steps["prep"].get_feature_names_out()
        importances = final_pipe.named_steps["clf"].feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df.sort_values("importance", ascending=False, inplace=True)
        imp_df.to_csv(results_dir / "feature_importances.csv", index=False)
        log.info(f"Top features:\n{imp_df.head(10).to_string(index=False)}")
    except Exception as e:
        log.warning(f"Could not extract feature importances: {e}")


if __name__ == "__main__":
    main()

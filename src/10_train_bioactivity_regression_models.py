#!/usr/bin/env python3
"""
10_train_bioactivity_regression_models.py

Trains and compares several regression algorithms predicting compound
bioactivity (pX = pIC50/pEC50/pKi-equivalent, -log10(M)) against CFTR from
the ChEMBL-derived feature table.

Two feature sets are evaluated separately, since they behave very
differently at small sample sizes:
  - "descriptors": the ~13 whole-molecule RDKit physicochemical descriptors
  - "descriptors+fp": descriptors + 1024-bit Morgan (ECFP4) fingerprint

Models compared (all scikit-learn, no GPU/exotic deps required):
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - ElasticNet
  - K-Nearest Neighbors Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Support Vector Regressor (RBF kernel)

Evaluation: 5-fold CV (or leave-one-out if the labeled set is very small),
reporting R^2, RMSE, MAE, and Pearson r for each (model x feature_set) pair.

Output:
  results/bioactivity_model_comparison.csv
  results/bioactivity_best_model_feature_importance.csv (if tree-based model wins)
  models/bioactivity_<model>_<feature_set>.joblib   (best model persisted)
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, check_version, save_json, PROJECT_ROOT

log = get_logger("train_bioactivity")

DESCRIPTOR_COLS = [
    "mol_weight", "logp", "tpsa", "h_bond_donors", "h_bond_acceptors",
    "rotatable_bonds", "aromatic_rings", "ring_count", "heavy_atom_count",
    "heteroatom_count", "fraction_csp3", "molar_refractivity",
    "qed_druglikeness", "lipinski_violations",
]

MODEL_FACTORIES = {
    "linear_regression": lambda: LinearRegression(),
    "ridge": lambda: Ridge(alpha=1.0, random_state=42),
    "lasso": lambda: Lasso(alpha=0.05, random_state=42, max_iter=5000),
    "elastic_net": lambda: ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42, max_iter=5000),
    "knn": lambda: KNeighborsRegressor(n_neighbors=5),
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42),
    "svr_rbf": lambda: SVR(kernel="rbf", C=2.0, epsilon=0.1),
}


def build_pipeline(model_name: str):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", MODEL_FACTORIES[model_name]()),
    ])


def evaluate(X: pd.DataFrame, y: pd.Series, model_name: str, cv):
    pipe = build_pipeline(model_name)
    y_pred = cross_val_predict(pipe, X, y, cv=cv)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y - y_pred)))
    try:
        r, _ = pearsonr(y, y_pred)
    except Exception:
        r = float("nan")
    return {"r2": round(float(r2), 3), "rmse": round(rmse, 3), "mae": round(mae, 3), "pearson_r": round(float(r), 3)}


def main():
    check_version()
    cfg = load_config()
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = proc_dir / "chembl_bioactivity_features.csv"
    if not data_path.exists():
        log.error(f"{data_path} not found. Run 08_fetch_chembl_bioactivity.py and 09_featurize_chembl_compounds.py first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    n = len(df)
    if n < 10:
        log.error(
            f"Only {n} compounds with bioactivity features -- too few to fit "
            "and cross-validate regression models meaningfully (need at "
            "least ~10-20, ideally hundreds). This usually means "
            "08_fetch_chembl_bioactivity.py could not reach the ChEMBL API "
            "in this environment. Run it on a machine with network access "
            "and copy data/raw/chembl_bioactivity_raw.csv here, then re-run "
            "steps 09-10."
        )
        sys.exit(1)

    # --- optionally merge real docking scores (13_run_docking.py) ---
    dock_path = results_dir / "docking_scores.csv"
    dock_cols = []
    if dock_path.exists():
        dock_df = pd.read_csv(dock_path)
        dock_wide = dock_df.pivot_table(
            index="molecule_id", columns="binding_site",
            values="best_affinity_kcal_mol", aggfunc="min",
        ).reset_index().rename(columns={"molecule_id": "molecule_chembl_id"})
        dock_wide.columns = [
            c if c == "molecule_chembl_id" else f"dock_affinity_{c}"
            for c in dock_wide.columns
        ]
        dock_cols = [c for c in dock_wide.columns if c.startswith("dock_affinity_")]
        df = df.merge(dock_wide, on="molecule_chembl_id", how="left")
        n_docked = df[dock_cols[0]].notna().sum() if dock_cols else 0
        log.info(f"Merged real docking scores for {n_docked}/{n} compounds ({len(dock_cols)} binding site(s))")
    else:
        log.info(
            f"{dock_path} not found -- skipping docking-augmented feature "
            "set. Run 11_extract_binding_pockets.py, 12_prepare_docking_"
            "inputs.py, and 13_run_docking.py first to enable it."
        )

    y = df["pX_median"]
    fp_cols = [c for c in df.columns if c.startswith("fp_")]
    feature_sets = {
        "descriptors": [c for c in DESCRIPTOR_COLS if c in df.columns],
        "descriptors+fp": [c for c in DESCRIPTOR_COLS if c in df.columns] + fp_cols,
    }
    if dock_cols:
        feature_sets["descriptors+docking"] = [c for c in DESCRIPTOR_COLS if c in df.columns] + dock_cols
        feature_sets["docking_only"] = dock_cols

    cv = LeaveOneOut() if n < 30 else KFold(n_splits=5, shuffle=True, random_state=42)
    cv_desc = "leave_one_out" if n < 30 else "5_fold"
    log.info(f"Evaluating on {n} compounds with {cv_desc} CV")

    all_results = []
    for fs_name, cols in feature_sets.items():
        X = df[cols]
        for model_name in MODEL_FACTORIES:
            try:
                metrics = evaluate(X, y, model_name, cv)
            except Exception as e:
                log.warning(f"{model_name} / {fs_name} failed: {e}")
                continue
            metrics.update({"model": model_name, "feature_set": fs_name, "n_features": len(cols)})
            all_results.append(metrics)
            log.info(f"{model_name:20s} [{fs_name:15s}] R2={metrics['r2']:+.3f} RMSE={metrics['rmse']:.3f} "
                      f"MAE={metrics['mae']:.3f} pearson_r={metrics['pearson_r']:.3f}")

    results_df = pd.DataFrame(all_results).sort_values("r2", ascending=False)
    results_path = results_dir / "bioactivity_model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"Wrote model comparison table -> {results_path}")

    # Refit and persist the best-performing (model, feature_set) pair on all data
    best = results_df.iloc[0]
    best_model_name, best_fs = best["model"], best["feature_set"]
    log.info(f"Best: {best_model_name} on {best_fs} (R2={best['r2']}, RMSE={best['rmse']})")
    X_best = df[feature_sets[best_fs]]
    final_pipe = build_pipeline(best_model_name)
    final_pipe.fit(X_best, y)
    joblib.dump(
        {"pipeline": final_pipe, "feature_cols": feature_sets[best_fs],
         "model_name": best_model_name, "feature_set": best_fs},
        models_dir / f"bioactivity_{best_model_name}_{best_fs.replace('+', '_')}.joblib",
    )

    if best_model_name in ("random_forest", "gradient_boosting"):
        importances = final_pipe.named_steps["model"].feature_importances_
        imp_df = pd.DataFrame({"feature": feature_sets[best_fs], "importance": importances})
        imp_df.sort_values("importance", ascending=False, inplace=True)
        imp_df.to_csv(results_dir / "bioactivity_best_model_feature_importance.csv", index=False)
        log.info(f"Top predictive features:\n{imp_df.head(10).to_string(index=False)}")

    save_json({
        "n_compounds": int(n), "cv_scheme": cv_desc,
        "best_model": best_model_name, "best_feature_set": best_fs,
        "best_r2": float(best["r2"]), "best_rmse": float(best["rmse"]),
        "note": "QSAR regression scaffold. Validate on a proper held-out "
                "test set and larger ChEMBL pull before drawing conclusions.",
    }, results_dir / "bioactivity_model_summary.json")


if __name__ == "__main__":
    main()

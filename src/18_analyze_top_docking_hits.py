#!/usr/bin/env python3
"""Create an auditable top-20 docking-hit table across all CFTR pockets."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    root = args.project.resolve()
    results = root / "results"

    docking = pd.read_csv(results / "docking_scores.csv")
    docking["best_affinity_kcal_mol"] = pd.to_numeric(
        docking["best_affinity_kcal_mol"], errors="coerce"
    )
    wide = docking.pivot_table(
        index="molecule_id", columns="binding_site",
        values="best_affinity_kcal_mol", aggfunc="min"
    )
    ordered = np.sort(wide.to_numpy(dtype=float), axis=1)
    summary = wide.copy()
    summary.insert(0, "best_affinity_kcal_mol", ordered[:, 0])
    summary.insert(1, "second_best_affinity_kcal_mol", ordered[:, 1])
    summary.insert(2, "pocket_selectivity_gap_kcal_mol", ordered[:, 1] - ordered[:, 0])
    summary.insert(3, "median_affinity_kcal_mol", np.nanmedian(ordered, axis=1))
    summary.insert(4, "best_binding_site", wide.idxmin(axis=1))
    summary = summary.sort_values("best_affinity_kcal_mol").head(args.top).reset_index()

    features_path = root / "data" / "processed" / "chembl_bioactivity_features.csv"
    if features_path.exists():
        keep = ["molecule_chembl_id", "canonical_smiles", "pX_median", "n_measurements",
                "mol_weight", "logp", "tpsa", "qed_druglikeness", "lipinski_violations"]
        features = pd.read_csv(features_path, usecols=lambda c: c in keep)
        summary = summary.merge(features, left_on="molecule_id",
                                right_on="molecule_chembl_id", how="left")
        summary = summary.drop(columns=["molecule_chembl_id"], errors="ignore")

    ranked_path = results / "ranked_cftr_all.csv"
    if ranked_path.exists():
        keep = ["molecule_chembl_id", "predicted_active_probability", "predicted_pX",
                "developability_score", "lead_score"]
        ranked = pd.read_csv(ranked_path, usecols=lambda c: c in keep)
        summary = summary.merge(ranked, left_on="molecule_id",
                                right_on="molecule_chembl_id", how="left")
        summary = summary.drop(columns=["molecule_chembl_id"], errors="ignore")

    summary.insert(0, "docking_rank", range(1, len(summary) + 1))
    summary["interpretation"] = (
        "docking hypothesis only; inspect pose/interactions and confirm experimentally"
    )
    output = results / "top20_docking_hits_deep_dive.csv"
    summary.to_csv(output, index=False)
    print(f"Wrote {len(summary)} distinct top docking hits -> {output}")


if __name__ == "__main__":
    main()

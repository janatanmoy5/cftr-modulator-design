#!/usr/bin/env python3
"""
07_predict_modulator_response.py

CLI to score a mutation against one or all five modulator products using
the trained pipeline. This runs the feature-extraction logic inline for a
single ad-hoc mutation (rather than requiring a full pipeline re-run) so it
can be used interactively.

Usage:
  python 07_predict_modulator_response.py --mutation G1244E
  python 07_predict_modulator_response.py --mutation G1244E --product Kaftrio
  python 07_predict_modulator_response.py --mutation G1244E --all-products
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, check_version, PROJECT_ROOT, atp_pocket_residues

# reuse feature logic from step 02 without re-importing the numbered module name
import importlib.util

log = get_logger("predict")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    check_version()
    parser = argparse.ArgumentParser(description="Predict CFTR modulator response for a mutation")
    parser.add_argument("--mutation", required=True, help='e.g. "G1244E", "F508del", "G551D"')
    parser.add_argument("--product", default=None, help="Restrict to one product, e.g. Kaftrio")
    parser.add_argument("--all-products", action="store_true", help="Score against all 5 products (default)")
    args = parser.parse_args()

    cfg = load_config()
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]
    src_dir = Path(__file__).resolve().parent

    model_path = models_dir / "cftr_modulator_response_model.joblib"
    if not model_path.exists():
        log.error(f"{model_path} not found. Run 06_train_model.py first.")
        sys.exit(1)
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]

    seq_mod = _load_module("seq_feat_mod", src_dir / "02_extract_sequence_features.py")
    parsed = seq_mod.parse_mutation(args.mutation)
    if parsed is None:
        log.error(f"Could not parse mutation notation: {args.mutation}")
        sys.exit(1)

    domains = cfg["domains"]
    f508_pos = cfg["key_sites"]["F508"]["residue"]
    atp_pockets = atp_pocket_residues(cfg)
    atp_all = [r for residues in atp_pockets.values() for r in residues]
    abp1_residues = atp_pockets.get("ABP1_noncanonical_degenerate", [])
    abp2_residues = atp_pockets.get("ABP2_canonical_hydrolytic", [])
    pos = parsed["pos"]

    base_row = {
        "mutation": args.mutation,
        "position": pos,
        "domain": seq_mod.domain_for_position(pos, domains),
        "mutation_type": parsed["mutation_type"],
        "dist_to_F508": abs(pos - f508_pos),
        "dist_to_nearest_ATP_pocket": min(abs(pos - s) for s in atp_all) if atp_all else None,
        "in_ABP1_degenerate_pocket": int(pos in abp1_residues),
        "in_ABP2_hydrolytic_pocket": int(pos in abp2_residues),
        "functional_class": "unknown",
        # structural features unknown for an ad-hoc query unless step 03 was
        # re-run for this mutation -- left as NaN, the trained pipeline's
        # median-imputer handles this gracefully.
        "plddt": None, "dist3d_to_F508_A": None, "dist3d_to_ATP_pocket_A": None,
    }
    if parsed["mt"] is not None:
        base_row["delta_hydrophobicity_kd"] = (
            seq_mod.KD_HYDROPATHY[parsed["mt"]] - seq_mod.KD_HYDROPATHY[parsed["wt"]]
        )
        base_row["delta_volume_A3"] = seq_mod.VOLUME[parsed["mt"]] - seq_mod.VOLUME[parsed["wt"]]
        base_row["delta_charge"] = seq_mod.CHARGE.get(parsed["mt"], 0) - seq_mod.CHARGE.get(parsed["wt"], 0)
        base_row["polarity_class_changed"] = int((parsed["mt"] in seq_mod.POLAR) != (parsed["wt"] in seq_mod.POLAR))

    lig_path = proc_dir / "ligand_features.csv"
    lig_df = pd.read_csv(lig_path) if lig_path.exists() else pd.DataFrame()
    product_map = {p["name"]: p["components"] for p in cfg["products"]}

    products = [args.product] if args.product else list(product_map.keys())
    rows = []
    for product_name in products:
        components = product_map[product_name]
        row = dict(base_row)
        row["product"] = product_name
        comp_rows = lig_df[lig_df["name"].isin(components)] if len(lig_df) else pd.DataFrame()
        for col in ["mol_weight", "logp", "tpsa", "h_bond_donors", "h_bond_acceptors",
                    "rotatable_bonds", "aromatic_rings", "fraction_csp3", "lipinski_violations"]:
            row[f"product_mean_{col}"] = comp_rows[col].mean() if col in comp_rows.columns and len(comp_rows) else None
        row["n_correctors_in_product"] = sum(
            1 for c in components for m in cfg["modulators"] if m["name"] == c and m["class"] == "corrector"
        )
        row["n_potentiators_in_product"] = sum(
            1 for c in components for m in cfg["modulators"] if m["name"] == c and m["class"] == "potentiator"
        )
        rows.append(row)

    X = pd.DataFrame(rows)
    for c in numeric_cols + categorical_cols:
        if c not in X.columns:
            X[c] = None
    X = X[numeric_cols + categorical_cols]

    probs = pipe.predict_proba(X)[:, 1]
    print(f"\nMutation: {args.mutation}  (domain={base_row['domain']}, type={base_row['mutation_type']})")
    print("-" * 60)
    for product_name, p in zip(products, probs):
        print(f"  {product_name:10s}  predicted response probability: {p:.3f}")
    print("-" * 60)
    print(
        "NOTE: this is a research scaffold model trained on a tiny hand-"
        "curated seed label set (see 06_train_model.py). It is NOT validated"
        " for clinical use. For actual treatment eligibility, consult "
        "CFTR2.org and the current FDA/EMA drug label.\n"
    )


if __name__ == "__main__":
    main()

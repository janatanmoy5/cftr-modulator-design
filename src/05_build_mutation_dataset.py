#!/usr/bin/env python3
"""
05_build_mutation_dataset.py

(a) Ensures data/raw/cf_mutations.csv exists (self-seeds via
    utils.ensure_mutations_seeded if not -- normally already created by
    02_extract_sequence_features.py's own self-seed, since pipeline.sh
    runs 02 before 05; this is just a safety net for out-of-order runs).
    SEED_MUTATIONS/KNOWN_LABELS live in utils.py so 02 and 05 share one
    definition rather than risking drift between two copies. This is a
    REFERENCE / DEMO labeled set for pipeline development, not a clinical
    decision tool -- for the current, complete, and authoritative list of
    eligible genotypes per drug, consult https://cftr2.org and each drug's
    current label.

(b) Joins sequence_features.csv + structural_features.csv with
    ligand_features.csv (cross product: every mutation x every modulator
    product) to build the model-ready dataset, with the seed label attached
    where known (else NaN, i.e. "unlabeled / to predict").

Output: data/processed/mutation_modulator_dataset.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, check_version, PROJECT_ROOT, ensure_mutations_seeded, KNOWN_LABELS

log = get_logger("build_dataset")


def main():
    check_version()
    cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    mut_path = raw_dir / "cf_mutations.csv"
    was_missing = not mut_path.exists()
    mut_path = ensure_mutations_seeded(raw_dir, log)
    if was_missing:
        log.info(
            "cf_mutations.csv was just seeded -- re-run steps 02 and 03 "
            "(sequence/structural feature extraction) before this step "
            "again so features exist for the newly seeded mutations."
        )
        return

    seq_path = proc_dir / "sequence_features.csv"
    struct_path = proc_dir / "structural_features.csv"
    lig_path = proc_dir / "ligand_features.csv"
    for p in (seq_path, lig_path):
        if not p.exists():
            log.error(f"Missing required input: {p}. Run the earlier pipeline steps first.")
            sys.exit(1)

    seq_df = pd.read_csv(seq_path)
    if struct_path.exists():
        struct_df = pd.read_csv(struct_path)
        # avoid duplicate columns from sequence_features being re-written into structural_features
        overlap = [c for c in struct_df.columns if c in seq_df.columns and c != "mutation"]
        merged = seq_df.merge(struct_df.drop(columns=overlap), on="mutation", how="left")
    else:
        log.warning(f"{struct_path} not found; proceeding sequence-only.")
        merged = seq_df

    lig_df = pd.read_csv(lig_path)

    # Map each modulator compound to its clinically relevant PRODUCT name
    # (products.yaml components), since clinical labels are per-product,
    # not per-molecule.
    product_map = {p["name"]: p["components"] for p in cfg["products"]}
    mutation_class_map = dict(zip(
        pd.read_csv(mut_path)["mutation"], pd.read_csv(mut_path)["functional_class"]
    ))

    records = []
    for _, mrow in merged.iterrows():
        for product_name, components in product_map.items():
            rec = mrow.to_dict()
            rec["product"] = product_name
            rec["product_components"] = "+".join(components)
            # aggregate ligand descriptors across the product's components (mean)
            comp_rows = lig_df[lig_df["name"].isin(components)]
            for col in ["mol_weight", "logp", "tpsa", "h_bond_donors",
                        "h_bond_acceptors", "rotatable_bonds", "aromatic_rings",
                        "fraction_csp3", "lipinski_violations"]:
                if col in comp_rows.columns and len(comp_rows):
                    rec[f"product_mean_{col}"] = comp_rows[col].mean()
                else:
                    rec[f"product_mean_{col}"] = None
            rec["n_correctors_in_product"] = sum(
                1 for c in components
                for m in cfg["modulators"] if m["name"] == c and m["class"] == "corrector"
            )
            rec["n_potentiators_in_product"] = sum(
                1 for c in components
                for m in cfg["modulators"] if m["name"] == c and m["class"] == "potentiator"
            )
            rec["functional_class"] = mutation_class_map.get(rec["mutation"])
            rec["known_response_label"] = KNOWN_LABELS.get((rec["mutation"], product_name))
            records.append(rec)

    out_df = pd.DataFrame(records)
    out_path = proc_dir / "mutation_modulator_dataset.csv"
    out_df.to_csv(out_path, index=False)
    n_labeled = out_df["known_response_label"].notna().sum()
    log.info(
        f"Wrote {len(out_df)} mutation x product rows -> {out_path} "
        f"({n_labeled} rows have a known seed label, rest are NaN/unlabeled)"
    )


if __name__ == "__main__":
    main()

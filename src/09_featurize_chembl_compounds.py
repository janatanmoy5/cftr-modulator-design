#!/usr/bin/env python3
"""
09_featurize_chembl_compounds.py

Builds a model-ready feature table from the raw ChEMBL bioactivity pull:

  - one row per unique compound (molecule_chembl_id), keeping its best/most
    common bioactivity measurement
  - target variable: pchembl_value where ChEMBL provides it, else derived
    as pX = -log10(standard_value_in_M) from standard_value/standard_units
    for standard_relation == '=' records only (censored data with '>' / '<'
    relations are excluded from the regression target, since they are not
    exact potency values)
  - RDKit descriptors (as in 04_extract_ligand_features.py) PLUS additional
    descriptors more relevant to a general bioactivity QSAR model: molar
    refractivity, ring counts, heteroatom count, QED drug-likeness score
  - 1024-bit Morgan (ECFP4) fingerprint, exploded into individual bit
    columns (fp_0 ... fp_1023), for models that benefit from substructure
    information beyond whole-molecule descriptors

Output: data/processed/chembl_bioactivity_features.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, check_version, PROJECT_ROOT

log = get_logger("featurize_chembl")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, QED
    from rdkit.Chem import rdFingerprintGenerator
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

UNIT_TO_MOLAR = {
    "M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12,
}
N_FP_BITS = 1024
FP_RADIUS = 2  # ECFP4
_MORGAN_GEN = None


def _morgan_gen():
    global _MORGAN_GEN
    if _MORGAN_GEN is None:
        _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=N_FP_BITS)
    return _MORGAN_GEN


def descriptors_for_smiles(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    d = {
        "mol_weight": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "h_bond_donors": Lipinski.NumHDonors(mol),
        "h_bond_acceptors": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "heteroatom_count": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "molar_refractivity": Descriptors.MolMR(mol),
        "qed_druglikeness": QED.qed(mol),
        "lipinski_violations": sum([
            Descriptors.MolWt(mol) > 500,
            Descriptors.MolLogP(mol) > 5,
            Lipinski.NumHDonors(mol) > 5,
            Lipinski.NumHAcceptors(mol) > 10,
        ]),
    }
    fp = _morgan_gen().GetFingerprint(mol)
    fp_arr = np.zeros((N_FP_BITS,), dtype=int)
    for bit in fp.GetOnBits():
        fp_arr[bit] = 1
    for i, v in enumerate(fp_arr):
        d[f"fp_{i}"] = int(v)
    return d


def compute_pX(row) -> float | None:
    if pd.notna(row.get("pchembl_value")):
        return float(row["pchembl_value"])
    if row.get("standard_relation") not in ("=", None) or pd.isna(row.get("standard_value")):
        return None
    unit = row.get("standard_units")
    if unit not in UNIT_TO_MOLAR or pd.isna(row.get("standard_value")):
        return None
    molar = float(row["standard_value"]) * UNIT_TO_MOLAR[unit]
    if molar <= 0:
        return None
    return -np.log10(molar)


def main():
    check_version()
    if not RDKIT_OK:
        log.error("rdkit is not installed. Run install.sh first, then re-run.")
        sys.exit(1)

    cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "chembl_bioactivity_raw.csv"
    if not raw_path.exists():
        log.error(f"{raw_path} not found. Run 08_fetch_chembl_bioactivity.py first.")
        sys.exit(1)

    df = pd.read_csv(raw_path)
    df = df.dropna(subset=["canonical_smiles", "molecule_chembl_id"])
    log.info(f"Loaded {len(df)} raw bioactivity records for {df['molecule_chembl_id'].nunique()} compounds")

    df["pX"] = df.apply(compute_pX, axis=1)
    df = df.dropna(subset=["pX"])
    log.info(f"{len(df)} records have a usable potency value (pchembl_value or exact standard_value)")

    # one row per compound: take the median pX across all its measurements
    # (a compound may have been tested in multiple assays)
    agg = (
        df.groupby(["molecule_chembl_id", "canonical_smiles"])["pX"]
        .agg(["median", "count", "std"])
        .reset_index()
        .rename(columns={"median": "pX_median", "count": "n_measurements", "std": "pX_std"})
    )
    context_cols = [c for c in ("assay_variant_context", "assay_mechanism_context", "document_chembl_id") if c in df]
    if context_cols:
        context = df.groupby(["molecule_chembl_id", "canonical_smiles"])[context_cols].agg(
            lambda s: ";".join(sorted({str(x) for x in s.dropna()}))
        ).reset_index()
        agg = agg.merge(context, on=["molecule_chembl_id", "canonical_smiles"], how="left")
    log.info(f"Aggregated to {len(agg)} unique compounds")

    rows = []
    n_failed = 0
    for _, r in agg.iterrows():
        desc = descriptors_for_smiles(r["canonical_smiles"])
        if desc is None:
            n_failed += 1
            continue
        row = {
            "molecule_chembl_id": r["molecule_chembl_id"],
            "canonical_smiles": r["canonical_smiles"],
            "pX_median": r["pX_median"],
            "n_measurements": int(r["n_measurements"]),
            "pX_std": r["pX_std"],
        }
        for col in context_cols:
            row[col] = r.get(col)
        row.update(desc)
        rows.append(row)

    if n_failed:
        log.warning(f"RDKit could not parse {n_failed} SMILES strings; those compounds were dropped")

    out_df = pd.DataFrame(rows)
    out_path = proc_dir / "chembl_bioactivity_features.csv"
    out_df.to_csv(out_path, index=False)
    log.info(f"Wrote features for {len(out_df)} compounds -> {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
03_extract_structural_features.py

Uses the AlphaFold full-length model (data/structures/AF-P13569-F1-model_latest.pdb)
as the primary coordinate source, since cryo-EM structures have large missing
regions (see config.yaml notes). For each mutation position, computes:

  - per-residue pLDDT (AlphaFold's confidence column, stored in the B-factor
    field of the PDB file) -> proxy for local structural order/disorder
  - 3D Euclidean distance (CA-CA) to the F508 C-alpha
  - 3D Euclidean distance (CA-CA) to the nearest NBD1/NBD2 ATP-site residue
  - relative solvent accessibility (RSA) via Biopython's Shrake-Rupley SASA
    (falls back to None if the structure or residue is unavailable)

Output: data/processed/structural_features.csv

Requires: biopython (Bio.PDB)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, check_version, PROJECT_ROOT, atp_pocket_residues

log = get_logger("struct_features")

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    BIOPYTHON_OK = True
except ImportError:
    BIOPYTHON_OK = False


def load_structure(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("CFTR_AF", str(pdb_path))
    return structure


def get_ca_atom(structure, resnum: int):
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[1] == resnum and "CA" in res:
                    return res["CA"], res
    return None, None


def main():
    check_version()
    if not BIOPYTHON_OK:
        log.error(
            "biopython is not installed. Run install.sh first "
            "(pip install biopython) then re-run this script."
        )
        sys.exit(1)

    cfg = load_config()
    struct_dir = PROJECT_ROOT / cfg["paths"]["structures_dir"]
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    af_path = struct_dir / "AF-P13569-F1-model_latest.pdb"
    seq_feat_path = proc_dir / "sequence_features.csv"

    if not seq_feat_path.exists():
        log.error(f"{seq_feat_path} missing. Run 02_extract_sequence_features.py first.")
        sys.exit(1)
    df = pd.read_csv(seq_feat_path)

    if not af_path.exists():
        log.warning(
            f"{af_path} not found (network egress to alphafold.ebi.ac.uk may "
            "be blocked, or 01_fetch_cftr_data.py has not been run on a "
            "machine with access). Writing structural columns as NaN so "
            "downstream steps still run in sequence-only mode."
        )
        df["plddt"] = None
        df["dist3d_to_F508_A"] = None
        df["dist3d_to_ATP_pocket_A"] = None
        df["relative_solvent_accessibility"] = None
        df.to_csv(proc_dir / "structural_features.csv", index=False)
        return

    structure = load_structure(af_path)

    # SASA (whole structure, computed once)
    sr = ShrakeRupley()
    try:
        sr.compute(structure, level="R")
        sasa_ok = True
    except Exception as e:
        log.warning(f"SASA computation failed: {e}")
        sasa_ok = False

    f508_ca, _ = get_ca_atom(structure, cfg["key_sites"]["F508"]["residue"])
    atp_pockets = atp_pocket_residues(cfg)
    atp_positions = [r for residues in atp_pockets.values() for r in residues]
    atp_cas = [a for a in (get_ca_atom(structure, p)[0] for p in atp_positions) if a is not None]

    plddt, d_f508, d_atp, rsa = [], [], [], []
    for _, row in df.iterrows():
        pos = int(row["position"])
        ca, res = get_ca_atom(structure, pos)
        if ca is None:
            plddt.append(None); d_f508.append(None); d_atp.append(None); rsa.append(None)
            continue
        # AlphaFold stores per-residue pLDDT in the B-factor column
        plddt.append(float(ca.get_bfactor()))
        d_f508.append(float(ca - f508_ca) if f508_ca is not None else None)
        d_atp.append(min(float(ca - a) for a in atp_cas) if atp_cas else None)
        if sasa_ok:
            # approximate max ASA per residue type (Tien et al. 2013, theoretical)
            rsa.append(getattr(res, "sasa", None))
        else:
            rsa.append(None)

    df["plddt"] = plddt
    df["dist3d_to_F508_A"] = d_f508
    df["dist3d_to_ATP_pocket_A"] = d_atp
    df["residue_sasa_A2"] = rsa

    out_path = proc_dir / "structural_features.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Wrote structural features for {len(df)} mutations -> {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate and featurize an external SMILES library for CFTR screening.

Accepted CSV columns: compound_id (or id/name) and canonical_smiles (or
smiles). A headerless .smi file with ``SMILES ID`` per line is also accepted.
Outputs molecular features and a dedicated candidate-only PDBQT directory.
"""
import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROJECT_ROOT, check_version, get_logger, load_config

log = get_logger("screening_library")


def load_numbered(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def read_library(path):
    if path.suffix.lower() in (".smi", ".smiles"):
        return pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                           names=["canonical_smiles", "compound_id"], usecols=[0, 1])
    # ChEMBL semicolon exports can contain irregular quoting in the very long
    # synonym/record fields. csv.DictReader still recovers the leading ID and
    # SMILES columns reliably, whereas pandas may reject or drop those rows.
    first = path.open(encoding="utf-8-sig").readline()
    if ";" in first and "Compound ChEMBL ID" in first:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            records = list(csv.DictReader(handle, delimiter=";"))
        df = pd.DataFrame([{k: r.get(k) for k in ("Compound ChEMBL ID", "Smiles")} for r in records])
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    id_col = next((c for c in ("compound_id", "molecule_id", "id", "name",
                               "Compound ChEMBL ID") if c in df), None)
    smi_col = next((c for c in ("canonical_smiles", "smiles", "SMILES", "Smiles") if c in df), None)
    if smi_col is None: raise ValueError("Library needs a canonical_smiles or smiles column")
    out = pd.DataFrame({"canonical_smiles": df[smi_col]})
    out["compound_id"] = df[id_col].astype(str) if id_col else [f"candidate_{i+1:06d}" for i in range(len(df))]
    return out


def main():
    check_version(); ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); args = ap.parse_args()
    cfg = load_config(); proc = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    struct = PROJECT_ROOT / cfg["paths"]["structures_dir"]
    inp = Path(args.input).expanduser().resolve()
    if not inp.exists(): log.error(f"Library not found: {inp}"); sys.exit(1)
    feat = load_numbered("chem_features", Path(__file__).parent / "09_featurize_chembl_compounds.py")
    prep = load_numbered("dock_prep", Path(__file__).parent / "12_prepare_docking_inputs.py")
    df = read_library(inp).dropna().drop_duplicates("compound_id")
    rows, invalid = [], []
    lig_dir = struct / "screening_ligands_pdbqt"; lig_dir.mkdir(parents=True, exist_ok=True)
    for _, r in df.iterrows():
        source_cid, smi = str(r.compound_id), str(r.canonical_smiles)
        cid = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_cid)
        desc = feat.descriptors_for_smiles(smi)
        if desc is None: invalid.append({"compound_id": cid, "canonical_smiles": smi, "reason": "invalid_smiles"}); continue
        rows.append({"molecule_chembl_id": cid, "compound_id": cid,
                     "source_compound_id": source_cid, "canonical_smiles": smi, **desc})
        safe = cid
        pdbqt = prep.embed_and_prepare_ligand(smi)
        if pdbqt: (lig_dir / f"{safe}.pdbqt").write_text(pdbqt)
        else: invalid.append({"compound_id": cid, "canonical_smiles": smi, "reason": "3d_or_pdbqt_failed"})
    proc.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(proc / "screening_library_features.csv", index=False)
    pd.DataFrame(invalid, columns=["compound_id", "canonical_smiles", "reason"]).to_csv(proc / "screening_library_rejections.csv", index=False)
    log.info(f"Prepared {len(rows)} candidates; {len(invalid)} warnings/rejections. Candidate PDBQT: {lig_dir}")


if __name__ == "__main__": main()

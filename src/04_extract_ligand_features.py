#!/usr/bin/env python3
"""
04_extract_ligand_features.py

Fetches canonical SMILES for each CFTR modulator compound from PubChem
PUG-REST (by name, cached locally so repeat runs don't hit the network),
then computes RDKit molecular descriptors relevant to CFTR corrector/
potentiator activity: molecular weight, logP, TPSA, H-bond donors/acceptors,
rotatable bonds, aromatic ring count, and Lipinski Rule-of-5 flags.

Output: data/processed/ligand_features.csv
Cache:  data/raw/ligands_cache.json

Requires: rdkit, requests
"""
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, check_version, load_json, save_json, PROJECT_ROOT

log = get_logger("ligand_features")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

PUBCHEM_SMILES_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/"
    "CanonicalSMILES,IsomericSMILES,MolecularFormula/JSON"
)


def fetch_smiles(name: str, cache: dict) -> dict | None:
    if name in cache:
        return cache[name]
    url = PUBCHEM_SMILES_URL.format(name=name)
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        props = resp.json()["PropertyTable"]["Properties"][0]
        result = {
            "smiles": props.get("IsomericSMILES") or props.get("CanonicalSMILES"),
            "molecular_formula": props.get("MolecularFormula"),
        }
        cache[name] = result
        return result
    except Exception as e:
        log.warning(f"PubChem lookup failed for '{name}': {e}")
        return None


def compute_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "mol_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "h_bond_donors": Lipinski.NumHDonors(mol),
        "h_bond_acceptors": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "fraction_csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "lipinski_violations": sum([
            Descriptors.MolWt(mol) > 500,
            Descriptors.MolLogP(mol) > 5,
            Lipinski.NumHDonors(mol) > 5,
            Lipinski.NumHAcceptors(mol) > 10,
        ]),
    }


def main():
    check_version()
    if not RDKIT_OK:
        log.error("rdkit is not installed. Run install.sh first (pip install rdkit) then re-run.")
        sys.exit(1)

    cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / "ligands_cache.json"
    cache = load_json(cache_path) if cache_path.exists() else {}

    rows = []
    for mod in cfg["modulators"]:
        pubchem_name = mod.get("pubchem_name", mod["name"])
        info = fetch_smiles(pubchem_name, cache)
        row = {
            "name": mod["name"],
            "brand": mod["brand"],
            "modulator_class": mod["class"],
        }
        if info and info.get("smiles"):
            row["smiles"] = info["smiles"]
            row["molecular_formula"] = info.get("molecular_formula")
            row.update(compute_descriptors(info["smiles"]))
        else:
            log.warning(
                f"No structure available for {mod['name']} "
                "(network egress to pubchem.ncbi.nlm.nih.gov may be blocked "
                "in this environment). Descriptor columns left blank -- "
                "populate data/raw/ligands_cache.json manually with "
                '{"<name>": {"smiles": "...", "molecular_formula": "..."}} '
                "entries if running offline."
            )
        rows.append(row)

    save_json(cache, cache_path)
    out_df = pd.DataFrame(rows)
    out_path = proc_dir / "ligand_features.csv"
    out_df.to_csv(out_path, index=False)
    log.info(f"Wrote ligand descriptors for {len(out_df)} compounds -> {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
12_prepare_docking_inputs.py

Prepares AutoDock Vina inputs:

  (a) RECEPTOR: converts the reference CFTR structure (8EIQ preferred --
      the real Trikafta-bound cryo-EM structure -- else the AlphaFold
      model) into a receptor PDBQT via meeko.Polymer.

      CAVEAT (stated plainly, not hidden): CFTR cryo-EM structures have
      large unresolved regions (residues 1-14, 645-843 [R-domain], 1173-
      1206, 1437-1480 -- see config.yaml notes), and it is a 1,480-residue
      polytopic membrane protein. Automated receptor prep on a structure
      like this commonly needs manual intervention (gap patching, chain
      selection, protonation state review, removing detergent/lipid
      HETATM records that aren't the docking target) before the output is
      trustworthy for real docking. This script does the automatable part
      and prints an explicit warning listing what to check by hand.

  (b) LIGANDS: for every compound in ligand_features.csv (04) and/or
      chembl_bioactivity_features.csv (09), embeds a 3D conformer (RDKit
      ETKDG + MMFF), then converts to PDBQT via meeko.MoleculePreparation.

Output:
  data/structures/receptor.pdbqt
  data/structures/ligands_pdbqt/<molecule_id>.pdbqt
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, check_version, PROJECT_ROOT

log = get_logger("prepare_docking")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import (
        MoleculePreparation, PDBQTWriterLegacy, Polymer, ResidueChemTemplates,
    )
    import gemmi
    DEPS_OK = True
except ImportError:
    DEPS_OK = False


def cif_to_pdb_string(cif_path: Path) -> str:
    """Convert mmCIF to a PDB-format string via gemmi (Polymer.from_pdb_string
    is the stable receptor-prep entry point in this meeko version; gemmi
    handles the CIF -> PDB text conversion cleanly, including for large
    multi-chain cryo-EM depositions)."""
    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()
    return structure.make_pdb_string()


def prepare_receptor(structure_path: Path, out_path: Path) -> bool:
    try:
        if structure_path.suffix.lower() == ".cif":
            pdb_string = cif_to_pdb_string(structure_path)
        else:
            pdb_string = structure_path.read_text()

        templates = ResidueChemTemplates.create_from_defaults()
        mk_prep = MoleculePreparation()
        # allow_bad_res=True: cryo-EM CFTR structures have large unresolved
        # gaps (see module docstring) -- residues meeko can't template
        # cleanly are skipped rather than aborting the whole receptor.
        polymer = Polymer.from_pdb_string(pdb_string, templates, mk_prep, allow_bad_res=True)
        pdbqt_string, err = PDBQTWriterLegacy.write_string_from_polymer(polymer)
        if not pdbqt_string:
            log.error(f"Receptor PDBQT export produced no output: {err}")
            return False
        out_path.write_text(pdbqt_string)
        log.info(f"Wrote receptor PDBQT -> {out_path} ({len(pdbqt_string)} bytes)")
        if err:
            log.warning(f"Receptor prep completed with warnings: {err}")
        return True
    except Exception as e:
        log.error(
            f"Receptor preparation failed for {structure_path.name}: {e}. "
            "This is common for large multi-domain membrane proteins with "
            "missing loops (see module docstring). Consider: (1) repairing "
            "gaps with PDBFixer or Modeller first, (2) manually selecting "
            "just the TMD region around the binding site rather than the "
            "full 1480-residue chain, or (3) using a pre-prepared receptor "
            "PDBQT from a docking study's supplementary data if available."
        )
        return False


def embed_and_prepare_ligand(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(fragments) > 1:
        # ChEMBL commonly stores salts/counterions (e.g. .[Cl-], .[Na+])
        # as disconnected fragments. Meeko requires exactly one fragment.
        # Retain the largest heavy-atom component as the parent structure;
        # this preserves charged CFTR-active chromophores while removing the
        # counterion. The original, unmodified SMILES remains in the CSV for
        # provenance and descriptor/QSAR use.
        mol = max(fragments, key=lambda frag: frag.GetNumHeavyAtoms())
    mol = Chem.AddHs(mol)
    embed_ok = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    if embed_ok != 0:
        # retry with more attempts for tricky/flexible molecules
        embed_ok = AllChem.EmbedMolecule(mol, randomSeed=1, useRandomCoords=True, maxAttempts=200)
        if embed_ok != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass  # fall back to unoptimized embedded geometry rather than failing outright

    preparator = MoleculePreparation()
    try:
        setups = preparator.prepare(mol)
    except Exception as exc:
        log.warning(f"Meeko ligand preparation failed after fragment standardization: {exc}")
        return None
    if not setups:
        return None
    pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not is_ok:
        log.warning(f"PDBQT writer reported an issue: {err}")
        return None
    return pdbqt_string


def main():
    check_version()
    if not DEPS_OK:
        log.error(
            "rdkit and/or meeko not installed. Run: "
            "pip install rdkit meeko  (then re-run this script)."
        )
        sys.exit(1)

    cfg = load_config()
    struct_dir = PROJECT_ROOT / cfg["paths"]["structures_dir"]
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    lig_out_dir = struct_dir / "ligands_pdbqt"
    lig_out_dir.mkdir(parents=True, exist_ok=True)

    # --- receptors: one per unique reference structure actually used by a
    # binding site, since the corrector pocket (7SVR/7SV7) is a materially
    # different conformational state from the potentiator/Trikafta
    # structure (8EIQ) -- docking every site against one global receptor
    # would silently mix conformational states. ---
    site_configs = list(cfg["binding_sites"].values()) + list(cfg["atp_binding_pockets"].values())
    needed_pdb_ids = set()
    for site_cfg in site_configs:
        for key in ("reference_pdb", "reference_pdb_alt"):
            if site_cfg.get(key):
                needed_pdb_ids.add(site_cfg[key])
    # always also try the generic docking-flagged structures as a fallback pool
    for entry in cfg["structures"]["full_length_cryo_em"]:
        if entry.get("use_for_docking"):
            needed_pdb_ids.add(entry["id"])

    n_receptors_ok = 0
    for pdb_id in sorted(needed_pdb_ids):
        struct_path = None
        for ext in (".cif", ".pdb"):
            p = struct_dir / f"{pdb_id}{ext}"
            if p.exists():
                struct_path = p
                break
        if struct_path is None:
            log.warning(f"{pdb_id}: structure file not found in {struct_dir}, skipping receptor prep for it.")
            continue
        out_path = struct_dir / f"receptor_{pdb_id}.pdbqt"
        if out_path.exists() and out_path.stat().st_size > 0:
            log.info(f"Reusing existing receptor PDBQT -> {out_path}")
            n_receptors_ok += 1
            continue
        if prepare_receptor(struct_path, out_path):
            n_receptors_ok += 1

    if n_receptors_ok == 0:
        af_path = None
        for ext in (".cif", ".pdb"):
            p = struct_dir / f"AF-P13569-F1-model_latest{ext}"
            if p.exists():
                af_path = p
                break
        if af_path is not None:
            log.info("No PDB-ID-specific structures available; preparing a fallback receptor from the AlphaFold model.")
            prepare_receptor(af_path, struct_dir / "receptor_AF-P13569-F1-model_latest.pdbqt")
        else:
            log.warning(
                "No reference structures found in data/structures/ at all -- "
                "run 01_fetch_cftr_data.py on a machine with network access "
                "first. Skipping receptor preparation; ligand preparation "
                "will still run below."
            )

    # --- ligands: pull from both the 5-drug ligand table and any ChEMBL pull ---
    smiles_by_id = {}
    lig_path = proc_dir / "ligand_features.csv"
    if lig_path.exists():
        df = pd.read_csv(lig_path)
        for _, r in df.iterrows():
            if pd.notna(r.get("smiles")):
                smiles_by_id[r["name"]] = r["smiles"]

    chembl_path = proc_dir / "chembl_bioactivity_features.csv"
    if chembl_path.exists():
        df = pd.read_csv(chembl_path)
        for _, r in df.iterrows():
            smiles_by_id[r["molecule_chembl_id"]] = r["canonical_smiles"]

    if not smiles_by_id:
        log.warning(
            "No ligand SMILES found (run 04_extract_ligand_features.py and/or "
            "08+09 ChEMBL steps first). Nothing to prepare."
        )
        return

    n_ok, n_fail = 0, 0
    for mol_id, smiles in smiles_by_id.items():
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(mol_id))
        out_path = lig_out_dir / f"{safe_id}.pdbqt"
        fail_path = lig_out_dir / f"{safe_id}.failed"
        if out_path.exists():
            n_ok += 1
            continue
        if fail_path.exists():
            log.warning(f"Skipping previously failed ligand {mol_id} ({fail_path.name})")
            n_fail += 1
            continue
        pdbqt = embed_and_prepare_ligand(smiles)
        if pdbqt is None:
            log.warning(f"Failed to prepare ligand {mol_id} (SMILES: {smiles})")
            fail_path.write_text(f"PDBQT preparation failed\nSMILES={smiles}\n")
            n_fail += 1
            continue
        out_path.write_text(pdbqt)
        n_ok += 1

    log.info(f"Ligand preparation: {n_ok} succeeded, {n_fail} failed -> {lig_out_dir}")


if __name__ == "__main__":
    main()

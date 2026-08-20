#!/usr/bin/env python3
"""
11_extract_binding_pockets.py

Loads the CIF structure(s) and, for each experimentally-defined binding
site in config.yaml (binding_sites: potentiator, elexacaftor, corrector),
extracts real pocket geometry rather than the single-residue distance
proxy used in 03_extract_structural_features.py:

  - pocket centroid (mean CA coordinate of the site's defining residues)
  - pocket "binding area": solvent-accessible surface area (Bio.PDB
    Shrake-Rupley) summed over the pocket-lining residues, i.e. the
    exposed surface available for a ligand to contact -- this is the
    literal binding-area feature requested, computed from real structure
    rather than approximated from sequence alone
  - pocket volume proxy: convex-hull volume (scipy) of the heavy-atom
    coordinates of the pocket residues, in cubic Angstrom
  - pocket hydrophobicity: mean Kyte-Doolittle hydropathy of the
    pocket-lining residues
  - for every mutation in sequence_features.csv: 3D distance from the
    mutated residue's CA to EACH pocket centroid (potentiator /
    elexacaftor / corrector), so downstream models can ask "does this
    mutation sit inside/near the potentiator pocket vs. the corrector
    region" directly, instead of only using the generic ATP-site proxy.

This directly informs (and can replace) a pharmacophore-distance docking
proxy: the docking box built in 12_prepare_docking_inputs.py is centered
on these same pocket centroids.

Output:
  data/processed/binding_pocket_geometry.csv   (one row per binding site)
  data/processed/mutation_pocket_distances.csv (mutation x site distances)

Requires: biopython, scipy, numpy
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, check_version, PROJECT_ROOT

log = get_logger("binding_pockets")

try:
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    from scipy.spatial import ConvexHull
    DEPS_OK = True
except ImportError:
    DEPS_OK = False

KD_HYDROPATHY = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5, "GLN": -3.5,
    "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5, "LEU": 3.8, "LYS": -3.9,
    "MET": 1.9, "PHE": 2.8, "PRO": -1.6, "SER": -0.8, "THR": -0.7, "TRP": -0.9,
    "TYR": -1.3, "VAL": 4.2,
}


def load_structure(path: Path):
    if path.suffix.lower() == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure("CFTR", str(path))


def find_structure_file(struct_dir: Path, pdb_id: str) -> Path | None:
    for ext in (".cif", ".pdb"):
        p = struct_dir / f"{pdb_id}{ext}"
        if p.exists():
            return p
    return None


def find_site_reference_structure(struct_dir: Path, site_cfg: dict, cfg: dict) -> Path | None:
    """Each binding site has its own correct reference structure -- the
    corrector pocket (7SVR/7SV7) is a materially different conformational
    state from the potentiator/Trikafta structure (8EIQ), so using one
    global structure for every site would be wrong. Falls back to any
    use_for_docking structure, then the AlphaFold model, if the site's
    preferred PDB wasn't fetched."""
    for key in ("reference_pdb", "reference_pdb_alt"):
        pdb_id = site_cfg.get(key)
        if pdb_id:
            p = find_structure_file(struct_dir, pdb_id)
            if p is not None:
                return p
    for entry in cfg["structures"]["full_length_cryo_em"]:
        if entry.get("use_for_docking"):
            p = find_structure_file(struct_dir, entry["id"])
            if p is not None:
                return p
    for name in ("AF-P13569-F1-model_latest",):
        p = find_structure_file(struct_dir, name)
        if p is not None:
            return p
    return None


def get_residue(structure, resnum: int):
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[1] == resnum and res.id[0] == " ":  # skip HETATM
                    return res
    return None


def pocket_geometry(structure, site_residues: list[dict]) -> dict:
    residues = []
    for r in site_residues:
        res = get_residue(structure, r["resnum"])
        if res is not None:
            residues.append(res)
    if not residues:
        return {"n_resolved_residues": 0}

    ca_coords = np.array([res["CA"].coord for res in residues if "CA" in res])
    heavy_coords = np.array([
        atom.coord for res in residues for atom in res
        if atom.element != "H"
    ])

    result = {"n_resolved_residues": len(residues), "n_requested_residues": len(site_residues)}
    if len(ca_coords):
        result["centroid_x"], result["centroid_y"], result["centroid_z"] = ca_coords.mean(axis=0)
    hydro_vals = [KD_HYDROPATHY.get(res.get_resname(), None) for res in residues]
    hydro_vals = [v for v in hydro_vals if v is not None]
    if hydro_vals:
        result["mean_hydrophobicity_kd"] = float(np.mean(hydro_vals))

    if len(heavy_coords) >= 4:
        try:
            hull = ConvexHull(heavy_coords)
            result["pocket_volume_A3"] = float(hull.volume)
        except Exception as e:
            log.warning(f"ConvexHull volume failed: {e}")
            result["pocket_volume_A3"] = None
    else:
        result["pocket_volume_A3"] = None
    return result


def pocket_sasa(structure, site_residues: list[dict]) -> float | None:
    """Sum per-residue SASA over the pocket-lining residues -- this is the
    'binding area' feature: total solvent-exposed surface these residues
    present for a ligand to contact."""
    sr = ShrakeRupley()
    try:
        sr.compute(structure, level="R")
    except Exception as e:
        log.warning(f"SASA computation failed for structure: {e}")
        return None
    total = 0.0
    found_any = False
    for r in site_residues:
        res = get_residue(structure, r["resnum"])
        if res is not None and hasattr(res, "sasa"):
            total += res.sasa
            found_any = True
    return total if found_any else None


def flatten_atp_pocket_residues(pocket_cfg: dict) -> list[dict]:
    """The atp_binding_pockets config entries mix single-residue dicts and
    lists of dicts/ints across several named fields (Walker A/B, signature
    motif, etc.) rather than one flat 'residues' list like binding_sites --
    flatten them into the same {resnum: ...} shape pocket_geometry/
    pocket_sasa expect."""
    residues = []
    for key, val in pocket_cfg.items():
        if key in ("description", "hydrolytic", "binding_affinity"):
            continue
        if isinstance(val, dict) and "resnum" in val:
            residues.append({"resnum": val["resnum"]})
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and "resnum" in item:
                    residues.append({"resnum": item["resnum"]})
                elif isinstance(item, int):
                    residues.append({"resnum": item})
    return residues


def main():
    check_version()
    if not DEPS_OK:
        log.error("biopython and/or scipy not installed. Run install.sh first, then re-run.")
        sys.exit(1)

    cfg = load_config()
    struct_dir = PROJECT_ROOT / cfg["paths"]["structures_dir"]
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    # combine ligand-binding sites (potentiator/elexacaftor/corrector) and
    # the two ATP-binding pockets (ABP1 degenerate, ABP2 hydrolytic) into
    # one uniform list of (name, residues, ref_structure_hint) so the rest
    # of the script treats them identically.
    all_sites = {}
    for site_name, site_cfg in cfg["binding_sites"].items():
        all_sites[site_name] = {"residues": site_cfg["residues"], "site_cfg": site_cfg}
    for site_name, pocket_cfg in cfg["atp_binding_pockets"].items():
        all_sites[site_name] = {"residues": flatten_atp_pocket_residues(pocket_cfg), "site_cfg": pocket_cfg}

    pocket_rows = []
    pocket_centroids = {}
    structures_used = {}
    for site_name, entry in all_sites.items():
        site_cfg, residues = entry["site_cfg"], entry["residues"]
        ref_path = find_site_reference_structure(struct_dir, site_cfg, cfg)
        if ref_path is None:
            log.warning(
                f"{site_name}: no reference structure available (need "
                f"{site_cfg.get('reference_pdb', 'a docking-flagged structure')} "
                "or the AlphaFold model in data/structures/ -- run "
                "01_fetch_cftr_data.py on a machine with network access "
                "first). Skipping this site."
            )
            pocket_rows.append({"binding_site": site_name, "n_resolved_residues": 0})
            continue

        if ref_path not in structures_used:
            structures_used[ref_path] = load_structure(ref_path)
        structure = structures_used[ref_path]

        geom = pocket_geometry(structure, residues)
        area = pocket_sasa(structure, residues)
        geom["binding_area_A2"] = area
        geom["binding_site"] = site_name
        geom["reference_structure"] = ref_path.name
        geom["reference_ligand"] = site_cfg.get("reference_ligand")
        pocket_rows.append(geom)
        if "centroid_x" in geom:
            pocket_centroids[site_name] = np.array(
                [geom["centroid_x"], geom["centroid_y"], geom["centroid_z"]]
            )
        log.info(
            f"{site_name} [{ref_path.name}]: resolved {geom.get('n_resolved_residues', 0)}/"
            f"{geom.get('n_requested_residues', len(residues))} pocket "
            f"residues, binding_area={area}, volume={geom.get('pocket_volume_A3')}"
        )

    geom_df = pd.DataFrame(pocket_rows)
    geom_path = proc_dir / "binding_pocket_geometry.csv"
    geom_df.to_csv(geom_path, index=False)
    log.info(f"Wrote pocket geometry for {len(geom_df)} binding site(s)/pocket(s) -> {geom_path}")

    if not structures_used:
        pd.DataFrame(columns=["mutation"]).to_csv(proc_dir / "mutation_pocket_distances.csv", index=False)
        return

    # for the mutation-distance table, prefer whichever structure resolved
    # the most sites (usually the AlphaFold model, since it has no missing
    # density) so distances are computed consistently across all sites.
    structure = max(structures_used.values(), key=lambda s: sum(1 for _ in s.get_residues()))

    # --- mutation x pocket 3D distances ---
    seq_feat_path = proc_dir / "sequence_features.csv"
    if not seq_feat_path.exists():
        log.warning(f"{seq_feat_path} not found; skipping mutation-pocket distance table (run 02 first).")
        return
    mut_df = pd.read_csv(seq_feat_path)

    dist_rows = []
    for _, row in mut_df.iterrows():
        res = get_residue(structure, int(row["position"]))
        rec = {"mutation": row["mutation"], "position": row["position"]}
        if res is not None and "CA" in res:
            ca = res["CA"].coord
            for site_name, centroid in pocket_centroids.items():
                rec[f"dist3d_to_{site_name}_A"] = float(np.linalg.norm(ca - centroid))
        else:
            for site_name in pocket_centroids:
                rec[f"dist3d_to_{site_name}_A"] = None
        dist_rows.append(rec)

    dist_df = pd.DataFrame(dist_rows)
    dist_path = proc_dir / "mutation_pocket_distances.csv"
    dist_df.to_csv(dist_path, index=False)
    log.info(f"Wrote mutation-to-pocket distances for {len(dist_df)} mutations -> {dist_path}")


if __name__ == "__main__":
    main()

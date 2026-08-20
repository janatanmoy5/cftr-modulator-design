#!/usr/bin/env python3
"""
13_run_docking.py

Runs real AutoDock Vina docking (not a distance/pharmacophore proxy) of
every prepared ligand (data/structures/ligands_pdbqt/*.pdbqt) against the
receptor(s) prepared per binding site, with the search box centered on
each site's pocket centroid computed in 11_extract_binding_pockets.py.

IMPLEMENTATION NOTE: two docking backends are supported, tried in order:

  1. STATIC BINARY (preferred) -- shells out to bin/vina (downloaded by
     install.sh) via subprocess. No compiling required on any platform.
  2. PYTHON BINDINGS (fallback) -- uses `from vina import Vina` if the
     `vina` pip package happens to be installed and importable (e.g. you
     installed it manually, or via conda + Boost/SWIG build deps). Used
     automatically if bin/vina isn't found, so an existing working
     install isn't wasted.

Either way the docking engine and scoring function are identical --
AutoDock Vina's own docs note the Python bindings and binary are "two
separate processes" wrapping the same underlying engine:
https://autodock-vina.readthedocs.io/en/latest/docking_requirements.html

This produces a real predicted binding affinity (kcal/mol, Vina scoring
function) per (ligand, binding_site) pair -- the computed-docking feature
that replaces the ligand-only descriptor proxy used in the bioactivity
regression branch (09/10).

Output: results/docking_scores.csv
  columns: molecule_id, binding_site, best_affinity_kcal_mol,
           n_poses, all_affinities_kcal_mol, receptor_used
"""
import argparse
import os
import re
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, check_version, PROJECT_ROOT

log = get_logger("run_docking")

try:
    from vina import Vina
    VINA_PYTHON_OK = True
except ImportError:
    VINA_PYTHON_OK = False

# Docking box size around each pocket centroid (Angstrom). The potentiator
# pocket is a tight, well-defined hydrophobic cradle (Liu et al. 2019); a
# 20 A cube is generous enough to accommodate ligand flexibility/pose
# search without extending across the whole membrane domain.
BOX_SIZE_A = (20.0, 20.0, 20.0)
EXHAUSTIVENESS = 8

AFFINITY_RE = re.compile(r"^REMARK VINA RESULT:\s*(-?\d+\.?\d*)", re.MULTILINE)


def find_vina_binary(project_root: Path) -> Path | None:
    """Prefer the project-local static binary (bin/vina, downloaded by
    install.sh) over any 'vina' on PATH, since the PATH one could be a
    stray Python-bindings console script rather than the actual engine."""
    local = project_root / "bin" / "vina"
    if local.exists() and local.stat().st_mode & stat.S_IXUSR:
        return local
    on_path = shutil.which("vina")
    return Path(on_path) if on_path else None


def dock_one_binary(vina_bin: Path, receptor: Path, ligand: Path, center: tuple,
                     box_size: tuple, out_dir: Path, exhaustiveness: int,
                     cpu: int | None = None) -> dict:
    out_path = out_dir / f"{ligand.stem}_docked.pdbqt"
    log_path = out_dir / f"{ligand.stem}_log.txt"
    cmd = [
        str(vina_bin),
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
        "--size_x", str(box_size[0]), "--size_y", str(box_size[1]), "--size_z", str(box_size[2]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", "9",
        "--out", str(out_path),
    ]
    if cpu is not None:
        cmd.extend(["--cpu", str(max(1, cpu))])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    log_path.write_text(result.stdout + "\n" + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"vina exited {result.returncode}: {result.stderr.strip()[-300:]}")

    if not out_path.exists():
        raise RuntimeError(f"vina reported success but no output file at {out_path}")

    out_text = out_path.read_text()
    affinities = [float(m) for m in AFFINITY_RE.findall(out_text)]
    if not affinities:
        raise RuntimeError(f"Could not parse any REMARK VINA RESULT lines from {out_path}")

    return {
        "best_affinity_kcal_mol": min(affinities),
        "n_poses": len(affinities),
        "all_affinities_kcal_mol": ";".join(f"{a:.2f}" for a in affinities),
    }


def dock_one_python(receptor: Path, ligand: Path, center: tuple, box_size: tuple,
                    exhaustiveness: int) -> dict:
    """Fallback path using the installed `vina` pip package's Python API.
    A fresh Vina() engine per call keeps receptor state isolated across
    the different per-site receptors used in this pipeline."""
    vina_engine = Vina(sf_name="vina", verbosity=0)
    vina_engine.set_receptor(str(receptor))
    vina_engine.set_ligand_from_file(str(ligand))
    vina_engine.compute_vina_maps(center=list(center), box_size=list(box_size))
    vina_engine.dock(exhaustiveness=exhaustiveness, n_poses=9)
    energies = vina_engine.energies()  # [affinity, inter, intra, torsions, intra_best_pose] per pose
    affinities = [float(row[0]) for row in energies]
    if not affinities:
        raise RuntimeError("vina Python API returned no poses")
    return {
        "best_affinity_kcal_mol": min(affinities),
        "n_poses": len(affinities),
        "all_affinities_kcal_mol": ";".join(f"{a:.2f}" for a in affinities),
    }


def resolve_receptor_for_site(struct_dir: Path, site_cfg: dict, cfg: dict) -> Path | None:
    """Mirrors 11_extract_binding_pockets.find_site_reference_structure's
    priority order, but returns the prepared receptor PDBQT (named
    receptor_<PDBID>.pdbqt by 12_prepare_docking_inputs.py) rather than the
    raw structure file."""
    for key in ("reference_pdb", "reference_pdb_alt"):
        pdb_id = site_cfg.get(key)
        if pdb_id:
            p = struct_dir / f"receptor_{pdb_id}.pdbqt"
            if p.exists():
                return p
    for entry in cfg["structures"]["full_length_cryo_em"]:
        if entry.get("use_for_docking"):
            p = struct_dir / f"receptor_{entry['id']}.pdbqt"
            if p.exists():
                return p
    p = struct_dir / "receptor_AF-P13569-F1-model_latest.pdbqt"
    return p if p.exists() else None


def main():
    check_version()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ligand-dir", default=None,
                        help="PDBQT directory; defaults to structures/ligands_pdbqt")
    parser.add_argument("--output", default="docking_scores.csv",
                        help="Output filename under results/")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="Parallel Vina processes (default: up to 8)")
    parser.add_argument("--exhaustiveness", type=int, default=EXHAUSTIVENESS)
    parser.add_argument("--max-ligands", type=int, default=None,
                        help="Dock a reproducible high-measurement calibration subset")
    parser.add_argument("--document-id", default=None,
                        help="Restrict ligands to a ChEMBL reference document")
    args = parser.parse_args()
    cfg = load_config()
    vina_bin = find_vina_binary(PROJECT_ROOT)
    if vina_bin is not None:
        backend = "binary"
        log.info(f"Using vina static binary: {vina_bin}")
    elif VINA_PYTHON_OK:
        backend = "python"
        log.info("bin/vina not found; using the installed `vina` Python package instead.")
    else:
        log.error(
            "No docking engine available: bin/vina static binary not "
            "found, and the `vina` Python package is not installed either. "
            "Run install.sh (it downloads the static binary automatically "
            "for your platform), or if you already have a working `vina` "
            "pip/conda install, just make sure it's importable in this "
            "environment. Manual binary download: "
            "https://github.com/ccsb-scripps/AutoDock-Vina/releases "
            "(place at bin/vina, chmod +x it; on macOS also run "
            "`xattr -d com.apple.quarantine bin/vina`)."
        )
        sys.exit(1)

    struct_dir = PROJECT_ROOT / cfg["paths"]["structures_dir"]
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    docking_out_dir = results_dir / "docking_outputs"
    docking_out_dir.mkdir(parents=True, exist_ok=True)

    lig_dir = Path(args.ligand_dir).expanduser().resolve() if args.ligand_dir else struct_dir / "ligands_pdbqt"
    geom_path = proc_dir / "binding_pocket_geometry.csv"

    if not lig_dir.exists() or not any(lig_dir.glob("*.pdbqt")):
        log.error(f"No ligand PDBQT files found in {lig_dir}. Run 12_prepare_docking_inputs.py first.")
        sys.exit(1)
    if not geom_path.exists():
        log.error(f"{geom_path} not found. Run 11_extract_binding_pockets.py first.")
        sys.exit(1)

    geom_df = pd.read_csv(geom_path)
    site_configs = {**cfg["binding_sites"], **cfg["atp_binding_pockets"]}

    sites = {}
    receptors = {}
    for _, row in geom_df.iterrows():
        site_name = row["binding_site"]
        if pd.isna(row.get("centroid_x")):
            continue
        receptor_path = resolve_receptor_for_site(struct_dir, site_configs.get(site_name, {}), cfg)
        if receptor_path is None:
            log.warning(f"{site_name}: no prepared receptor found -- run 12_prepare_docking_inputs.py first. Skipping.")
            continue
        sites[site_name] = (row["centroid_x"], row["centroid_y"], row["centroid_z"])
        receptors[site_name] = receptor_path

    if not sites:
        log.error(
            f"{geom_path} has no resolved pocket centroids with a matching "
            "prepared receptor -- docking needs both a real 3D centroid "
            "(11_extract_binding_pockets.py) and a prepared receptor "
            "(12_prepare_docking_inputs.py) for at least one site."
        )
        sys.exit(1)

    log.info(f"Docking against {len(sites)} binding site(s): {list(sites.keys())}")

    ligand_files = sorted(lig_dir.glob("*.pdbqt"))
    if args.document_id:
        feature_path = proc_dir / "chembl_bioactivity_features.csv"
        feat = pd.read_csv(feature_path, usecols=lambda c: c in ("molecule_chembl_id", "document_chembl_id"))
        if "document_chembl_id" not in feat:
            raise RuntimeError("Feature table has no document_chembl_id column; rerun steps 08-09")
        wanted = set(feat.loc[feat["document_chembl_id"].astype(str).str.contains(
            args.document_id, regex=False), "molecule_chembl_id"].astype(str))
        ligand_files = [p for p in ligand_files if p.stem in wanted]
        log.info(f"Selected {len(ligand_files)} ligands from document {args.document_id}")
    if args.max_ligands and len(ligand_files) > args.max_ligands:
        feature_path = proc_dir / "chembl_bioactivity_features.csv"
        if feature_path.exists():
            feat = pd.read_csv(feature_path, usecols=lambda c: c in ("molecule_chembl_id", "n_measurements"))
            wanted = set(feat.sort_values(["n_measurements", "molecule_chembl_id"], ascending=[False, True])
                         .head(args.max_ligands)["molecule_chembl_id"].astype(str))
            ligand_files = [p for p in ligand_files if p.stem in wanted]
        ligand_files = ligand_files[:args.max_ligands]
        log.info(f"Using calibration subset of {len(ligand_files)} ligands (--max-ligands)")
    log.info(f"Docking {len(ligand_files)} ligands x {len(sites)} sites "
             f"= {len(ligand_files) * len(sites)} docking runs "
             f"(exhaustiveness={args.exhaustiveness}, workers={args.workers})")

    def run_task(lig_path, site_name, center):
        mol_id = lig_path.stem
        receptor_path = receptors[site_name]
        site_out_dir = docking_out_dir / site_name
        site_out_dir.mkdir(parents=True, exist_ok=True)
        try:
            if backend == "binary":
                result = dock_one_binary(vina_bin, receptor_path, lig_path, center,
                                         BOX_SIZE_A, site_out_dir, args.exhaustiveness)
            else:
                result = dock_one_python(receptor_path, lig_path, center,
                                         BOX_SIZE_A, args.exhaustiveness)
            result.update({"molecule_id": mol_id, "binding_site": site_name,
                           "receptor_used": receptor_path.name})
            return result
        except Exception as e:
            return {"molecule_id": mol_id, "binding_site": site_name,
                    "receptor_used": receptor_path.name,
                    "best_affinity_kcal_mol": None, "n_poses": 0,
                    "all_affinities_kcal_mol": None, "error": str(e)}

    tasks = [(lig, site, center) for lig in ligand_files for site, center in sites.items()]
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_task, *task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result(); rows.append(result)
            if result.get("best_affinity_kcal_mol") is not None:
                log.info(f"[{i}/{len(tasks)}] {result['molecule_id']} x {result['binding_site']}: "
                         f"{result['best_affinity_kcal_mol']:.2f} kcal/mol")
            else:
                log.warning(f"[{i}/{len(tasks)}] failed {result['molecule_id']} x "
                            f"{result['binding_site']}: {result.get('error')}")
            if i % 25 == 0:
                pd.DataFrame(rows).to_csv(results_dir / args.output, index=False)

    out_df = pd.DataFrame(rows)
    out_path = results_dir / args.output
    out_df.to_csv(out_path, index=False)
    n_success = out_df["best_affinity_kcal_mol"].notna().sum()
    log.info(f"Wrote {len(out_df)} docking results ({n_success} succeeded) -> {out_path}")
    log.info(
        "NOTE: exhaustiveness is set low (8) for pipeline speed. For "
        "results you'd actually rely on, raise EXHAUSTIVENESS to 16-32 "
        "and re-run -- this increases runtime roughly linearly."
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
13_run_docking.py

Runs real AutoDock Vina docking (not a distance/pharmacophore proxy) of
every prepared ligand (data/structures/ligands_pdbqt/*.pdbqt) against the
receptor(s) prepared per binding site, with the search box centered on
each site's pocket centroid computed in 11_extract_binding_pockets.py.

IMPLEMENTATION NOTE: two docking backends are supported, tried in order:

  1. STATIC BINARY (preferred) -- shells out to bin/vina (downloaded by
     install.sh) via subprocess. No compiling required on any platform.
  2. PYTHON BINDINGS (fallback) -- uses `from vina import Vina` if the
     `vina` pip package happens to be installed and importable (e.g. you
     installed it manually, or via conda + Boost/SWIG build deps). Used
     automatically if bin/vina isn't found, so an existing working
     install isn't wasted.

Either way the docking engine and scoring function are identical --
AutoDock Vina's own docs note the Python bindings and binary are "two
separate processes" wrapping the same underlying engine:
https://autodock-vina.readthedocs.io/en/latest/docking_requirements.html

This produces a real predicted binding affinity (kcal/mol, Vina scoring
function) per (ligand, binding_site) pair -- the computed-docking feature
that replaces the ligand-only descriptor proxy used in the bioactivity
regression branch (09/10).

Output: results/docking_scores.csv
  columns: molecule_id, binding_site, best_affinity_kcal_mol,
           n_poses, all_affinities_kcal_mol, receptor_used
"""
import argparse
import os
import re
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, check_version, PROJECT_ROOT

log = get_logger("run_docking")

try:
    from vina import Vina
    VINA_PYTHON_OK = True
except ImportError:
    VINA_PYTHON_OK = False

# Docking box size around each pocket centroid (Angstrom). The potentiator
# pocket is a tight, well-defined hydrophobic cradle (Liu et al. 2019); a
# 20 A cube is generous enough to accommodate ligand flexibility/pose
# search without extending across the whole membrane domain.
BOX_SIZE_A = (20.0, 20.0, 20.0)
EXHAUSTIVENESS = 8

AFFINITY_RE = re.compile(r"^REMARK VINA RESULT:\s*(-?\d+\.?\d*)", re.MULTILINE)


def find_vina_binary(project_root: Path) -> Path | None:
    """Prefer the project-local static binary (bin/vina, downloaded by
    install.sh) over any 'vina' on PATH, since the PATH one could be a
    stray Python-bindings console script rather than the actual engine."""
    local = project_root / "bin" / "vina"
    if local.exists() and local.stat().st_mode & stat.S_IXUSR:
        return local
    on_path = shutil.which("vina")
    return Path(on_path) if on_path else None


def dock_one_binary(vina_bin: Path, receptor: Path, ligand: Path, center: tuple,
                     box_size: tuple, out_dir: Path, exhaustiveness: int) -> dict:
    out_path = out_dir / f"{ligand.stem}_docked.pdbqt"
    log_path = out_dir / f"{ligand.stem}_log.txt"
    cmd = [
        str(vina_bin),
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
        "--size_x", str(box_size[0]), "--size_y", str(box_size[1]), "--size_z", str(box_size[2]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", "9",
        "--out", str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    log_path.write_text(result.stdout + "\n" + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"vina exited {result.returncode}: {result.stderr.strip()[-300:]}")

    if not out_path.exists():
        raise RuntimeError(f"vina reported success but no output file at {out_path}")

    out_text = out_path.read_text()
    affinities = [float(m) for m in AFFINITY_RE.findall(out_text)]
    if not affinities:
        raise RuntimeError(f"Could not parse any REMARK VINA RESULT lines from {out_path}")

    return {
        "best_affinity_kcal_mol": min(affinities),
        "n_poses": len(affinities),
        "all_affinities_kcal_mol": ";".join(f"{a:.2f}" for a in affinities),
    }


def dock_one_python(receptor: Path, ligand: Path, center: tuple, box_size: tuple,
                    exhaustiveness: int) -> dict:
    """Fallback path using the installed `vina` pip package's Python API.
    A fresh Vina() engine per call keeps receptor state isolated across
    the different per-site receptors used in this pipeline."""
    vina_engine = Vina(sf_name="vina", verbosity=0)
    vina_engine.set_receptor(str(receptor))
    vina_engine.set_ligand_from_file(str(ligand))
    vina_engine.compute_vina_maps(center=list(center), box_size=list(box_size))
    vina_engine.dock(exhaustiveness=exhaustiveness, n_poses=9)
    energies = vina_engine.energies()  # [affinity, inter, intra, torsions, intra_best_pose] per pose
    affinities = [float(row[0]) for row in energies]
    if not affinities:
        raise RuntimeError("vina Python API returned no poses")
    return {
        "best_affinity_kcal_mol": min(affinities),
        "n_poses": len(affinities),
        "all_affinities_kcal_mol": ";".join(f"{a:.2f}" for a in affinities),
    }


def resolve_receptor_for_site(struct_dir: Path, site_cfg: dict, cfg: dict) -> Path | None:
    """Mirrors 11_extract_binding_pockets.find_site_reference_structure's
    priority order, but returns the prepared receptor PDBQT (named
    receptor_<PDBID>.pdbqt by 12_prepare_docking_inputs.py) rather than the
    raw structure file."""
    for key in ("reference_pdb", "reference_pdb_alt"):
        pdb_id = site_cfg.get(key)
        if pdb_id:
            p = struct_dir / f"receptor_{pdb_id}.pdbqt"
            if p.exists():
                return p
    for entry in cfg["structures"]["full_length_cryo_em"]:
        if entry.get("use_for_docking"):
            p = struct_dir / f"receptor_{entry['id']}.pdbqt"
            if p.exists():
                return p
    p = struct_dir / "receptor_AF-P13569-F1-model_latest.pdbqt"
    return p if p.exists() else None


def main():
    check_version()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ligand-dir", default=None,
                        help="PDBQT directory; defaults to structures/ligands_pdbqt")
    parser.add_argument("--output", default="docking_scores.csv",
                        help="Output filename under results/")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="Parallel Vina processes (default: up to 8)")
    parser.add_argument("--exhaustiveness", type=int, default=EXHAUSTIVENESS)
    parser.add_argument("--max-ligands", type=int, default=None,
                        help="Dock a reproducible high-measurement calibration subset")
    parser.add_argument("--document-id", default=None,
                        help="Restrict ligands to a ChEMBL reference document")
    args = parser.parse_args()
    cfg = load_config()
    vina_bin = find_vina_binary(PROJECT_ROOT)
    if vina_bin is not None:
        backend = "binary"
        log.info(f"Using vina static binary: {vina_bin}")
    elif VINA_PYTHON_OK:
        backend = "python"
        log.info("bin/vina not found; using the installed `vina` Python package instead.")
    else:
        log.error(
            "No docking engine available: bin/vina static binary not "
            "found, and the `vina` Python package is not installed either. "
            "Run install.sh (it downloads the static binary automatically "
            "for your platform), or if you already have a working `vina` "
            "pip/conda install, just make sure it's importable in this "
            "environment. Manual binary download: "
            "https://github.com/ccsb-scripps/AutoDock-Vina/releases "
            "(place at bin/vina, chmod +x it; on macOS also run "
            "`xattr -d com.apple.quarantine bin/vina`)."
        )
        sys.exit(1)

    struct_dir = PROJECT_ROOT / cfg["paths"]["structures_dir"]
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    docking_out_dir = results_dir / "docking_outputs"
    docking_out_dir.mkdir(parents=True, exist_ok=True)

    lig_dir = Path(args.ligand_dir).expanduser().resolve() if args.ligand_dir else struct_dir / "ligands_pdbqt"
    geom_path = proc_dir / "binding_pocket_geometry.csv"

    if not lig_dir.exists() or not any(lig_dir.glob("*.pdbqt")):
        log.error(f"No ligand PDBQT files found in {lig_dir}. Run 12_prepare_docking_inputs.py first.")
        sys.exit(1)
    if not geom_path.exists():
        log.error(f"{geom_path} not found. Run 11_extract_binding_pockets.py first.")
        sys.exit(1)

    geom_df = pd.read_csv(geom_path)
    site_configs = {**cfg["binding_sites"], **cfg["atp_binding_pockets"]}

    sites = {}
    receptors = {}
    for _, row in geom_df.iterrows():
        site_name = row["binding_site"]
        if pd.isna(row.get("centroid_x")):
            continue
        receptor_path = resolve_receptor_for_site(struct_dir, site_configs.get(site_name, {}), cfg)
        if receptor_path is None:
            log.warning(f"{site_name}: no prepared receptor found -- run 12_prepare_docking_inputs.py first. Skipping.")
            continue
        sites[site_name] = (row["centroid_x"], row["centroid_y"], row["centroid_z"])
        receptors[site_name] = receptor_path

    if not sites:
        log.error(
            f"{geom_path} has no resolved pocket centroids with a matching "
            "prepared receptor -- docking needs both a real 3D centroid "
            "(11_extract_binding_pockets.py) and a prepared receptor "
            "(12_prepare_docking_inputs.py) for at least one site."
        )
        sys.exit(1)

    log.info(f"Docking against {len(sites)} binding site(s): {list(sites.keys())}")

    ligand_files = sorted(lig_dir.glob("*.pdbqt"))
    if args.document_id:
        feature_path = proc_dir / "chembl_bioactivity_features.csv"
        feat = pd.read_csv(feature_path, usecols=lambda c: c in ("molecule_chembl_id", "document_chembl_id"))
        if "document_chembl_id" not in feat:
            raise RuntimeError("Feature table has no document_chembl_id column; rerun steps 08-09")
        wanted = set(feat.loc[feat["document_chembl_id"].astype(str).str.contains(
            args.document_id, regex=False), "molecule_chembl_id"].astype(str))
        ligand_files = [p for p in ligand_files if p.stem in wanted]
        log.info(f"Selected {len(ligand_files)} ligands from document {args.document_id}")
    if args.max_ligands and len(ligand_files) > args.max_ligands:
        feature_path = proc_dir / "chembl_bioactivity_features.csv"
        if feature_path.exists():
            feat = pd.read_csv(feature_path, usecols=lambda c: c in ("molecule_chembl_id", "n_measurements"))
            wanted = set(feat.sort_values(["n_measurements", "molecule_chembl_id"], ascending=[False, True])
                         .head(args.max_ligands)["molecule_chembl_id"].astype(str))
            ligand_files = [p for p in ligand_files if p.stem in wanted]
        ligand_files = ligand_files[:args.max_ligands]
        log.info(f"Using calibration subset of {len(ligand_files)} ligands (--max-ligands)")
    log.info(f"Docking {len(ligand_files)} ligands x {len(sites)} sites "
             f"= {len(ligand_files) * len(sites)} docking runs "
             f"(exhaustiveness={args.exhaustiveness}, workers={args.workers})")

    def run_task(lig_path, site_name, center):
        mol_id = lig_path.stem
        receptor_path = receptors[site_name]
        site_out_dir = docking_out_dir / site_name
        site_out_dir.mkdir(parents=True, exist_ok=True)
        try:
            if backend == "binary":
                result = dock_one_binary(vina_bin, receptor_path, lig_path, center,
                                         BOX_SIZE_A, site_out_dir, args.exhaustiveness)
            else:
                result = dock_one_python(receptor_path, lig_path, center,
                                         BOX_SIZE_A, args.exhaustiveness)
            result.update({"molecule_id": mol_id, "binding_site": site_name,
                           "receptor_used": receptor_path.name})
            return result
        except Exception as e:
            return {"molecule_id": mol_id, "binding_site": site_name,
                    "receptor_used": receptor_path.name,
                    "best_affinity_kcal_mol": None, "n_poses": 0,
                    "all_affinities_kcal_mol": None, "error": str(e)}

    tasks = [(lig, site, center) for lig in ligand_files for site, center in sites.items()]
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_task, *task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result(); rows.append(result)
            if result.get("best_affinity_kcal_mol") is not None:
                log.info(f"[{i}/{len(tasks)}] {result['molecule_id']} x {result['binding_site']}: "
                         f"{result['best_affinity_kcal_mol']:.2f} kcal/mol")
            else:
                log.warning(f"[{i}/{len(tasks)}] failed {result['molecule_id']} x "
                            f"{result['binding_site']}: {result.get('error')}")
            if i % 25 == 0:
                pd.DataFrame(rows).to_csv(results_dir / args.output, index=False)

    out_df = pd.DataFrame(rows)
    out_path = results_dir / args.output
    out_df.to_csv(out_path, index=False)
    n_success = out_df["best_affinity_kcal_mol"].notna().sum()
    log.info(f"Wrote {len(out_df)} docking results ({n_success} succeeded) -> {out_path}")
    log.info(
        "NOTE: exhaustiveness is set low (8) for pipeline speed. For "
        "results you'd actually rely on, raise EXHAUSTIVENESS to 16-32 "
        "and re-run -- this increases runtime roughly linearly."
    )


if __name__ == "__main__":
    main()

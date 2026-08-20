#!/usr/bin/env python3
"""Single-molecule, five-pocket CFTR docking service used by app.py."""
import importlib.util
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def strict_json_value(value):
    """Return values that JSON.parse can consume without NaN/Infinity extensions."""
    if isinstance(value, dict):
        return {str(key): strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return strict_json_value(value.item())
    return value


def load_numbered(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def dock_smiles(root: Path, smiles: str, job_id: str, progress=lambda message: None):
    src = root / "src"
    prep = load_numbered(f"prep_{job_id}", src / "12_prepare_docking_inputs.py")
    docking = load_numbered(f"dock_{job_id}", src / "13_run_docking.py")
    export = load_numbered(f"export_{job_id}", src / "19_export_top_docked_complexes.py")
    render = load_numbered(f"render_{job_id}", src / "20_render_best_docked_poses.py")
    utils = load_numbered(f"utils_{job_id}", src / "utils.py")
    cfg = utils.load_config(); structures = root / cfg["paths"]["structures_dir"]
    processed = root / cfg["paths"]["processed_dir"]
    job_dir = root / "results" / "web_docking" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    progress("Preparing 3D ligand and PDBQT")
    ligand_text = prep.embed_and_prepare_ligand(smiles)
    if not ligand_text: raise RuntimeError("Ligand 3D embedding or PDBQT preparation failed")
    ligand = job_dir / f"{job_id}.pdbqt"; ligand.write_text(ligand_text)
    vina = docking.find_vina_binary(root)
    if vina is None: raise RuntimeError("AutoDock Vina is unavailable; run ./install.sh")
    geom_path = processed / "binding_pocket_geometry.csv"
    if not geom_path.exists(): raise RuntimeError("Pocket geometry missing; run step 11 first")
    geometry = pd.read_csv(geom_path)
    site_cfg = {**cfg["binding_sites"], **cfg["atp_binding_pockets"]}
    tasks = []
    for row in geometry.itertuples(index=False):
        if pd.isna(row.centroid_x): continue
        receptor = docking.resolve_receptor_for_site(structures, site_cfg.get(row.binding_site, {}), cfg)
        if receptor:
            tasks.append((row.binding_site, receptor,
                          (float(row.centroid_x), float(row.centroid_y), float(row.centroid_z))))
    if not tasks: raise RuntimeError("No prepared CFTR receptors/pockets found; run steps 11–12")

    progress(f"Docking against {len(tasks)} CFTR pockets")
    def one(site, receptor, center):
        out = job_dir / "poses" / site; out.mkdir(parents=True, exist_ok=True)
        result = docking.dock_one_binary(vina, receptor, ligand, center,
                                         docking.BOX_SIZE_A, out, 4)
        return {**result, "binding_site": site, "receptor_used": receptor.name,
                "pose_file": str((out / f"{job_id}_docked.pdbqt").relative_to(root))}

    rows = []
    with ThreadPoolExecutor(max_workers=min(5, len(tasks))) as pool:
        futures = {pool.submit(one, *task): task[0] for task in tasks}
        for future in as_completed(futures): rows.append(future.result())
    rows.sort(key=lambda row: float(row["best_affinity_kcal_mol"]))
    best = rows[0]; receptor = structures / best["receptor_used"]
    all_pose = root / best["pose_file"]; pose = export.first_model(all_pose.read_text())
    best_pose = job_dir / f"{job_id}_{best['binding_site']}_best_pose.pdbqt"
    best_pose.write_text(pose)
    complex_pdb = job_dir / f"{job_id}_{best['binding_site']}_complex.pdb"
    export.make_complex_pdb(receptor, pose, complex_pdb, job_id,
                            best["binding_site"], float(best["best_affinity_kcal_mol"]))

    progress("Rendering best receptor–ligand complex")
    png = job_dir / f"{job_id}_{best['binding_site']}_complex.png"
    pymol = shutil.which("pymol"); rendered = False
    if pymol:
        label = f"{job_id} | Vina: {float(best['best_affinity_kcal_mol']):.3f} kcal/mol"
        rendered, _ = render.render(pymol, complex_pdb, png, 1600, 1200, 5.0,
                                    True, "full", label, "rainbow")
    result = {
        "job_id": job_id, "scores": rows,
        "best_binding_site": best["binding_site"],
        "best_affinity_kcal_mol": float(best["best_affinity_kcal_mol"]),
        "best_pose_pdbqt": str(best_pose.relative_to(root)),
        "complex_pdb": str(complex_pdb.relative_to(root)),
        "complex_png": str(png.relative_to(root)) if rendered else None,
        "result_json": str((job_dir / "result.json").relative_to(root)),
        "warning": "Docking is a computational hypothesis and requires pose review and experimental confirmation.",
    }
    result = strict_json_value(result)
    (job_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result

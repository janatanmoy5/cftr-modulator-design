#!/usr/bin/env python3
"""Re-dock the strongest hits to their winning CFTR pocket and export complexes."""
import argparse
import csv
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


AFFINITY = re.compile(r"^REMARK VINA RESULT:\s*(-?\d+(?:\.\d+)?)", re.MULTILINE)


def first_model(text: str) -> str:
    lines, inside = [], False
    for line in text.splitlines():
        if line.startswith("MODEL"):
            if inside: break
            inside = True
            continue
        if line.startswith("ENDMDL") and inside: break
        if inside or not any(x.startswith("MODEL") for x in text.splitlines()):
            lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def pdb_atom(line: str, serial: int, ligand: bool) -> str | None:
    if not line.startswith(("ATOM", "HETATM")): return None
    atom = line[12:16] if len(line) >= 16 else " C  "
    res = "LIG" if ligand else (line[17:20].strip() or "UNK")
    chain = "Z" if ligand else (line[21:22].strip() or "A")
    resid = 1 if ligand else int((line[22:26].strip() or "1"))
    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
    # PDBQT uses AutoDock atom types (OA, NA, HD...) where PDB expects a
    # chemical element. Translate these explicitly for viewer compatibility.
    ad_type = line.split()[-1] if line.split() else ""
    element = {"OA": "O", "NA": "N", "SA": "S", "HD": "H", "HS": "H",
               "A": "C", "C": "C", "N": "N", "O": "O", "S": "S",
               "P": "P", "F": "F", "Cl": "Cl", "Br": "Br", "I": "I"}.get(ad_type)
    if not element:
        clean_atom = re.sub(r"[^A-Za-z]", "", atom)
        element = (clean_atom[:1] or "C").upper()
    rec = "HETATM" if ligand else "ATOM  "
    return f"{rec}{serial:5d} {atom:<4} {res:>3} {chain}{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}"


def make_complex_pdb(receptor: Path, ligand_pose: str, output: Path,
                     molecule_id: str = "", binding_site: str = "",
                     affinity: float | None = None):
    lines = [f"REMARK 900 MOLECULE {molecule_id}",
             f"REMARK 900 BINDING_SITE {binding_site}"]
    if affinity is not None:
        lines.append(f"REMARK 900 VINA_AFFINITY_KCAL_MOL {affinity:.3f}")
    serial = 1
    for source, is_ligand in ((receptor.read_text().splitlines(), False),
                              (ligand_pose.splitlines(), True)):
        for line in source:
            converted = pdb_atom(line, serial, is_ligand)
            if converted:
                lines.append(converted); serial += 1
        if not is_ligand: lines.append("TER")
    lines.extend(["TER", "END"])
    output.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--output-dir", default="results/top20_docked_complexes")
    args = ap.parse_args(); root = args.project.resolve()
    results, structures = root / "results", root / "data" / "structures"
    out = root / args.output_dir; out.mkdir(parents=True, exist_ok=True)
    vina = root / "bin" / "vina"
    if not vina.exists(): raise SystemExit(f"Vina executable missing: {vina}")

    scores = pd.read_csv(results / "docking_scores.csv")
    scores["best_affinity_kcal_mol"] = pd.to_numeric(scores["best_affinity_kcal_mol"], errors="coerce")
    best = (scores.sort_values("best_affinity_kcal_mol")
            .drop_duplicates("molecule_id").head(args.top).copy())
    geom = pd.read_csv(root / "data" / "processed" / "binding_pocket_geometry.csv").set_index("binding_site")

    def run(row):
        mol, site = str(row.molecule_id), str(row.binding_site)
        ligand = structures / "ligands_pdbqt" / f"{mol}.pdbqt"
        receptor = structures / str(row.receptor_used)
        if not ligand.exists() or not receptor.exists():
            raise FileNotFoundError(f"Missing input for {mol}: {ligand} / {receptor}")
        g = geom.loc[site]; hit_dir = out / f"{int(row.docking_rank):02d}_{mol}"
        hit_dir.mkdir(parents=True, exist_ok=True)
        multi = hit_dir / f"{mol}_{site}_all_poses.pdbqt"
        cmd = [str(vina), "--receptor", str(receptor), "--ligand", str(ligand),
               "--center_x", str(g.centroid_x), "--center_y", str(g.centroid_y), "--center_z", str(g.centroid_z),
               "--size_x", "20", "--size_y", "20", "--size_z", "20",
               "--exhaustiveness", str(args.exhaustiveness), "--num_modes", "9", "--out", str(multi)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        (hit_dir / "vina.log").write_text(proc.stdout + "\n" + proc.stderr)
        if proc.returncode or not multi.exists(): raise RuntimeError(f"Vina failed for {mol}: {proc.stderr[-300:]}")
        text = multi.read_text(); pose = first_model(text)
        affinity = min(map(float, AFFINITY.findall(text)))
        pose_path = hit_dir / f"{mol}_{site}_best_pose.pdbqt"; pose_path.write_text(pose)
        complex_path = hit_dir / f"{mol}_{site}_complex.pdb"
        make_complex_pdb(receptor, pose, complex_path, mol, site, affinity)
        return {"docking_rank": int(row.docking_rank), "molecule_id": mol,
                "binding_site": site, "original_affinity_kcal_mol": row.best_affinity_kcal_mol,
                "redocked_affinity_kcal_mol": affinity, "receptor": receptor.name,
                "best_pose_pdbqt": str(pose_path.relative_to(root)),
                "complex_pdb": str(complex_path.relative_to(root)), "status": "success"}

    best["docking_rank"] = range(1, len(best) + 1)
    records = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run, row): row for row in best.itertuples(index=False)}
        for future in as_completed(futures):
            row = futures[future]
            try: records.append(future.result())
            except Exception as exc:
                records.append({"docking_rank": int(row.docking_rank), "molecule_id": row.molecule_id,
                                "binding_site": row.binding_site, "status": "failed", "error": str(exc)})
    records.sort(key=lambda x: x["docking_rank"])
    manifest = out / "top20_docked_complexes_manifest.csv"
    with manifest.open("w", newline="") as handle:
        fields = sorted({k for row in records for k in row})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    ok = sum(r["status"] == "success" for r in records)
    print(f"Exported {ok}/{len(records)} complexes -> {out}")
    if ok != len(records): raise SystemExit(1)


if __name__ == "__main__": main()

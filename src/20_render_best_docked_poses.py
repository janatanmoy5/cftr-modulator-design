#!/usr/bin/env python3
"""Render best docked CFTR poses to publication-ready PNG files with PyMOL."""
import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def pymol_quote(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')


def render(pymol: str, structure: Path, output: Path, width: int, height: int,
           distance: float, ray: bool, view: str, label_text: str = "",
           style: str = "gray") -> tuple[bool, str]:
    is_complex = structure.name.endswith("_complex.pdb")
    ligand_selection = "chain Z and resn LIG" if is_complex else "all"
    output.parent.mkdir(parents=True, exist_ok=True)
    commands = [
        "reinitialize",
        f'load "{pymol_quote(structure)}", docked_structure',
        "remove solvent",
        "hide everything, all",
        f"select ligand, {ligand_selection}",
        "show sticks, ligand",
        "color magenta, ligand" if style == "rainbow" else "color cyan, ligand",
        "set stick_radius, 0.22, ligand",
    ]
    if is_complex:
        commands += [
            "select target, polymer",
            "show cartoon, target",
            "spectrum count, blue_green_yellow_red, target" if style == "rainbow" else "color gray70, target",
            "set cartoon_transparency, 0.10, target" if style == "rainbow" else "set cartoon_transparency, 0.55, target",
            f"select pocket, byres (target within {distance:.2f} of ligand)",
            "show sticks, pocket",
            "color white, pocket" if style == "rainbow" else "color salmon, pocket",
            "set stick_radius, 0.14, pocket",
            "select polar_contacts, (pocket and (donor or acceptor))",
            "distance contacts, ligand, polar_contacts, 3.6, mode=2",
            "color yellow, contacts",
            "hide labels, contacts",
            "set dash_width, 2.0",
        ]
    commands += [
        "set orthoscopic, on",
        "set antialias, 2",
        "set ray_shadows, off",
        "bg_color black" if style == "rainbow" else "bg_color white",
        "orient target" if is_complex and view == "full" else "orient ligand",
        "zoom target, 4" if is_complex and view == "full" else "zoom ligand, 9",
        # PyMOL's `png` command treats quote characters as part of the file
        # name in some builds. Project paths contain no spaces, so pass the
        # absolute path directly.
        f'png {pymol_quote(output)}, width={width}, height={height}, dpi=300, ray={1 if ray else 0}',
        "quit",
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".pml", delete=False) as handle:
        handle.write("\n".join(commands) + "\n"); pml = Path(handle.name)
    try:
        proc = subprocess.run([pymol, "-cq", str(pml)], capture_output=True,
                              text=True, timeout=600)
        ok = proc.returncode == 0 and output.exists() and output.stat().st_size > 0
        if ok and label_text:
            image = Image.open(output).convert("RGB")
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 38)
            except OSError:
                font = ImageFont.load_default()
            banner_h = 64
            banner_color = "black" if style == "rainbow" else "white"
            text_color = "white" if style == "rainbow" else "black"
            draw.rectangle((0, 0, image.width, banner_h), fill=banner_color)
            box = draw.textbbox((0, 0), label_text, font=font)
            x = max(12, (image.width - (box[2] - box[0])) // 2)
            draw.text((x, 10), label_text, fill=text_color, font=font)
            image.save(output, dpi=(300, 300))
        return ok, (proc.stderr or proc.stdout)[-500:]
    finally:
        pml.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--input", type=Path, help="One complex PDB or best-pose PDBQT")
    ap.add_argument("--input-dir", type=Path,
                    help="Directory containing per-hit complex PDB/best-pose PDBQT files")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--pocket-distance", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-ray", action="store_true", help="Faster preview rendering")
    ap.add_argument("--view", choices=("pocket", "full"), default="pocket",
                    help="Pocket close-up or complete receptor with highlighted ligand")
    ap.add_argument("--style", choices=("gray", "rainbow"), default="gray",
                    help="Gray publication style or rainbow-on-black receptor style")
    args = ap.parse_args(); root = args.project.resolve()
    pymol = shutil.which("pymol")
    if not pymol: raise SystemExit("PyMOL executable not found on PATH")
    source_dir = (args.input_dir or root / "results" / "top20_docked_complexes").resolve()
    output_dir = (args.output_dir or source_dir / "png").resolve()

    if args.input:
        structures = [args.input.resolve()]
    else:
        complexes = sorted(source_dir.glob("*/*_complex.pdb"))
        structures = complexes or sorted(source_dir.glob("*/*_best_pose.pdbqt"))
    if args.limit is not None: structures = structures[:args.limit]
    if not structures: raise SystemExit(f"No renderable docked structures found in {source_dir}")

    affinity_by_molecule = {}
    docking_manifest = source_dir / "top20_docked_complexes_manifest.csv"
    if docking_manifest.exists():
        with docking_manifest.open() as handle:
            for row in csv.DictReader(handle):
                affinity_by_molecule[row.get("molecule_id", "")] = row.get("redocked_affinity_kcal_mol", "")
    records = []
    for i, structure in enumerate(structures, 1):
        stem = structure.name.removesuffix("_complex.pdb").removesuffix("_best_pose.pdbqt")
        output = output_dir / f"{stem}_best_pose.png"
        print(f"[{i}/{len(structures)}] Rendering {structure.name}", flush=True)
        molecule_id = stem.split("_")[0]
        affinity = affinity_by_molecule.get(molecule_id, "")
        label = f"{molecule_id} | Vina: {float(affinity):.3f} kcal/mol" if affinity else molecule_id
        ok, message = render(pymol, structure, output, args.width, args.height,
                             args.pocket_distance, not args.no_ray, args.view, label, args.style)
        records.append({"structure": str(structure.relative_to(root)),
                        "png": str(output.relative_to(root)),
                        "status": "success" if ok else "failed", "message": message})
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "rendered_pose_images_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    succeeded = sum(r["status"] == "success" for r in records)
    print(f"Rendered {succeeded}/{len(records)} PNG files -> {output_dir}")
    if succeeded != len(records): raise SystemExit(1)


if __name__ == "__main__": main()

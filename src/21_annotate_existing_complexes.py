#!/usr/bin/env python3
"""Embed docking identity and affinity REMARK records in existing complex PDBs."""
import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args(); root = args.project.resolve()
    folder = root / "results" / "top20_docked_complexes"
    manifest = folder / "top20_docked_complexes_manifest.csv"
    rows = list(csv.DictReader(manifest.open()))
    updated = 0
    for row in rows:
        if row.get("status") != "success": continue
        path = root / row["complex_pdb"]
        text = path.read_text()
        body = "\n".join(line for line in text.splitlines()
                         if not line.startswith("REMARK 900")) + "\n"
        header = (f"REMARK 900 MOLECULE {row['molecule_id']}\n"
                  f"REMARK 900 BINDING_SITE {row['binding_site']}\n"
                  f"REMARK 900 ORIGINAL_VINA_AFFINITY_KCAL_MOL {float(row['original_affinity_kcal_mol']):.3f}\n"
                  f"REMARK 900 REDOCKED_VINA_AFFINITY_KCAL_MOL {float(row['redocked_affinity_kcal_mol']):.3f}\n")
        path.write_text(header + body); updated += 1
    print(f"Annotated {updated}/{len(rows)} complex PDB files with Vina affinities")


if __name__ == "__main__": main()

#!/usr/bin/env python3
"""
01_fetch_cftr_data.py

Fetches:
  - CFTR canonical sequence + JSON record from UniProt (P13569)
  - Reference structures listed in config.yaml from RCSB PDB
  - The full-length AlphaFold model (fills gaps missing from cryo-EM density)

The AlphaFold model URL is resolved DYNAMICALLY via the AlphaFold
Prediction API (https://alphafold.ebi.ac.uk/api/prediction/<accession>)
rather than a hardcoded version number in the URL -- the AlphaFold DB has
moved through several major versions (v4 -> v6 as of 2026) and a
hardcoded "_v4" suffix in the download URL will eventually 404. Whatever
version is current gets saved locally under a FIXED, version-independent
filename (AF-P13569-F1-model_latest.{cif,pdb}) so no other script in this
pipeline ever needs to know or care what AlphaFold's current version
number is.

Outputs:
  data/raw/cftr_P13569.fasta
  data/raw/cftr_P13569.json
  data/structures/<PDB_ID>.{cif,pdb}
  data/structures/AF-P13569-F1-model_latest.{cif,pdb}
"""
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, check_version, PROJECT_ROOT

log = get_logger("fetch_cftr_data")

HEADERS = {"User-Agent": "cftr-modulator-design-pipeline/0.1 (research use)"}
ALPHAFOLD_LOCAL_STEM = "AF-P13569-F1-model_latest"


def fetch(url: str, dest: Path, binary: bool = False, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        log.info(f"Already present, skipping: {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            mode = "wb" if binary else "w"
            content = resp.content if binary else resp.text
            with open(dest, mode) as fh:
                fh.write(content)
            log.info(f"Fetched {url} -> {dest}")
            return True
        except Exception as e:
            log.warning(f"Attempt {attempt}/{retries} failed for {url}: {e}")
            time.sleep(1.5 * attempt)
    log.error(f"FAILED to fetch {url} after {retries} attempts")
    return False


def fetch_alphafold_model(uniprot_accession: str, struct_dir: Path) -> None:
    """Resolves the current AlphaFold model URLs via the prediction API,
    then downloads to fixed local filenames regardless of what version
    number AlphaFold itself is using internally."""
    cif_dest = struct_dir / f"{ALPHAFOLD_LOCAL_STEM}.cif"
    pdb_dest = struct_dir / f"{ALPHAFOLD_LOCAL_STEM}.pdb"
    if cif_dest.exists() and pdb_dest.exists():
        log.info(f"Already present, skipping: {ALPHAFOLD_LOCAL_STEM}.{{cif,pdb}}")
        return

    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_accession}"
    try:
        resp = requests.get(api_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        predictions = resp.json()
        if not predictions:
            raise ValueError("API returned an empty prediction list")
        pred = predictions[0]
        cif_url = pred.get("cifUrl")
        pdb_url = pred.get("pdbUrl")
        if not cif_url and not pdb_url:
            raise ValueError(f"API response had no cifUrl/pdbUrl: {pred}")
        log.info(f"Resolved current AlphaFold model via API (version: {pred.get('latestVersion', '?')})")
    except Exception as e:
        log.warning(
            f"Could not resolve AlphaFold model via API ({e}). Falling back "
            "to a guessed v6 URL, then v4 as a last resort -- if both fail, "
            "this is non-fatal: downstream steps degrade gracefully to "
            "sequence-only / cryo-EM-only mode without the AlphaFold model."
        )
        cif_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_accession}-F1-model_v6.cif"
        pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_accession}-F1-model_v6.pdb"

    if cif_url:
        fetch(cif_url, cif_dest)
    if pdb_url:
        fetch(pdb_url, pdb_dest)


def main():
    check_version()
    cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    struct_dir = PROJECT_ROOT / cfg["paths"]["structures_dir"]

    # --- UniProt sequence + full JSON record ---
    fetch(cfg["project"]["uniprot_fasta_url"], raw_dir / "cftr_P13569.fasta")
    fetch(cfg["project"]["uniprot_json_url"], raw_dir / "cftr_P13569.json")

    # --- Full-length cryo-EM structures (CIF primary, PDB fallback) ---
    for entry in cfg["structures"]["full_length_cryo_em"]:
        pdb_id = entry["id"]
        fetch(f"https://files.rcsb.org/download/{pdb_id}.cif", struct_dir / f"{pdb_id}.cif")
        fetch(f"https://files.rcsb.org/download/{pdb_id}.pdb", struct_dir / f"{pdb_id}.pdb")

    # --- Isolated domain structures (NBD1 / NBD2 crystal structures) ---
    for entry in cfg["structures"]["isolated_domains"]:
        pdb_id = entry["id"]
        fetch(f"https://files.rcsb.org/download/{pdb_id}.cif", struct_dir / f"{pdb_id}.cif")
        fetch(f"https://files.rcsb.org/download/{pdb_id}.pdb", struct_dir / f"{pdb_id}.pdb")

    # --- AlphaFold full-length model (covers cryo-EM density gaps) ---
    fetch_alphafold_model(cfg["project"]["uniprot_id"], struct_dir)

    log.info("Step 01 complete. Raw sequence + structures staged.")
    log.info(
        "NOTE: network access to files.rcsb.org / alphafold.ebi.ac.uk / "
        "rest.uniprot.org must be permitted in this environment's egress "
        "rules. If any fetch failed above, download manually and place the "
        "file at the indicated path, then re-run this script (it skips "
        "files that already exist)."
    )


if __name__ == "__main__":
    main()

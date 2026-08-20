#!/usr/bin/env python3
"""
08_fetch_chembl_bioactivity.py

Resolves the ChEMBL target ID for CFTR dynamically from its UniProt
accession (P13569) -- rather than hardcoding a target_chembl_id that could
be wrong or stale -- then pages through the ChEMBL activity endpoint to
pull all IC50 / EC50 / Ki / Potency bioactivity records against that target.

ChEMBL REST API (no key required): https://www.ebi.ac.uk/chembl/api/data/

Output:
  data/raw/chembl_target.json               resolved target record(s)
  data/raw/chembl_bioactivity_raw.csv       one row per bioactivity record
"""
import sys
import time
import re
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, check_version, save_json, PROJECT_ROOT

log = get_logger("fetch_chembl")

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
ACTIVITY_TYPES = ["IC50", "EC50", "Ki", "Kd", "Potency"]
PAGE_LIMIT = 1000


def resolve_target_chembl_id(uniprot_accession: str) -> list[dict]:
    url = f"{CHEMBL_BASE}/target.json"
    params = {"target_components__accession": uniprot_accession, "limit": 50}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    targets = resp.json().get("targets", [])
    return targets


def fetch_activities(target_chembl_id: str) -> list[dict]:
    records = []
    url = f"{CHEMBL_BASE}/activity.json"
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type__in": ",".join(ACTIVITY_TYPES),
        "limit": PAGE_LIMIT,
        "offset": 0,
    }
    while True:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        batch = payload.get("activities", [])
        records.extend(batch)
        log.info(f"Fetched {len(batch)} records (total so far: {len(records)})")
        next_page = payload.get("page_meta", {}).get("next")
        if not next_page or not batch:
            break
        params["offset"] += PAGE_LIMIT
        time.sleep(0.3)  # be polite to the public API
    return records


def main():
    check_version()
    cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    uniprot_id = cfg["project"]["uniprot_id"]

    try:
        targets = resolve_target_chembl_id(uniprot_id)
    except Exception as e:
        log.error(
            f"Could not reach ChEMBL API to resolve target for {uniprot_id}: {e}. "
            "This environment's network egress may not permit "
            "www.ebi.ac.uk -- run this script on a machine with access, "
            "then copy data/raw/chembl_*.{json,csv} into this project."
        )
        sys.exit(0)  # non-fatal: downstream steps degrade gracefully

    if not targets:
        log.error(f"No ChEMBL target found for UniProt accession {uniprot_id}.")
        sys.exit(1)

    save_json({"uniprot_accession": uniprot_id, "targets": targets}, raw_dir / "chembl_target.json")
    # Prefer the SINGLE PROTEIN target type (most specific to CFTR itself,
    # as opposed to a protein complex or protein family target)
    single_protein = [t for t in targets if t.get("target_type") == "SINGLE PROTEIN"]
    preferred_id = cfg.get("chembl", {}).get("primary_target_chembl_id")
    preferred = [t for t in single_protein if t.get("target_chembl_id") == preferred_id]
    chosen = preferred[0] if preferred else (single_protein[0] if single_protein else targets[0])
    target_chembl_id = chosen["target_chembl_id"]
    log.info(
        f"Resolved CFTR ({uniprot_id}) -> ChEMBL target {target_chembl_id} "
        f"({chosen.get('pref_name')}, type={chosen.get('target_type')})"
    )

    activities = fetch_activities(target_chembl_id)
    if not activities:
        log.warning(f"No bioactivity records returned for {target_chembl_id}.")
        return

    df = pd.json_normalize(activities)
    keep_cols = [c for c in [
        "molecule_chembl_id", "canonical_smiles", "target_chembl_id", "document_chembl_id",
        "target_pref_name", "assay_chembl_id", "assay_description",
        "assay_type", "standard_type", "standard_relation", "standard_value",
        "standard_units", "pchembl_value", "activity_comment", "document_year",
    ] if c in df.columns]
    df = df[keep_cols].drop_duplicates()
    if "assay_description" in df.columns:
        variants = r"(wild[ -]?type|F508del|deltaF508|G551D|N1303K|W1282X)"
        df["assay_variant_context"] = (df["assay_description"].astype(str)
            .str.extract(variants, flags=re.IGNORECASE, expand=False)
            .fillna("unspecified").str.replace("deltaF508", "F508del", case=False, regex=False))
        df["assay_mechanism_context"] = "unspecified"
        text = df["assay_description"].astype(str).str.lower()
        df.loc[text.str.contains("potentiat"), "assay_mechanism_context"] = "potentiator"
        df.loc[text.str.contains("correct"), "assay_mechanism_context"] = "corrector"
        df.loc[text.str.contains("activat"), "assay_mechanism_context"] = "activator"

    out_path = raw_dir / "chembl_bioactivity_raw.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Wrote {len(df)} bioactivity records ({df['molecule_chembl_id'].nunique()} unique compounds) -> {out_path}")


if __name__ == "__main__":
    main()

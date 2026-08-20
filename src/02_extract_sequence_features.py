#!/usr/bin/env python3
"""
02_extract_sequence_features.py

For every mutation in data/raw/cf_mutations.csv (mutation notation like
"F508del", "G551D", "N1303K"), compute sequence-derived features:

  - domain membership (from config.yaml domain boundaries)
  - distance (in residues) to key functional sites (F508, ATP pockets)
  - physicochemical property deltas between wild-type and mutant residue
    (hydrophobicity [Kyte-Doolittle], charge, volume, polarity)
  - local sequence window (+/- 7 residues) around the mutation site
  - whether the mutation is a deletion, missense, nonsense, or frameshift/splice

Output: data/processed/sequence_features.csv
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, check_version, PROJECT_ROOT, ensure_mutations_seeded, atp_pocket_residues

log = get_logger("seq_features")

# Kyte-Doolittle hydrophobicity
KD_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Approximate residue volume (A^3), Zamyatnin 1972
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}
CHARGE = {**{aa: 0 for aa in KD_HYDROPATHY}, "D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1}
POLAR = set("STNQCYHKRDE")  # polar/charged residues

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}

MUT_RE = re.compile(r"^([A-Za-z]{1,3})(\d+)(del|fs.*|[A-Za-z]{1,3}|X|\*)?$")


def parse_mutation(notation: str):
    """Parse mutation notation such as F508del, G551D, N1303K, W1282X."""
    notation = notation.strip()
    m = MUT_RE.match(notation)
    if not m:
        return None
    wt_raw, pos, mut_raw = m.groups()
    wt = AA3_TO_1.get(wt_raw, wt_raw.upper()[:1] if len(wt_raw) <= 1 else None)
    if wt is None or wt not in KD_HYDROPATHY:
        return None
    pos = int(pos)

    if mut_raw is None:
        mtype, mt = "unknown", None
    elif mut_raw.lower() == "del":
        mtype, mt = "deletion", None
    elif mut_raw.lower().startswith("fs"):
        mtype, mt = "frameshift", None
    elif mut_raw.upper() in ("X", "*"):
        mtype, mt = "nonsense", None
    else:
        mt = AA3_TO_1.get(mut_raw, mut_raw.upper()[:1] if len(mut_raw) <= 1 else None)
        mtype = "missense" if mt in KD_HYDROPATHY else "unknown"

    return {"wt": wt, "pos": pos, "mt": mt, "mutation_type": mtype}


def domain_for_position(pos: int, domains: list) -> str:
    for d in domains:
        lo, hi = d["range"]
        if lo <= pos <= hi:
            return d["name"]
    return "unknown"


def sequence_window(seq: str, pos: int, window: int = 7) -> str:
    idx = pos - 1  # 1-indexed -> 0-indexed
    lo = max(0, idx - window)
    hi = min(len(seq), idx + window + 1)
    return seq[lo:hi]


def read_fasta_seq(fasta_path: Path) -> str:
    lines = fasta_path.read_text().splitlines()
    return "".join(l.strip() for l in lines if not l.startswith(">"))


def main():
    check_version()
    cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = raw_dir / "cftr_P13569.fasta"

    if not fasta_path.exists():
        log.warning(
            f"{fasta_path} not found (run 01_fetch_cftr_data.py first, or "
            "network egress to rest.uniprot.org is blocked in this "
            "environment). Falling back to config-embedded length-only mode "
            "-- hydrophobicity/volume deltas will still compute from mutation "
            "notation, but the local sequence WINDOW column will be empty."
        )
        seq = None
    else:
        seq = read_fasta_seq(fasta_path)
        log.info(f"Loaded CFTR sequence, length={len(seq)} aa")

    # Self-seed the mutation list on first run rather than requiring
    # 05_build_mutation_dataset.py to have run first -- pipeline.sh runs
    # 01->02->03->04->05->06 in that order, so 02 can't depend on 05's output.
    mut_path = ensure_mutations_seeded(raw_dir, log)

    mut_df = pd.read_csv(mut_path)
    domains = cfg["domains"]
    f508_pos = cfg["key_sites"]["F508"]["residue"]
    atp_pockets = atp_pocket_residues(cfg)
    atp_all = [r for residues in atp_pockets.values() for r in residues]
    abp1_residues = atp_pockets.get("ABP1_noncanonical_degenerate", [])
    abp2_residues = atp_pockets.get("ABP2_canonical_hydrolytic", [])

    rows = []
    for _, r in mut_df.iterrows():
        notation = str(r["mutation"])
        parsed = parse_mutation(notation)
        if parsed is None:
            log.warning(f"Could not parse mutation notation: {notation}, skipping")
            continue
        pos = parsed["pos"]
        wt, mt = parsed["wt"], parsed["mt"]

        d_hydro = d_vol = d_charge = polar_change = None
        if mt is not None and wt in KD_HYDROPATHY and mt in KD_HYDROPATHY:
            d_hydro = KD_HYDROPATHY[mt] - KD_HYDROPATHY[wt]
            d_vol = VOLUME[mt] - VOLUME[wt]
            d_charge = CHARGE.get(mt, 0) - CHARGE.get(wt, 0)
            polar_change = int((mt in POLAR) != (wt in POLAR))

        window = sequence_window(seq, pos) if seq else None

        rows.append({
            "mutation": notation,
            "wt_residue": wt,
            "position": pos,
            "mt_residue": mt,
            "mutation_type": parsed["mutation_type"],
            "domain": domain_for_position(pos, domains),
            "dist_to_F508": abs(pos - f508_pos),
            "dist_to_nearest_ATP_pocket": min(abs(pos - s) for s in atp_all) if atp_all else None,
            "in_ABP1_degenerate_pocket": int(pos in abp1_residues),
            "in_ABP2_hydrolytic_pocket": int(pos in abp2_residues),
            "delta_hydrophobicity_kd": d_hydro,
            "delta_volume_A3": d_vol,
            "delta_charge": d_charge,
            "polarity_class_changed": polar_change,
            "local_sequence_window": window,
        })

    out_df = pd.DataFrame(rows)
    out_path = proc_dir / "sequence_features.csv"
    out_df.to_csv(out_path, index=False)
    log.info(f"Wrote {len(out_df)} mutation feature rows -> {out_path}")


if __name__ == "__main__":
    main()

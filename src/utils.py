"""
utils.py
Shared helpers used across the CFTR modulator pipeline scripts.
"""
import json
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_config(config_path: str | Path = None) -> dict:
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def resolve_path(relative: str) -> Path:
    """Resolve a path relative to the project root (as given in config.yaml)."""
    p = PROJECT_ROOT / relative
    p.mkdir(parents=True, exist_ok=True) if p.suffix == "" else p.parent.mkdir(
        parents=True, exist_ok=True
    )
    return p


def load_json(path: Path) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def check_version():
    """Print VERSION.txt contents so stale-copy runs (e.g. from Trash/old
    clones) are caught immediately, mirroring the safeguard used in the
    hla-eplet-immunogenicity pipeline."""
    version_file = PROJECT_ROOT / "VERSION.txt"
    if version_file.exists():
        print(f"[VERSION] {version_file.read_text().strip()}  (root={PROJECT_ROOT})")
    else:
        print(f"[VERSION] WARNING: no VERSION.txt found at {PROJECT_ROOT}")


# -----------------------------------------------------------------------------
# Seed mutation list, shared between 02_extract_sequence_features.py (which
# needs it to exist BEFORE it can extract features) and
# 05_build_mutation_dataset.py (which uses the labels below to build the
# training set). Living here means step 02 can self-seed on first run
# instead of requiring step 05 to have already run -- pipeline.sh runs
# 01->02->03->04->05->06 in that order, so 02 can't depend on 05's output.
#
# (notation, functional class, brief public annotation)
# -----------------------------------------------------------------------------
SEED_MUTATIONS = [
    ("F508del", "II", "Most common CF allele; NBD1 misfolding; corrector-responsive"),
    ("G551D",   "III", "Signature motif gating mutation; index mutation for ivacaftor"),
    ("G178R",   "III", "Gating mutation; ivacaftor-responsive per label"),
    ("S549N",   "III", "Gating mutation; NBD1 signature motif region"),
    ("R117H",   "IV",  "Conductance mutation; ivacaftor-responsive (with 5T/7T modifier)"),
    ("N1303K",  "II",  "Severe NBD2 misfolding; historically poor modulator response"),
    ("W1282X",  "I",   "Premature stop codon (nonsense); no protein made"),
    ("G542X",   "I",   "Premature stop codon (nonsense); no protein made"),
    ("R553X",   "I",   "Premature stop codon (nonsense); no protein made"),
    ("621+1G->T", "I", "Canonical splice-site mutation; no functional protein"),
    ("3849+10kbC->T", "V", "Splicing mutation; reduced normal transcript"),
    ("A455E",   "II/V", "Mild processing mutation; residual function retained"),
    ("R560T",   "II",  "NBD1 processing mutation near signature motif"),
    ("D1152H",  "IV",  "Conductance mutation; ivacaftor-responsive per label"),
]

# Seed clinical-response labels: 1 = approved/labeled as responsive,
# 0 = not indicated / not responsive, NaN = unknown/not in current label.
# This mirrors the CLASS-level mechanism logic in config.yaml
# (correctors for Class II, potentiators for Class III/IV) plus a few
# well-known label-specific exceptions (e.g. F508del homozygous -> Kaftrio).
KNOWN_LABELS = {
    ("F508del", "Kaftrio"): 1,
    ("F508del", "Symkevi"): 1,
    ("F508del", "Orkambi"): 1,
    ("F508del", "Alyftrek"): 1,
    ("F508del", "Kalydeco"): 0,
    ("G551D", "Kalydeco"): 1,
    ("G551D", "Kaftrio"): 1,
    ("G551D", "Orkambi"): 0,
    ("G178R", "Kalydeco"): 1,
    ("S549N", "Kalydeco"): 1,
    ("R117H", "Kalydeco"): 1,
    ("D1152H", "Kalydeco"): 1,
    ("N1303K", "Kaftrio"): 0,
    ("N1303K", "Kalydeco"): 0,
    ("W1282X", "Kalydeco"): 0,
    ("W1282X", "Kaftrio"): 0,
    ("G542X", "Kalydeco"): 0,
    ("R553X", "Kalydeco"): 0,
    ("3849+10kbC->T", "Kalydeco"): 1,
}


def ensure_mutations_seeded(raw_dir: Path, logger: logging.Logger = None) -> Path:
    """Returns the path to data/raw/cf_mutations.csv, creating it from
    SEED_MUTATIONS if it doesn't exist yet. Safe to call from any script
    that needs the mutation list -- idempotent, never overwrites an
    existing (possibly user-edited) file."""
    import pandas as pd  # local import: keeps utils.py's baseline import
                          # cost low for scripts that don't need pandas

    mut_path = raw_dir / "cf_mutations.csv"
    if mut_path.exists():
        return mut_path

    raw_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(SEED_MUTATIONS, columns=["mutation", "functional_class", "notes"])
    df.to_csv(mut_path, index=False)
    if logger:
        logger.info(f"Seeded {len(df)} reference mutations -> {mut_path}")
    return mut_path


def atp_pocket_residues(cfg: dict) -> dict:
    """Flattens the corrected atp_binding_pockets.ABP1/ABP2 config
    structure (which mixes single-residue dicts and lists of dicts across
    several named fields -- Walker A/B, signature motif, etc.) into simple
    {pocket_name: [resnum, ...]} lists, for scripts that just need
    generic 'is this position in an ATP pocket' membership/distance
    features rather than the full residue-role metadata."""
    def _flatten(pocket_cfg: dict) -> list[int]:
        residues = []
        for key, val in pocket_cfg.items():
            if key in ("description", "hydrolytic", "binding_affinity"):
                continue
            if isinstance(val, dict) and "resnum" in val:
                residues.append(val["resnum"])
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and "resnum" in item:
                        residues.append(item["resnum"])
                    elif isinstance(item, int):
                        residues.append(item)
        return residues

    pockets = cfg.get("atp_binding_pockets", {})
    return {name: _flatten(pocket_cfg) for name, pocket_cfg in pockets.items()}

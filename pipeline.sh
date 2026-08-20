#!/usr/bin/env bash
# =============================================================================
# pipeline.sh - Runs the full CFTR modulator response prediction pipeline
#
# Usage:
#   ./pipeline.sh                  # run mutation-response steps 01-06
#   ./pipeline.sh --from 03        # resume from step 03 onward
#   ./pipeline.sh --bioactivity    # run ChEMBL QSAR regression branch (08-10)
#   ./pipeline.sh --docking        # run real structural docking branch (11-13)
#   ./pipeline.sh --integrated     # paired activity classifier + pX regressor (14)
#   ./pipeline.sh --rank-leads     # integrated lead ranking (15)
#   ./pipeline.sh --plots          # generate HTML report and plots (16)
#   ./pipeline.sh --only 07 --mutation G1244E --product Kaftrio   # ad-hoc predict
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
SRC_DIR="${PROJECT_ROOT}/src"

echo "=================================================================="
cat "${PROJECT_ROOT}/VERSION.txt" 2>/dev/null || echo "WARNING: no VERSION.txt"
echo " Running from: ${PROJECT_ROOT}"
echo "=================================================================="

if [ -d "${VENV_DIR}" ]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
else
  echo "WARNING: ${VENV_DIR} not found. Run ./install.sh first. Continuing with system python3."
fi

FROM_STEP="01"
ONLY_STEP=""
BIOACTIVITY=false
DOCKING=false
INTEGRATED=false
RANK_LEADS=false
PLOTS=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM_STEP="$2"; shift 2 ;;
    --only) ONLY_STEP="$2"; shift 2 ;;
    --bioactivity) BIOACTIVITY=true; shift ;;
    --docking) DOCKING=true; shift ;;
    --integrated) INTEGRATED=true; shift ;;
    --rank-leads) RANK_LEADS=true; shift ;;
    --plots) PLOTS=true; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

run_step () {
  local step_num="$1"
  local script
  script="$(ls "${SRC_DIR}"/${step_num}_*.py 2>/dev/null | head -n1)"
  if [ -z "${script}" ]; then
    echo "ERROR: no script found for step ${step_num}"
    exit 1
  fi
  echo ""
  echo "------------------------------------------------------------------"
  echo ">>> Step ${step_num}: $(basename "${script}")"
  echo "------------------------------------------------------------------"
  # Guarded expansion: macOS ships bash 3.2 by default (Apple stopped
  # updating it for licensing reasons), and bash <4.4 throws "unbound
  # variable" under `set -u` when expanding an EMPTY array, even though
  # the array itself exists and was initialized. This explicit length
  # check avoids ever expanding EXTRA_ARGS[@] when it's empty.
  if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    python3 "${script}" "${EXTRA_ARGS[@]}"
  else
    python3 "${script}"
  fi
}

if [ "${INTEGRATED}" = true ]; then
  run_step "14"
  exit 0
fi

if [ "${RANK_LEADS}" = true ]; then
  run_step "15"
  exit 0
fi

if [ "${PLOTS}" = true ]; then
  run_step "16"
  exit 0
fi

if [ -n "${ONLY_STEP}" ]; then
  run_step "${ONLY_STEP}"
  exit 0
fi

if [ "${DOCKING}" = true ]; then
  echo "Running structural docking branch (steps 11-13) ..."
  echo "Requires: 01_fetch_cftr_data.py already run (need 8EIQ.cif + ligand SMILES from 04/09)"
  for step in "11" "12" "13"; do
    run_step "${step}"
  done
  exit 0
fi

if [ "${BIOACTIVITY}" = true ]; then
  echo "Running ChEMBL bioactivity QSAR branch (steps 08-10) ..."
  for step in "08" "09" "10"; do
    run_step "${step}"
  done
  exit 0
fi

STEPS=("01" "02" "03" "04" "05" "06")
for step in "${STEPS[@]}"; do
  if [[ "${step}" < "${FROM_STEP}" ]]; then
    continue
  fi
  run_step "${step}"
done

echo ""
echo "=================================================================="
echo " Pipeline complete."
echo " Predict on a new mutation with:"
echo "   python3 ${SRC_DIR}/07_predict_modulator_response.py --mutation G1244E"
echo "=================================================================="

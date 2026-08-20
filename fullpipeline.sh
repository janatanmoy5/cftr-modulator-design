#!/usr/bin/env bash
# =============================================================================
# Complete CFTR small-molecule discovery and optional virtual-screening run.
#
# Usage:
#   ./fullpipeline.sh
#   ./fullpipeline.sh --library compounds.csv --top 100
#   ./fullpipeline.sh --skip-docking
#   ./fullpipeline.sh --skip-install --library compounds.smi
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBRARY=""
TOP_N="100"
SKIP_INSTALL=false
SKIP_DOCKING=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --library) LIBRARY="$2"; shift 2 ;;
    --top) TOP_N="$2"; shift 2 ;;
    --skip-install) SKIP_INSTALL=true; shift ;;
    --skip-docking) SKIP_DOCKING=true; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown argument: $1"; exit 2 ;;
  esac
done

if [ -n "${LIBRARY}" ]; then
  LIBRARY="$(cd "$(dirname "${LIBRARY}")" && pwd)/$(basename "${LIBRARY}")"
  [ -f "${LIBRARY}" ] || { echo "ERROR: compound library not found: ${LIBRARY}"; exit 2; }
fi
[[ "${TOP_N}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --top must be a positive integer"; exit 2; }

mkdir -p "${PROJECT_ROOT}/results/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_ROOT}/results/logs/fullpipeline_${RUN_ID}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE="startup"
on_error () {
  code=$?
  echo "ERROR: full pipeline stopped in stage '${CURRENT_STAGE}' (exit ${code})"
  echo "Log: ${LOG_FILE}"
  exit "${code}"
}
trap on_error ERR

stage () {
  CURRENT_STAGE="$1"; shift
  echo ""
  echo "=================================================================="
  echo ">>> ${CURRENT_STAGE}"
  echo "=================================================================="
  "$@"
}

echo "CFTR novel-small-molecule full pipeline"
echo "Run ID: ${RUN_ID}"
echo "Project: ${PROJECT_ROOT}"
echo "Candidate library: ${LIBRARY:-none (training/reference analysis only)}"
echo "Docking: $([ "${SKIP_DOCKING}" = true ] && echo skipped || echo enabled)"
cd "${PROJECT_ROOT}"

if [ "${SKIP_INSTALL}" = false ] && [ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
  stage "Environment installation" bash "${PROJECT_ROOT}/install.sh"
fi
[ -x "${PROJECT_ROOT}/.venv/bin/python" ] || {
  echo "ERROR: .venv is missing. Run without --skip-install first."; exit 1;
}
source "${PROJECT_ROOT}/.venv/bin/activate"
export MPLCONFIGDIR="${PROJECT_ROOT}/results/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

# Reference protein, structures, known-modulator controls and secondary
# mutation context. These steps do not define the novel-compound endpoint.
stage "01/17 Fetch CFTR sequence and structures" python3 src/01_fetch_cftr_data.py
stage "02/17 Extract mutation sequence context" python3 src/02_extract_sequence_features.py
stage "03/17 Extract mutation structural context" python3 src/03_extract_structural_features.py
stage "04/17 Featurize approved reference modulators" python3 src/04_extract_ligand_features.py
stage "05/17 Build secondary mutation-response dataset" python3 src/05_build_mutation_dataset.py
stage "06/17 Train secondary mutation-response model" python3 src/06_train_model.py

# Primary discovery endpoint: direct human CFTR CHEMBL4051.
stage "08/17 Fetch ChEMBL4051 bioactivities" python3 src/08_fetch_chembl_bioactivity.py
stage "09/17 Featurize CFTR-active chemical space" python3 src/09_featurize_chembl_compounds.py
if [ "${SKIP_DOCKING}" = false ]; then
  stage "11/17 Extract experimentally grounded CFTR pockets" python3 src/11_extract_binding_pockets.py
  stage "12/17 Prepare receptors and training ligands" python3 src/12_prepare_docking_inputs.py
  stage "13/17 Dock CHEMBL5523680 reference compounds" python3 src/13_run_docking.py \
    --document-id CHEMBL5523680 --workers 8 --exhaustiveness 4
fi

stage "10/17 Compare potency regression models" python3 src/10_train_bioactivity_regression_models.py
stage "14/17 Train integrated activity and potency models" python3 src/14_train_integrated_qsar.py
stage "15/17 Rank measured CFTR reference compounds" python3 src/15_rank_cftr_leads.py

if [ -n "${LIBRARY}" ]; then
  stage "17/17 Validate and featurize candidate library" python3 src/17_prepare_screening_library.py --input "${LIBRARY}"
  if [ "${SKIP_DOCKING}" = false ]; then
    stage "Candidate-library CFTR docking" python3 src/13_run_docking.py \
      --ligand-dir "${PROJECT_ROOT}/data/structures/screening_ligands_pdbqt" \
      --output screening_docking_scores.csv
    SCREEN_DOCKING="${PROJECT_ROOT}/results/screening_docking_scores.csv"
  else
    SCREEN_DOCKING="${PROJECT_ROOT}/results/nonexistent_screening_docking.csv"
  fi
  stage "Candidate activity, potency and lead ranking" python3 src/15_rank_cftr_leads.py \
    --input "${PROJECT_ROOT}/data/processed/screening_library_features.csv" \
    --docking "${SCREEN_DOCKING}" --output virtual_screening_hits.csv --top "${TOP_N}"
fi

stage "16/17 Generate plots, manifest and HTML report" python3 src/16_generate_results_report.py \
  --run-id "${RUN_ID}" --log "${LOG_FILE}"

trap - ERR
echo ""
echo "FULL PIPELINE COMPLETE"
echo "Report: ${PROJECT_ROOT}/results/report/index.html"
echo "Log: ${LOG_FILE}"
if [ -n "${LIBRARY}" ]; then
  echo "Virtual-screening hits: ${PROJECT_ROOT}/results/virtual_screening_hits.csv"
fi

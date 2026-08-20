#!/usr/bin/env bash
# =============================================================================
# install.sh - CFTR Modulator Response Prediction Pipeline
#
# Installs everything with a plain venv + pip -- no conda, no compiling.
#
# This project previously tried `pip install vina` (the AutoDock Vina
# Python bindings package), which compiles C++/SWIG code against Boost at
# install time and commonly fails on macOS without a hand-configured
# toolchain. That package is NOT actually needed: AutoDock Vina also ships
# as a STATIC, PRECOMPILED BINARY on its GitHub releases, and the docking
# script (13_run_docking.py) shells out to that binary directly. Meeko
# (which handles all PDBQT preparation) and everything else in this
# project -- rdkit, biopython, gemmi, scipy, etc. -- are pure pip wheels
# with no compiling required on any platform.
#
# Steps:
#   1. venv + pip install for all Python dependencies (including meeko,
#      gemmi, rdkit -- none of these need compiling)
#   2. download the static `vina` binary for the current OS/arch from
#      https://github.com/ccsb-scripps/AutoDock-Vina/releases into bin/vina
#
# Tested target: macOS (Apple Silicon), Linux x86_64, Python 3.10+
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
BIN_DIR="${PROJECT_ROOT}/bin"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VINA_VERSION="1.2.7"

echo "=================================================================="
echo " CFTR Modulator Pipeline - Install"
echo " Project root: ${PROJECT_ROOT}"
echo "=================================================================="
cat "${PROJECT_ROOT}/VERSION.txt" 2>/dev/null || echo "WARNING: VERSION.txt missing at project root -- is this the right copy?"
echo ""

if ! command -v "${PYTHON_BIN}" &> /dev/null; then
  echo "ERROR: ${PYTHON_BIN} not found. Install Python 3.10+ first (e.g. via brew, pyenv, or conda)."
  exit 1
fi

PY_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Using ${PYTHON_BIN} (version ${PY_VERSION})"

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating virtual environment at ${VENV_DIR} ..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Virtual environment already exists at ${VENV_DIR}, reusing."
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip/setuptools/wheel ..."
pip install --upgrade pip setuptools wheel --quiet

echo "Installing pipeline dependencies from requirements.txt ..."
pip install --quiet -r "${PROJECT_ROOT}/requirements.txt"

# --- static AutoDock Vina binary (Branch C: structural docking) ------------
echo ""
echo "Downloading AutoDock Vina v${VINA_VERSION} static binary (no compiling) ..."
mkdir -p "${BIN_DIR}"

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
VINA_ASSET=""
case "${OS_NAME}_${ARCH_NAME}" in
  Darwin_arm64)  VINA_ASSET="vina_${VINA_VERSION}_mac_aarch64" ;;
  Darwin_x86_64) VINA_ASSET="vina_${VINA_VERSION}_mac_x86_64" ;;
  Linux_x86_64)  VINA_ASSET="vina_${VINA_VERSION}_linux_x86_64" ;;
  *)
    echo "WARNING: no known Vina binary for ${OS_NAME} ${ARCH_NAME}."
    echo "Download one manually from:"
    echo "  https://github.com/ccsb-scripps/AutoDock-Vina/releases"
    echo "and place it at ${BIN_DIR}/vina (chmod +x)."
    ;;
esac

if [ -n "${VINA_ASSET}" ]; then
  VINA_URL="https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v${VINA_VERSION}/${VINA_ASSET}"
  if curl -sL -f "${VINA_URL}" -o "${BIN_DIR}/vina"; then
    chmod +x "${BIN_DIR}/vina"
    # macOS Gatekeeper quarantines downloaded executables; clear it so the
    # binary can actually run without a manual right-click-Open dance.
    if [ "${OS_NAME}" = "Darwin" ]; then
      xattr -d com.apple.quarantine "${BIN_DIR}/vina" 2>/dev/null || true
    fi
    if "${BIN_DIR}/vina" --help &> /dev/null; then
      echo "Vina binary installed and verified -> ${BIN_DIR}/vina"
    else
      echo "WARNING: downloaded vina binary did not run successfully."
      echo "On macOS, try: xattr -d com.apple.quarantine ${BIN_DIR}/vina"
      echo "and confirm Xcode Command Line Tools are installed (xcode-select -p)."
    fi
  else
    echo "WARNING: failed to download ${VINA_URL}"
    echo "(network access to github.com/release-assets may be blocked in this"
    echo "environment -- download it manually and place at ${BIN_DIR}/vina)."
  fi
fi

echo ""
echo "=================================================================="
echo " Install complete."
echo " Activate with:   source ${VENV_DIR}/bin/activate"
echo " Run pipeline:    ./pipeline.sh"
echo "=================================================================="

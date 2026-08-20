#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="${PYTHON_BIN:-python3}"
fi

# Cloud platforms inject PORT and require binding on all network interfaces.
HOST_ARG="${CFTR_APP_HOST:-${HOST:-0.0.0.0}}"
PORT_ARG="${CFTR_APP_PORT:-${PORT:-8765}}"
exec "${PYTHON}" "${ROOT}/app.py" --host "${HOST_ARG}" --port "${PORT_ARG}" "$@"

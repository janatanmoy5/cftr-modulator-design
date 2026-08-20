#!/usr/bin/env bash
# Virtual-screen a new CSV/SMI library through the canonical full pipeline.
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "Usage: ./screen_library.sh path/to/library.csv [top_n]"
  echo "CSV columns: compound_id,canonical_smiles"
  exit 2
fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${PROJECT_ROOT}/fullpipeline.sh" --library "$1" --top "${2:-100}"

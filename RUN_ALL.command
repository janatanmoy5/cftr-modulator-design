#!/usr/bin/env bash
# Double-click this file in Finder on macOS.
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
bash "${PROJECT_ROOT}/fullpipeline.sh"
echo ""
echo "Finished. Results are in ${PROJECT_ROOT}/results/report"
read -r -p "Press Return to close..." _

#!/usr/bin/env bash
# Backward-compatible alias for the canonical full pipeline.
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${PROJECT_ROOT}/fullpipeline.sh" "$@"

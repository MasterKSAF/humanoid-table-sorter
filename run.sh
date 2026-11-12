#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec python3 -u "$root/scripts/remote/run-fast-sort.py" \
  --fast-batch --edge-batch --upright-box --force-box --cup-mouse \
  --cup-trial-release --ignore-force-checks --timed-batch --sim-budget 260 \
  --adaptive-batch "$@"

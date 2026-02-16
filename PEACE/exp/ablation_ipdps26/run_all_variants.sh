#!/usr/bin/env bash
# Run the workload replay for all PEACE ablation variants.
#
# NOTE:
# 1) You must start the correct server variant BEFORE running each replay.
# 2) This script only runs the client-side workload runner.
#
# Usage:
#   BASE_URL=http://127.0.0.1:8000 MODEL=/path/to/model TRACE=./traces/azure_prepared.jsonl ./run_all_variants.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
API_KEY="${API_KEY:-EMPTY}"
ENDPOINT="${ENDPOINT:-chat}"
MODEL="${MODEL:?MODEL is required}"
TRACE="${TRACE:?TRACE is required}"
CONCURRENCY="${CONCURRENCY:-64}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-1.0}"

for VAR in peace pe dis col fsp; do
  echo "============================================================"
  echo "Variant: ${VAR}"
  echo "Make sure the server is started with the corresponding ablation flags."
  echo "============================================================"
  python run_workload.py \
    --base-url "${BASE_URL}" \
    --api-key "${API_KEY}" \
    --endpoint "${ENDPOINT}" \
    --model "${MODEL}" \
    --trace "${TRACE}" \
    --out-dir "./results/${VAR}" \
    --concurrency "${CONCURRENCY}" \
    --arrival-scale "${ARRIVAL_SCALE}"
done

echo "Done. Now run:"
echo "  python analyze_ablation.py --results-root ./results --out-dir ./figures"

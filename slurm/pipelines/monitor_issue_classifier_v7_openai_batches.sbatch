#!/bin/bash
# Monitor OpenAI Batch jobs for issue-level classification v7 and download outputs as they complete.
#
# Required env:
#   - RUN_ROOT=/scratch/.../issue_classifier_v7_*_YYYYMMDD_HHMMSS
#
# Optional env:
#   - SECRETS_FILE (default: /scratch/sxr203/.secrets/issue_classifier.env)
#   - RUNNER_ROOT (default: /scratch/sxr203/issue_classifier_v7_runner)
#   - POLL_SECONDS (default: 600)
#
# Notes:
#   - This job exits when all batches are final OR when walltime is reached.
#   - Re-submit if walltime expires before completion.

#SBATCH -J issue-cls-v7-monitor
#SBATCH -o /scratch/sxr203/newspaper-downloads/dedupe-webp/logs/issue_cls_v7_monitor_%j.out
#SBATCH -e /scratch/sxr203/newspaper-downloads/dedupe-webp/logs/issue_cls_v7_monitor_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -p cpu_short
#SBATCH -A torch_pr_609_general

set -euo pipefail

RUN_ROOT=${RUN_ROOT:-}
if [[ -z "${RUN_ROOT}" ]]; then
  echo "Missing required env: RUN_ROOT" >&2
  exit 1
fi

SECRETS_FILE=${SECRETS_FILE:-/scratch/sxr203/.secrets/issue_classifier.env}
RUNNER_ROOT=${RUNNER_ROOT:-/scratch/sxr203/issue_classifier_v7_runner}
POLL_SECONDS=${POLL_SECONDS:-600}

python -u "${RUNNER_ROOT}/scripts/openai_batch_monitor_curl.py" \
  --record-path "${RUN_ROOT}/openai_batch_submission/batch_jobs.jsonl" \
  --env-file "${SECRETS_FILE}" \
  --poll-seconds "${POLL_SECONDS}" \
  --curl-retries 8 \
  --curl-timeout-seconds 90 \
  --max-consecutive-errors 25 \
  --stop-when-final

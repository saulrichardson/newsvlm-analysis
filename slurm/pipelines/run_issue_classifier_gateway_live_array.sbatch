#!/bin/bash
# Run a prepared issue-classifier live gateway rerun as a Slurm job array.
#
# Required env:
#   - PREPARED_RUN_ROOT=/scratch/.../issue_classifier_v7_gateway_live_*
#
# Optional env:
#   - WORKSET=full|smoke                         (default: smoke)
#   - RUNNER_ROOT=/scratch/sxr203/issue_classifier_v7_gateway_runner
#   - VENV_PATH=/scratch/sxr203/issue_classifier_v7_gateway_runner/venv
#   - SECRETS_FILE=/scratch/sxr203/.secrets/issue_classifier.env
#   - MODEL=openai:gpt-5-mini
#   - WORKERS_PER_JOB=1
#   - MAX_CONCURRENCY=12
#   - TIMEOUT_SECONDS=300
#   - MAX_RETRIES=2
#   - PORT_BASE=8300
#
# Usage:
#   sbatch --array=0-3 slurm/run_issue_classifier_gateway_live_array.sbatch

#SBATCH -J issue-cls-gw
#SBATCH -o /scratch/sxr203/newspaper-downloads/dedupe-webp/logs/issue_cls_gw_%A_%a.out
#SBATCH -e /scratch/sxr203/newspaper-downloads/dedupe-webp/logs/issue_cls_gw_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -p cpu_short
#SBATCH -A torch_pr_609_general

set -euo pipefail

PREPARED_RUN_ROOT=${PREPARED_RUN_ROOT:-}
if [[ -z "${PREPARED_RUN_ROOT}" ]]; then
  echo "Missing required env: PREPARED_RUN_ROOT" >&2
  exit 1
fi

WORKSET=${WORKSET:-smoke}
RUNNER_ROOT=${RUNNER_ROOT:-/scratch/sxr203/issue_classifier_v7_gateway_runner}
VENV_PATH=${VENV_PATH:-${RUNNER_ROOT}/venv}
SECRETS_FILE=${SECRETS_FILE:-/scratch/sxr203/.secrets/issue_classifier.env}
MODEL=${MODEL:-openai:gpt-5-mini}
WORKERS_PER_JOB=${WORKERS_PER_JOB:-1}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-12}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-300}
MAX_RETRIES=${MAX_RETRIES:-2}
PORT_BASE=${PORT_BASE:-8300}

if [[ ! -f "${SECRETS_FILE}" ]]; then
  echo "Missing secrets file: ${SECRETS_FILE}" >&2
  exit 1
fi
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing runner virtualenv: ${VENV_PATH}" >&2
  exit 1
fi

REQUESTS_ROOT="${PREPARED_RUN_ROOT}/${WORKSET}/workers"
RESULTS_DIR="${PREPARED_RUN_ROOT}/${WORKSET}/results"

if [[ ! -d "${REQUESTS_ROOT}" ]]; then
  echo "Missing worker request root: ${REQUESTS_ROOT}" >&2
  exit 1
fi

mapfile -t WORKER_DIRS < <(find "${REQUESTS_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ "${#WORKER_DIRS[@]}" -eq 0 ]]; then
  echo "No worker dirs found under ${REQUESTS_ROOT}" >&2
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 )); then
  echo "SLURM_ARRAY_TASK_ID must be >= 0" >&2
  exit 1
fi

if (( WORKERS_PER_JOB <= 0 )); then
  echo "WORKERS_PER_JOB must be > 0" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

set -a
source "${SECRETS_FILE}"
set +a
source "${VENV_PATH}/bin/activate"

if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ -z "${OPENAI_KEY:-}" ]] && [[ -z "${CODEX_API_KEY:-}" ]]; then
  echo "Missing OpenAI key after sourcing ${SECRETS_FILE}" >&2
  exit 1
fi

START_INDEX=$((TASK_ID * WORKERS_PER_JOB))
if (( START_INDEX >= ${#WORKER_DIRS[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} with WORKERS_PER_JOB=${WORKERS_PER_JOB} starts beyond ${#WORKER_DIRS[@]} workers" >&2
  exit 1
fi
END_INDEX=$((START_INDEX + WORKERS_PER_JOB - 1))
if (( END_INDEX >= ${#WORKER_DIRS[@]} )); then
  END_INDEX=$((${#WORKER_DIRS[@]} - 1))
fi

declare -a PIDS=()
FAIL=0

for (( GLOBAL_IDX=START_INDEX; GLOBAL_IDX<=END_INDEX; GLOBAL_IDX++ )); do
  WORKER_DIR="${WORKER_DIRS[$GLOBAL_IDX]}"
  REQUEST_DIR="${WORKER_DIR}/requests"
  PORT=$((PORT_BASE + GLOBAL_IDX))
  STDOUT_LOG="${RESULTS_DIR}/worker_${GLOBAL_IDX}.stdout.log"
  STDERR_LOG="${RESULTS_DIR}/worker_${GLOBAL_IDX}.stderr.log"

  python -u "${RUNNER_ROOT}/scripts/run_openai_requests_via_gateway.py" \
    --request-dir "${REQUEST_DIR}" \
    --output-dir "${RESULTS_DIR}" \
    --model "${MODEL}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --timeout "${TIMEOUT_SECONDS}" \
    --max-retries "${MAX_RETRIES}" \
    --port "${PORT}" \
    > "${STDOUT_LOG}" 2> "${STDERR_LOG}" &
  PIDS+=("$!")
done

for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAIL=1
  fi
done

exit "${FAIL}"

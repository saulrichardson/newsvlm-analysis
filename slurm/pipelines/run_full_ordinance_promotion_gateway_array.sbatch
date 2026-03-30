#!/bin/bash
# Run a prepared second-stage full-ordinance promotion gateway job as a Slurm array.
#
# Required env:
#   - PREPARED_RUN_ROOT=/scratch/.../promotion_prepared_*
#
# Optional env:
#   - RUNNER_ROOT=/scratch/.../issue_classifier_v7_runner
#   - VENV_PATH=/scratch/.../issue_classifier_v7_runner/venv_postgate or /venv_py39 or /venv
#   - SECRETS_FILE=/scratch/.../.secrets/issue_classifier.env
#   - GATEWAY_MODEL=gemini:gemini-3-flash-preview
#   - MAX_CONCURRENCY=4
#   - TIMEOUT_SECONDS=21600
#   - PORT_BASE=9300 optional; otherwise a free localhost port is chosen
#
# Usage:
#   sbatch --array=0-3 slurm/run_full_ordinance_promotion_gateway_array.sbatch

#SBATCH -J promote-gw
#SBATCH -o /scratch/sxr203/newspaper-downloads/dedupe-webp/logs/promote_gw_%A_%a.out
#SBATCH -e /scratch/sxr203/newspaper-downloads/dedupe-webp/logs/promote_gw_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -p cs
#SBATCH -A torch_pr_609_general

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_RUNNER_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

PREPARED_RUN_ROOT=${PREPARED_RUN_ROOT:-}
if [[ -z "${PREPARED_RUN_ROOT}" ]]; then
  echo "Missing required env: PREPARED_RUN_ROOT" >&2
  exit 1
fi

SUMMARY_JSON="${PREPARED_RUN_ROOT}/summary.json"
if [[ ! -f "${SUMMARY_JSON}" ]]; then
  echo "Missing prepared summary: ${SUMMARY_JSON}" >&2
  exit 1
fi

RUNNER_ROOT=${RUNNER_ROOT:-${DEFAULT_RUNNER_ROOT}}
if [[ ! -d "${RUNNER_ROOT}" ]]; then
  echo "Missing RUNNER_ROOT: ${RUNNER_ROOT}" >&2
  exit 1
fi

if [[ -z "${VENV_PATH:-}" ]]; then
  if [[ -f "${RUNNER_ROOT}/venv_postgate/bin/activate" ]]; then
    VENV_PATH="${RUNNER_ROOT}/venv_postgate"
  elif [[ -f "${RUNNER_ROOT}/venv_py39/bin/activate" ]]; then
    VENV_PATH="${RUNNER_ROOT}/venv_py39"
  elif [[ -f "${RUNNER_ROOT}/venv/bin/activate" ]]; then
    VENV_PATH="${RUNNER_ROOT}/venv"
  else
    echo "Missing runner virtualenv under ${RUNNER_ROOT}/venv_postgate, ${RUNNER_ROOT}/venv_py39, or ${RUNNER_ROOT}/venv" >&2
    exit 1
  fi
fi

SECRETS_FILE=${SECRETS_FILE:-/scratch/sxr203/.secrets/issue_classifier.env}
if [[ ! -f "${SECRETS_FILE}" ]]; then
  echo "Missing secrets file: ${SECRETS_FILE}" >&2
  exit 1
fi
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing runner virtualenv: ${VENV_PATH}" >&2
  exit 1
fi

_summary_default() {
  local key="$1"
  python - "$SUMMARY_JSON" "$key" <<'PY'
import json, sys
path, key = sys.argv[1], sys.argv[2]
obj = json.load(open(path, "r", encoding="utf-8"))
value = obj.get(key, "")
if isinstance(value, bool):
    print("1" if value else "0")
elif value is None:
    print("")
else:
    print(str(value))
PY
}

GATEWAY_MODEL=${GATEWAY_MODEL:-$(_summary_default gateway_model)}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-$(_summary_default max_concurrency)}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-$(_summary_default timeout)}
PORT_BASE=${PORT_BASE:-}

WORKERS_ROOT="${PREPARED_RUN_ROOT}/workers"
RESULTS_ROOT="${PREPARED_RUN_ROOT}/results"
mkdir -p "${RESULTS_ROOT}"

if [[ ! -d "${WORKERS_ROOT}" ]]; then
  echo "Missing workers root: ${WORKERS_ROOT}" >&2
  exit 1
fi

mapfile -t WORKER_DIRS < <(find "${WORKERS_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ "${#WORKER_DIRS[@]}" -eq 0 ]]; then
  echo "No worker dirs found under ${WORKERS_ROOT}" >&2
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#WORKER_DIRS[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} outside worker range 0..$((${#WORKER_DIRS[@]} - 1))" >&2
  exit 1
fi

set -a
source "${SECRETS_FILE}"
set +a
source "${VENV_PATH}/bin/activate"

if [[ -z "${GEMINI_KEY:-}" ]]; then
  echo "Missing GEMINI_KEY after sourcing ${SECRETS_FILE}" >&2
  exit 1
fi

WORKER_DIR="${WORKER_DIRS[$TASK_ID]}"
WORKER_NAME=$(basename "${WORKER_DIR}")
REQUEST_DIR="${WORKER_DIR}/requests"
if [[ ! -d "${REQUEST_DIR}" ]]; then
  echo "Missing request dir: ${REQUEST_DIR}" >&2
  exit 1
fi

WORKER_OUT="${RESULTS_ROOT}/${WORKER_NAME}"
STDOUT_LOG="${WORKER_OUT}.stdout.log"
STDERR_LOG="${WORKER_OUT}.stderr.log"

_pick_free_local_port() {
  python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

if [[ -n "${PORT_BASE}" ]]; then
  PORT=$((PORT_BASE + TASK_ID))
else
  PORT=$(_pick_free_local_port)
fi

python -u "${RUNNER_ROOT}/scripts/run_openai_requests_via_gateway.py" \
  --request-dir "${REQUEST_DIR}" \
  --output-dir "${WORKER_OUT}" \
  --model "${GATEWAY_MODEL}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --timeout "${TIMEOUT_SECONDS}" \
  --port "${PORT}" \
  > "${STDOUT_LOG}" 2> "${STDERR_LOG}"

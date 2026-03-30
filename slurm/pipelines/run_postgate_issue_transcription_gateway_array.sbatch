#!/bin/bash
# Run a prepared post-gate issue transcription manifest as a Slurm job array.
#
# Required env:
#   - PREPARED_RUN_ROOT=/scratch/.../postgate_issue_transcription_gateway_*
#
# Optional env:
#   - RUNNER_ROOT=/scratch/.../newspaper-analysis
#   - VENV_PATH=/scratch/.../newspaper-analysis/venv_postgate or /venv_py39 or /venv
#   - SECRETS_FILE=/scratch/.../.secrets/issue_classifier.env
#   - PROMPT_PATH=/scratch/.../prompts/transcription_v13_issue_selected_images_postgate.txt
#   - REQUEST_MODEL=gemini-3.1-pro-preview
#   - GATEWAY_MODEL=gemini:gemini-3.1-pro-preview
#   - REASONING_EFFORT=high
#   - WORKERS_PER_JOB=1
#   - MAX_CONCURRENCY=4
#   - TIMEOUT_SECONDS=21600
#   - PORT_BASE=8300 optional, for deterministic ports; otherwise a free localhost port is chosen per worker
#
# Usage:
#   sbatch --array=0-99 slurm/run_postgate_issue_transcription_gateway_array.sbatch

#SBATCH -J postgate-gw
#SBATCH -o /scratch/sxr203/newspaper-downloads/dedupe-webp/logs/postgate_gw_%A_%a.out
#SBATCH -e /scratch/sxr203/newspaper-downloads/dedupe-webp/logs/postgate_gw_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
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

PROMPT_PATH=${PROMPT_PATH:-$(_summary_default prompt_path)}
REQUEST_MODEL=${REQUEST_MODEL:-$(_summary_default request_model)}
GATEWAY_MODEL=${GATEWAY_MODEL:-$(_summary_default gateway_model)}
REASONING_EFFORT=${REASONING_EFFORT:-$(_summary_default reasoning_effort)}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-$(_summary_default max_concurrency)}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-$(_summary_default timeout)}
WORKERS_PER_JOB=${WORKERS_PER_JOB:-1}
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
if (( TASK_ID < 0 )); then
  echo "SLURM_ARRAY_TASK_ID must be >= 0" >&2
  exit 1
fi
if (( WORKERS_PER_JOB <= 0 )); then
  echo "WORKERS_PER_JOB must be > 0" >&2
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
declare -a USED_PORTS=()
FAIL=0

_pick_free_local_port() {
  python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

_port_already_reserved() {
  local candidate="$1"
  for existing in "${USED_PORTS[@]:-}"; do
    if [[ "${existing}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

for (( GLOBAL_IDX=START_INDEX; GLOBAL_IDX<=END_INDEX; GLOBAL_IDX++ )); do
  WORKER_DIR="${WORKER_DIRS[$GLOBAL_IDX]}"
  MANIFEST_PATH="${WORKER_DIR}/manifest.jsonl"
  if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "Missing worker manifest: ${MANIFEST_PATH}" >&2
    exit 1
  fi

  WORKER_NAME=$(basename "${WORKER_DIR}")
  WORKER_OUT="${RESULTS_ROOT}/${WORKER_NAME}"
  if [[ -n "${PORT_BASE}" ]]; then
    PORT=$((PORT_BASE + GLOBAL_IDX))
  else
    while true; do
      PORT=$(_pick_free_local_port)
      if ! _port_already_reserved "${PORT}"; then
        USED_PORTS+=("${PORT}")
        break
      fi
    done
  fi
  STDOUT_LOG="${WORKER_OUT}.stdout.log"
  STDERR_LOG="${WORKER_OUT}.stderr.log"

  python -u "${RUNNER_ROOT}/scripts/run_postgate_issue_transcription.py" \
    --manifest-path "${MANIFEST_PATH}" \
    --prompt-path "${PROMPT_PATH}" \
    --output-dir "${WORKER_OUT}" \
    --env-file "${SECRETS_FILE}" \
    --request-model "${REQUEST_MODEL}" \
    --gateway-model "${GATEWAY_MODEL}" \
    --reasoning-effort "${REASONING_EFFORT}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --timeout "${TIMEOUT_SECONDS}" \
    --gateway-port "${PORT}" \
    > "${STDOUT_LOG}" 2> "${STDERR_LOG}" &
  PIDS+=("$!")
done

for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAIL=1
  fi
done

exit "${FAIL}"

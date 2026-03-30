#!/bin/bash
# Periodically download newly-completed OpenAI Batch outputs for issue topic embeddings,
# then rehydrate per-issue embedding matrices.
#
# Required env:
#   - RUN_ROOT=/vast/.../issue_topics_<source>_<label>_YYYYMMDD_HHMMSS
#
# Optional env:
#   - SLEEP_SECONDS (default 600)
#   - ITERATIONS    (default 36; 36*600s = 6h)
#   - SECRETS_FILE  (default: /scratch/sxr203/.secrets/openai.env)
#
# Output:
#   - ${RUN_ROOT}/results/openai_results_shard*.jsonl (+ downloaded_batches.jsonl)
#   - ${RUN_ROOT}/embeddings/{issue_embeddings.npy,issue_ids.txt,issue_metadata.jsonl}

#SBATCH -J openai-issue-embed-sweep
#SBATCH -o /vast/sxr203/newspaper-downloads/dedupe-webp/logs/openai_issue_embed_sweep_%j.out
#SBATCH -e /vast/sxr203/newspaper-downloads/dedupe-webp/logs/openai_issue_embed_sweep_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -p short

set -euo pipefail

PROJECT_ROOT=/scratch/sxr203/newspaper-parsing-export
VENV=/scratch/sxr203/newspaper-parsing/venv
SECRETS_FILE=${SECRETS_FILE:-/scratch/sxr203/.secrets/openai.env}

RUN_ROOT=${RUN_ROOT:-}
SLEEP_SECONDS=${SLEEP_SECONDS:-600}
ITERATIONS=${ITERATIONS:-36}

if [[ -z "${RUN_ROOT}" ]]; then
  echo "Missing required env: RUN_ROOT" >&2
  exit 1
fi

REQUEST_DIR="${RUN_ROOT}/requests"
RESULTS_DIR="${RUN_ROOT}/results"
EMB_DIR="${RUN_ROOT}/embeddings"

if [[ ! -d "${REQUEST_DIR}" ]]; then
  echo "Missing request dir: ${REQUEST_DIR}" >&2
  exit 1
fi

if [[ ! -f "${SECRETS_FILE}" ]]; then
  echo "Missing secrets file: ${SECRETS_FILE}" >&2
  exit 1
fi

source "${SECRETS_FILE}"
if [[ -z "${PROJECT_OPENAI_KEY:-}" ]] && [[ -z "${OPENAI_API_KEY:-}" ]] && [[ -z "${OPENAI_KEY:-}" ]]; then
  echo "No OpenAI key found after sourcing ${SECRETS_FILE} (expected PROJECT_OPENAI_KEY or OPENAI_API_KEY/OPENAI_KEY)" >&2
  exit 1
fi

source "${VENV}/bin/activate"

mkdir -p "${RESULTS_DIR}" "${EMB_DIR}"

for ((i=1; i<=ITERATIONS; i++)); do
  echo "=== openai issue embedding sweep ${i}/${ITERATIONS} ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="

  # Download any newly-completed batch outputs (idempotent)
  PYTHONUNBUFFERED=1 python "${PROJECT_ROOT}/scripts/download_openai_batch_results.py" \
    --request-dir "${REQUEST_DIR}" \
    --out-dir "${RESULTS_DIR}" \
    --skip-existing \
    --skip-not-completed \
    --openai-key-mode project_first || true

  submitted=0
  downloaded=0
  if [[ -f "${REQUEST_DIR}/submitted_batches.jsonl" ]]; then
    submitted=$(wc -l "${REQUEST_DIR}/submitted_batches.jsonl" | awk '{print $1}')
  fi
  if [[ -f "${RESULTS_DIR}/downloaded_batches.jsonl" ]]; then
    downloaded=$(wc -l "${RESULTS_DIR}/downloaded_batches.jsonl" | awk '{print $1}')
  fi

  echo "downloaded_batches ${downloaded}/${submitted}"
  if [[ "${submitted}" != "0" ]] && [[ "${downloaded}" -ge "${submitted}" ]]; then
    touch "${RESULTS_DIR}/DOWNLOADED_OPENAI_ALL"
    echo "All batches downloaded. Wrote: ${RESULTS_DIR}/DOWNLOADED_OPENAI_ALL"

    PYTHONUNBUFFERED=1 python "${PROJECT_ROOT}/scripts/rehydrate_issue_topic_embeddings_openai_batch_results.py" \
      --request-dir "${REQUEST_DIR}" \
      --results-dir "${RESULTS_DIR}" \
      --output-dir "${EMB_DIR}"

    exit 0
  fi

  sleep "${SLEEP_SECONDS}"
done

echo "Sweep finished (time limit)."


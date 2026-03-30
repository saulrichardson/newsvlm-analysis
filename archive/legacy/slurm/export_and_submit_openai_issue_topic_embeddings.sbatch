#!/bin/bash
# Export + submit OpenAI Batch jobs for ISSUE-level topic embeddings.
#
# Required env:
#   - RUN_ROOT=/vast/.../issue_topics_<source>_<label>_YYYYMMDD_HHMMSS
#   - ISSUE_ZONING_MANIFEST=/vast/.../issue_zoning_.../outputs/manifest.jsonl
#   - LABELS=full_ordinance|amendment_substantial|amendment_targeted (comma-separated OK)
#
# Optional env:
#   - EMBEDDING_MODEL (default: text-embedding-3-small)
#   - PAGE_SELECTION (default: pages_with_zoning)
#   - CHUNK_SIZE_CHARS (default: 16000)
#   - MAX_PAGE_CHARS (default: 0)
#   - REQUESTS_PER_SHARD (default: 5000)
#   - MAX_BYTES_PER_SHARD (default: 180000000)
#   - SECRETS_FILE (default: /scratch/sxr203/.secrets/openai.env)
#
# Output:
#   - ${RUN_ROOT}/requests/{openai_requests_shard*.jsonl,mapping_shard*.jsonl,inputs/*,submitted_batches.jsonl}

#SBATCH -J openai-issue-embed-submit
#SBATCH -o /vast/sxr203/newspaper-downloads/dedupe-webp/logs/openai_issue_embed_submit_%j.out
#SBATCH -e /vast/sxr203/newspaper-downloads/dedupe-webp/logs/openai_issue_embed_submit_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -p short

set -euo pipefail

PROJECT_ROOT=/scratch/sxr203/newspaper-parsing-export
VENV=/scratch/sxr203/newspaper-parsing/venv

SECRETS_FILE=${SECRETS_FILE:-/scratch/sxr203/.secrets/openai.env}

RUN_ROOT=${RUN_ROOT:-}
ISSUE_ZONING_MANIFEST=${ISSUE_ZONING_MANIFEST:-}
LABELS=${LABELS:-}

EMBEDDING_MODEL=${EMBEDDING_MODEL:-text-embedding-3-small}
PAGE_SELECTION=${PAGE_SELECTION:-pages_with_zoning}
CHUNK_SIZE_CHARS=${CHUNK_SIZE_CHARS:-16000}
MAX_PAGE_CHARS=${MAX_PAGE_CHARS:-0}
REQUESTS_PER_SHARD=${REQUESTS_PER_SHARD:-5000}
MAX_BYTES_PER_SHARD=${MAX_BYTES_PER_SHARD:-180000000}

if [[ -z "${RUN_ROOT}" ]]; then
  echo "Missing required env: RUN_ROOT" >&2
  exit 1
fi

if [[ -z "${ISSUE_ZONING_MANIFEST}" ]]; then
  echo "Missing required env: ISSUE_ZONING_MANIFEST" >&2
  exit 1
fi

if [[ -z "${LABELS}" ]]; then
  echo "Missing required env: LABELS" >&2
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

REQUEST_DIR="${RUN_ROOT}/requests"
mkdir -p "${REQUEST_DIR}"

cd "${PROJECT_ROOT}"

echo "RUN_ROOT=${RUN_ROOT}"
echo "ISSUE_ZONING_MANIFEST=${ISSUE_ZONING_MANIFEST}"
echo "LABELS=${LABELS}"
echo "EMBEDDING_MODEL=${EMBEDDING_MODEL}"

PYTHONUNBUFFERED=1 python "${PROJECT_ROOT}/scripts/export_issue_topic_embedding_batch_requests.py" \
  --issue-zoning-manifest "${ISSUE_ZONING_MANIFEST}" \
  --labels "${LABELS}" \
  --output-dir "${REQUEST_DIR}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --page-selection "${PAGE_SELECTION}" \
  --chunk-size-chars "${CHUNK_SIZE_CHARS}" \
  --max-page-chars "${MAX_PAGE_CHARS}" \
  --requests-per-shard "${REQUESTS_PER_SHARD}" \
  --max-bytes-per-shard "${MAX_BYTES_PER_SHARD}"

PYTHONUNBUFFERED=1 python "${PROJECT_ROOT}/scripts/submit_batch_shards.py" \
  --request-dir "${REQUEST_DIR}" \
  --providers openai \
  --openai-endpoint /v1/embeddings \
  --openai-key-mode project_first \
  --display-name-prefix issue-embeddings


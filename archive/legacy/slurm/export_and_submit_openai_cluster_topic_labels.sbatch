#!/bin/bash
# Export + submit OpenAI Batch jobs for cluster topic labels (chunk clusters → human labels).
#
# Required env:
#   - RUN_ROOT=/vast/.../issue_topics_chunks3000_weight_<label>_<timestamp>
#
# Optional env:
#   - OPENAI_MODEL (default: gpt-5-nano)
#   - PROMPT_PATH  (default: prompts/cluster_topic_labeler_prompt_text.txt)
#   - SCHEMA_VERSION (default: v1)  # v2 adds structured mechanics tags
#   - LABEL_DIR_NAME (default: cluster_topic_labels)
#   - MAX_EXAMPLES_PER_CLUSTER (default: 8)
#   - EXAMPLE_MAX_CHARS (default: 1100)
#   - CANDIDATE_POOL_PER_CLUSTER (default: 180)
#   - CLUSTER_KEYWORDS_K (default: 12)
#   - EXAMPLE_SELECTION (default: informative) # informative|legacy
#   - NO_MECHANICS_TAGS (default: 0) # when 1, do not include heuristic mechanics tags in prompt
#   - EXCERPT_WINDOW_CHARS (default: 240)
#   - TFIDF_MAX_CHARS (default: 20000)
#   - MIN_CLUSTER_SIZE (default: 1)   # 1 = label all clusters (except noise)
#   - MAX_CLUSTERS (default: 0)       # 0 = all clusters
#   - SECRETS_FILE (default: /scratch/sxr203/.secrets/openai.env)
#
# Output:
#   - ${RUN_ROOT}/cluster_topic_labels/requests/{openai_requests_shard*.jsonl,mapping_shard*.jsonl,inputs/*,submitted_batches.jsonl}

#SBATCH -J openai-cluster-label-submit
#SBATCH -o /vast/sxr203/newspaper-downloads/dedupe-webp/logs/openai_cluster_label_submit_%j.out
#SBATCH -e /vast/sxr203/newspaper-downloads/dedupe-webp/logs/openai_cluster_label_submit_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -p short

set -euo pipefail

PROJECT_ROOT=/scratch/sxr203/newspaper-parsing-export
VENV=/scratch/sxr203/newspaper-parsing/venv
SECRETS_FILE=${SECRETS_FILE:-/scratch/sxr203/.secrets/openai.env}

RUN_ROOT=${RUN_ROOT:-}

OPENAI_MODEL=${OPENAI_MODEL:-gpt-5-nano}
PROMPT_PATH=${PROMPT_PATH:-prompts/cluster_topic_labeler_prompt_text.txt}
SCHEMA_VERSION=${SCHEMA_VERSION:-v1}
LABEL_DIR_NAME=${LABEL_DIR_NAME:-cluster_topic_labels}
MAX_EXAMPLES_PER_CLUSTER=${MAX_EXAMPLES_PER_CLUSTER:-8}
EXAMPLE_MAX_CHARS=${EXAMPLE_MAX_CHARS:-1100}
CANDIDATE_POOL_PER_CLUSTER=${CANDIDATE_POOL_PER_CLUSTER:-180}
CLUSTER_KEYWORDS_K=${CLUSTER_KEYWORDS_K:-12}
EXAMPLE_SELECTION=${EXAMPLE_SELECTION:-informative}
NO_MECHANICS_TAGS=${NO_MECHANICS_TAGS:-0}
EXCERPT_WINDOW_CHARS=${EXCERPT_WINDOW_CHARS:-240}
TFIDF_MAX_CHARS=${TFIDF_MAX_CHARS:-20000}
MIN_CLUSTER_SIZE=${MIN_CLUSTER_SIZE:-1}
MAX_CLUSTERS=${MAX_CLUSTERS:-0}

if [[ -z "${RUN_ROOT}" ]]; then
  echo "Missing required env: RUN_ROOT" >&2
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

REQUEST_DIR="${RUN_ROOT}/${LABEL_DIR_NAME}/requests"
mkdir -p "${REQUEST_DIR}"

cd "${PROJECT_ROOT}"

echo "RUN_ROOT=${RUN_ROOT}"
echo "OPENAI_MODEL=${OPENAI_MODEL}"
echo "SCHEMA_VERSION=${SCHEMA_VERSION}"
echo "EXAMPLE_SELECTION=${EXAMPLE_SELECTION}"

NO_MECHANICS_FLAG=""
if [[ "${NO_MECHANICS_TAGS}" == "1" ]]; then
  NO_MECHANICS_FLAG="--no-mechanics-tags"
fi

PYTHONUNBUFFERED=1 python "${PROJECT_ROOT}/scripts/export_cluster_topic_labeling_batch_requests.py" \
  --run-root "${RUN_ROOT}" \
  --output-dir "${REQUEST_DIR}" \
  --prompt-path "${PROMPT_PATH}" \
  --schema-version "${SCHEMA_VERSION}" \
  --openai-model "${OPENAI_MODEL}" \
  --openai-text-format json_schema \
  --example-selection "${EXAMPLE_SELECTION}" \
  --max-examples-per-cluster "${MAX_EXAMPLES_PER_CLUSTER}" \
  --example-max-chars "${EXAMPLE_MAX_CHARS}" \
  --candidate-pool-per-cluster "${CANDIDATE_POOL_PER_CLUSTER}" \
  --cluster-keywords-k "${CLUSTER_KEYWORDS_K}" \
  ${NO_MECHANICS_FLAG} \
  --excerpt-window-chars "${EXCERPT_WINDOW_CHARS}" \
  --tfidf-max-chars "${TFIDF_MAX_CHARS}" \
  --min-cluster-size "${MIN_CLUSTER_SIZE}" \
  --max-clusters "${MAX_CLUSTERS}"

PYTHONUNBUFFERED=1 python "${PROJECT_ROOT}/scripts/submit_batch_shards.py" \
  --request-dir "${REQUEST_DIR}" \
  --providers openai \
  --openai-endpoint /v1/responses \
  --openai-key-mode project_first \
  --display-name-prefix cluster-topic-labels

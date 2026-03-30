#!/bin/bash
# Periodically download newly-completed OpenAI Batch outputs for cluster topic labels,
# then rehydrate cluster_labels.jsonl and re-plot cluster frequencies with human-readable names.
#
# Required env:
#   - RUN_ROOT=/vast/.../issue_topics_chunks3000_weight_<label>_<timestamp>
#
# Optional env:
#   - SLEEP_SECONDS (default 600)
#   - ITERATIONS    (default 36; 36*600s = 6h)
#   - SECRETS_FILE  (default: /scratch/sxr203/.secrets/openai.env)
#   - LABEL_DIR_NAME (default: cluster_topic_labels)
#   - PLOT_DIR_NAME  (default: plots_chunks_labeled)
#   - COUNT_MODE    (default: weight)       # rows|unique_issue|weight
#   - WEIGHT_FIELD  (default: doc_weight)
#   - TOP_K         (default: 20)
#   - OMIT_OTHER    (default: 0)            # when 1, do not plot/CSV an "other" series
#
# Output:
#   - ${RUN_ROOT}/cluster_topic_labels/results/openai_results_shard*.jsonl (+ downloaded_batches.jsonl)
#   - ${RUN_ROOT}/cluster_topic_labels/outputs/cluster_labels.jsonl
#   - ${RUN_ROOT}/plots_chunks_labeled/cluster_frequency_year.png

#SBATCH -J openai-cluster-label-sweep
#SBATCH -o /vast/sxr203/newspaper-downloads/dedupe-webp/logs/openai_cluster_label_sweep_%j.out
#SBATCH -e /vast/sxr203/newspaper-downloads/dedupe-webp/logs/openai_cluster_label_sweep_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -p short

set -euo pipefail

PROJECT_ROOT=/scratch/sxr203/newspaper-parsing-export
VENV=/scratch/sxr203/newspaper-parsing/venv
PLOT_VENV=${PLOT_VENV:-/scratch/sxr203/newspaper-parsing/venv-topics}
SECRETS_FILE=${SECRETS_FILE:-/scratch/sxr203/.secrets/openai.env}

RUN_ROOT=${RUN_ROOT:-}
SLEEP_SECONDS=${SLEEP_SECONDS:-600}
ITERATIONS=${ITERATIONS:-36}

LABEL_DIR_NAME=${LABEL_DIR_NAME:-cluster_topic_labels}
PLOT_DIR_NAME=${PLOT_DIR_NAME:-plots_chunks_labeled}

COUNT_MODE=${COUNT_MODE:-weight}
WEIGHT_FIELD=${WEIGHT_FIELD:-doc_weight}
TOP_K=${TOP_K:-20}
OMIT_OTHER=${OMIT_OTHER:-0}

if [[ -z "${RUN_ROOT}" ]]; then
  echo "Missing required env: RUN_ROOT" >&2
  exit 1
fi

REQUEST_DIR="${RUN_ROOT}/${LABEL_DIR_NAME}/requests"
RESULTS_DIR="${RUN_ROOT}/${LABEL_DIR_NAME}/results"
OUT_DIR="${RUN_ROOT}/${LABEL_DIR_NAME}/outputs"

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

mkdir -p "${RESULTS_DIR}" "${OUT_DIR}"

for ((i=1; i<=ITERATIONS; i++)); do
  echo "=== openai cluster label sweep ${i}/${ITERATIONS} ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="

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

    PYTHONUNBUFFERED=1 python "${PROJECT_ROOT}/scripts/rehydrate_cluster_topic_labels_openai_batch_results.py" \
      --request-dir "${REQUEST_DIR}" \
      --results-dir "${RESULTS_DIR}" \
      --output-dir "${OUT_DIR}"

    # Replot with human-readable topic names.
    # Use venv-topics because it includes matplotlib (base venv does not necessarily).
    source "${PLOT_VENV}/bin/activate"
    PLOT_DIR="${RUN_ROOT}/${PLOT_DIR_NAME}"
    mkdir -p "${PLOT_DIR}"
    plot_args=( \
      --clusters-jsonl "${RUN_ROOT}/clusters_chunks/clusters.jsonl" \
      --cluster-labels-jsonl "${OUT_DIR}/cluster_labels.jsonl" \
      --output-dir "${PLOT_DIR}" \
      --bucket year \
      --top-k "${TOP_K}" \
      --count-mode "${COUNT_MODE}" \
    )
    if [[ "${COUNT_MODE}" == "weight" ]]; then
      plot_args+=( --weight-field "${WEIGHT_FIELD}" )
    fi
    if [[ "${OMIT_OTHER}" == "1" ]]; then
      plot_args+=( --omit-other )
    fi

    PYTHONUNBUFFERED=1 python "${PROJECT_ROOT}/scripts/plot_cluster_frequencies_over_time.py" "${plot_args[@]}"
    exit 0
  fi

  sleep "${SLEEP_SECONDS}"
done

echo "Sweep finished (time limit) without downloading all batches."
touch "${RESULTS_DIR}/SWEEP_TIMED_OUT"
echo "Wrote: ${RESULTS_DIR}/SWEEP_TIMED_OUT"
exit 1

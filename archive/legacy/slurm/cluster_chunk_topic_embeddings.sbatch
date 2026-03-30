#!/bin/bash
# Cluster chunk embeddings (UMAP→HDBSCAN) + plot frequencies over time.
#
# Required env:
#   - RUN_ROOT=/vast/.../issue_topics_<label>_YYYYMMDD_HHMMSS
#
# Optional env:
#   - VENV (default: /scratch/sxr203/newspaper-parsing/venv-topics)
#   - UMAP_N_NEIGHBORS (default: 25)
#   - UMAP_MIN_DIST (default: 0.0)
#   - UMAP_N_COMPONENTS (default: 5)
#   - HDBSCAN_MIN_CLUSTER_SIZE (default: 30)
#   - HDBSCAN_MIN_SAMPLES (default: 10)
#   - BUCKET (default: year)
#   - TOP_K (default: 12)
#   - COUNT_MODE (default: rows)  [rows|unique_issue|weight]
#   - WEIGHT_FIELD (default: doc_weight)

#SBATCH -J chunk-embed-cluster
#SBATCH -o /vast/sxr203/newspaper-downloads/dedupe-webp/logs/chunk_embed_cluster_%j.out
#SBATCH -e /vast/sxr203/newspaper-downloads/dedupe-webp/logs/chunk_embed_cluster_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -p short

set -euo pipefail

PROJECT_ROOT=/scratch/sxr203/newspaper-parsing-export
VENV=${VENV:-/scratch/sxr203/newspaper-parsing/venv-topics}

RUN_ROOT=${RUN_ROOT:-}

UMAP_N_NEIGHBORS=${UMAP_N_NEIGHBORS:-25}
UMAP_MIN_DIST=${UMAP_MIN_DIST:-0.0}
UMAP_N_COMPONENTS=${UMAP_N_COMPONENTS:-5}
HDBSCAN_MIN_CLUSTER_SIZE=${HDBSCAN_MIN_CLUSTER_SIZE:-30}
HDBSCAN_MIN_SAMPLES=${HDBSCAN_MIN_SAMPLES:-10}

BUCKET=${BUCKET:-year}
TOP_K=${TOP_K:-12}
COUNT_MODE=${COUNT_MODE:-rows}
WEIGHT_FIELD=${WEIGHT_FIELD:-doc_weight}

if [[ -z "${RUN_ROOT}" ]]; then
  echo "Missing required env: RUN_ROOT" >&2
  exit 1
fi

source "${VENV}/bin/activate"

EMB_DIR="${RUN_ROOT}/embeddings_chunks"
OUT_DIR="${RUN_ROOT}/clusters_chunks"
PLOT_DIR="${RUN_ROOT}/plots_chunks"
mkdir -p "${OUT_DIR}" "${PLOT_DIR}"

cd "${PROJECT_ROOT}"

PYTHONUNBUFFERED=1 python "${PROJECT_ROOT}/scripts/cluster_chunk_topic_embeddings.py" \
  --embedding-dir "${EMB_DIR}" \
  --output-dir "${OUT_DIR}" \
  --umap-n-neighbors "${UMAP_N_NEIGHBORS}" \
  --umap-min-dist "${UMAP_MIN_DIST}" \
  --umap-n-components "${UMAP_N_COMPONENTS}" \
  --hdbscan-min-cluster-size "${HDBSCAN_MIN_CLUSTER_SIZE}" \
  --hdbscan-min-samples "${HDBSCAN_MIN_SAMPLES}"

plot_args=( \
  --clusters-jsonl "${OUT_DIR}/clusters.jsonl" \
  --output-dir "${PLOT_DIR}" \
  --bucket "${BUCKET}" \
  --top-k "${TOP_K}" \
  --count-mode "${COUNT_MODE}" \
)

if [[ "${COUNT_MODE}" == "weight" ]]; then
  plot_args+=( --weight-field "${WEIGHT_FIELD}" )
fi

PYTHONUNBUFFERED=1 python "${PROJECT_ROOT}/scripts/plot_cluster_frequencies_over_time.py" "${plot_args[@]}"


#!/bin/bash
# Export + submit ISSUE-level classification batches (OpenAI Batch API; optional Gemini Batch).
#
# This job:
#   1) (optional) exports Paddle VL1.5 Markdown -> single JSONL.GZ (inode-friendly)
#   2) exports issue-level batch request shards
#   3) submits OpenAI Batch jobs (Responses API)
#
# Required env:
#   - RUN_ROOT=/scratch/sxr203/newspaper-downloads/dedupe-webp/issue_classifier_v7_<label>_YYYYMMDD_HHMMSS
#
# Optional env:
#   - UNIQUE_PNG_ROOT (default: /scratch/sxr203/newspaper-downloads/dedupe-webp/unique_png)
#   - PADDLE_RUN_ROOT (default: /scratch/$USER/paddleocr_vl15/runs)
#   - PADDLE_RUN_PREFIX (default: layout_bagging_unique_png_sharded_20260220_200103_manifest_)
#   - PADDLE_LAYOUT_VARIANT (default: pld_v3_thr03)
#   - PADDLE_VL15_VARIANT (default: pvl15_docparser_v15)
#   - VLM_PAGE_ROOTS (default: empty; set to enable per-page *.vlm.json cache scanning)
#   - OPENAI_BOX_RESULTS (default: /scratch/sxr203/newspaper-downloads/dedupe-webp/batch_requests_dedup_jpeg1024_mp1_q80/part_*/openai_gpt52_reasoning_medium_split/results)
#   - GEMINI_BOX_RESULTS (default: /scratch/sxr203/newspaper-downloads/dedupe-webp/batch_requests_dedup_jpeg1024_mp1_q80/part_*/results_gemini,/scratch/sxr203/newspaper-downloads/dedupe-webp/batch_requests_openai_no_ok_gemini3_flash3prep_20260121_005716/results_gemini)
#   - OPENAI_MODEL (default: gpt-5-mini)
#   - CLASSIFIER_PROVIDER (default: openai; choices: openai, gemini, both)
#   - GEMINI_MODEL (default: models/gemini-2.5-flash; only when CLASSIFIER_PROVIDER includes gemini)
#   - ISSUES_PER_SHARD (default: 5000)
#   - MAX_BYTES_PER_SHARD (default: 180000000)
#   - MISSING_PAGE_POLICY (default: skip)
#   - SOURCE_PRIORITY (default: paddle_vl15_md,vlm_page,gemini_box_results,openai_box_results)
#   - MAX_ISSUES (default: 0 = all)
#   - DRY_RUN (default: 0; if 1, do not submit batches)
#   - SECRETS_FILE (default: /scratch/sxr203/.secrets/issue_classifier.env)
#   - RUNNER_ROOT (default: /scratch/sxr203/issue_classifier_v7_runner)
#
# Outputs (under RUN_ROOT):
#   - requests/{openai_requests_shard*.jsonl,mapping_shard*.jsonl} (and gemini_requests_shard*.jsonl if enabled)
#   - provenance.json, missing_pages.csv, skipped_issues.csv
#   - openai_batch_submission/batch_jobs.jsonl
#   - requests/submitted_gemini_batches.jsonl (if enabled)
#
# Notes:
#   - This job submits batches but does not download results. Run monitors/downloaders later.

#SBATCH -J issue-cls-v7-submit
#SBATCH -o /scratch/sxr203/newspaper-downloads/dedupe-webp/logs/issue_cls_v7_submit_%j.out
#SBATCH -e /scratch/sxr203/newspaper-downloads/dedupe-webp/logs/issue_cls_v7_submit_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -p cpu_short
#SBATCH -A torch_pr_609_general

set -euo pipefail

RUN_ROOT=${RUN_ROOT:-}
if [[ -z "${RUN_ROOT}" ]]; then
  echo "Missing required env: RUN_ROOT" >&2
  exit 1
fi

UNIQUE_PNG_ROOT=${UNIQUE_PNG_ROOT:-/scratch/sxr203/newspaper-downloads/dedupe-webp/unique_png}
PADDLE_RUN_ROOT=${PADDLE_RUN_ROOT:-/scratch/$USER/paddleocr_vl15/runs}
PADDLE_RUN_PREFIX=${PADDLE_RUN_PREFIX:-layout_bagging_unique_png_sharded_20260220_200103_manifest_}
PADDLE_LAYOUT_VARIANT=${PADDLE_LAYOUT_VARIANT:-pld_v3_thr03}
PADDLE_VL15_VARIANT=${PADDLE_VL15_VARIANT:-pvl15_docparser_v15}
VLM_PAGE_ROOTS=${VLM_PAGE_ROOTS:-}
OPENAI_BOX_RESULTS=${OPENAI_BOX_RESULTS:-/scratch/sxr203/newspaper-downloads/dedupe-webp/batch_requests_dedup_jpeg1024_mp1_q80/part_*/openai_gpt52_reasoning_medium_split/results}
GEMINI_BOX_RESULTS=${GEMINI_BOX_RESULTS:-/scratch/sxr203/newspaper-downloads/dedupe-webp/batch_requests_dedup_jpeg1024_mp1_q80/part_*/results_gemini,/scratch/sxr203/newspaper-downloads/dedupe-webp/batch_requests_openai_no_ok_gemini3_flash3prep_20260121_005716/results_gemini}

OPENAI_MODEL=${OPENAI_MODEL:-gpt-5-mini}
GEMINI_MODEL=${GEMINI_MODEL:-models/gemini-2.5-flash}
CLASSIFIER_PROVIDER=${CLASSIFIER_PROVIDER:-openai}

ISSUES_PER_SHARD=${ISSUES_PER_SHARD:-5000}
MAX_BYTES_PER_SHARD=${MAX_BYTES_PER_SHARD:-180000000}
MISSING_PAGE_POLICY=${MISSING_PAGE_POLICY:-skip}
SOURCE_PRIORITY=${SOURCE_PRIORITY:-paddle_vl15_md,vlm_page,gemini_box_results,openai_box_results}
MAX_ISSUES=${MAX_ISSUES:-0}
DRY_RUN=${DRY_RUN:-0}

SECRETS_FILE=${SECRETS_FILE:-/scratch/sxr203/.secrets/issue_classifier.env}
RUNNER_ROOT=${RUNNER_ROOT:-/scratch/sxr203/issue_classifier_v7_runner}

PROMPT_PATH="${RUNNER_ROOT}/prompts/classification_prompt_v7.txt"

mkdir -p "${RUN_ROOT}"

if [[ ! -f "${SECRETS_FILE}" ]]; then
  echo "Missing secrets file: ${SECRETS_FILE}" >&2
  exit 1
fi

# Load keys into env for subprocesses (do NOT echo values).
set -a
source "${SECRETS_FILE}"
set +a

if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ -z "${OPENAI_KEY:-}" ]]; then
  echo "Missing OpenAI key (expected OPENAI_API_KEY or OPENAI_KEY after sourcing ${SECRETS_FILE})" >&2
  exit 1
fi
if [[ "${CLASSIFIER_PROVIDER}" == "gemini" || "${CLASSIFIER_PROVIDER}" == "both" ]]; then
  if [[ -z "${GEMINI_KEY:-}" ]]; then
    echo "Missing Gemini key (expected GEMINI_KEY after sourcing ${SECRETS_FILE})" >&2
    exit 1
  fi
  # Activate venv that has google-genai (required for Gemini Batch submission).
  VENV_PATH=${VENV_PATH:-${RUNNER_ROOT}/venv_py39}
  if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "Missing runner venv: ${VENV_PATH} (expected google-genai installed on compute nodes)" >&2
    exit 1
  fi
  source "${VENV_PATH}/bin/activate"
fi

echo "RUN_ROOT=${RUN_ROOT}"
echo "UNIQUE_PNG_ROOT=${UNIQUE_PNG_ROOT}"
echo "OPENAI_MODEL=${OPENAI_MODEL}"
echo "GEMINI_MODEL=${GEMINI_MODEL}"

MAX_ISSUES_ARG=()
if [[ "${MAX_ISSUES}" != "0" ]]; then
  MAX_ISSUES_ARG=(--max-issues "${MAX_ISSUES}")
fi

python "${RUNNER_ROOT}/scripts/export_issue_zoning_issue_classifier_batch_requests.py" \
  --output-dir "${RUN_ROOT}" \
  --prompt-path "${PROMPT_PATH}" \
  --provider "${CLASSIFIER_PROVIDER}" \
  --openai-model "${OPENAI_MODEL}" \
  --unique-png-root "${UNIQUE_PNG_ROOT}" \
  --paddle-run-root "${PADDLE_RUN_ROOT}" \
  --paddle-run-prefix "${PADDLE_RUN_PREFIX}" \
  --paddle-layout-variant "${PADDLE_LAYOUT_VARIANT}" \
  --paddle-vl15-variant "${PADDLE_VL15_VARIANT}" \
  --openai-box-results-jsonl "${OPENAI_BOX_RESULTS}" \
  --gemini-box-results-jsonl "${GEMINI_BOX_RESULTS}" \
  --vlm-source-priority "${SOURCE_PRIORITY}" \
  --missing-page-policy "${MISSING_PAGE_POLICY}" \
  --issues-per-shard "${ISSUES_PER_SHARD}" \
  --max-bytes-per-shard "${MAX_BYTES_PER_SHARD}" \
  ${VLM_PAGE_ROOTS:+--vlm-page-roots "${VLM_PAGE_ROOTS}"} \
  "${MAX_ISSUES_ARG[@]}"

REQUEST_DIR="${RUN_ROOT}/requests"
if [[ ! -d "${REQUEST_DIR}" ]]; then
  echo "Missing request dir after export: ${REQUEST_DIR}" >&2
  exit 1
fi

# Pre-flight validation + whitelist (fails fast if anything is inconsistent).
python "${RUNNER_ROOT}/scripts/validate_issue_classifier_export.py" \
  --request-dir "${REQUEST_DIR}" \
  --provider "${CLASSIFIER_PROVIDER}" \
  --write-whitelist \
  --check-pages \
  --max-check-issues 25

# OpenAI Batch submissions (Responses API).
OPENAI_DRY=()
GEMINI_DRY=()
if [[ "${DRY_RUN}" == "1" ]]; then
  OPENAI_DRY=(--dry-run)
  GEMINI_DRY=(--dry-run)
fi

python "${RUNNER_ROOT}/scripts/openai_batch_submit_curl.py" \
  --request-dir "${REQUEST_DIR}" \
  --record-path "${RUN_ROOT}/openai_batch_submission/batch_jobs.jsonl" \
  --env-file "${SECRETS_FILE}" \
  --endpoint /v1/responses \
  --completion-window 24h \
  --run-label issue_classifier_v7 \
  --model-label "${OPENAI_MODEL}" \
  "${OPENAI_DRY[@]}"

if [[ "${CLASSIFIER_PROVIDER}" == "gemini" || "${CLASSIFIER_PROVIDER}" == "both" ]]; then
  # Gemini Batch submissions.
  python "${RUNNER_ROOT}/scripts/gemini_batch_submit.py" \
    --request-dir "${REQUEST_DIR}" \
    --env-file "${SECRETS_FILE}" \
    --model "${GEMINI_MODEL}" \
    --display-name-prefix issue_classifier_v7 \
    "${GEMINI_DRY[@]}"
fi

echo "done"

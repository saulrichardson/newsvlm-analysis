#!/usr/bin/env bash
set -euo pipefail

# Refresh + export ordinance questionnaire results to a single XLSX.
#
# This script is designed to be re-runnable:
#   - It checks OpenAI batch status (via submitted_batches.jsonl).
#   - If completed, it downloads any missing results.
#   - It normalizes results into normalized.jsonl + invalid_rows.jsonl.
#   - It exports a merged XLSX across run #1 (+ run #2 if present).
#
# Usage:
#   bash scripts/refresh_ordinance_questionnaire_excel.sh <OUT_XLSX>
#
# Optional env overrides:
#   QUESTIONS_XLSX=~/Downloads/Questions.xlsx
#   RUN1_REQ_DIR=...
#   RUN1_RES_DIR=...
#   RUN2_REQ_DIR=...
#   RUN2_RES_DIR=...
#   SKIP_RUN2=1  (refresh/export run #1 only)
#
# Auth:
#   Uses scripts/report_openai_batch_status.py and scripts/download_openai_batch_results.py,
#   both configured here to use the "OPENAI_KEY" env var in .env.

OUT_XLSX="${1:-}"
if [[ -z "${OUT_XLSX}" ]]; then
  echo "Usage: $0 <OUT_XLSX>" >&2
  exit 1
fi

QUESTIONS_XLSX="${QUESTIONS_XLSX:-${HOME}/Downloads/Questions.xlsx}"

RUN1_REQ_DIR="${RUN1_REQ_DIR:-}"
RUN1_RES_DIR="${RUN1_RES_DIR:-}"
if [[ -z "${RUN1_REQ_DIR}" || -z "${RUN1_RES_DIR}" ]]; then
  echo "Missing required env vars:" >&2
  echo "  RUN1_REQ_DIR=/path/to/request_dir" >&2
  echo "  RUN1_RES_DIR=/path/to/results_dir" >&2
  exit 1
fi

RUN2_REQ_DIR="${RUN2_REQ_DIR:-}"
RUN2_RES_DIR="${RUN2_RES_DIR:-}"

SKIP_RUN2="${SKIP_RUN2:-0}"

# Pass-through so downstream scripts can be configured consistently via env.
export QUESTIONS_XLSX RUN1_REQ_DIR RUN1_RES_DIR RUN2_REQ_DIR RUN2_RES_DIR SKIP_RUN2

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

ensure_normalized() {
  local req_dir="$1"
  local res_dir="$2"
  local run_label="$3"

  if [[ ! -d "${req_dir}" ]]; then
    echo "Missing request dir for ${run_label}: ${req_dir}" >&2
    exit 1
  fi

  mkdir -p "${res_dir}"

  local norm_path="${res_dir}/normalized.jsonl"
  local bad_path="${res_dir}/invalid_rows.jsonl"

  if [[ -f "${norm_path}" && -f "${bad_path}" ]]; then
    echo "[${run_label}] normalized present; skipping download/normalize" >&2
    return 0
  fi

  local submitted_record="${req_dir}/submitted_batches.jsonl"
  require_file "${submitted_record}" "${run_label} submitted record"

  echo "[${run_label}] checking batch status..." >&2
  status_out="$(
    python scripts/report_openai_batch_status.py \
      --submitted-record "${submitted_record}" \
      --openai-key-mode openai_only
  )"
  echo "${status_out}" >&2

  python - <<'PY' <<<"${status_out}"
import json
import sys

lines = [ln.rstrip("\n") for ln in sys.stdin.read().splitlines() if ln.strip()]
status_line = next((ln for ln in lines if ln.startswith("status_counts\t")), None)
if status_line is None:
    print("refresh: could not parse status_counts from report_openai_batch_status.py output", file=sys.stderr)
    sys.exit(2)

counts = json.loads(status_line.split("\t", 1)[1])
bad = {k: v for k, v in counts.items() if k != "completed" and int(v) > 0}
if bad:
    print(f"refresh: batch is not completed yet (status_counts={counts})", file=sys.stderr)
    print("refresh: wait for completion, then re-run this command.", file=sys.stderr)
    sys.exit(3)
PY

  echo "[${run_label}] downloading results (if missing)..." >&2
  python scripts/download_openai_batch_results.py \
    --request-dir "${req_dir}" \
    --out-dir "${res_dir}" \
    --openai-key-mode openai_only

  echo "[${run_label}] normalizing to normalized.jsonl..." >&2
  python scripts/normalize_ordinance_questionnaire_openai_batch_results.py \
    --request-dir "${req_dir}" \
    --results-dir "${res_dir}" \
    --questions-xlsx "${QUESTIONS_XLSX}"

  require_file "${norm_path}" "${run_label} normalized.jsonl"
  require_file "${bad_path}" "${run_label} invalid_rows.jsonl"
}

ensure_normalized "${RUN1_REQ_DIR}" "${RUN1_RES_DIR}" "run1"

if [[ "${SKIP_RUN2}" != "1" ]]; then
  if [[ -d "${RUN2_REQ_DIR}" ]]; then
    ensure_normalized "${RUN2_REQ_DIR}" "${RUN2_RES_DIR}" "run2"
  else
    echo "[run2] request dir not found; exporting run1 only" >&2
    SKIP_RUN2="1"
  fi
fi

echo "[excel] exporting workbook..." >&2
PYTHONUNBUFFERED=1 bash scripts/export_ordinance_questionnaire_answers_to_excel.sh "${OUT_XLSX}"

echo "[done] wrote ${OUT_XLSX}" >&2

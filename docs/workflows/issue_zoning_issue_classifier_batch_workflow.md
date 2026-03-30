# Issue Zoning Classifier (Issue-Level) — Batch + Gateway Workflow

This repo’s “issue zoning classifier” prompt is designed to run **once per issue** (newspaper slug + date), using the
concatenation of **all page transcripts** for that issue.

We run it in two execution modes:

1. **OpenAI Batch API** (true async batch; upload JSONL → poll → download JSONL)
2. **Gemini Batch API** (true async batch; upload JSONL → poll → download JSONL)
3. **vendor/agent-gateway runner** (fast synchronous “pseudo-batch”; concurrency over live provider APIs)

The request JSONL format is the same in both cases: **OpenAI Batch–shaped** `POST /v1/responses` lines.

## 0) Keys / env

This repo’s scripts read provider keys from `.env` (gitignored). The minimum you’ll want:

- `OPENAI_API_KEY` (or `OPENAI_KEY`) for OpenAI Batch and/or gateway OpenAI runs
- `GEMINI_KEY` for gateway Gemini runs

## 1) Export issue-level request shards (packaging step)

Script:

- `scripts/pipelines/export_issue_zoning_issue_classifier_batch_requests.py`

This exporter:

- takes a canonical page list from `unique_png/*.png` (Torch: `/scratch/<user>/newspaper-downloads/dedupe-webp/unique_png/*.png`)
- chooses the best available transcript per page using a **source priority list**
- concatenates page text into **issue transcripts**
- writes:
  - `requests/openai_requests_shardNNN.jsonl`
  - `requests/gemini_requests_shardNNN.jsonl` (optional)
  - `requests/mapping_shardNNN.jsonl`
  - provenance CSV/JSON for missing pages / skipped issues

### Typical Torch “full run” inputs

- Paddle VL1.5 Markdown export JSONL (inode-friendly): from `newspaper-parsing/scripts/export_vl15_markdown_jsonl.py`
- Gemini per-page transcripts: `.../vlm_out_gemini/*.vlm.json`
- OpenAI box-level results: `.../openai_results_shard*.jsonl`
- (Optional) extra Gemini box-level results: `.../gemini_results_shard*.jsonl`

Example:

```bash
python scripts/pipelines/export_issue_zoning_issue_classifier_batch_requests.py \
  --output-dir /scratch/$USER/issue_zoning_classifier_run_$(date +%Y%m%d_%H%M%S) \
  --prompt-path prompts/pipelines/zoning_issue_classifier_prompt_v5_issue_schema_city_state_v4_3_strict_json_single_line.txt \
  --provider both \
  --openai-model gpt-5-mini \
  --unique-png-root /scratch/$USER/newspaper-downloads/dedupe-webp/unique_png \
  --paddle-vl15-jsonl /scratch/$USER/paddleocr_vl15/exports/vl15_markdown.jsonl.gz \
  --vlm-page-roots '/scratch/sxr203/newspaper-downloads/dedupe-webp/batch_requests_dedup_jpeg1024_mp1_q80/part_*/vlm_out_gemini' \
  --openai-box-results-jsonl '/scratch/sxr203/newspaper-downloads/dedupe-webp/batch_requests_dedup_jpeg1024_mp1_q80/part_*/openai_gpt52_reasoning_medium_split/results' \
  --gemini-box-results-jsonl /scratch/sxr203/newspaper-downloads/dedupe-webp/batch_requests_openai_no_ok_gemini3_flash3prep_20260121_005716/results_gemini \
  --vlm-source-priority paddle_vl15_md,vlm_page,gemini_box_results,openai_box_results \
  --missing-page-policy skip \
  --max-bytes-per-shard 180000000
```

Notes:

- `--openai-box-results-jsonl` and `--gemini-box-results-jsonl` accept either:
  - a comma-separated list of files, or
  - a directory (the exporter will glob common shard names inside it).
- `--vlm-page-roots` accepts comma-separated directories and supports simple globs (quote the argument to prevent your shell from expanding it too early).
- `--missing-page-policy=require` is safest, but will fail fast if any page in an issue is missing transcript text.
- `--missing-page-policy=skip` is usually the pragmatic option for large runs.
- `--max-bytes-per-shard` helps avoid provider file-size limits (recommended for OpenAI Batch).

## 2A) Execute via OpenAI Batch API (async)

1) Submit shards:

```bash
python scripts/platform/openai_batch_submit_curl.py \
  --request-dir /path/to/run/requests \
  --env-file /path/to/repo/.env \
  --endpoint /v1/responses
```

2) Poll + download outputs:

```bash
python scripts/platform/openai_batch_monitor_curl.py \
  --record-path /path/to/run/openai_batch_submission/batch_jobs_<timestamp>.jsonl \
  --env-file /path/to/repo/.env \
  --stop-when-final
```

When completed, you’ll have downloaded `*_output.jsonl` (and possibly `*_error.jsonl`).

## 2B) Execute via Gemini Batch API (async)

This requires `requests/gemini_requests_shardNNN.jsonl` (export with `--provider gemini` or `--provider both`).

Submit shards (creates Gemini Batch jobs and records them in an append-only JSONL):

```bash
python scripts/platform/gemini_batch_submit.py \
  --request-dir /path/to/run/requests \
  --env-file /path/to/repo/.env \
  --model models/gemini-2.5-flash \
  --display-name-prefix issue-zoning
```

Outputs:

- `requests/submitted_gemini_batches.jsonl`

Downloading Gemini batch outputs is a separate step (not covered in this doc yet).

## 2C) Execute via vendor/agent-gateway (sync, concurrent)

This runs the already-exported request JSONLs over live APIs through the local `vendor/agent-gateway` submodule.

```bash
python scripts/platform/run_openai_requests_via_gateway.py \
  --request-dir /path/to/run/requests \
  --output-dir /path/to/run/results_gemini \
  --model gemini:gemini-2.5-flash \
  --max-concurrency 4 \
  --timeout 240
```

Outputs:

- `openai_results_shardNNN.jsonl`
- `openai_errors_shardNNN.jsonl`

## 3) Rehydrate results into parsed issue outputs

Script:

- `scripts/pipelines/rehydrate_issue_zoning_issue_classifier_results.py`

Gateway example:

```bash
python scripts/pipelines/rehydrate_issue_zoning_issue_classifier_results.py \
  --request-dir /path/to/run/requests \
  --results-dir /path/to/run/results_gemini \
  --output-dir /path/to/run/outputs_gemini
```

OpenAI Batch example:

```bash
python scripts/pipelines/rehydrate_issue_zoning_issue_classifier_results.py \
  --request-dir /path/to/run/requests \
  --results-dir /path/to/run/openai_batch_submission/completed_outputs \
  --output-dir /path/to/run/outputs_openai_batch
```

Key outputs:

- `issue_zoning_parsed_outputs.jsonl` (one line per issue; includes parsed JSON + mapping provenance)
- `manifest.jsonl` (one line per issue; includes `output_path` and status)
- `*.issue_zoning.json` (optional per-issue file; on by default)

## 4) Smoke test (recommended after key updates)

Run the end-to-end smoke test (gateway OpenAI + gateway Gemini + OpenAI Batch):

```bash
python scripts/pipelines/run_issue_zoning_issue_classifier_smoke.py
```

This writes a timestamped run directory under `artifacts/runs/` and fails loudly if any step is broken.

# Post-Gate Multimodal Transcription

This is the repo path for producing a clean zoning-ordinance transcription after an upstream gate has already decided an issue is likely relevant.

The production-scale design is:

- one request per gated issue
- full issue OCR transcript in the prompt
- one attached image per selected issue page, in explicit order
- gateway live execution from many independent workers
- no `max_output_tokens` cap in the generated request body
- long per-request timeout defaults for whole-issue multimodal runs

The implemented default is:

- full issue OCR transcript for context
- selected issue page images for visual repair
- image-first reconstruction prompt
- strict output contract:
  - `SOURCE`
  - `=== ORDINANCE TEXT ===`
  - `--- NOTES ---`

## Why this path

Grounded local evidence already points in this direction:

- [postgate_operative_transcription_models_20260306.md](/Users/saulrichardson/projects/newspapers/newspaper-analysis/artifacts/reports/postgate_operative_transcription_models_20260306.md)
  - shows that the post-gate image-backed prompt materially improves reconstruction quality on positive ordinance pages
- [issue_image_multimodal_pilot_20260306.md](/Users/saulrichardson/projects/newspapers/newspaper-analysis/artifacts/reports/issue_image_multimodal_pilot_20260306.md)
  - shows that full issue OCR plus issue page images can recover cleaner multi-page ordinance text than OCR-only

The production prompt is:

- [transcription_v13_issue_selected_images_postgate.txt](/Users/saulrichardson/projects/newspapers/newspaper-analysis/prompts/pipelines/transcription_v13_issue_selected_images_postgate.txt)

The runner is:

- [run_postgate_issue_transcription.py](/Users/saulrichardson/projects/newspapers/newspaper-analysis/scripts/pipelines/run_postgate_issue_transcription.py)

## Input shape

The runner always needs:

- either an `issue_txt_dir` containing `<issue_id>.txt`
- or explicit `transcript_path` values in the manifest

It can optionally take a manifest to restrict which issues and pages are attached as images, or to provide direct image paths.

Supported manifest formats:

1. `CSV`

Columns:

- `issue_id`
- `request_id` optional
- `transcript_path` optional if `issue_txt_dir` is provided
- `page_ids` optional, using `|` or `,` separators
- `image_paths` optional, using `|` separators

Example:

```csv
request_id,issue_id,transcript_path,page_ids,image_paths
oxnard-scale-smoke,oxnard-press-courier__1968-05-24,/abs/path/oxnard-press-courier__1968-05-24.txt,oxnard-press-courier-may-24-1968-p-43|oxnard-press-courier-may-24-1968-p-46,/abs/path/oxnard-press-courier-may-24-1968-p-43.png|/abs/path/oxnard-press-courier-may-24-1968-p-46.png
```

2. `JSONL`

Each line may include:

- `request_id`
- `issue_id`
- `transcript_path`
- `page_ids` as a list or string
- `image_paths` as a list or string
- `images` as a list of strings or objects with `image_path` and optional `page_id`

3. plain `TXT`

One `issue_id` per line. In this mode the runner attaches all pages found in the issue transcript.

If no manifest is provided, the runner transcribes every `*.txt` file in `issue_txt_dir`.

## Image resolution

The runner supports two image sources:

1. direct `image_paths` in the manifest
2. local image root via `--local-png-root`
3. Torch remote root via `--torch-host` and `--torch-png-root`

Resolved images are cached under the run directory in `images/<issue_id>/`.

If `image_paths` are provided directly, the runner uses those files as-is and does not clip, recompress, or transform them.

## Request contract

The generated `requests/openai_requests_shard000.jsonl` lines intentionally:

- include the full issue OCR transcript inside the prompt text
- include one full base64 data URL per attached issue image
- preserve image ordering by inserting an alignment text block before each image
- omit `max_output_tokens`

The request body keys are expected to be:

- `model`
- `input`
- `reasoning` when requested
- `stream`

There should be no `max_output_tokens` key unless someone adds it explicitly in code.

## Timeout contract

Whole-issue multimodal requests are slow. The repo is now configured so this path defaults to a long timeout:

- [run_postgate_issue_transcription.py](/Users/saulrichardson/projects/newspapers/newspaper-analysis/scripts/pipelines/run_postgate_issue_transcription.py): `--timeout 21600`
- [prepare_postgate_issue_transcription_gateway_run.py](/Users/saulrichardson/projects/newspapers/newspaper-analysis/scripts/pipelines/prepare_postgate_issue_transcription_gateway_run.py): records `timeout=21600` by default
- [run_openai_requests_via_gateway.py](/Users/saulrichardson/projects/newspapers/newspaper-analysis/scripts/platform/run_openai_requests_via_gateway.py): `--timeout 21600` by default and propagates that into `GATEWAY_TIMEOUT_SECONDS`

This does not guarantee a provider never stalls, but it removes the avoidable short local timeout failure mode from this repo path.

## Typical usage

Dry run to build requests and verify page/image resolution:

```bash
python scripts/pipelines/run_postgate_issue_transcription.py \
  --issue-txt-dir /path/to/issue_txt \
  --manifest-path /path/to/gated_issues.csv \
  --local-png-root /path/to/png_root \
  --gateway-model openai:gpt-5.4 \
  --request-model gpt-5.4 \
  --dry-run
```

Live run through the gateway:

```bash
python scripts/pipelines/run_postgate_issue_transcription.py \
  --issue-txt-dir /path/to/issue_txt \
  --manifest-path /path/to/gated_issues.csv \
  --torch-host torch \
  --torch-png-root /scratch/sxr203/newspaper-downloads/dedupe-webp/unique_png \
  --gateway-model openai:gpt-5.4 \
  --request-model gpt-5.4 \
  --reasoning-effort high
```

Prepared scale-out run with direct transcript and image paths:

```bash
python scripts/pipelines/prepare_postgate_issue_transcription_gateway_run.py \
  --manifest-path /path/to/postgate_manifest.jsonl \
  --output-dir /scratch/sxr203/postgate_issue_transcription_gateway_20260306 \
  --worker-count 100 \
  --request-model gemini-3.1-pro-preview \
  --gateway-model gemini:gemini-3.1-pro-preview \
  --reasoning-effort high \
  --timeout 21600
```

Then run the prepared worker manifests through Slurm:

```bash
PREPARED_RUN_ROOT=/scratch/sxr203/postgate_issue_transcription_gateway_20260306 \
sbatch --array=0-99 slurm/pipelines/run_postgate_issue_transcription_gateway_array.sbatch
```

Each worker process starts its own local gateway port and sends requests from its worker manifest through that gateway.

By default the Slurm wrapper now picks a free localhost port per worker. That avoids the fragile fixed-port assumption when many workers or multiple jobs land on the same node. If deterministic ports are needed for debugging, set `PORT_BASE` explicitly.

## Outputs

Each run writes:

- `build_manifest.csv`
- `prompt_used.txt`
- `requests/openai_requests_shard000.jsonl`
- `results/openai_results_shard000.jsonl`
- `rendered_outputs/raw/<issue_id>.txt`
- `rendered_outputs/ordinance_text/<issue_id>.txt`
- `rendered_outputs/notes/<issue_id>.txt`
- `transcriptions.csv`
- `transcriptions.jsonl`
- `summary.json`

## Design choice

This runner uses the full issue transcript plus a selected set of issue page images, rather than requiring every page image from the issue by default.

That is a deliberate payload-control choice:

- the OCR transcript still gives issue-wide context
- the attached images give the visual evidence needed for cleanup and repair
- upstream gating can decide which page images are worth attaching

If `page_ids` are omitted, the runner falls back to attaching all pages present in the issue transcript.

For production scale-out, the manifest can instead provide explicit `image_paths` for every page the caller wants attached. That is the intended interface for Torch Slurm runs where upstream selection has already happened.

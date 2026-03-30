# `newspaper-analysis`

This repository contains **analysis workflows** built on top of the upstream extraction artifacts produced by the
`newsvlm` engine repo (OCR `*.vlm.json`, zoning labels, issue-level outputs, stitched ordinance docs, etc.).

It vendors the `agent-gateway` repo as an optional git submodule under `vendor/agent-gateway/` so that scripts can run
many concurrent LLM requests locally without depending on a separate checkout.

If you need to make new extraction outputs, do that in the engine repo first, then point these scripts at the resulting
artifacts.

## What lives here

- `src/newsvlm_analysis/frontier/`: active modular analysis code
- `scripts/frontier/`: frontier entrypoints and report builders
- `scripts/pipelines/`: active issue-classifier, transcription, and recovery workflows
- `scripts/platform/`: gateway and batch execution utilities
- `prompts/frontier/`: frontier prompt bundles
- `prompts/pipelines/`: active workflow prompts
- `docs/workflows/`: current workflow documentation
- `reports/curated/`: commit-worthy report bundles
- `artifacts/`: local run outputs, scratch work, and generated reports
- `archive/legacy/`: quarantined legacy workflows, docs, prompts, and reports
- `vendor/agent-gateway/`: optional gateway submodule

## Repository organization framework

The repo now uses a strict split between active code, curated outputs, and local artifacts:

- `reports/curated/`: curated report bundles kept in git.
- `artifacts/runs/`: local run roots. Ignored by git except `artifacts/runs/README.md`.
- `artifacts/scratch/`: local one-off experiments and temporary files. Ignored by git except `artifacts/scratch/README.md`.
- `artifacts/reports/`: generated or exploratory report material not meant for version control.
- `archive/legacy/`: older flat workflows kept out of the active surface area.

Rule of thumb:
- Commit active code, workflow docs, prompts, and curated reports.
- Do not commit raw batch outputs, temporary run roots, ad-hoc scratch experiments, or generated report dumps.

## Setup (local dev)

1. Create a virtualenv:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Initialize submodules (optional but recommended for LLM-backed workflows):
   ```bash
   git submodule update --init --recursive
   ```
3. Install the engine package (local path or a pinned git ref):
   ```bash
   # local sibling checkout
   pip install -e ../newspaper-parsing
   ```
4. Install analysis package + dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. (Optional) Install the vendored gateway into the same env (needed for `scripts/platform/run_openai_requests_via_gateway.py` and other LLM-backed workflows):
   ```bash
   pip install -e vendor/agent-gateway
   ```

## Notes on reproducibility

For any run that generates downstream datasets or reports, record:
- the `newsvlm` engine commit/tag used to generate inputs
- the exact prompt files used
- the source manifest(s) and run roots on VAST/Greene
- the corresponding local `artifacts/runs/...` or `artifacts/reports/...` output location

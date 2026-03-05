# `newspaper-analysis`

This repository contains **analysis workflows** built on top of the upstream extraction artifacts produced by the
`newsvlm` engine repo (OCR `*.vlm.json`, zoning labels, issue-level outputs, stitched ordinance docs, etc.).

It includes the `agent-gateway` repo as an (optional) git submodule (`agent-gateway/`) so that scripts can easily run
**many concurrent LLM requests** locally without depending on a separate checkout.

If you need to make new extraction outputs, do that in the engine repo first, then point these scripts at the resulting
artifacts.

## What lives here

- `docs/`: workflow documentation (topic discovery, hybrid regulatory topics, etc.)
- `scripts/`: clustering, plotting, report builders, RHS construction, etc.
- `prompts/`: analysis-only prompts (e.g. cluster labeling / instrument labeling)
- `slurm/`: HPC wrappers for analysis runs
- `reports/`: report outputs and report templates

## Repository organization framework

The repo uses a strict split between commit-worthy artifacts and local run outputs:

- `reports/hybrid_regulatory_topics_report`, `reports/issue_topics_report`, `reports/rhs_v4_full_writeup`: curated report bundles kept in git.
- `reports/runs/`: local run artifacts (large/intermediate outputs). Ignored by git except `reports/runs/README.md`.
- `scratch/`: local one-off experiments and temporary files. Ignored by git except `scratch/README.md`.

Rule of thumb:
- Commit code, docs, prompts, and curated reports.
- Do not commit raw batch outputs, temporary run roots, or ad-hoc scratch experiments.

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
5. (Optional) Install `agent-gateway` into the same env (needed for `scripts/run_openai_requests_via_gateway.py` and other LLM-backed workflows):
   ```bash
   pip install -e agent-gateway
   ```

## Notes on reproducibility

For any run that generates downstream datasets or reports, record:
- the `newsvlm` engine commit/tag used to generate inputs
- the exact prompt files used
- the source manifest(s) and run roots on VAST/Greene

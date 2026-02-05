# `newspaper-analysis`

This repository contains **analysis workflows** built on top of the upstream extraction artifacts produced by the
`newsvlm` engine repo (OCR `*.vlm.json`, zoning labels, issue-level outputs, stitched ordinance docs, etc.).

It intentionally **does not** vendor or run the `agent-gateway` submodule. If you need to make new extraction outputs,
do that in the engine repo first, then point these scripts at the resulting artifacts.

## What lives here

- `docs/`: workflow documentation (topic discovery, hybrid regulatory topics, etc.)
- `scripts/`: clustering, plotting, report builders, RHS construction, etc.
- `prompts/`: analysis-only prompts (e.g. cluster labeling / instrument labeling)
- `slurm/`: HPC wrappers for analysis runs
- `reports/`: example report sources + bundles (recommended: commit sources, ignore build artifacts)

## Setup (local dev)

1. Create a virtualenv:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install the engine package (local path or a pinned git ref):
   ```bash
   # local sibling checkout
   pip install -e ../newspaper-parsing
   ```
3. Install analysis package + dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Notes on reproducibility

For any run that generates downstream datasets or reports, record:
- the `newsvlm` engine commit/tag used to generate inputs
- the exact prompt files used
- the source manifest(s) and run roots on VAST/Greene

# Scripts Layout

Scripts are organized by naming convention so workflows are easy to locate:

- `run_*`: end-to-end pipeline entrypoints.
- `build_*`: table/figure/model/report builders.
- `export_*`: request exporters for external/LLM batch jobs.
- `rehydrate_*`: parsers for returned batch results.
- `compute_*`, `analyze_*`, `evaluate_*`, `compare_*`: analytic transformations and diagnostics.
- `plot_*`, `render_*`, `summarize_*`: reporting and visualization utilities.

Notable entrypoints:

- `scripts/extract_zoning_text_from_panels.py`: build zoning-only corpora from full newspaper panel transcripts (rules + optional LLM hybrid).
  - Supports an `llm_only` mode (no deterministic candidate filtering): the LLM selects which transcript blocks to keep, then the script stitches those original blocks into a zoning-only corpus.
- `scripts/run_openai_requests_via_gateway.py`: run `openai_requests_shard*.jsonl` locally through `agent-gateway/` with configurable concurrency (fast iteration without provider Batch APIs).
- `scripts/build_geo_backbone_crosswalk.py`: construct a robust Census geography backbone for each city panel (place/tract/county/PUMA/ZCTA/UA/CBSA plus 2010↔2020 crosswalks with weights).

Recommended workflow:

1. Start with a `run_*` entrypoint when available.
2. Use `build_*` scripts to materialize specific outputs.
3. Use `render_*`/`build_*_latex_report.py` for publication-ready writeups.

Conventions:

- All scripts should support `--help`.
- New scripts should avoid hardcoded local paths; prefer explicit CLI arguments.
- Large generated outputs should be written to `reports/runs/` (gitignored).

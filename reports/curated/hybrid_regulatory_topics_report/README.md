# Hybrid regulatory topic discovery report (LaTeX)

This folder contains a small LaTeX writeup that documents the PI-motivated
shift from **chunk-level clustering** to a **hybrid clause-level instrument discovery**
pipeline.

## Build

```bash
cd reports/hybrid_regulatory_topics_report
make
```

This produces `report.pdf`.

## Regenerate tables/figures from a hybrid run

The report is meant to be reproducible from a concrete hybrid run directory
(e.g. `tmp_hybrid_pi_real_1/`):

```bash
python scripts/build_hybrid_regulatory_topics_latex_report.py \
  --hybrid-run-dir tmp_hybrid_pi_real_1 \
  --output-dir reports/hybrid_regulatory_topics_report
```

Then rebuild the PDF:

```bash
cd reports/hybrid_regulatory_topics_report
make
```

## What gets pulled in

The builder script tries to include (when present):
- section segmentation summaries (`sections_out/sections.jsonl`)
- chunk motive labels (`motive_out/chunk_motives.jsonl`)
- extracted clauses (`clause_out/clauses.jsonl` or `clause_out_salvage/clauses.jsonl`)
- clause cluster assignments + UMAP scatter (`clause_clusters_*/clusters.jsonl`, `scatter.png`)
- LLM instrument labels (`instrument_label_out/cluster_labels.jsonl`)
- document-level purposes (`purpose_out/doc_purposes.jsonl`)

It also optionally includes one baseline artifact from the prior chunk-topic pipeline:
`reports/issue_topics_report/figures/full_ordinance__umap_scatter.png`.


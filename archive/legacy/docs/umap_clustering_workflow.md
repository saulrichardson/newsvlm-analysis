# UMAP + Clustering Workflow (Newspaper Text Exploration)

This repo already contains a full “topic discovery” pipeline built around:

- **Documents**: newspaper OCR transcripts (page → issue → issue-chunks)
- **Embeddings**: provider embeddings via Batch APIs (OpenAI embeddings)
- **Discovery**: **UMAP → HDBSCAN** clustering (“tweet topics” style)
- **Interpretation**: cluster labeling (LLM) + frequency-over-time plots + a LaTeX report

It also now includes a **no-API local helper** to do quick exploratory clustering directly from
`*.vlm.json` transcripts using **TF‑IDF → SVD → UMAP → HDBSCAN**.

## Mental Model

There are three natural “document levels” you can cluster:

1. **Page-level** (each `*.vlm.json` is a document)
2. **Issue-level** (concatenate all pages from the same issue)
3. **Chunk-level** (split an issue’s text into fixed-size chunks; cluster chunks, not whole issues)

The “issue topic” work in this repo is primarily **chunk-level clustering**, because a single issue often mixes topics.

## Existing Issue-Topic Pipeline (Batch Embeddings)

This is the pipeline that produced the bundled example report under `reports/issue_topics_report/`.

### 0) Inputs you need

- Per-page OCR outputs: `*.vlm.json` files (from the VLM OCR pipeline).
- Issue zoning classifier outputs + manifest (so you can filter to issues with zoning text):
  - The issue-level outputs are `*.issue_zoning.json`
  - The manifest is `manifest.jsonl` (one row per issue output)

The key utility used across the pipeline to extract page transcript text is:

- `src/newsvlm/issue_zoning_classifier.py` → `load_page_result()` + `page_text_from_boxes()`

### 1) Build “issue documents” and export embedding Batch requests

- Script: `scripts/export_issue_topic_embedding_batch_requests.py`
- Inputs: an issue-zoning `manifest.jsonl` and (referenced) per-page `*.vlm.json` files
- Outputs (request_dir):
  - `openai_requests_shardNNN.jsonl` (POST `/v1/embeddings`)
  - `mapping_shardNNN.jsonl` (provenance: which issue/chunk each request corresponds to)

On Greene/VAST this is usually driven via:

- `slurm/export_and_submit_openai_issue_topic_embeddings.sbatch`

### 2) Submit requests, then download results

- Submit: `scripts/submit_batch_shards.py` (OpenAI provider, endpoint `/v1/embeddings`)
- Download: `scripts/download_openai_batch_results.py`

### 3) Rehydrate results into dense embedding matrices

There are two rehydration scripts depending on what you want to cluster:

- **Chunk-level embeddings** (recommended for topic discovery):
  - `scripts/rehydrate_issue_topic_chunk_embeddings_openai_batch_results.py`
  - Output:
    - `chunk_ids.txt`
    - `chunk_embeddings.npy`
    - `chunk_metadata.jsonl`

- **Issue-level embeddings** (single averaged vector per issue):
  - `scripts/rehydrate_issue_topic_embeddings_openai_batch_results.py`
  - Output:
    - `issue_ids.txt`
    - `issue_embeddings.npy`
    - `issue_metadata.jsonl`

### 4) Cluster embeddings with UMAP → HDBSCAN

Scripts:

- Chunk-level: `scripts/cluster_chunk_topic_embeddings.py`
- Issue-level: `scripts/cluster_issue_topic_embeddings.py`

Each writes:

- `clusters.jsonl` (cluster assignment per row)
- `cluster_summary.json`
- `umap_2d.npy` + `scatter.png` (sanity check)

On Greene/VAST, the chunk-level path is usually driven via:

- `slurm/cluster_chunk_topic_embeddings.sbatch`

### 5) Label clusters (optional but recommended)

This is the “make clusters human-readable” step.

1. Export labeler prompts as OpenAI Batch requests:
   - `scripts/export_cluster_topic_labeling_batch_requests.py`
   - Uses:
     - `clusters_chunks/clusters.jsonl` (to pick representative examples)
     - `requests/openai_requests_shard*.jsonl` (to recover exact chunk text for examples)
   - Output:
     - `cluster_topic_labels/requests/openai_requests_shard*.jsonl` (POST `/v1/responses`)
     - `cluster_topic_labels/requests/mapping_shard*.jsonl`

2. Submit + download results:
   - `scripts/submit_batch_shards.py` (endpoint `/v1/responses`)
   - `scripts/download_openai_batch_results.py`

3. Rehydrate validated labels:
   - `scripts/rehydrate_cluster_topic_labels_openai_batch_results.py`
   - Output:
     - `cluster_topic_labels/outputs/cluster_labels.jsonl`

Prompt text lives at:

- `prompts/cluster_topic_labeler_prompt_text.txt`

### 6) Plot cluster frequencies over time

- Script: `scripts/plot_cluster_frequencies_over_time.py`
- Input: `clusters.jsonl` (must include `issue_date`)
- Output: `cluster_counts_by_{month|year}.csv` + `cluster_frequency_{month|year}.png`

Chunk-level runs often use `--count-mode weight --weight-field doc_weight` to avoid overweighting multi-chunk issues.

### 7) Build a LaTeX report across multiple runs

- Script: `scripts/build_issue_topics_latex_report.py`
- Output bundle: `reports/issue_topics_report/` (tables + figures + `report.tex`)

The example `reports/issue_topics_report/provenance.json` shows the VAST run roots used to build the bundled PDF.

## Local No-API Exploration (Quick UMAP + Clustering)

If you want to “poke around” in transcripts without doing any provider embedding work, use:

- `scripts/explore_vlm_text_umap_clusters.py`

It runs:

**TF‑IDF → TruncatedSVD → UMAP → HDBSCAN**

### Example: page-level clustering (sample N pages)

```bash
python scripts/explore_vlm_text_umap_clusters.py \
  --pages "/path/to/vlm_out/*.vlm.json" \
  --output-dir "/path/to/umap_explore_pages_smoke" \
  --doc-level page \
  --max-docs 2000 \
  --sample-mode random \
  --sample-seed 0 \
  --min-chars 1200
```

### Example: issue-level clustering (group pages into issues)

```bash
python scripts/explore_vlm_text_umap_clusters.py \
  --pages "/path/to/vlm_out/*.vlm.json" \
  --output-dir "/path/to/umap_explore_issues_smoke" \
  --doc-level issue \
  --min-chars 5000
```

### Outputs to inspect

- `scatter.png`: quick sanity check (colored by `cluster_id`, noise is `-1`)
- `cluster_keywords.jsonl`: top TF‑IDF terms per cluster (fast way to get a feel for what a cluster is)
- `clusters.jsonl`: per-doc cluster assignments (join with `docs.jsonl` via `doc_id`)
- `cluster_summary.json`: parameter snapshot for reproducibility

## Dependency Note

The clustering scripts require:

- `umap-learn` (imported as `umap`)
- `hdbscan`

They’re now listed in `requirements.txt`.

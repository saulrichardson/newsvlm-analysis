# Hybrid Regulatory Topics Workflow (PI feedback)

Problem statement (PI)
----------------------
The pure clustering approach is interesting, but it does not surface regulatory topics/questions at a sufficiently granular level.

This workflow implements a hybrid approach:

1) **Chunk/section** the document
2) Ask an LLM for the **primary regulatory motive** per chunk/section (controlled taxonomy)
3) Ask an LLM to **extract atomic clauses/requirements** per chunk/section
4) **Embed + cluster** clauses to discover granular instruments
5) Ask an LLM to **label clause clusters** into human-readable regulatory instruments

Key implementation choices
--------------------------
- We avoid brittle heading regexes by supporting **LLM-driven section segmentation** via **line-number ranges** (optional).
- We keep the core semantic steps (motive labeling + clause extraction + cluster labeling) **LLM-driven**.
- We support both:
  - **Local no-API clustering** for fast iteration (TF‑IDF → SVD → UMAP → HDBSCAN)
  - **Provider embeddings** (OpenAI embeddings) for higher-fidelity clustering at scale


## Step 0: Inputs

You need chunk/section text in a JSONL with at least:

```json
{"chunk_id":"...","doc_id":"...","text":"..."}
```

Two common ways to get that:

### Option A) Reuse existing issue-topic embedding request shards (recommended)

If you already ran `scripts/export_issue_topic_embedding_batch_requests.py`, you already have chunk text embedded in:
`<RUN_ROOT>/requests/openai_requests_shard*.jsonl`.

Extract that into a `chunks.jsonl`:

```bash
python scripts/extract_chunks_from_openai_embedding_requests.py \
  --request-dir /path/to/RUN_ROOT/requests \
  --mapping-dir /path/to/RUN_ROOT/requests \
  --output /path/to/RUN_ROOT/hybrid_chunks.jsonl
```

### Option B) Start from a docs JSONL and do mechanical chunking

If you have a docs JSONL like:
```json
{"doc_id":"...","text":"..."}
```

```bash
python scripts/write_text_chunks_jsonl.py \
  --input /path/to/docs.jsonl \
  --output /path/to/chunks.jsonl \
  --id-field doc_id \
  --text-field text \
  --chunk-size-chars 8000 \
  --copy-other-fields
```


## Step 1 (optional but PI-recommended): LLM-driven section segmentation

If fixed-size chunks are still too incoherent, segment into sections/subsections first.

Export section segmentation requests:

```bash
python scripts/export_regulatory_section_segmentation_batch_requests.py \
  --input-jsonl /path/to/docs.jsonl \
  --output-dir /path/to/section_requests \
  --openai-model gpt-5-nano \
  --wrap-width 120 \
  --max-doc-chars 120000 \
  --max-lines 2500
```

Run the resulting `openai_requests_shard*.jsonl` through OpenAI Batch (or locally through `scripts/run_openai_requests_via_gateway.py` if you have a configured gateway).

Then materialize sections as chunks:

```bash
python scripts/rehydrate_regulatory_section_segmentation_openai_batch_results.py \
  --request-dir /path/to/section_requests \
  --results-dir /path/to/section_results \
  --input-jsonl /path/to/docs.jsonl \
  --output-dir /path/to/sections_out
```

The output `sections.jsonl` can be used anywhere `chunks.jsonl` is expected.


## Step 2: Motive labeling (chunk → primary regulatory motive)

Export motive labeling requests:

```bash
python scripts/export_regulatory_motive_batch_requests.py \
  --chunks-jsonl /path/to/chunks.jsonl \
  --output-dir /path/to/motive_requests \
  --openai-model gpt-5-nano
```

Run the requests via OpenAI Batch, then rehydrate:

```bash
python scripts/rehydrate_regulatory_motive_openai_batch_results.py \
  --request-dir /path/to/motive_requests \
  --results-dir /path/to/motive_results \
  --output-dir /path/to/motive_out
```


## Step 3: Clause extraction (chunk → list of atomic requirements)

Export clause extraction requests (optionally including the motive guess):

```bash
python scripts/export_regulatory_clause_extraction_batch_requests.py \
  --chunks-jsonl /path/to/chunks.jsonl \
  --motive-jsonl /path/to/motive_out/chunk_motives.jsonl \
  --output-dir /path/to/clause_requests \
  --openai-model gpt-5-nano
```

Run the requests via OpenAI Batch, then rehydrate into `clauses.jsonl`:

```bash
python scripts/rehydrate_regulatory_clause_extraction_openai_batch_results.py \
  --request-dir /path/to/clause_requests \
  --results-dir /path/to/clause_results \
  --output-dir /path/to/clause_out
```

If you are running *locally via the gateway* (not OpenAI Batch) and you see
validation failures (e.g., oversized `clause_text` quotes or non-taxonomy `motive`
strings), you can salvage those chunks instead of dropping them:

```bash
python scripts/rehydrate_regulatory_clause_extraction_openai_batch_results.py \
  --request-dir /path/to/clause_requests \
  --results-dir /path/to/clause_results \
  --output-dir /path/to/clause_out \
  --salvage-invalid \
  --allow-errors
```


## Step 4: Embedding + clustering on clauses (instrument discovery)

### Option A) Fast local clustering (no API)

```bash
python scripts/cluster_regulatory_clauses_local.py \
  --clauses-jsonl /path/to/clause_out/clauses.jsonl \
  --output-dir /path/to/clause_clusters_local \
  --text-field requirement
```

To focus on one PI motive category (recommended):

```bash
python scripts/cluster_regulatory_clauses_local.py \
  --clauses-jsonl /path/to/clause_out/clauses.jsonl \
  --output-dir /path/to/clause_clusters_exclusion \
  --filter-motive exclusion \
  --text-field requirement
```

### Option B) Provider embeddings (OpenAI /v1/embeddings)

Export embedding requests:

```bash
python scripts/export_clause_embedding_batch_requests.py \
  --clauses-jsonl /path/to/clause_out/clauses.jsonl \
  --output-dir /path/to/clause_embedding_requests \
  --embedding-model text-embedding-3-small \
  --text-field requirement
```

Run via OpenAI Batch, then rehydrate:

```bash
python scripts/rehydrate_clause_embeddings_openai_batch_results.py \
  --request-dir /path/to/clause_embedding_requests \
  --results-dir /path/to/clause_embedding_results \
  --output-dir /path/to/clause_embedding_dir
```

Cluster the embeddings:

```bash
python scripts/cluster_clause_embeddings.py \
  --embedding-dir /path/to/clause_embedding_dir \
  --output-dir /path/to/clause_clusters_embedding
```


## Step 5: Label clause clusters into instruments (LLM)

Export cluster labeling requests:

```bash
python scripts/export_regulatory_instrument_cluster_labeling_batch_requests.py \
  --clusters-jsonl /path/to/clause_clusters_local/clusters.jsonl \
  --clauses-jsonl /path/to/clause_out/clauses.jsonl \
  --cluster-keywords-jsonl /path/to/clause_clusters_local/cluster_keywords.jsonl \
  --output-dir /path/to/instrument_label_requests \
  --openai-model gpt-5-nano \
  --min-cluster-size 10
```

Run via OpenAI Batch, then rehydrate:

```bash
python scripts/rehydrate_regulatory_instrument_cluster_labels_openai_batch_results.py \
  --request-dir /path/to/instrument_label_requests \
  --results-dir /path/to/instrument_label_results \
  --output-dir /path/to/instrument_label_out
```

This yields `cluster_labels.jsonl` with human-readable instrument names/descriptions.

## Step 6 (optional but recommended): Build a human inspection packet

Once you have clause clusters + (optionally) instrument labels, generate a
Markdown packet you can skim with the PI:

```bash
python scripts/build_regulatory_instrument_cluster_report.py \
  --clauses-jsonl /path/to/clause_out/clauses.jsonl \
  --clusters-jsonl /path/to/clause_clusters_local/clusters.jsonl \
  --cluster-keywords-jsonl /path/to/clause_clusters_local/cluster_keywords.jsonl \
  --cluster-labels-jsonl /path/to/instrument_label_out/cluster_labels.jsonl \
  --sections-jsonl /path/to/sections_out/sections.jsonl \
  --output-dir /path/to/instrument_cluster_report \
  --top-clauses 12
```

Outputs:
- `report.md` (overview + per-cluster links)
- `cluster_packets/*.md` (cluster-by-cluster examples for manual review)

## Step 7 (optional): Build RHS panels from instrument clusters

If you want to use discovered instruments as **city×time RHS covariates** (similar
to the existing mechanics-tag RHS suite), you can build long-form panels from the
clause clusters:

```bash
python scripts/build_regression_rhs_from_instrument_clusters.py \
  --clauses-jsonl /path/to/clause_out/clauses.jsonl \
  --clusters-jsonl /path/to/clause_clusters_local/clusters.jsonl \
  --cluster-labels-jsonl /path/to/instrument_label_out/cluster_labels.jsonl \
  --output-dir /path/to/rhs_from_instruments \
  --bucket month \
  --unit paper
```

To aggregate at `city_state` or `state`, pass publication metadata:

```bash
python scripts/build_regression_rhs_from_instrument_clusters.py \
  --clauses-jsonl /path/to/clause_out/clauses.jsonl \
  --clusters-jsonl /path/to/clause_clusters_local/clusters.jsonl \
  --cluster-labels-jsonl /path/to/instrument_label_out/cluster_labels.jsonl \
  --pub-metadata /path/to/locations_headful.parquet \
  --output-dir /path/to/rhs_from_instruments \
  --bucket month \
  --unit city_state
```

Outputs include:
- issue-level summary + issue×cluster long panel
- unit×bucket×cluster long panel with mean shares (including zeros)

## What to look at during iteration

- `clauses.jsonl`: Are clauses atomic and specific, or overly broad?
- `cluster_keywords.jsonl` + `scatter.png`: Are clusters organizing around instruments (parking, setbacks, rezoning, etc.)?
- `cluster_labels.jsonl`: Are instrument labels stable and specific enough for research use?

If clusters are still too coarse, the highest-leverage knob is usually:
- improving clause extraction prompting (smaller/atomic clauses),
- segmenting documents into better sections before extraction,
- clustering within motive categories instead of globally.

# Issue Classifier Review App

This is the local browser workflow for manually validating issue-level classifier outputs, especially the likely printed-law positives (`code_publication_*`, `amendment_*`, `zoning_ordinance_limited_scope`, `map_rezoning_order`, `variance_special_use_order`).

## 1) Build a review packet

Input must be the parsed JSONL produced by `scripts/pipelines/rehydrate_issue_zoning_issue_classifier_results.py`.

Typical command:

```bash
python scripts/pipelines/build_issue_classifier_review_packet.py \
  --parsed-jsonl /path/to/issue_zoning_parsed_outputs.jsonl \
  --out-dir /path/to/review_packet
```

Default behavior:

- includes only rows with `zoning_presence.primary_class = zoning_legal_text`
- excludes rows whose predicted operativity is already `proposed`
- requires `status=ok`
- fails if the original transcript source files needed for review are missing

Useful flags:

- `--labels label_a,label_b,...` to change the predicted-label subset
- `--include-all` to build a packet for every parsed issue
- `--include-proposed` to keep issues that the model already marked `proposed`
- `--include-non-zoning-legal-text` to bypass the printed-law gate and include non-`zoning_legal_text` rows
- `--on-missing-source skip_issue` to skip issues whose original source text cannot be reconstructed

Packet outputs:

- `metadata.json`
- `index.jsonl`
- `review_sheet.csv`
- `review_events.jsonl`
- `review_snapshot.csv`
- one JSON file per issue under `<predicted_label>/`

Each per-issue JSON contains:

- full issue transcript
- full parsed model output
- source provenance
- per-page source text when available
- derived heuristic signals for quick review

## 2) Run the browser app

```bash
python scripts/pipelines/run_issue_classifier_review_app.py \
  --packet-dir /path/to/review_packet \
  --port 8787 \
  --open-browser
```

The UI supports:

- a one-issue-at-a-time review queue
- transcript inspection
- a compact model summary focused on label, operativity, and printed-law evidence
- manual label override
- manual operativity override
- strict operative-target verdict (`yes|no|unclear`)
- notes
- append-only review event logging plus current-state CSV snapshot

## 3) Saved review state

The app writes two persistent artifacts inside the packet directory:

- `review_events.jsonl`: append-only event log
- `review_snapshot.csv`: latest review state, one row per reviewed issue

`review_snapshot.csv` is the easiest export to use for downstream analysis.

## 4) Keyboard shortcuts

- `j`: next issue
- `k`: previous issue
- `a`: accept current prediction and save
- `f`: set manual label to `code_publication_full_issue`
- `e`: set manual label to `code_publication_excerpt_or_installment`
- `Cmd+S` / `Ctrl+S`: save current review

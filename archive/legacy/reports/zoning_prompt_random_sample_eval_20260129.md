# Zoning page classifier — random sample A/B vs existing labels (2026-01-29)

Mode: **Interpretive** (per AGENTS.md). You asked for the “best way” to build an LLM-driven classifier that detects when a newspaper OCR transcript **prints actual zoning law text** (not stories/editorials *about* zoning). I used repo artifacts (existing label outputs + OCR transcripts), a random sample run, and stress tests to find concrete failure modes and then iterated the prompt to address them.

## What I evaluated

- **Baseline labels (existing):** `newspaper-parsing-local/data/zoning_labels_openai_gpt5nano/` (278 pages; last-modified Dec 12, 2025).
  - These are prior LLM outputs, not human ground truth, but they are a useful “baseline A” to see what the new prompt changes.
- **New prompt (B):** `prompts/zoning_ocr_classifier_prompt_text.txt`
- **Model:** `openai:gpt-5-nano` via local gateway (`http://127.0.0.1:8000`)
- **Run settings:** `temperature=0`, `max_concurrency=4`
- **Random sample:** 50 pages (`seed=20260129`)

Command used:

```bash
python scripts/eval_zoning_classifier_random_sample.py \
  --baseline-dir newspaper-parsing-local/data/zoning_labels_openai_gpt5nano \
  --output-dir tmp/zoning_prompt_eval_random_sample_20260129 \
  --sample-size 50 --seed 20260129 \
  --model openai:gpt-5-nano --temperature 0
```

Artifacts from the run:
- `tmp/zoning_prompt_eval_random_sample_20260129/summary.txt`
- `tmp/zoning_prompt_eval_random_sample_20260129/comparison.jsonl`
- `tmp/zoning_prompt_eval_random_sample_20260129/new_outputs/*.zoning.json`

## High-level results (sample=50)

- Agreement vs baseline labels: **38/50 (76%)**
- Disagreements vs baseline labels: **12/50 (24%)**

This “disagreement rate” is not a quality metric by itself (baseline is not ground truth). So I manually inspected **all 12 disagreements** against the underlying page OCR transcripts.

## Manual review of disagreements (what the page actually contains)

Summary of what changed (12 cases):
- **Stories/minutes that only *describe* zoning** were pushed to `unrelated` (fixes the false-positive mode you described).
- **Formal hearing notices** previously missed were upgraded to `public_hearing`.
- **True printed zoning changes** previously missed were upgraded to `amendment_targeted`.
- One category-level issue (positive-but-wrong *type*) was found and then fixed by prompt iteration.

Below: baseline → new, plus what the transcript supports.

### 1) `anniston-star-nov-01-1970-p-25`
- Baseline: `amendment_substantial`
- New: `full_ordinance` (**law_text_stage=proposed**)
- Transcript evidence: prints substantial zoning provisions (district uses/standards) in an explanatory format (e.g., “R- — One-Family Residential. Uses…”), and signals it is not yet official (“…before the ordinance can become official.”).
- Manual verdict: **printed zoning law text is present**; “proposed” nuance is real. Primary label could be debated (it’s proposed + article-ish), but it is *not* “unrelated”.

### 2) `argus-jan-20-1965-p-1`
- Baseline: `amendment_targeted`
- New: `unrelated`
- Transcript evidence: news story about a rezoning request (“Rezoning of 92 Acres”, “PUD ordinance”) but **no ordinance text** or formal notice logistics.
- Manual verdict: **`unrelated`** (correct under “must print the law” rule).

### 3) `arlington-heights-daily-herald-suburban-chicago-oct-29-1994-p-374`
- Baseline: `unrelated`
- New: `public_hearing`
- Transcript evidence: explicit hearing notice with logistics (date + time + location) tied to zoning matters.
- Manual verdict: **`public_hearing`**.

### 4) `bucks-county-courier-times-nov-15-1976-p-39`
- Baseline: `amendment_targeted`
- New: `public_hearing`
- Transcript evidence: zoning hearing board meeting notice with date/time/location (variance/appeal), **no printed zoning code/rezoning clauses**.
- Manual verdict: **`public_hearing`**.

### 5) `carbondale-southern-illinoisan-aug-30-1965-p-2`
- Baseline: `full_ordinance`
- New: `unrelated`
- Transcript evidence: housing-code enforcement story (“violation of the ordinance… fine…”) + unrelated items; no zoning law text.
- Manual verdict: **`unrelated`** (baseline was a classic “ordinance != zoning ordinance” false positive).

### 6) `cedar-rapids-gazette-apr-19-2003-p-63`
- Baseline: `unrelated`
- New: `amendment_targeted` (**law_text_stage=proposed**)
- Transcript evidence: public hearing notice that **prints the specific replacement clause** (“Replace condition 6)… ‘The berm shall not be required…’”).
- Manual verdict: **`amendment_targeted`** (because actual amendment text is printed; it’s not only a notice).

### 7) `cedar-rapids-gazette-jan-07-1999-p-17`
- Baseline: `public_hearing`
- New: `unrelated`
- Transcript evidence: board minutes mention “Public Hearing… Street Address Ordinance Amendment” and separately “Second Consideration of Zoning Ordinance Amendment” but **no zoning hearing logistics** and **no zoning text printed**.
- Manual verdict: **`unrelated`**.

### 8) `cedar-rapids-gazette-mar-30-2006-p-76`
- Baseline: `unrelated`
- New: `public_hearing`
- Transcript evidence: formal rezoning hearing notice with logistics (“Monday April 3rd, 2006 at 7:00 p.m.” in City Hall).
- Manual verdict: **`public_hearing`**.

### 9) `cedar-rapids-gazette-oct-17-1988-p-23`
- Baseline: `full_ordinance`
- New: `amendment_targeted`
- Transcript evidence: ordinance changes zoning district for a specific property and includes operative effect language.
- Manual verdict: **`amendment_targeted`** (site-specific rezoning ≠ full ordinance).

### 10) `colorado-springs-gazette-jul-18-1970-p-49`
- Baseline: `amendment_substantial`
- New (initial sample run): `full_ordinance`
- Transcript evidence: page prints multiple ordinances, including “repealed and reenacted, as amended to read as follows” (substantial replacement) and also specific zone establishment/map changes.
- Manual verdict: **should be `amendment_substantial`** (because substantial amendment text is present; priority should pick it).
- Follow-up: prompt iteration (see below) reclassified this page to `amendment_substantial` and set `present` flags more correctly.

### 11) `coronado-eagle-and-journal-may-20-1943-p-1`
- Baseline: `public_hearing`
- New: `unrelated`
- Transcript evidence: news story says “May 24 is set… as the date for hearing…” but **no formal notice + no time/location**; not printed zoning text.
- Manual verdict: **`unrelated`**.

### 12) `coronado-eagle-and-journal-sep-28-1972-p-11`
- Baseline: `amendment_substantial`
- New: `amendment_targeted`
- Transcript evidence: ordinance adds a new subsection and changes one lot’s classification (“changed from R-3 to C-2”), with effective language.
- Manual verdict: **`amendment_targeted`**.

## Prompt iterations after the sample (to fix discovered issues)

The sample review surfaced a real prompt weakness: on pages with **multiple ordinance types**, the model could pick a lower-priority type (e.g., `full_ordinance`) even when a higher-priority `amendment_substantial` was present.

I updated `prompts/zoning_ocr_classifier_prompt_text.txt` to:
- Make public-hearing qualification explicit (standard logistics vs formal notice-ordinance exception).
- Strengthen amendment-targeted definition to include:
  - parcel/site-specific rezoning and map amendments,
  - special use / conditional use / planned development approvals (even if long),
  - clause-level edits (e.g., adding a single permitted use).
- Make the “primary label = highest-priority type present anywhere” selection procedure explicit.

Then I reran the prompt edge-case stress test:

```bash
python scripts/stress_test_zoning_page_prompt.py \
  --cases scripts/fixtures/zoning_page_prompt_edge_cases.jsonl \
  --output-dir tmp/zoning_page_prompt_stress_post_samplefix6_20260129 \
  --model openai:gpt-5-nano \
  --prompt-paths prompts/zoning_ocr_classifier_prompt_text.txt \
  --trials 1 --temperature 0
```

Result: **27/27 passing** in `tmp/zoning_page_prompt_stress_post_samplefix6_20260129/stress_test_page_zoning_summary.txt`.

## Takeaways / next step options

1) If you want “human-like ground truth”, the next step is to sample from a set you trust (human-coded labels, or a curated “gold” set) and compute precision/recall for:
   - “printed zoning law text present” (any of {full_ordinance, amendment_substantial, amendment_targeted})
   - “formal zoning hearing notice present” (public_hearing)
2) If you want to run this on a much larger slice (hundreds/thousands of pages), we can:
   - stratify the sample by label (oversample `full_ordinance`/`amendment_*` to focus on false positives),
   - write a review queue (CSV/JSONL) with page_id + evidence + direct transcript path for fast inspection.


# RHS v4 Suite Status (gpt-5-nano relabel)

Created: 2026-01-30

This file is an index of the artifacts produced for the **v4 mechanics relabel** and the
downstream **city×time RHS measurement suite** (no treatment definition chosen yet).

## 1) v4 cluster/topic labels (OpenAI Batch, gpt-5-nano)

Local snapshot (used for all downstream work):

- `newspaper-parsing-local/vast_snap_20260130_041623/full_ordinance/cluster_topic_labels_mechanics_v4_gpt5nano_20260130_013333/outputs/cluster_labels.jsonl`
- `newspaper-parsing-local/vast_snap_20260130_041623/amendment_substantial/cluster_topic_labels_mechanics_v4_gpt5nano_20260130_013333/outputs/cluster_labels.jsonl`
- `newspaper-parsing-local/vast_snap_20260130_041623/amendment_targeted/cluster_topic_labels_mechanics_v4_gpt5nano_20260130_013333/outputs/cluster_labels.jsonl`

## 2) Mechanics-tag RHS panels (city×month)

Observed (only months with issues):

- `newspaper-parsing-local/rhs_panel_mechanics_v4_gpt5nano_20260130_013333_citytime/rhs_city_state_month_panel_20260129_233814.parquet`
- `newspaper-parsing-local/rhs_panel_mechanics_v4_gpt5nano_20260130_013333_citytime/rhs_issue_panel_20260129_233814.parquet`

Step / forward-filled (piecewise-constant between observed months):

- `newspaper-parsing-local/rhs_panel_mechanics_v4_gpt5nano_20260130_013333_citytime_step/step_panel_20260129_233834.parquet`

Type-separated (classification_label-stratified) and pivoted wide:

- `newspaper-parsing-local/rhs_panel_mechanics_v4_gpt5nano_20260130_013333_citytime_bylabel/rhs_city_state_month_panel_20260129_233907.parquet`
- `newspaper-parsing-local/rhs_panel_mechanics_v4_gpt5nano_20260130_013333_citytime_bylabel_wide/rhs_city_time_wide_by_label_20260129_233926.parquet`

Catalogs (quick feature summaries):

- `newspaper-parsing-local/rhs_panel_mechanics_v4_gpt5nano_20260130_013333_citytime/catalog_pooled/rhs_panel_summary_20260129_233947.md`
- `newspaper-parsing-local/rhs_panel_mechanics_v4_gpt5nano_20260130_013333_citytime_step/catalog_step/rhs_panel_summary_20260129_233947.md`

## 3) Embedding policy-state panels (city×month)

Embedding PCA “policy state” and jump metrics (built earlier; reused here):

- `newspaper-parsing-local/policy_state_embeddings_pca10_20260130_005107/step/city_state_month_step_policy_panel_20260130_005607.csv`

## 4) Merged STEP panel (mechanics + embeddings + mechanics PCA)

Canonical input for event discovery / regimes / overlays:

- `newspaper-parsing-local/rhs_merged_STEP_v4_embeddings_pca10_plus_mech_pca8_20260130/merged_city_month_panel_STEP_v4_20260129_234126.parquet`

## 5) Event candidates (no treatment definition chosen)

Events from **embedding-jump** intensity (`jump_cosine_pca`):

- `newspaper-parsing-local/rhs_event_candidates_STEP_v4_emb_jump_20260130/events_by_city_top5_20260129_234154.csv`
- `newspaper-parsing-local/rhs_event_candidates_STEP_v4_emb_jump_20260130/event_packets/` (per-event markdown packets)

Events from **mechanics-jump** intensity (`mechanics_jump_cosine_pca`):

- `newspaper-parsing-local/rhs_event_candidates_STEP_v4_mech_jump_20260130/events_by_city_top5_20260129_234155.csv`
- `newspaper-parsing-local/rhs_event_candidates_STEP_v4_mech_jump_20260130/event_packets/`

## 6) Discrete regime clustering (option A-style RHS)

KMeans regimes over `pc1..pc3` + `mechanics_pc1..pc3` (fit on observed months):

- `newspaper-parsing-local/policy_regimes_v4_emb_plus_mech_20260130/regime_assignments_20260129_234217.parquet`
- `newspaper-parsing-local/policy_regimes_v4_emb_plus_mech_20260130/regime_events_20260129_234217.csv`
- `newspaper-parsing-local/policy_regimes_v4_emb_plus_mech_20260130/regime_summaries_k*_20260129_234217.csv`

## 7) Outcome overlays (lightweight exploratory)

FRED FHFA HPI series cache (3 cities):

- `newspaper-parsing-local/outcomes/fred_fhfa_20260130/fred_city_outcomes_20260129_235434.parquet`

Overlay figures (RHS vs FHFA HPI + event markers):

- `newspaper-parsing-local/outcome_overlay_v4_20260130_emb_jump/figures/`
- `newspaper-parsing-local/outcome_overlay_v4_20260130_mech_jump/figures/`

Scripts:

- `scripts/fetch_fred_series.py`
- `scripts/plot_rhs_with_outcomes.py`

## 8) LaTeX writeup updated to v4 cluster labels

- `reports/issue_topics_report/report.pdf`
- `reports/issue_topics_report/provenance.json`

The v4 “mountain” plots were regenerated with legends **outside** and **no “other” band**:

- `newspaper-parsing-local/vast_snap_20260130_041623/*/plots_chunks_labeled_top50_noother_mechanics_v4_gpt5nano_legend_outside_20260130_235600/cluster_frequency_year.png`

## 9) Expanded inspection sample (Sample A = top 32 city_states)

Definition:

- Sample A = the **top 32** city_states by total issue count in the v4 observed city×month panel.
- This is exactly the set with `n_issues_sum >= 100` (32 city_states).

Artifacts:

- `newspaper-parsing-local/sample_A_top32_v4_20260130/top32_city_state.csv`
- `newspaper-parsing-local/sample_A_top32_v4_20260130/event_plots_embedding_jump/` (32 cities)
- `newspaper-parsing-local/sample_A_top32_v4_20260130/event_plots_mechanics_jump/` (32 cities)
- `newspaper-parsing-local/rhs_explore_citytime_STEP_mechanics_v4_plus_embeddings_pca10_20260130_top32/figures/` (32 dashboards)

## 10) Consolidated writeup (topics + city-by-city RHS)

- `reports/rhs_v4_full_writeup/report.pdf`
- `reports/rhs_v4_full_writeup/provenance.json`
- Builder: `scripts/build_rhs_v4_full_latex_report.py`

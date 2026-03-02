# Zoning Prompt Accuracy Audit + Manual Validation (2026-02-28)

## Scope
- Diagnose why ordinance-presence and granularity classification prompts are inaccurate.
- Run A/B prompt tests on the same diagnostic set using the gateway (Gemini).
- Perform manual validation on representative error cases.

## Runs Executed
- Existing baseline comparison:
  - `v2_full243` (existing in repo)
- New runs in this audit:
  - `v3_full243_gemini_20260228`
  - `v4_manual123_gemini_20260228` (ultra-strict follow-up on manual subset only)

## Gateway/Model Notes
- Gemini via gateway is operational.
- OpenAI via current gateway runner/client path returns empty extracted text (even when provider call returns HTTP 200), so production A/B execution here used Gemini.
- Direct OpenAI API checks show responses are present; the break appears to be in response-text extraction in the gateway client/runner path.

## Quantitative Results (Manual-Bucket Benchmark, n=123)

### Variant B (ontology prompt) across iterations

| Run | Full precision | Full recall | Full F1 | Non-full false-full rate | False-full count |
|---|---:|---:|---:|---:|---:|
| v2 | 0.0513 | 1.0000 | 0.0976 | 0.3058 | 37 |
| v3 | 0.0556 | 1.0000 | 0.1053 | 0.2810 | 34 |
| v4 | 0.0303 | 0.5000 | 0.0571 | 0.2645 | 32 |

### Interpretation
- `v3` is the best balance in this audit:
  - modest full-code precision/F1 gain over `v2`,
  - reduced non-full false-full rate (`0.3058 -> 0.2810`),
  - preserved full recall on this benchmark.
- `v4` reduced false-full count further, but at unacceptable cost:
  - full recall collapse (`1.00 -> 0.50`),
  - lower parse stability,
  - worse full-code precision/F1 than `v3`.

## Ordinance-Presence vs Granularity
- On a high-confidence subset (`n=73`: likely zoning buckets + likely non-zoning bucket), presence is already strong.
- Using `primary_class != unrelated` as presence:
  - `v2`: precision `0.9275`, recall `1.0000`, F1 `0.9624`
  - `v3`: precision `0.9275`, recall `1.0000`, F1 `0.9624`
- Conclusion: the bigger problem is **granularity calibration** (especially full-code inflation), not binary presence recall.

## Manual Validation Findings

I manually inspected representative disagreements and high-impact mistakes (focus: false-full, notice-vs-full, non-zoning leakage).

### Cases where model over-calls full code (true model issue)
1. `albion-evening-recorder__1970-01-23`
   - Evidence is mixed: map boundary edits + selected sections + adoption line.
   - Better treated as amendment/partial recodification evidence, not definitive full ordinance.
2. `albion-evening-recorder__1968-12-19`
   - Strong procedural/board text with some ordinance clauses, but fragmentary.
   - Full code is too aggressive.
3. `alamogordo-daily-news__1959-01-14`
   - Contains codified sections, but context looks like targeted zoning-boundary mechanics.
   - Likely over-escalated to full.

### Cases where benchmark label appears noisy or too coarse (ground-truth issue)
1. `bountiful-davis-county-clipper__1961-11-17`
   - Printed comprehensive zoning ordinance text (title/purpose/enforcement/administration).
   - Manual bucket says hearing-like; text strongly supports full recodification/proposed code.
2. `arkansas-city-traveler__1964-12-24`
   - Multi-article zoning legal text with repeal language and occupancy provisions.
   - Manual hearing bucket likely under-calls legal-text depth.
3. `american-fork-citizen__1963-08-22`
   - Extensive zone/use/map ordinance structure; likely full/recodification class.
4. `appleton-post-crescent__1929-12-13`
   - Includes industrial district ordinance sections; non-zoning manual bucket likely incorrect.
5. `athens-messenger__1960-09-30`
   - Explicit zoning article text; non-zoning manual bucket likely incorrect.
6. `argus__1976-05-12`
   - Mixed page, but includes zoning standards text (nursery school + district constraints); not clean non-zoning.

### Genuine non-zoning leakage example
1. `athens-messenger__1965-05-24`
   - Ordinance-like legal language without explicit zoning anchors.
   - Should remain non-zoning ordinance; model still tends to over-map generic ordinance form to zoning.

## Root-Cause Diagnosis
1. **Label noise in evaluation set**
   - Manual buckets are intentionally coarse (`likely_*`, `unclear*`) and contain clear under-calls/over-calls in both directions.
   - This caps measurable prompt gains and can penalize correct model behavior.

2. **Over-reliance on ordinance form over zoning specificity**
   - Model still maps “ordinance + sections + adoption” to zoning full-code too easily when zoning anchors are weak or scope is narrow.

3. **Mixed-content OCR pages**
   - Single issue transcript can contain unrelated legal notices plus zoning snippets.
   - Without stronger “dominant legal object” and “scope lock” rules, class inflation persists.

4. **Notice vs full-code ambiguity**
   - Public-notice language that includes excerpted zoning text can be interpreted as full code.
   - Needs more explicit tie-breaks to prevent full escalation unless comprehensive criteria are fully met.

5. **OpenAI path instrumentation gap in gateway runner**
   - Current local gateway runner/client extraction drops OpenAI output text, blocking fair cross-provider A/B within this exact harness.

## Prompt Recommendation
- Use `prompts/zoning_issue_classifier_prompt_v3_ontology_precision.txt` as the best current option.
- Do **not** adopt v4.

## Practical Next Steps
1. Fix OpenAI extraction in gateway runner/client, then rerun `v3` with OpenAI on the same `manual123` set for true provider comparison.
2. Build a cleaner adjudicated benchmark (replace `likely_*` with final labels for a 150-300 row eval pack).
3. Add a hard zoning-anchor requirement for non-zoning suppression:
   - if no explicit zoning anchor, force `non_zoning_ordinance` or `none`.
4. Add a post-classification rule layer:
   - veto `full_code_*` when scope is parcel-specific/mixed and dimensional evidence is inferred rather than quoted.

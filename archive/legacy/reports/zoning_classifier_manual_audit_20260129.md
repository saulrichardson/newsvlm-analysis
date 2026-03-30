# Zoning page classifier — manual audit notes (2026-01-29)

Goal (primary): **high-precision detection of whether a newspaper OCR transcript prints operative zoning law text** (ordinance/amendment/rezoning order text), not merely news/editorials/minutes/notices *about* zoning.

Rubric used in this audit:
- **Printed zoning law text present?** = YES only if the page reproduces enforceable ordinance/amendment clauses (e.g., `Section 1...`, `BE IT ORDAINED...`, `is hereby amended to read as follows`, `zoning classification ... is hereby changed from ... to ...`).
- Zoning stories, meeting minutes, agenda summaries, “introduced Ordinance No. X…”, or notices saying a copy is available elsewhere = **NO** (even if they describe rezoning or adoption).
- A page can still be relevant as a **public hearing notice**, but that is **not** “printed zoning law text” for the primary goal.

This file records a focused audit of the 6 pages where the updated prompt (prompt-fix) flipped from **non-law → law-text** labels.

## Summary of reviewed flips (non-law → law-text)

| page_id | old label | new label | manual label | printed zoning law text? | notes |
|---|---:|---:|---:|---:|---|
| bucks-county-courier-times-jan-21-1976-p-103 | public_hearing | full_ordinance | public_hearing | NO | Formal zoning hearing notices + proposed ordinance **summary** only |
| anniston-star-nov-02-1970-p-15 | public_hearing | amendment_substantial | public_hearing (or unrelated) | NO | “Notice ordinance” scheduling future adoption; does **not** print the zoning amendment text |
| cedar-rapids-gazette-mar-30-2006-p-76 | unrelated | amendment_targeted | public_hearing | NO | Contains a zoning hearing notice, plus council minutes that **reference** an ordinance but don’t print it |
| cedar-rapids-gazette-jan-07-1999-p-17 | public_hearing | amendment_targeted | unrelated | NO | Board minutes list rezoning cases + ordinance numbers; no ordinance text printed |
| bucks-county-courier-times-mar-12-1969-p-129 | public_hearing | amendment_targeted | unrelated | NO | News story about a proposed zoning amendment; “hearing date will be set” |
| athens-news-courier-dec-28-2007-p-39 | unrelated | full_ordinance | full_ordinance | YES | Zoning ordinance definitional text (clearly code-like) |

## Evidence snippets (from OCR page text)

### bucks-county-courier-times-jan-21-1976-p-103
Manual call: **public_hearing** (NOT law text).

Key evidence it’s a notice/summary, not operative ordinance clauses:
- `NOTICE IS HEREBY GIVEN ... the enactment of an Ordinance hereinafter set-forth in summary will be considered at a public hearing ...`
- `ORDINANCE NO. AN ORDINANCE REGULATING AND RESTRICTING ... DIVIDING THE TOWNSHIP INTO ZONING DISTRICTS ...`
- `Copy of aforesaid Proposed Ordinance and Map can be examined at ...`

Also includes a separate zoning hearing board notice:
- `Take notice that the Zoning Hearing Board ... will meet on Monday, February 9, 1976, at 7:30 P.M. ... for variances to Sections ... of the ... Zoning Ordinance ...`

### anniston-star-nov-02-1970-p-15
Manual call: **public_hearing** (or **unrelated** if we require time/location), but **NOT** an amendment text.

Key evidence it’s scheduling future adoption / describing intent, not printing amended zoning code:
- `WHEREAS ... zoning laws ... be amended by adding thereto "Planned Residential A"; ...`
- `WHEREAS, any such changes may be made only after a public hearing.`
- `Section 1. That on November 17, 1970, the Council ... will consider the adoption and passage of an ordinance which shall amend the existing zoning laws ...`
- `Section 2 ... directed to cause a copy of this ordinance to be published ...`

### cedar-rapids-gazette-mar-30-2006-p-76
Manual call: **public_hearing** (NOT law text).

Key evidence of a real zoning hearing notice:
- `TO WHOM IT MAY CONCERN: ... Public Hearing ... parcel ... designated a zoning of R-1 ... hearing ... Monday April 3rd, 2006 at 7:00 p.m. ... Robins City Hall ...`

Also contains city council minutes that only *reference* an ordinance:
- `Wainwright introduced Ordinance No. 54, AN ORDINANCE AMENDING THE FAIRFAX ZONING ORDINANCE OF 2000 ...`
- `... be considered for the third time ... adopted ... Ordinance No. 54 is declared to have been enacted.`

No ordinance/amendment clauses are printed.

### cedar-rapids-gazette-jan-07-1999-p-17
Manual call: **unrelated** (NOT law text).

Key evidence this is meeting minutes (procedural summaries), not ordinance clauses:
- `Motion by ... seconded by ...`
- `... approve upon second consideration Rezoning Case ... request ... to rezone from ... to ...`
- `... adopt upon third and final consideration Ordinance #... request ... to rezone ...`
- `... Ordinance #... amending Zoning Ordinance Chapter 1.`

No “Section 1…”, “Be it ordained…”, “to read as follows…”, etc. The text reads like minutes.

### bucks-county-courier-times-mar-12-1969-p-129
Manual call: **unrelated** (NOT law text).

Key evidence this is a news story describing an amendment, with future hearing unspecified:
- `... zoning amendment discussed last night ... Half-acre lots ... would be deleted from the township zoning ordinance.`
- `... must be approved ... and then a public hearing date will be set ...`

### athens-news-courier-dec-28-2007-p-39
Manual call: **full_ordinance** (printed law text present).

Key evidence this is actual ordinance/code-like text:
- `Hardship. A circumstance existing when the conditions imposed by the Zoning Ordinance would deprive ...`
- Many definition-style entries + ordinance-internal references.

## Failure modes observed

The updated prompt reduced several “zoning story → ordinance” false positives, but these flips show remaining failure modes:
1. **“Notice of public hearing” + “ordinance summary”** being treated as full ordinance text (should be `public_hearing`).
2. **Meeting minutes** that reference ordinances/rezoning cases being treated as printed amendment text (should be `unrelated`).
3. **Procedural “notice ordinance”** that schedules a hearing / future adoption being treated as an amendment (should be `public_hearing` or `unrelated` depending on the strictness of the hearing-definition rule).
4. **News stories that describe proposed ordinance standards (numeric rules) in narrative form** being treated as printed law text.

## Additional spot checks (2026-01-29)

### albert-lea-evening-tribune-jul-06-1972-p-9
Manual call: **unrelated** (NOT printed law text).

Why: this page reads like a news story about planners recommending a shoreline ordinance, including a byline and narrative framing (not ordinance sections).
Key evidence it’s reporting, not reproduced ordinance text:
- `Planners Recommend Ordinance: County to Control Shorelines`
- `By THE ASSOCIATED PRESS`
- Narrative rule descriptions like `200-foot frontage and 80,000 square-foot lot minimums. Buildings must be located at least 200 feet from the shore...`

Action taken:
- Updated prompt to require a **legal/ordinance register** (e.g., `ORDINANCE NO.`, `BE IT ORDAINED`, numbered `Section`, explicit `is hereby amended`, explicit rezoning order language) before labeling something as printed law text.
- Added a synthetic fixture case: `article_describes_zoning_standards_no_ordinance_text` to prevent regressions.

### alton-telegraph-nov-08-1996-p-35
Manual call: **unrelated** (NOT printed law text).

Why: this is a rezoning news report (mentions `Zoning Board`, “rezone”, etc.) but does not reproduce ordinance clauses/sections.

#!/usr/bin/env python3
"""
Audit: questionnaire answers vs source OCR text.

Goal
----
Detect *systemic* problems where the questionnaire model produces:
  - non-null answers without evidence,
  - evidence strings that cannot be found in the source OCR text,
  - answers that appear to be semantically off-topic (heuristic keyword anchors).

This script is intentionally conservative and artifact-driven:
  - It only uses local request mappings + VLM page JSONs + normalized answers.
  - It does NOT "judge correctness" of zoning policy; it flags likely alignment failures.

Primary use case right now
--------------------------
Slug-year grouped questionnaire runs where:
  - mapping_shard*.jsonl contains pages[] with page_path + selected_box_ids
  - normalized.jsonl contains answers_by_id[qid].evidence

Outputs
-------
Writes a markdown report to --out-md and prints a short summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from os.path import expanduser
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:  # noqa: BLE001
            raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(obj, dict):
            raise SystemExit(f"Expected JSON object in {path}, got {type(obj).__name__}")
        out.append(obj)
    return out


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit(f"Expected JSON object in {path}, got {type(obj).__name__}")
    return obj


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _normalize_for_match(s: str) -> str:
    # Lowercase + collapse whitespace.
    s = s.lower()
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _normalize_for_match_no_inword_hyphen(s: str) -> str:
    # Also drop hyphens between letters: "fam-ily" -> "family".
    s = _normalize_for_match(s)
    # Handle OCR hyphenation splitting words across line/space breaks:
    #   "con- tain" or "con-\n tain" -> "contain"
    s = re.sub(r"(?<=[a-z])-\s+(?=[a-z])", "", s)
    s = re.sub(r"(?<=[a-z])-(?=[a-z])", "", s)
    return s


def _normalize_for_match_alnum(s: str) -> str:
    # Keep only letters/digits/spaces; useful when OCR injects punctuation noise.
    s = _normalize_for_match(s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _evidence_segments(evidence: str) -> list[str]:
    # Evidence often contains "..." indicating omission; we match segments independently.
    parts = re.split(r"\.\.\.|…", evidence)
    segs = []
    for p in parts:
        s = _norm_ws(p)
        if len(s) >= 8:
            segs.append(s)
    # Prefer longer segments first for matching.
    return sorted(segs, key=len, reverse=True)


def _segment_in_text(segment: str, source_text: str) -> bool:
    seg = segment
    src = source_text
    # Pass A: whitespace-normalized substring.
    if _normalize_for_match(seg) in _normalize_for_match(src):
        return True
    # Pass B: also remove in-word hyphens (handles "fam-ily" vs "family").
    if _normalize_for_match_no_inword_hyphen(seg) in _normalize_for_match_no_inword_hyphen(src):
        return True
    # Pass C: alphanumeric-only fallback.
    if _normalize_for_match_alnum(seg) in _normalize_for_match_alnum(src):
        return True
    return False


def _evidence_found(evidence: str, source_text: str) -> bool:
    segs = _evidence_segments(evidence)
    if not segs:
        return False
    # If any substantial segment is found, treat as found.
    return any(_segment_in_text(seg, source_text) for seg in segs[:5])


def _extract_selected_box_texts_from_page(page_json: dict[str, Any], selected_box_ids: list[int]) -> list[str]:
    boxes = page_json.get("boxes")
    if not isinstance(boxes, list):
        return []
    by_id: dict[int, str] = {}
    for b in boxes:
        if not isinstance(b, dict):
            continue
        bid = b.get("id")
        if not isinstance(bid, int):
            continue
        t = b.get("transcript")
        if isinstance(t, str) and t.strip():
            by_id[bid] = t
    out: list[str] = []
    for bid in selected_box_ids:
        t = by_id.get(int(bid))
        if t:
            out.append(t)
    return out


@dataclass(frozen=True)
class GroupSource:
    group_id: str
    group_mode: str
    slug: str | None
    year: int | None
    page_refs: list[tuple[str, Path, list[int]]]  # (page_id, page_path, selected_box_ids)
    source_text: str


def _load_group_sources_from_mapping(mapping_rows: list[dict[str, Any]]) -> dict[str, GroupSource]:
    out: dict[str, GroupSource] = {}

    for m in mapping_rows:
        gid = m.get("id")
        if not isinstance(gid, str) or not gid.strip():
            continue
        group_mode = str(m.get("group_mode") or "box").strip()

        slug = m.get("slug") if isinstance(m.get("slug"), str) else None
        year = m.get("page_year") if isinstance(m.get("page_year"), int) else None

        page_refs: list[tuple[str, Path, list[int]]] = []
        source_chunks: list[str] = []

        if group_mode == "slug_year":
            pages = m.get("pages")
            if not isinstance(pages, list):
                raise SystemExit(f"Expected pages[] for slug_year group id={gid}")
            for p in pages:
                if not isinstance(p, dict):
                    continue
                page_id = p.get("page_id")
                page_path = p.get("page_path")
                sel = p.get("selected_box_ids")
                if not isinstance(page_id, str) or not isinstance(page_path, str) or not isinstance(sel, list):
                    continue
                sel_int = [int(x) for x in sel if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit())]
                page_refs.append((page_id, Path(page_path).expanduser(), sel_int))
            # Load each referenced page and concatenate selected box transcripts.
            for page_id, page_path, sel in page_refs:
                page_json = _load_json(page_path)
                texts = _extract_selected_box_texts_from_page(page_json, sel)
                if not texts:
                    continue
                # Keep light structure so humans can read snippets in reports.
                joined = "\n\n".join(_norm_ws(t) for t in texts if isinstance(t, str) and t.strip())
                if joined.strip():
                    source_chunks.append(f"[{page_id}]\n{joined}")
        else:
            # box-level
            page_id = m.get("page_id")
            page_path = m.get("page_path")
            box_id = m.get("box_id")
            if isinstance(page_id, str) and isinstance(page_path, str) and isinstance(box_id, int):
                page_refs.append((page_id, Path(page_path).expanduser(), [int(box_id)]))
                page_json = _load_json(Path(page_path).expanduser())
                texts = _extract_selected_box_texts_from_page(page_json, [int(box_id)])
                joined = "\n\n".join(_norm_ws(t) for t in texts if isinstance(t, str) and t.strip())
                if joined.strip():
                    source_chunks.append(f"[{page_id} box={box_id}]\n{joined}")

        out[gid] = GroupSource(
            group_id=gid,
            group_mode=group_mode,
            slug=slug,
            year=year,
            page_refs=page_refs,
            source_text="\n\n".join(source_chunks).strip(),
        )

    return out


def _default_anchor_map() -> dict[str, list[str]]:
    # These are intentionally conservative “topic anchors”, not a full correctness spec.
    # If an answer is non-null but its evidence contains *none* of these anchors, it is
    # likely (not guaranteed) to be semantically off-topic.
    return {
        # Accessory dwelling unit family-occupancy restriction
        "Z10": ["accessory", "adu", "in-law", "inlaw", "accessory apartment", "second unit", "secondary unit"],
        # ADU allowed categorical
        "Z9": ["accessory", "adu", "in-law", "inlaw", "accessory apartment", "second unit", "secondary unit"],
        # ADU allowed (binary)
        "11": ["accessory", "adu", "in-law", "inlaw", "accessory apartment", "second unit", "secondary unit"],
        # Mobile homes
        "Z12": ["mobile home", "manufactured", "trailer", "mobilehome", "mobile-home", "mobile home park"],
        # Septic regs
        "S1": ["septic", "sewage", "disposal", "soil absorption", "percolation", "lagoon", "sanitary"],
        # Lot sizes
        "28Min": ["acre", "acres", "square feet", "sq", "ft", "feet", "lot area", "minimum"],
        "28Max": ["acre", "acres", "square feet", "sq", "ft", "feet", "lot area", "minimum"],
        # Wetlands bylaw/reg
        "W1": ["wetland", "wetlands", "vernal", "pool", "buffer", "flood", "marsh", "swamp"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit questionnaire evidence alignment against OCR source text.")
    p.add_argument("--request-dir", required=True, help="Questionnaire request dir containing mapping_shard*.jsonl")
    p.add_argument("--results-dir", required=True, help="Questionnaire results dir containing normalized.jsonl")
    p.add_argument(
        "--out-md",
        required=True,
        help="Where to write the markdown report",
    )
    p.add_argument(
        "--max-examples-per-qid",
        type=int,
        default=10,
        help="Cap examples per QID in the report (keeps it readable)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    request_dir = Path(expanduser(args.request_dir))
    results_dir = Path(expanduser(args.results_dir))
    out_md = Path(expanduser(args.out_md))

    mapping_paths = sorted(request_dir.glob("mapping_shard*.jsonl"))
    if not mapping_paths:
        raise SystemExit(f"No mapping_shard*.jsonl found in {request_dir}")

    norm_path = results_dir / "normalized.jsonl"
    if not norm_path.is_file():
        raise SystemExit(f"normalized.jsonl not found: {norm_path}")

    mapping_rows: list[dict[str, Any]] = []
    for mp in mapping_paths:
        mapping_rows.extend(_read_jsonl(mp))

    sources = _load_group_sources_from_mapping(mapping_rows)
    if not sources:
        raise SystemExit("No usable mapping rows found (empty sources).")

    normalized_rows = _read_jsonl(norm_path)

    anchors_by_qid = _default_anchor_map()

    # Aggregations
    group_missing_source = 0
    non_null_total = 0
    non_null_missing_evidence = 0
    non_null_evidence_not_found = 0

    per_qid = defaultdict(lambda: Counter())
    examples_by_qid: dict[str, list[str]] = defaultdict(list)

    def add_example(qid: str, line: str) -> None:
        if int(len(examples_by_qid[qid])) >= int(args.max_examples_per_qid):
            return
        examples_by_qid[qid].append(line)

    for row in normalized_rows:
        cid = row.get("custom_id")
        if not isinstance(cid, str) or not cid.strip():
            continue

        src = sources.get(cid)
        if src is None:
            group_missing_source += 1
            continue

        answers_by_id = row.get("normalized", {}).get("answers_by_id")
        if not isinstance(answers_by_id, dict):
            continue

        for qid, av in answers_by_id.items():
            if not isinstance(qid, str) or not isinstance(av, dict):
                continue
            ans = av.get("answer")
            if ans is None:
                continue

            non_null_total += 1
            per_qid[qid]["non_null"] += 1

            evidence = av.get("evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                non_null_missing_evidence += 1
                per_qid[qid]["missing_evidence"] += 1
                add_example(qid, f"- {cid}: answer={ans} (MISSING evidence)")
                continue

            # Evidence containment check
            if not src.source_text.strip():
                per_qid[qid]["missing_source_text"] += 1
            else:
                found = _evidence_found(evidence, src.source_text)
                if not found:
                    non_null_evidence_not_found += 1
                    per_qid[qid]["evidence_not_found"] += 1
                    ev_short = _norm_ws(evidence)
                    if len(ev_short) > 220:
                        ev_short = ev_short[:217] + "..."
                    add_example(qid, f"- {cid}: answer={ans} evidence_not_found ev='{ev_short}'")

            # Topic anchor check (heuristic)
            anchors = anchors_by_qid.get(qid)
            if anchors:
                ev_l = evidence.lower()
                if not any(a in ev_l for a in anchors):
                    per_qid[qid]["anchor_mismatch"] += 1
                    ev_short = _norm_ws(evidence)
                    if len(ev_short) > 220:
                        ev_short = ev_short[:217] + "..."
                    add_example(qid, f"- {cid}: answer={ans} anchor_mismatch ev='{ev_short}'")
                else:
                    per_qid[qid]["anchor_match"] += 1

    # Render report
    lines: list[str] = []
    lines.append("# Questionnaire answer alignment audit")
    lines.append("")
    lines.append("This report audits *non-null* questionnaire answers against the **local OCR source text** used to answer them.")
    lines.append("It flags three classes of potential systemic problems:")
    lines.append("- non-null answers with missing evidence")
    lines.append("- evidence strings that cannot be found in the source OCR text")
    lines.append("- evidence that looks off-topic for the question (heuristic topic anchors)")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- request_dir: `{request_dir}`")
    lines.append(f"- results_dir: `{results_dir}`")
    lines.append(f"- normalized: `{norm_path}`")
    lines.append("")
    lines.append("## Top-line counts")
    lines.append(f"- mapped groups available: **{len(sources)}**")
    lines.append(f"- normalized rows scanned: **{len(normalized_rows)}**")
    lines.append(f"- groups missing mapping source: **{group_missing_source}** (should be 0 for consistent runs)")
    lines.append(f"- non-null answers total: **{non_null_total}**")
    lines.append(f"- non-null answers missing evidence: **{non_null_missing_evidence}** ({(non_null_missing_evidence/non_null_total*100.0 if non_null_total else 0.0):.1f}%)")
    lines.append(f"- non-null answers with evidence_not_found: **{non_null_evidence_not_found}** ({(non_null_evidence_not_found/non_null_total*100.0 if non_null_total else 0.0):.1f}%)")
    lines.append("")
    lines.append("## Per-question breakdown (most problematic first)")
    lines.append("")
    lines.append("| qid | non_null | missing_evidence | evidence_not_found | anchor_mismatch | anchor_match |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    def sort_key(item: tuple[str, Counter]) -> tuple[int, int, int, str]:
        qid, c = item
        return (
            int(c.get("missing_evidence", 0)) + int(c.get("evidence_not_found", 0)) + int(c.get("anchor_mismatch", 0)),
            int(c.get("anchor_mismatch", 0)),
            int(c.get("missing_evidence", 0)),
            qid,
        )

    for qid, c in sorted(per_qid.items(), key=sort_key, reverse=True):
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                qid,
                int(c.get("non_null", 0)),
                int(c.get("missing_evidence", 0)),
                int(c.get("evidence_not_found", 0)),
                int(c.get("anchor_mismatch", 0)),
                int(c.get("anchor_match", 0)),
            )
        )

    lines.append("")
    lines.append("## Heuristic anchor definitions")
    lines.append("These are the topic anchors used for the `anchor_mismatch` flag (only a small subset of qids):")
    for qid in sorted(anchors_by_qid.keys()):
        lines.append(f"- `{qid}`: {json.dumps(anchors_by_qid[qid], ensure_ascii=False)}")

    lines.append("")
    lines.append("## Examples (capped)")
    lines.append("Examples are grouped by QID and include the group `custom_id` (slug__year for grouped runs).")
    lines.append("")
    for qid in sorted(examples_by_qid.keys()):
        lines.append(f"### {qid}")
        lines.extend(examples_by_qid[qid])
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote report:", out_md)
    print(
        "non_null_total",
        non_null_total,
        "missing_evidence",
        non_null_missing_evidence,
        "evidence_not_found",
        non_null_evidence_not_found,
    )


if __name__ == "__main__":
    main()

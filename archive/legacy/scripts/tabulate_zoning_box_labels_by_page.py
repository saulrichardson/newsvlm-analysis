#!/usr/bin/env python3
"""
Tabulate zoning box-classifier labels at the *page/png* level.

Input artifacts (from the box-classifier run):
  - mapping_shard*.jsonl  (from the classifier request dir)
  - openai_results_shard*.jsonl (from the classifier results dir)

We count a label for a page_id if ANY box on that page_id received that label.

Notes:
  - A single page can (and usually will) count toward multiple labels.
  - By default we only include boxes whose responses are HTTP 200 + body.status=completed
    + parseable JSON output that contains a string `label`.
  - Failures are tracked separately (box-level and page-level).
  - OpenAI batch output order is not assumed to match mapping order; we join by custom_id.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from os.path import expanduser
from pathlib import Path
from typing import Any


ORDINANCE_AMENDMENT_LABELS = {"full_ordinance", "amendment_substantial", "amendment_targeted"}


def _extract_openai_output_text(body: dict[str, Any]) -> str:
    out = body.get("output")
    if not isinstance(out, list):
        return ""
    parts: list[str] = []
    for item in out:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or []
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get("type") == "output_text":
                parts.append(str(c.get("text") or ""))
    return "".join(parts)


def _coerce_json(text: str) -> dict[str, Any]:
    stripped = str(text).strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", stripped)
        return json.loads(cleaned)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tabulate zoning box labels by unique page/png (page_id).")
    ap.add_argument(
        "--classification-request-dir",
        required=True,
        help="Directory containing mapping_shard*.jsonl from the box-classifier export",
    )
    ap.add_argument(
        "--classification-results-dir",
        required=True,
        help="Directory containing openai_results_shard*.jsonl from the box-classifier download",
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only count boxes with classifier confidence >= this value (default: 0.0)",
    )
    ap.add_argument(
        "--include-labels",
        default="",
        help="Optional comma-separated label allowlist. If set, only those labels are counted.",
    )
    ap.add_argument(
        "--show-ordinance-combos",
        action="store_true",
        help="Also print combination counts for {full_ordinance, amendment_substantial, amendment_targeted}.",
    )
    ap.add_argument(
        "--limit-lines",
        type=int,
        default=0,
        help="Debug: stop after N boxes processed (0 = no limit).",
    )
    return ap.parse_args()


def _fmt_pct(n: int, d: int) -> str:
    return f"{(n / d * 100.0):.1f}%" if d else "0.0%"


def main() -> None:
    args = _parse_args()
    req_dir = Path(expanduser(args.classification_request_dir)).resolve()
    res_dir = Path(expanduser(args.classification_results_dir)).resolve()
    if not req_dir.is_dir():
        raise SystemExit(f"--classification-request-dir not found: {req_dir}")
    if not res_dir.is_dir():
        raise SystemExit(f"--classification-results-dir not found: {res_dir}")

    include_labels = {s.strip() for s in str(args.include_labels).split(",") if s.strip()}

    mapping_paths = sorted(req_dir.glob("mapping_shard*.jsonl"))
    result_paths = sorted(res_dir.glob("openai_results_shard*.jsonl"))
    error_paths = sorted(res_dir.glob("openai_errors_shard*.jsonl"))
    if not mapping_paths:
        raise SystemExit(f"No mapping_shard*.jsonl found in {req_dir}")
    if not result_paths:
        raise SystemExit(f"No openai_results_shard*.jsonl found in {res_dir}")
    if not error_paths:
        raise SystemExit(f"No openai_errors_shard*.jsonl found in {res_dir}")

    pages_total: set[str] = set()
    pages_with_any_ok_box: set[str] = set()
    pages_with_any_failure: set[str] = set()

    mapping_by_id: dict[str, str] = {}

    box_label_counts = Counter()
    page_label_counts = Counter()
    page_label_sets: dict[str, set[str]] = defaultdict(set)
    page_ord_label_sets: dict[str, set[str]] = defaultdict(set)

    boxes_total = 0  # total requests (mapping lines)
    boxes_ok = 0
    boxes_failed = 0
    fail_reasons = Counter()
    missing_mapping = 0

    min_conf = float(args.min_confidence)
    limit = int(args.limit_lines or 0)

    for mp in mapping_paths:
        for raw in mp.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("id")
            page_id = obj.get("page_id")
            if not isinstance(cid, str) or not cid:
                continue
            if not isinstance(page_id, str) or not page_id:
                continue
            mapping_by_id[cid] = page_id
            pages_total.add(page_id)
            boxes_total += 1

    if not mapping_by_id:
        raise SystemExit(f"Loaded 0 mapping ids from {req_dir}")

    def note_failure(custom_id: str, reason: str) -> None:
        nonlocal boxes_failed, missing_mapping
        boxes_failed += 1
        fail_reasons[reason] += 1
        page_id = mapping_by_id.get(custom_id)
        if page_id is None:
            missing_mapping += 1
            return
        pages_with_any_failure.add(page_id)

    def note_ok(custom_id: str, label: str) -> None:
        nonlocal boxes_ok
        boxes_ok += 1
        box_label_counts[label] += 1
        page_id = mapping_by_id.get(custom_id)
        if page_id is None:
            missing_mapping += 1
            return
        pages_with_any_ok_box.add(page_id)
        page_label_sets[page_id].add(label)
        if label in ORDINANCE_AMENDMENT_LABELS:
            page_ord_label_sets[page_id].add(label)

    # Process successful result lines (may still be incomplete/parse-fail)
    processed = 0
    for rp in result_paths:
        with rp.open("r", encoding="utf-8") as f:
            for raw in f:
                if limit and processed >= limit:
                    break
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                custom_id = obj.get("custom_id")
                if not isinstance(custom_id, str) or not custom_id:
                    continue
                processed += 1

                resp = obj.get("response")
                if not isinstance(resp, dict):
                    note_failure(custom_id, "no_response")
                    continue

                status_code = resp.get("status_code")
                if status_code != 200:
                    note_failure(custom_id, f"http_{status_code}")
                    continue

                body = resp.get("body")
                if not isinstance(body, dict) or not body:
                    note_failure(custom_id, "empty_body")
                    continue

                if body.get("status") != "completed":
                    note_failure(custom_id, f"body_status_{body.get('status')}")
                    continue

                output_text = _extract_openai_output_text(body)
                if not output_text.strip():
                    note_failure(custom_id, "empty_output_text")
                    continue

                try:
                    parsed = _coerce_json(output_text)
                except Exception:  # noqa: BLE001
                    note_failure(custom_id, "json_parse_fail")
                    continue

                label = parsed.get("label")
                conf = parsed.get("confidence")

                try:
                    conf_f = float(conf) if isinstance(conf, (int, float)) else 0.0
                except Exception:
                    conf_f = 0.0

                if conf_f < min_conf:
                    continue

                if not isinstance(label, str) or not label:
                    note_failure(custom_id, "missing_label")
                    continue

                if include_labels and label not in include_labels:
                    continue

                note_ok(custom_id, label)

        if limit and processed >= limit:
            break

    # Process OpenAI error-file lines (failed requests)
    for ep in error_paths:
        with ep.open("r", encoding="utf-8") as f:
            for raw in f:
                if limit and processed >= limit:
                    break
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                custom_id = obj.get("custom_id")
                if not isinstance(custom_id, str) or not custom_id:
                    continue
                processed += 1
                note_failure(custom_id, "openai_error_file")

    for pid, labels in page_label_sets.items():
        for lab in labels:
            page_label_counts[lab] += 1

    print(f"classification_request_dir\t{req_dir}")
    print(f"classification_results_dir\t{res_dir}")
    print(f"boxes_total\t{boxes_total}")
    print(f"boxes_ok_counted\t{boxes_ok}")
    print(f"boxes_failed\t{boxes_failed}")
    print(f"missing_mapping_for_result_rows\t{missing_mapping}")
    print(f"unique_pages_total\t{len(pages_total)}")
    print(f"unique_pages_with_any_ok_box\t{len(pages_with_any_ok_box)}")
    print(f"unique_pages_with_any_failure\t{len(pages_with_any_failure)}")
    pages_any_ord = len(page_ord_label_sets)
    print(f"unique_pages_with_any_ordinance_or_amendment\t{pages_any_ord}")
    print(f"unique_pages_with_any_ordinance_or_amendment_pct\t{_fmt_pct(pages_any_ord, len(pages_total))}")
    if fail_reasons:
        print("fail_reason_counts\t" + json.dumps(dict(fail_reasons), ensure_ascii=False))

    total_pages = len(pages_total)
    print("\nlabel\tpages_with_label\tpct_pages\tboxes_with_label")
    for label, n_pages in page_label_counts.most_common():
        print(f"{label}\t{n_pages}\t{_fmt_pct(n_pages, total_pages)}\t{box_label_counts.get(label, 0)}")

    if args.show_ordinance_combos:
        combo_counts = Counter()
        for labels in page_ord_label_sets.values():
            ord_labels = sorted(labels)
            combo_counts[",".join(ord_labels)] += 1
        print("\nordinance_amendment_combo\tpages\tpct_pages")
        for combo, n in combo_counts.most_common():
            print(f"{combo}\t{n}\t{_fmt_pct(n, total_pages)}")


if __name__ == "__main__":
    main()

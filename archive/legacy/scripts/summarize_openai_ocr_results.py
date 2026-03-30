#!/usr/bin/env python3
"""
Summarize downloaded OpenAI OCR batch results (Dell layout text boxes only).

This scans the on-disk JSONLs produced by:
  - scripts/download_openai_batch_results.py

Expected directory layout:
  <RESULTS_ROOT>/part_XXXX/<OPENAI_SUBDIR>/results/
    - openai_results_shardNNN.jsonl
    - openai_errors_shardNNN.jsonl

It reports:
  - how many pages (png stems) are represented
  - how many box requests succeeded vs unreadable vs failed (and why)
  - page-level readiness buckets for the next pipeline stage (e.g. zoning classification)

Optionally, it writes useful page-id manifests (one per line) into --out-dir.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from os.path import expanduser
from pathlib import Path

from newsvlm.models import TEXT_CLASSES


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize OpenAI OCR batch results JSONLs.")
    ap.add_argument(
        "--results-root",
        required=True,
        help="Root containing part_XXXX directories",
    )
    ap.add_argument(
        "--openai-subdir",
        default="openai_gpt52_reasoning_medium_split",
        help="Subdirectory under each part dir that contains OpenAI request/results",
    )
    ap.add_argument(
        "--layout-root",
        default=None,
        help="Optional: directory containing <page_id>.json Dell layout files (for existence sanity-check)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Optional: write page-id manifests and summary.json here",
    )
    ap.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional: stop after seeing this many unique pages (useful for quick smoke checks)",
    )
    ap.add_argument(
        "--top-k-slugs",
        type=int,
        default=10,
        help="How many newspaper slugs to print (best-effort parse from page_id)",
    )
    return ap.parse_args()


def _extract_output_text(body: dict) -> str:
    # OpenAI Responses API returns `output: [{... content: [{type: output_text, text: ...}, ...]}]`.
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


def _quantile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        raise ValueError("empty values")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    idx = int(round((len(sorted_vals) - 1) * q))
    return sorted_vals[idx]


def _safe_write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(f"{ln}\n" for ln in lines), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = _parse_args()
    results_root = Path(expanduser(args.results_root)).resolve()
    openai_subdir = args.openai_subdir.strip()
    layout_root = Path(expanduser(args.layout_root)).resolve() if args.layout_root else None
    out_dir = Path(expanduser(args.out_dir)).resolve() if args.out_dir else None

    if not results_root.is_dir():
        raise SystemExit(f"--results-root is not a directory: {results_root}")
    if layout_root is not None and not layout_root.is_dir():
        raise SystemExit(f"--layout-root is not a directory: {layout_root}")

    pattern = f"part_*/{openai_subdir}/results"
    result_files = sorted(results_root.glob(f"{pattern}/openai_results_shard*.jsonl"))
    error_files = sorted(results_root.glob(f"{pattern}/openai_errors_shard*.jsonl"))
    if not result_files and not error_files:
        raise SystemExit(f"No OpenAI result/error shard JSONLs found under: {results_root}/{pattern}")

    # Per-page stats
    per_page: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "results": 0,
            "errors": 0,
            "ok": 0,
            "unreadable": 0,
            "parsefail": 0,
            "invalid_status": 0,
            "ok_chars": 0,
        }
    )

    status_counts: Counter[str | None] = Counter()
    error_type_counts: Counter[str] = Counter()

    # Parse result JSONLs
    pages_seen: set[str] = set()

    for fp in result_files:
        with fp.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = str(obj.get("custom_id") or "")
                page_id = cid.rsplit(":", 1)[0] if ":" in cid else cid
                per_page[page_id]["results"] += 1
                pages_seen.add(page_id)

                body = ((obj.get("response") or {}).get("body") or {})
                if not isinstance(body, dict):
                    per_page[page_id]["parsefail"] += 1
                    continue
                text = _extract_output_text(body)
                try:
                    parsed = json.loads(text)
                except Exception:
                    per_page[page_id]["parsefail"] += 1
                    continue
                if not isinstance(parsed, dict):
                    per_page[page_id]["parsefail"] += 1
                    continue
                st = parsed.get("status")
                status_counts[st] += 1
                if st == "ok":
                    per_page[page_id]["ok"] += 1
                    tx = str(parsed.get("transcript") or "")
                    if tx.strip():
                        per_page[page_id]["ok_chars"] += len(tx)
                elif st == "unreadable":
                    per_page[page_id]["unreadable"] += 1
                else:
                    # Parsed JSON but did not match our constrained OCR schema.
                    # Treat as unusable for downstream pipelines.
                    per_page[page_id]["invalid_status"] += 1

                if args.max_pages and len(pages_seen) >= int(args.max_pages):
                    break
        if args.max_pages and len(pages_seen) >= int(args.max_pages):
            break

    # Parse error JSONLs
    for fp in error_files:
        with fp.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = str(obj.get("custom_id") or "")
                page_id = cid.rsplit(":", 1)[0] if ":" in cid else cid
                per_page[page_id]["errors"] += 1
                pages_seen.add(page_id)

                body = ((obj.get("response") or {}).get("body") or {})
                if isinstance(body, dict):
                    err = body.get("error")
                    if isinstance(err, dict):
                        et = err.get("type")
                        if isinstance(et, str) and et.strip():
                            error_type_counts[et.strip()] += 1

                if args.max_pages and len(pages_seen) >= int(args.max_pages):
                    break
        if args.max_pages and len(pages_seen) >= int(args.max_pages):
            break

    pages = sorted(per_page.keys())

    # Buckets for downstream pipeline use
    pages_with_ok = [pid for pid in pages if per_page[pid]["ok"] > 0]
    pages_no_ok = [pid for pid in pages if per_page[pid]["ok"] == 0]
    pages_with_any_error = [pid for pid in pages if per_page[pid]["errors"] > 0]
    pages_with_no_error = [pid for pid in pages if per_page[pid]["errors"] == 0]
    pages_no_err_no_parse = [pid for pid in pages if per_page[pid]["errors"] == 0 and per_page[pid]["parsefail"] == 0]
    pages_no_err_no_parse_or_invalid = [
        pid
        for pid in pages
        if per_page[pid]["errors"] == 0 and per_page[pid]["parsefail"] == 0 and per_page[pid]["invalid_status"] == 0
    ]
    pages_all_boxes_ok = [
        pid
        for pid in pages
        if per_page[pid]["errors"] == 0 and per_page[pid]["parsefail"] == 0 and per_page[pid]["unreadable"] == 0
    ]

    total_results = sum(per_page[pid]["results"] for pid in pages)
    total_errors = sum(per_page[pid]["errors"] for pid in pages)
    total_ok = sum(per_page[pid]["ok"] for pid in pages)
    total_unreadable = sum(per_page[pid]["unreadable"] for pid in pages)
    total_parsefail = sum(per_page[pid]["parsefail"] for pid in pages)
    total_invalid_status = sum(per_page[pid]["invalid_status"] for pid in pages)

    boxes_per_page = sorted(per_page[pid]["results"] + per_page[pid]["errors"] for pid in pages)
    ok_chars_per_page = sorted(per_page[pid]["ok_chars"] for pid in pages_with_ok)

    # Year/slug parse (best effort)
    year_re = re.compile(r"-(\d{4})-p-\d+$")
    decade_counts: Counter[int] = Counter()
    year_counts: Counter[int] = Counter()
    slug_counts: Counter[str] = Counter()
    years_parsed = 0

    for pid in pages:
        m = year_re.search(pid)
        if m:
            y = int(m.group(1))
            years_parsed += 1
            year_counts[y] += 1
            decade_counts[(y // 10) * 10] += 1
        parts = pid.split("-")
        if len(parts) >= 6:
            slug = "-".join(parts[:-5])
            if slug:
                slug_counts[slug] += 1

    # Layout existence + expected text-box count sanity-check (optional).
    #
    # If layout_root is provided, we compute the expected number of text boxes
    # from the Dell layout JSONs (filtered to TEXT_CLASSES), and compare it to
    # the observed count in the OpenAI batch output (results + errors).
    layout_exists: int | None = None
    layout_missing: int | None = None
    expected_text_boxes: dict[str, int] | None = None

    pages_fully_covered: list[str] | None = None
    pages_missing_boxes_vs_layout: list[str] | None = None
    pages_extra_boxes_vs_layout: list[str] | None = None
    pages_fully_covered_no_error: list[str] | None = None
    pages_fully_covered_no_error_no_parse_or_invalid: list[str] | None = None
    pages_fully_covered_all_boxes_ok: list[str] | None = None

    if layout_root is not None:
        expected_text_boxes = {}
        exists = 0
        missing = 0

        for pid in pages:
            lp = layout_root / f"{pid}.json"
            if not lp.is_file():
                missing += 1
                continue
            exists += 1
            try:
                data = json.loads(lp.read_text(encoding="utf-8"))
            except Exception:
                # Treat unreadable JSON as missing for our expected-count check.
                missing += 1
                exists -= 1
                continue
            bboxes = data.get("bboxes") or []
            if not isinstance(bboxes, list):
                continue
            expected = 0
            for b in bboxes:
                if not isinstance(b, dict):
                    continue
                if b.get("class") in TEXT_CLASSES:
                    expected += 1
            expected_text_boxes[pid] = expected

        layout_exists = exists
        layout_missing = missing

        pages_fully_covered = []
        pages_missing_boxes_vs_layout = []
        pages_extra_boxes_vs_layout = []

        for pid, expected in expected_text_boxes.items():
            observed = per_page[pid]["results"] + per_page[pid]["errors"]
            if observed == expected:
                pages_fully_covered.append(pid)
            elif observed < expected:
                pages_missing_boxes_vs_layout.append(pid)
            else:
                pages_extra_boxes_vs_layout.append(pid)

        pages_fully_covered.sort()
        pages_missing_boxes_vs_layout.sort()
        pages_extra_boxes_vs_layout.sort()

        pages_fully_covered_no_error = [pid for pid in pages_fully_covered if per_page[pid]["errors"] == 0]
        pages_fully_covered_no_error_no_parse_or_invalid = [
            pid
            for pid in pages_fully_covered_no_error
            if per_page[pid]["parsefail"] == 0 and per_page[pid]["invalid_status"] == 0
        ]
        pages_fully_covered_all_boxes_ok = [
            pid
            for pid in pages_fully_covered_no_error_no_parse_or_invalid
            if per_page[pid]["unreadable"] == 0
        ]

    print(f"results_root\t{results_root}")
    print(f"openai_subdir\t{openai_subdir}")
    print(f"result_files\t{len(result_files)}")
    print(f"error_files\t{len(error_files)}")
    if layout_root is not None:
        print(f"layout_root\t{layout_root}")
        print(f"layouts_found\t{layout_exists}")
        print(f"layouts_missing\t{layout_missing}")

    print(f"unique_pages\t{len(pages)}")
    print(f"boxes_total\t{total_results + total_errors}")
    print(f"boxes_results\t{total_results}")
    print(f"boxes_errors\t{total_errors}")
    print(f"boxes_ok\t{total_ok}")
    print(f"boxes_unreadable\t{total_unreadable}")
    print(f"boxes_parsefail\t{total_parsefail}")
    print(f"boxes_invalid_status\t{total_invalid_status}")
    print(f"status_counts\t{json.dumps(dict(status_counts), ensure_ascii=False)}")
    if error_type_counts:
        print(f"error_types\t{json.dumps(dict(error_type_counts), ensure_ascii=False)}")

    print(f"pages_with_ok\t{len(pages_with_ok)}")
    print(f"pages_no_ok\t{len(pages_no_ok)}")
    print(f"pages_with_any_error\t{len(pages_with_any_error)}")
    print(f"pages_with_no_error\t{len(pages_with_no_error)}")
    print(f"pages_no_err_no_parse\t{len(pages_no_err_no_parse)}")
    print(f"pages_no_err_no_parse_or_invalid\t{len(pages_no_err_no_parse_or_invalid)}")
    print(f"pages_all_boxes_ok\t{len(pages_all_boxes_ok)}")

    if layout_root is not None and expected_text_boxes is not None:
        assert pages_fully_covered is not None
        assert pages_missing_boxes_vs_layout is not None
        assert pages_extra_boxes_vs_layout is not None
        assert pages_fully_covered_no_error is not None
        assert pages_fully_covered_no_error_no_parse_or_invalid is not None
        assert pages_fully_covered_all_boxes_ok is not None
        print(f"pages_obs_eq_expected\t{len(pages_fully_covered)}")
        print(f"pages_obs_lt_expected\t{len(pages_missing_boxes_vs_layout)}")
        print(f"pages_obs_gt_expected\t{len(pages_extra_boxes_vs_layout)}")
        print(f"pages_fully_covered_no_error\t{len(pages_fully_covered_no_error)}")
        print(
            f"pages_fully_covered_no_error_no_parse_or_invalid\t{len(pages_fully_covered_no_error_no_parse_or_invalid)}"
        )
        print(f"pages_fully_covered_all_boxes_ok\t{len(pages_fully_covered_all_boxes_ok)}")

    if boxes_per_page:
        print(f"boxes_per_page_p50\t{_quantile(boxes_per_page, 0.50)}")
        print(f"boxes_per_page_p90\t{_quantile(boxes_per_page, 0.90)}")
        print(f"boxes_per_page_max\t{boxes_per_page[-1]}")

    if ok_chars_per_page:
        print(f"ok_chars_per_page_p50\t{_quantile(ok_chars_per_page, 0.50)}")
        print(f"ok_chars_per_page_p90\t{_quantile(ok_chars_per_page, 0.90)}")
        print(f"ok_chars_per_page_p99\t{_quantile(ok_chars_per_page, 0.99)}")
        print(f"ok_chars_per_page_max\t{ok_chars_per_page[-1]}")

    if years_parsed:
        years = sorted(year_counts.keys())
        print(f"year_min\t{years[0]}")
        print(f"year_max\t{years[-1]}")
        # Print a compact decade histogram line for quick scanning.
        decades = sorted(decade_counts.keys())
        decade_str = ", ".join(f"{d}:{decade_counts[d]}" for d in decades)
        print(f"decades\t{decade_str}")

    top_k = max(0, int(args.top_k_slugs))
    if top_k and slug_counts:
        print(f"top_slugs\t{json.dumps(slug_counts.most_common(top_k), ensure_ascii=False)}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "results_root": str(results_root),
            "openai_subdir": openai_subdir,
            "result_files": len(result_files),
            "error_files": len(error_files),
            "unique_pages": len(pages),
            "boxes_total": total_results + total_errors,
            "boxes_results": total_results,
            "boxes_errors": total_errors,
            "boxes_ok": total_ok,
            "boxes_unreadable": total_unreadable,
            "boxes_parsefail": total_parsefail,
            "boxes_invalid_status": total_invalid_status,
            "status_counts": dict(status_counts),
            "error_types": dict(error_type_counts),
            "pages_with_ok": len(pages_with_ok),
            "pages_no_ok": len(pages_no_ok),
            "pages_with_any_error": len(pages_with_any_error),
            "pages_with_no_error": len(pages_with_no_error),
            "pages_no_err_no_parse": len(pages_no_err_no_parse),
            "pages_no_err_no_parse_or_invalid": len(pages_no_err_no_parse_or_invalid),
            "pages_all_boxes_ok": len(pages_all_boxes_ok),
            "year_min": min(year_counts.keys()) if year_counts else None,
            "year_max": max(year_counts.keys()) if year_counts else None,
            "top_slugs": slug_counts.most_common(top_k) if top_k else [],
        }
        if layout_root is not None and expected_text_boxes is not None:
            assert pages_fully_covered is not None
            assert pages_missing_boxes_vs_layout is not None
            assert pages_extra_boxes_vs_layout is not None
            assert pages_fully_covered_no_error is not None
            assert pages_fully_covered_no_error_no_parse_or_invalid is not None
            assert pages_fully_covered_all_boxes_ok is not None
            summary["layout_root"] = str(layout_root)
            summary["expected_box_counts"] = {
                "pages_obs_eq_expected": len(pages_fully_covered),
                "pages_obs_lt_expected": len(pages_missing_boxes_vs_layout),
                "pages_obs_gt_expected": len(pages_extra_boxes_vs_layout),
                "pages_fully_covered_no_error": len(pages_fully_covered_no_error),
                "pages_fully_covered_no_error_no_parse_or_invalid": len(pages_fully_covered_no_error_no_parse_or_invalid),
                "pages_fully_covered_all_boxes_ok": len(pages_fully_covered_all_boxes_ok),
            }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        _safe_write_lines(out_dir / "pages_with_ok.txt", pages_with_ok)
        _safe_write_lines(out_dir / "pages_no_err_no_parse.txt", pages_no_err_no_parse)
        _safe_write_lines(out_dir / "pages_with_no_error.txt", pages_with_no_error)
        _safe_write_lines(out_dir / "pages_all_boxes_ok.txt", pages_all_boxes_ok)
        _safe_write_lines(out_dir / "pages_with_any_error.txt", pages_with_any_error)
        _safe_write_lines(out_dir / "pages_no_ok.txt", pages_no_ok)
        _safe_write_lines(out_dir / "pages_no_err_no_parse_or_invalid.txt", pages_no_err_no_parse_or_invalid)
        if layout_root is not None and expected_text_boxes is not None:
            assert pages_fully_covered is not None
            assert pages_missing_boxes_vs_layout is not None
            assert pages_extra_boxes_vs_layout is not None
            assert pages_fully_covered_no_error is not None
            assert pages_fully_covered_no_error_no_parse_or_invalid is not None
            assert pages_fully_covered_all_boxes_ok is not None
            _safe_write_lines(out_dir / "pages_fully_covered.txt", pages_fully_covered)
            _safe_write_lines(out_dir / "pages_missing_boxes_vs_layout.txt", pages_missing_boxes_vs_layout)
            _safe_write_lines(out_dir / "pages_extra_boxes_vs_layout.txt", pages_extra_boxes_vs_layout)
            _safe_write_lines(out_dir / "pages_fully_covered_no_error.txt", pages_fully_covered_no_error)
            _safe_write_lines(
                out_dir / "pages_fully_covered_no_error_no_parse_or_invalid.txt",
                pages_fully_covered_no_error_no_parse_or_invalid,
            )
            _safe_write_lines(out_dir / "pages_fully_covered_all_boxes_ok.txt", pages_fully_covered_all_boxes_ok)

        print(f"wrote_out_dir\t{out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from os.path import expanduser
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize page QA outputs (tabular).")
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing <page_id>.qa.json outputs and manifest.jsonl",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many examples to print for reruns/errors (0 disables)",
    )
    return p.parse_args()


def _fmt_pct(n: int, d: int) -> str:
    return f"{(n / d * 100.0):.1f}%" if d else "0.0%"


def _quantile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    idx = int(round((len(sorted_vals) - 1) * q))
    return sorted_vals[idx]


def _parse_slug_from_page_id(page_id: Any) -> str | None:
    if not isinstance(page_id, str):
        return None
    pid = page_id.strip()
    if not pid:
        return None
    parts = pid.split("-")
    if len(parts) < 6:
        return None
    slug = "-".join(parts[:-5]).strip()
    return slug or None


def load_manifest_rows(out_dir: Path) -> list[dict[str, Any]]:
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest.jsonl not found in {out_dir}")
    rows: list[dict[str, Any]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise SystemExit(f"Expected JSON object in {manifest_path}, got {type(obj).__name__}")
        rows.append(obj)
    return rows


def _load_page_qa(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def main() -> None:
    args = parse_args()
    out_dir = Path(expanduser(args.output_dir))
    if not out_dir.is_dir():
        raise SystemExit(f"--output-dir not found or not a directory: {out_dir}")

    rows = load_manifest_rows(out_dir)
    total = len(rows)

    status_counts = Counter((r.get("status") or "<missing>") for r in rows)
    err_rows = [r for r in rows if r.get("status") == "error" or r.get("quality_score") is None]
    okish_rows = [r for r in rows if r.get("status") in {"ok", "needs_rerun"}]

    print(f"output_dir\t{out_dir}")
    print(f"total_pages\t{total}")
    for k, v in status_counts.most_common():
        print(f"status.{k}\t{v}\t{_fmt_pct(v, total)}")

    qa_models = Counter(r.get("qa_model") for r in rows)
    src_models = Counter(r.get("source_model") for r in rows)
    if qa_models:
        print("qa_models\t" + json.dumps(dict(qa_models), ensure_ascii=False))
    if src_models:
        print("source_models\t" + json.dumps(dict(src_models), ensure_ascii=False))

    scores = [float(r["quality_score"]) for r in okish_rows if isinstance(r.get("quality_score"), (int, float))]
    if scores:
        scores_sorted = sorted(scores)
        print("\nquality_score_avg\t{:.1f}".format(sum(scores_sorted) / len(scores_sorted)))
        print("quality_score_p10\t{:.1f}".format(_quantile(scores_sorted, 0.10) or 0.0))
        print("quality_score_p25\t{:.1f}".format(_quantile(scores_sorted, 0.25) or 0.0))
        print("quality_score_p50\t{:.1f}".format(statistics.median(scores_sorted)))
        print("quality_score_p75\t{:.1f}".format(_quantile(scores_sorted, 0.75) or 0.0))
        print("quality_score_p90\t{:.1f}".format(_quantile(scores_sorted, 0.90) or 0.0))
        print("quality_score_min\t{:.1f}".format(scores_sorted[0]))
        print("quality_score_max\t{:.1f}".format(scores_sorted[-1]))

    # Slug breakdown
    slug_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        slug = _parse_slug_from_page_id(r.get("page_id"))
        if slug:
            slug_rows[slug].append(r)

    if slug_rows:
        print("\nslug\tpages\tneeds_rerun\tneeds_rerun_pct\tavg_quality_score")
        for slug, srows in sorted(slug_rows.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            n = len(srows)
            nr = sum(1 for r in srows if r.get("status") == "needs_rerun")
            qs = [float(r["quality_score"]) for r in srows if isinstance(r.get("quality_score"), (int, float))]
            avg_q = (sum(qs) / len(qs)) if qs else None
            print(f"{slug}\t{n}\t{nr}\t{_fmt_pct(nr, n)}\t{avg_q:.1f}" if avg_q is not None else f"{slug}\t{n}\t{nr}\t{_fmt_pct(nr, n)}\t")

    # Issues: load from per-page QA JSON outputs (manifest doesn't include issue types).
    issue_type_counts = Counter()
    issue_severity_counts = Counter()
    for r in okish_rows:
        out_path = r.get("output_path")
        if not isinstance(out_path, str) or not out_path.strip():
            continue
        qa_obj = _load_page_qa(Path(out_path))
        if not qa_obj:
            continue
        qa = qa_obj.get("qa")
        if not isinstance(qa, dict):
            continue
        issues = qa.get("issues")
        if not isinstance(issues, list):
            continue
        for iss in issues:
            if not isinstance(iss, dict):
                continue
            t = iss.get("type")
            sev = iss.get("severity")
            if isinstance(t, str) and t.strip():
                issue_type_counts[t.strip()] += 1
            if isinstance(sev, str) and sev.strip():
                issue_severity_counts[sev.strip()] += 1

    if issue_severity_counts:
        print("\nissue_severity\tcount")
        for k, v in issue_severity_counts.most_common():
            print(f"{k}\t{v}")
    if issue_type_counts:
        print("\nissue_type\tcount")
        for k, v in issue_type_counts.most_common(25):
            print(f"{k}\t{v}")

    # Reruns + errors (examples)
    if args.top_k and args.top_k > 0:
        needs_rerun = [r for r in rows if r.get("status") == "needs_rerun"]
        needs_rerun_sorted = sorted(
            needs_rerun,
            key=lambda r: (
                float(r["quality_score"]) if isinstance(r.get("quality_score"), (int, float)) else 1e9,
                str(r.get("page_id") or ""),
            ),
        )
        if needs_rerun_sorted:
            print("\nneeds_rerun_examples_low_quality")
            for r in needs_rerun_sorted[: int(args.top_k)]:
                pid = r.get("page_id")
                qs = r.get("quality_score")
                op = r.get("output_path")
                print(f"- {pid}\tquality_score={qs}\t{op}")

        if err_rows:
            err_counts = Counter()
            for r in err_rows:
                e = r.get("error")
                if isinstance(e, dict):
                    msg = e.get("message") or json.dumps(e, ensure_ascii=False)
                else:
                    msg = str(e)
                err_counts[msg] += 1
            print("\nerror_count\tmessage")
            for msg, n in err_counts.most_common(20):
                one_line = " ".join(str(msg).split())
                if len(one_line) > 160:
                    one_line = one_line[:157] + "..."
                print(f"{n}\t{one_line}")


if __name__ == "__main__":
    main()


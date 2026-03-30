#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from os.path import expanduser
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize zoning classifier outputs (tabular).")
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing <page_id>.zoning.json outputs and manifest.jsonl",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="If >0, print top-k highest confidence examples per label",
    )
    return p.parse_args()


def _fmt_pct(n: int, d: int) -> str:
    return f"{(n / d * 100.0):.1f}%" if d else "0.0%"


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("empty")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    idx = int(round((len(sorted_vals) - 1) * q))
    return sorted_vals[idx]


def load_manifest_rows(out_dir: Path) -> list[dict]:
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest.jsonl not found in {out_dir}")
    rows: list[dict] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(expanduser(args.output_dir))
    if not out_dir.is_dir():
        raise SystemExit(f"--output-dir not found or not a directory: {out_dir}")

    rows = load_manifest_rows(out_dir)
    err_rows = [r for r in rows if r.get("error")]
    ok_rows = [r for r in rows if not r.get("error")]

    print(f"output_dir\t{out_dir}")
    print(f"total_pages\t{len(rows)}")
    print(f"ok_pages\t{len(ok_rows)}")
    print(f"error_pages\t{len(err_rows)}")

    # Models sanity
    src_models = Counter(r.get("source_model") for r in rows)
    cls_models = Counter(r.get("classifier_model") for r in rows)
    if src_models:
        print("source_models\t" + json.dumps(dict(src_models), ensure_ascii=False))
    if cls_models:
        print("classifier_models\t" + json.dumps(dict(cls_models), ensure_ascii=False))

    # Label distribution + confidence stats
    label_counts = Counter((r.get("label") or "<missing>") for r in ok_rows)
    conf_by_label: dict[str, list[float]] = defaultdict(list)
    for r in ok_rows:
        label = r.get("label") or "<missing>"
        c = r.get("confidence")
        if isinstance(c, (int, float)):
            conf_by_label[label].append(float(c))

    total_ok = len(ok_rows)
    print("\nlabel\tcount\tpct\tavg_conf\tmedian_conf\tmin_conf\tmax_conf")
    for label, count in label_counts.most_common():
        confs = sorted(conf_by_label.get(label, []))
        if confs:
            avg = sum(confs) / len(confs)
            med = statistics.median(confs)
            print(
                f"{label}\t{count}\t{_fmt_pct(count, total_ok)}\t{avg:.3f}\t{med:.3f}\t{confs[0]:.3f}\t{confs[-1]:.3f}"
            )
        else:
            print(f"{label}\t{count}\t{_fmt_pct(count, total_ok)}\t\t\t\t")

    # Present flags
    present_counts = Counter()
    for r in ok_rows:
        present = r.get("present")
        if isinstance(present, dict):
            for k, v in present.items():
                if v is True:
                    present_counts[k] += 1

    print("\npresent_flag\ttrue_count\tpct_of_pages")
    for k, v in present_counts.most_common():
        print(f"{k}\t{v}\t{_fmt_pct(v, total_ok)}")

    # Performance
    durs = [float(r["duration_ms"]) for r in rows if isinstance(r.get("duration_ms"), (int, float))]
    attempts = [int(r["attempts"]) for r in rows if isinstance(r.get("attempts"), int)]
    if durs:
        durs_sorted = sorted(durs)
        print("\nduration_ms_avg\t{:.1f}".format(sum(durs) / len(durs)))
        print("duration_ms_p50\t{:.1f}".format(_quantile(durs_sorted, 0.50)))
        print("duration_ms_p95\t{:.1f}".format(_quantile(durs_sorted, 0.95)))
        print("duration_ms_min\t{:.1f}".format(durs_sorted[0]))
        print("duration_ms_max\t{:.1f}".format(durs_sorted[-1]))
    if attempts:
        print("\nattempts_avg\t{:.3f}".format(sum(attempts) / len(attempts)))
        print("attempts_max\t{}".format(max(attempts)))

    # Errors
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

    # Top-k examples
    if args.top_k and args.top_k > 0:
        best: dict[str, list[tuple[float, dict]]] = defaultdict(list)
        for r in ok_rows:
            label = r.get("label") or "<missing>"
            c = r.get("confidence")
            if not isinstance(c, (int, float)):
                continue
            best[label].append((float(c), r))
        print("\nexamples_by_label")
        for label in sorted(best.keys()):
            items = sorted(best[label], key=lambda t: t[0], reverse=True)[: int(args.top_k)]
            print(f"{label}:")
            for c, r in items:
                print(f"  - {r.get('page_id')}\t{c:.3f}")


if __name__ == "__main__":
    main()


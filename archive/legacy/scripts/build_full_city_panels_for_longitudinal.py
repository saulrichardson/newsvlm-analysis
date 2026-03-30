#!/usr/bin/env python3
"""
Build a forked run directory with full per-city issue coverage for selected city keys.

This script reconstructs issue_texts.jsonl for each selected city_key using all
available local-text issues (not evenly-spaced sampled panels).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _load_proto_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("prototype_zoning_panel_analysis", str(script_path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load module spec from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build full-coverage city panels for longitudinal fork runs.")
    ap.add_argument(
        "--source-run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60",
        help="Source run directory containing provenance + pilot manifests.",
    )
    ap.add_argument(
        "--output-run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
        help="Output fork run directory.",
    )
    ap.add_argument(
        "--city-keys-file",
        default="",
        help="Path to JSON/CSV/TXT with city keys. Default: <source-run-dir>/longitudinal/pilot20_selection_manifest.json",
    )
    ap.add_argument(
        "--prototype-script",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/scripts/prototype_zoning_panel_analysis.py",
        help="Path to prototype_zoning_panel_analysis.py for helper functions.",
    )
    ap.add_argument("--issue-panel", default="", help="Override issue panel path.")
    ap.add_argument("--merged-manifest", default="", help="Override merged manifest path.")
    ap.add_argument(
        "--vlm-roots",
        default="",
        help="Override comma-separated VLM roots. Default from source provenance.",
    )
    return ap.parse_args()


def _parse_city_keys(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"City-keys file not found: {path}")
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("selected_city_keys"), list):
            return [str(x).strip() for x in obj["selected_city_keys"] if str(x).strip()]
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        raise SystemExit(f"Unsupported JSON for city keys: {path}")
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        d = pd.read_csv(path, sep=sep)
        if "city_key" not in d.columns:
            raise SystemExit(f"city_key column missing in {path}")
        return [str(x).strip() for x in d["city_key"].tolist() if str(x).strip()]
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.extend([x.strip() for x in s.split(",") if x.strip()])
    return out


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    source_run_dir = Path(args.source_run_dir).expanduser().resolve()
    output_run_dir = Path(args.output_run_dir).expanduser().resolve()
    output_run_dir.mkdir(parents=True, exist_ok=True)

    provenance_path = source_run_dir / "provenance.json"
    if not provenance_path.is_file():
        raise SystemExit(f"Missing source provenance: {provenance_path}")
    prov = json.loads(provenance_path.read_text(encoding="utf-8"))

    city_keys_file = (
        Path(args.city_keys_file).expanduser().resolve()
        if str(args.city_keys_file).strip()
        else (source_run_dir / "longitudinal" / "pilot20_selection_manifest.json")
    )
    city_keys = _parse_city_keys(city_keys_file)
    if not city_keys:
        raise SystemExit("No city keys found for fork build.")

    issue_panel_path = Path(args.issue_panel).expanduser().resolve() if str(args.issue_panel).strip() else Path(str(prov.get("issue_panel", ""))).expanduser().resolve()
    merged_manifest_path = (
        Path(args.merged_manifest).expanduser().resolve()
        if str(args.merged_manifest).strip()
        else Path(str(prov.get("merged_manifest", ""))).expanduser().resolve()
    )
    vlm_roots_arg = str(args.vlm_roots).strip()
    if vlm_roots_arg:
        vlm_roots_csv = vlm_roots_arg
    else:
        roots = prov.get("vlm_roots") or []
        if isinstance(roots, list):
            vlm_roots_csv = ",".join(str(x) for x in roots if str(x).strip())
        else:
            vlm_roots_csv = str(roots)

    prototype_script = Path(args.prototype_script).expanduser().resolve()
    proto = _load_proto_module(prototype_script)

    issue_df = proto._load_table(issue_panel_path)
    issue_df["state_abbr"] = issue_df["state_abbr"].astype(str).str.lower()
    issue_df["issue_date"] = pd.to_datetime(issue_df["issue_date"], errors="coerce")
    issue_df = issue_df.dropna(subset=["issue_date"]).copy()

    issue_manifest = proto._load_issue_manifest(merged_manifest_path)
    vlm_roots = proto._parse_csv_paths(vlm_roots_csv)

    selected_issue_source = source_run_dir / "panels" / "selected_panel_issues.csv"
    if not selected_issue_source.is_file():
        raise SystemExit(f"Missing source selected_panel_issues.csv: {selected_issue_source}")
    source_sel = pd.read_csv(selected_issue_source)
    meta_cols = ["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy"]
    source_meta = source_sel[meta_cols].drop_duplicates(subset=["city_key"]).copy()
    source_meta["state_abbr"] = source_meta["state_abbr"].astype(str).str.lower()
    meta_map = {str(r.city_key): r for r in source_meta.itertuples(index=False)}

    panels_dir = output_run_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    selected_issue_rows: list[dict[str, Any]] = []
    selected_panel_rows: list[dict[str, Any]] = []

    for city_key in city_keys:
        m = meta_map.get(str(city_key))
        if m is None:
            raise SystemExit(f"City key missing from source selected_panel_issues metadata: {city_key}")

        city_name = str(m.city_name)
        st = str(m.state_abbr).lower()
        region = str(m.region)
        urbanicity_proxy = str(m.urbanicity_proxy)

        g = issue_df[(issue_df["city_name"] == city_name) & (issue_df["state_abbr"] == st)].copy()
        if g.empty:
            continue
        g = g.sort_values(["issue_date", "issue_id"]).drop_duplicates(subset=["issue_id"], keep="first").reset_index(drop=True)

        rows_for_jsonl: list[dict[str, Any]] = []
        for r in g.itertuples(index=False):
            iid = str(r.issue_id)
            man = issue_manifest.get(iid)
            if not man:
                continue
            pages = [p for p in (man.get("pages_with_zoning") or []) if isinstance(p, dict) and p.get("page_id")]
            pages = sorted(pages, key=lambda x: proto._page_sort_key(str(x.get("page_id"))))

            page_texts: list[str] = []
            page_ids: list[str] = []
            for p in pages:
                pid = str(p.get("page_id"))
                fp = proto._find_page_file(pid, vlm_roots)
                if fp is None:
                    continue
                txt = proto._extract_text_from_vlm_page(fp)
                if txt:
                    page_texts.append(txt)
                    page_ids.append(pid)
            text = "\n\n".join(page_texts).strip()
            if not text:
                continue

            issue_date = str(pd.Timestamp(r.issue_date).date())
            row = {
                "issue_id": iid,
                "city_name": city_name,
                "state_abbr": st,
                "city_key": city_key,
                "region": region,
                "urbanicity_proxy": urbanicity_proxy,
                "issue_date": issue_date,
                "classification_label": str(r.classification_label),
                "pub_slug": str(getattr(r, "pub_slug", "") or ""),
                "page_ids": page_ids,
                "text": text,
                "text_chars": int(len(text)),
                "page_count": int(len(page_ids)),
            }
            rows_for_jsonl.append(row)
            selected_issue_rows.append(
                {
                    "city_key": city_key,
                    "city_name": city_name,
                    "state_abbr": st,
                    "region": region,
                    "urbanicity_proxy": urbanicity_proxy,
                    "issue_id": iid,
                    "issue_date": issue_date,
                    "classification_label": str(r.classification_label),
                    "text_chars": int(len(text)),
                    "page_count": int(len(page_ids)),
                }
            )

        rows_for_jsonl = sorted(rows_for_jsonl, key=lambda x: (str(x.get("issue_date")), str(x.get("issue_id"))))
        if not rows_for_jsonl:
            continue

        _write_jsonl(rows_for_jsonl, panels_dir / city_key / "issue_texts.jsonl")
        selected_panel_rows.append(
            {
                "city_key": city_key,
                "city_name": city_name,
                "state_abbr": st,
                "region": region,
                "urbanicity_proxy": urbanicity_proxy,
                "selected_issue_count": int(len(rows_for_jsonl)),
                "issue_date_min": str(rows_for_jsonl[0]["issue_date"]),
                "issue_date_max": str(rows_for_jsonl[-1]["issue_date"]),
            }
        )

    selected_issue_df = pd.DataFrame.from_records(selected_issue_rows)
    if selected_issue_df.empty:
        raise SystemExit("No full-coverage issue rows were reconstructed.")
    selected_issue_df = selected_issue_df.sort_values(["city_key", "issue_date", "issue_id"]).reset_index(drop=True)
    selected_issue_df.to_csv(panels_dir / "selected_panel_issues.csv", index=False)

    selected_panels_df = pd.DataFrame.from_records(selected_panel_rows).sort_values(["city_key"]).reset_index(drop=True)
    selected_panels_df.to_csv(panels_dir / "selected_panels.csv", index=False)

    # Carry over contamination table for downstream compatibility (not used in fixed-key mode).
    src_contam = source_run_dir / "tables" / "contamination_panel_summary.csv"
    if src_contam.is_file():
        dst_contam = output_run_dir / "tables"
        dst_contam.mkdir(parents=True, exist_ok=True)
        pd.read_csv(src_contam).to_csv(dst_contam / "contamination_panel_summary.csv", index=False)

    out_prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_run_dir": str(source_run_dir),
        "output_run_dir": str(output_run_dir),
        "city_keys_file": str(city_keys_file),
        "selected_city_keys": city_keys,
        "issue_panel": str(issue_panel_path),
        "merged_manifest": str(merged_manifest_path),
        "vlm_roots": [str(p) for p in vlm_roots],
        "selected_panel_issues_rows": int(len(selected_issue_df)),
        "selected_city_count_written": int(selected_issue_df["city_key"].nunique()),
    }
    (output_run_dir / "fork_build_provenance.json").write_text(json.dumps(out_prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Done. "
        f"cities={selected_issue_df['city_key'].nunique()} "
        f"rows={len(selected_issue_df)} "
        f"output={output_run_dir}"
    )


if __name__ == "__main__":
    main()

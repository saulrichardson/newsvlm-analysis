#!/usr/bin/env python3
"""
Augment an existing issue-organized bundle with post-gate ordinance reconstructions.

This script is intentionally additive:
- it does not rebuild the existing bundle
- it does not create a second top-level metadata file
- it does not modify summary.json

Instead it:
- pulls the needed post-gate artifacts from a local or Torch run root
- writes ordinance_raw.txt / ordinance.txt / ordinance_notes.txt into each issue folder
- updates the existing metadata.csv with ordinance retrieval/provenance fields
- updates each issue.json with an ordinance_reconstruction object

The primary use case is the full-code Dropbox bundle paired with the finished Torch
OpenAI post-gate reconstruction run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
csv.field_size_limit(sys.maxsize)

DEFAULT_BUNDLE_DIR = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/newspapers/code_publication_full_issue_bundle_20260306"
)
DEFAULT_RUN_ROOT = Path(
    "/scratch/sxr203/newspaper-downloads/dedupe-webp/postgate_fullcode_openai_prepared_20260306_733workers"
)
DEFAULT_TORCH_HOST = "torch"

ISSUE_JSON_NAME = "issue.json"
METADATA_CSV_NAME = "metadata.csv"
README_NAME = "README.md"

ORDINANCE_METADATA_FIELDS = [
    "ordinance_status",
    "ordinance_relpath",
    "ordinance_raw_relpath",
    "ordinance_notes_relpath",
    "ordinance_contract_ok",
    "ordinance_repair_used",
    "ordinance_chars",
    "ordinance_model",
    "ordinance_prompt",
    "ordinance_source_line",
]

README_APPENDIX = """

## Ordinance Reconstruction

This bundle may include a second-stage ordinance reconstruction layer produced from the
post-gate multimodal transcription run.

- `issues/<issue_id>/ordinance/ordinance_raw.txt`: raw per-issue model output, including `SOURCE`,
  `=== ORDINANCE TEXT ===`, and notes markers when present.
- `issues/<issue_id>/ordinance/ordinance.txt`: cleaned ordinance body, repaired from raw output when
  the strict parser dropped a usable response.
- `issues/<issue_id>/ordinance/ordinance_notes.txt`: reconstruction notes when present.

Additional `ordinance_*` columns in `metadata.csv` and `issue.json` identify the preferred
bundle-local ordinance file paths and whether repair-from-raw was used.
""".strip()


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    tmp.replace(path)


def _boolish(value: Any) -> bool:
    return _norm(value).lower() in {"1", "true", "yes"}


def _ensure_readme_mentions_ordinances(path: Path) -> None:
    text = _read_text(path) if path.is_file() else ""
    if "## Ordinance Reconstruction" in text:
        updated = re.sub(
            r"\n## Ordinance Reconstruction\n.*\Z",
            "\n\n" + README_APPENDIX + "\n",
            text,
            flags=re.S,
        )
        if updated != text:
            _write_text_atomic(path, updated)
        return
    suffix = "\n\n" if text and not text.endswith("\n\n") else ""
    _write_text_atomic(path, text + suffix + README_APPENDIX + "\n")


def _find_repo_prompt_label(prompt_text: str) -> str:
    target_hash = _sha256_text(prompt_text)
    for prompt_path in sorted((REPO_ROOT / "prompts").glob("*.txt")):
        try:
            if _sha256_text(_read_text(prompt_path)) == target_hash:
                return prompt_path.name
        except Exception:
            continue
    return f"sha256:{target_hash[:16]}"


@dataclass
class RunIssueRecord:
    issue_id: str
    worker_name: str
    contract_ok: int
    source_line: str
    raw_path: Path | None
    ordinance_path: Path | None
    notes_path: Path | None


def _parse_contract_tolerant(raw_text: str) -> dict[str, str]:
    raw = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return {"source_line": "", "ordinance_text": "", "notes_text": ""}

    strict = re.search(
        r"^\s*SOURCE:\s*(?P<source>.*?)\n\s*=== ORDINANCE TEXT ===\s*\n(?P<ord>.*?)\n\s*--- NOTES ---\s*\n(?P<notes>.*)\Z",
        raw,
        flags=re.S,
    )
    if strict:
        return {
            "source_line": _norm(strict.group("source")),
            "ordinance_text": _norm(strict.group("ord")),
            "notes_text": _norm(strict.group("notes")),
        }

    source_line = ""
    remainder = raw
    source_match = re.match(r"^\s*SOURCE:\s*(?P<source>.*?)(?:\n\s*\n|\n(?=\s*=== ORDINANCE TEXT ===))", raw, flags=re.S)
    if source_match:
        source_line = _norm(source_match.group("source"))
        remainder = raw[source_match.end() :].lstrip()

    ord_marker = re.search(r"^\s*=== ORDINANCE TEXT ===\s*$", remainder, flags=re.M)
    if ord_marker:
        ordinance_block = remainder[ord_marker.end() :].lstrip("\n")
    else:
        ordinance_block = remainder

    notes_match = re.search(r"\n\s*--- NOTES ---\s*\n", ordinance_block, flags=re.S)
    if notes_match:
        ordinance_text = _norm(ordinance_block[: notes_match.start()])
        notes_text = _norm(ordinance_block[notes_match.end() :])
    else:
        ordinance_text = _norm(ordinance_block)
        notes_text = ""

    return {
        "source_line": source_line,
        "ordinance_text": ordinance_text,
        "notes_text": notes_text,
    }


def _stage_results_if_needed(*, run_root: str, torch_host: str, tmp_dir: Path) -> Path:
    local_candidate = Path(run_root).expanduser()
    if local_candidate.exists():
        results_dir = local_candidate.resolve() / "results"
        if not results_dir.is_dir():
            raise SystemExit(f"Local run root is missing results/: {results_dir}")
        return results_dir

    staged_results = tmp_dir / "results"
    staged_results.parent.mkdir(parents=True, exist_ok=True)
    remote_source = f"{torch_host}:{run_root.rstrip('/')}/results/"
    cmd = [
        "rsync",
        "-a",
        "--prune-empty-dirs",
        "--include=*/",
        "--include=worker*/transcriptions.csv",
        "--include=worker*/prompt_used.txt",
        "--include=worker*/results/openai_results_shard000.jsonl",
        "--include=worker*/rendered_outputs/raw/*.txt",
        "--include=worker*/rendered_outputs/ordinance_text/*.txt",
        "--include=worker*/rendered_outputs/notes/*.txt",
        "--exclude=*",
        remote_source,
        str(staged_results),
    ]
    subprocess.run(cmd, check=True)
    if not staged_results.is_dir():
        raise SystemExit(f"Failed to stage remote results from {remote_source}")
    return staged_results


def _infer_run_model(results_root: Path) -> str:
    for jsonl_path in sorted(results_root.glob("worker*/results/openai_results_shard000.jsonl")):
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                model = _norm((((payload.get("response") or {}).get("body") or {}).get("model")))
                if model:
                    return model
    return ""


def _infer_prompt_label(results_root: Path) -> str:
    for prompt_path in sorted(results_root.glob("worker*/prompt_used.txt")):
        prompt_text = _read_text(prompt_path)
        if prompt_text:
            return _find_repo_prompt_label(prompt_text)
    return ""


def _load_run_index(results_root: Path) -> dict[str, RunIssueRecord]:
    out: dict[str, RunIssueRecord] = {}
    for csv_path in sorted(results_root.glob("worker*/transcriptions.csv")):
        worker_dir = csv_path.parent
        worker_name = worker_dir.name
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                issue_id = _norm(row.get("issue_id"))
                if not issue_id:
                    continue
                if issue_id in out:
                    raise SystemExit(f"Duplicate issue_id in run results: {issue_id}")
                raw_path = worker_dir / "rendered_outputs" / "raw" / f"{issue_id}.txt"
                ordinance_path = worker_dir / "rendered_outputs" / "ordinance_text" / f"{issue_id}.txt"
                notes_path = worker_dir / "rendered_outputs" / "notes" / f"{issue_id}.txt"
                out[issue_id] = RunIssueRecord(
                    issue_id=issue_id,
                    worker_name=worker_name,
                    contract_ok=1 if _boolish(row.get("contract_ok")) else 0,
                    source_line=_norm(row.get("source_line")),
                    raw_path=raw_path if raw_path.is_file() else None,
                    ordinance_path=ordinance_path if ordinance_path.is_file() else None,
                    notes_path=notes_path if notes_path.is_file() else None,
                )
    if not out:
        raise SystemExit(f"No transcriptions.csv rows found under {results_root}")
    return out


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _remove_dir_if_empty(path: Path) -> None:
    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()


def _issue_json_path(bundle_dir: Path, row: dict[str, Any]) -> Path:
    rel = _norm(row.get("issue_json_relpath"))
    if rel:
        return bundle_dir / rel
    issue_id = _norm(row.get("issue_id"))
    if not issue_id:
        raise SystemExit("metadata row is missing issue_id")
    return bundle_dir / "issues" / issue_id / ISSUE_JSON_NAME


def _issue_dir_path(bundle_dir: Path, row: dict[str, Any]) -> Path:
    rel = _norm(row.get("issue_dir_rel"))
    if rel:
        return bundle_dir / rel
    issue_id = _norm(row.get("issue_id"))
    if not issue_id:
        raise SystemExit("metadata row is missing issue_id")
    return bundle_dir / "issues" / issue_id


def _read_metadata_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [{k: _norm(v) for k, v in row.items()} for row in reader]
    if not fieldnames:
        raise SystemExit(f"metadata.csv has no header: {path}")
    return fieldnames, rows


def _augment_bundle(
    *,
    bundle_dir: Path,
    run_index: dict[str, RunIssueRecord],
    run_model: str,
    prompt_label: str,
) -> dict[str, int]:
    metadata_path = bundle_dir / METADATA_CSV_NAME
    if not metadata_path.is_file():
        raise SystemExit(f"Missing metadata.csv: {metadata_path}")

    fieldnames, rows = _read_metadata_csv(metadata_path)
    for name in ORDINANCE_METADATA_FIELDS:
        if name not in fieldnames:
            fieldnames.append(name)

    counts = {
        "issues_total": 0,
        "issues_ok": 0,
        "issues_repaired": 0,
        "issues_raw_only": 0,
        "issues_empty_output": 0,
        "issues_missing_result": 0,
    }

    for row in rows:
        counts["issues_total"] += 1
        issue_id = _norm(row.get("issue_id"))
        if not issue_id:
            raise SystemExit("metadata.csv row is missing issue_id")

        issue_dir = _issue_dir_path(bundle_dir, row)
        issue_json_path = _issue_json_path(bundle_dir, row)
        if not issue_dir.is_dir():
            raise SystemExit(f"Missing issue directory for issue_id={issue_id}: {issue_dir}")
        if not issue_json_path.is_file():
            raise SystemExit(f"Missing issue.json for issue_id={issue_id}: {issue_json_path}")

        record = run_index.get(issue_id)
        status = "missing_result"
        repair_used = False
        contract_ok = 0
        source_line = ""
        raw_text = ""
        ordinance_text = ""
        notes_text = ""

        if record is not None:
            contract_ok = int(record.contract_ok)
            source_line = record.source_line
            raw_text = _read_text(record.raw_path) if record.raw_path is not None else ""
            ordinance_text = _read_text(record.ordinance_path) if record.ordinance_path is not None else ""
            notes_text = _read_text(record.notes_path) if record.notes_path is not None else ""

            if not ordinance_text and raw_text:
                repaired = _parse_contract_tolerant(raw_text)
                if repaired["ordinance_text"]:
                    ordinance_text = repaired["ordinance_text"]
                    if not notes_text:
                        notes_text = repaired["notes_text"]
                    if not source_line:
                        source_line = repaired["source_line"]
                    repair_used = True

            if ordinance_text:
                status = "repaired_from_raw" if repair_used else "ok"
            elif raw_text:
                status = "raw_only"
            else:
                status = "empty_output"

        ordinance_dir = issue_dir / "ordinance"
        ordinance_raw_path = ordinance_dir / "ordinance_raw.txt"
        ordinance_text_path = ordinance_dir / "ordinance.txt"
        ordinance_notes_path = ordinance_dir / "ordinance_notes.txt"
        legacy_raw_path = issue_dir / "ordinance_raw.txt"
        legacy_text_path = issue_dir / "ordinance.txt"
        legacy_notes_path = issue_dir / "ordinance_notes.txt"

        if raw_text:
            _write_text_atomic(ordinance_raw_path, raw_text)
        else:
            _remove_if_exists(ordinance_raw_path)

        if ordinance_text:
            _write_text_atomic(ordinance_text_path, ordinance_text)
        else:
            _remove_if_exists(ordinance_text_path)

        if notes_text:
            _write_text_atomic(ordinance_notes_path, notes_text)
        else:
            _remove_if_exists(ordinance_notes_path)

        _remove_if_exists(legacy_raw_path)
        _remove_if_exists(legacy_text_path)
        _remove_if_exists(legacy_notes_path)
        _remove_dir_if_empty(ordinance_dir)

        ordinance_raw_relpath = ""
        ordinance_relpath = ""
        ordinance_notes_relpath = ""
        if ordinance_raw_path.is_file():
            ordinance_raw_relpath = ordinance_raw_path.relative_to(bundle_dir).as_posix()
        if ordinance_text_path.is_file():
            ordinance_relpath = ordinance_text_path.relative_to(bundle_dir).as_posix()
        if ordinance_notes_path.is_file():
            ordinance_notes_relpath = ordinance_notes_path.relative_to(bundle_dir).as_posix()

        ordinance_chars = len(ordinance_text)

        row.update(
            {
                "ordinance_status": status,
                "ordinance_relpath": ordinance_relpath,
                "ordinance_raw_relpath": ordinance_raw_relpath,
                "ordinance_notes_relpath": ordinance_notes_relpath,
                "ordinance_contract_ok": str(contract_ok),
                "ordinance_repair_used": "1" if repair_used else "0",
                "ordinance_chars": str(ordinance_chars),
                "ordinance_model": run_model,
                "ordinance_prompt": prompt_label,
                "ordinance_source_line": source_line,
            }
        )

        issue_payload = json.loads(_read_text(issue_json_path))
        metadata_payload = issue_payload.get("metadata")
        if not isinstance(metadata_payload, dict):
            raise SystemExit(f"issue.json metadata is not an object: {issue_json_path}")
        metadata_payload.update({name: row.get(name, "") for name in ORDINANCE_METADATA_FIELDS})
        issue_payload["metadata"] = metadata_payload
        issue_payload["ordinance_reconstruction"] = {
            "status": status,
            "model": run_model,
            "prompt": prompt_label,
            "contract_ok": contract_ok,
            "repair_used": repair_used,
            "source_line": source_line,
            "ordinance_chars": ordinance_chars,
            "ordinance_raw_relpath": ordinance_raw_relpath,
            "ordinance_relpath": ordinance_relpath,
            "ordinance_notes_relpath": ordinance_notes_relpath,
        }
        _write_json_atomic(issue_json_path, issue_payload)

        if status == "ok":
            counts["issues_ok"] += 1
        elif status == "repaired_from_raw":
            counts["issues_repaired"] += 1
        elif status == "raw_only":
            counts["issues_raw_only"] += 1
        elif status == "empty_output":
            counts["issues_empty_output"] += 1
        elif status == "missing_result":
            counts["issues_missing_result"] += 1
        else:
            raise SystemExit(f"Unhandled ordinance_status={status!r} for issue_id={issue_id}")

    _write_csv_atomic(metadata_path, rows, fieldnames)
    _ensure_readme_mentions_ordinances(bundle_dir / README_NAME)
    return counts


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    ap.add_argument(
        "--run-root",
        default=str(DEFAULT_RUN_ROOT),
        help="Local or Torch-side post-gate run root. If the path does not exist locally, rsync is used via --torch-host.",
    )
    ap.add_argument("--torch-host", default=DEFAULT_TORCH_HOST)
    ap.add_argument(
        "--ordinance-model",
        default="",
        help="Optional explicit model label to write into metadata. If omitted, the script tries to infer it from staged results.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    bundle_dir = args.bundle_dir.expanduser().resolve()
    if not bundle_dir.is_dir():
        raise SystemExit(f"--bundle-dir is not a directory: {bundle_dir}")

    with tempfile.TemporaryDirectory(prefix="postgate_bundle_stage_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        results_root = _stage_results_if_needed(
            run_root=str(args.run_root),
            torch_host=str(args.torch_host),
            tmp_dir=tmp_dir,
        )
        run_index = _load_run_index(results_root)
        run_model = _norm(args.ordinance_model) or _infer_run_model(results_root)
        prompt_label = _infer_prompt_label(results_root)
        counts = _augment_bundle(
            bundle_dir=bundle_dir,
            run_index=run_index,
            run_model=run_model,
            prompt_label=prompt_label,
        )

    print(json.dumps(
        {
            "bundle_dir": str(bundle_dir),
            "run_root": str(args.run_root),
            "issues_with_results": len(run_index),
            "run_model": run_model,
            "prompt_label": prompt_label,
            **counts,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run post-gate issue transcription using full issue OCR plus selected page images.

This is the productionized version of the earlier pilots:
  - read one or more gated issue transcripts
  - optionally restrict each issue to a gated subset of page_ids
  - fetch or resolve the corresponding original PNGs
  - build multimodal Responses API requests
  - execute them through the local vendor/agent-gateway
  - parse the required SOURCE / ORDINANCE TEXT / NOTES contract
  - emit clean transcription artifacts for downstream use
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from fullnewspaper_v3_common import extract_openai_output_text, iter_jsonl, run_gateway_requests
from PIL import Image

ISSUE_PAGE_DELIMITER = "\n\n----- PAGE BREAK -----\n\n"
GEMINI_IMAGE_MAX_EDGE = 2200
GEMINI_IMAGE_JPEG_QUALITY = 85


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _split_page_ids(raw: Any) -> list[str]:
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                page_id = _norm(item.get("page_id"))
                if page_id:
                    out.append(page_id)
                continue
            value = _norm(item)
            if value:
                out.append(value)
        return out
    s = _norm(raw)
    if not s:
        return []
    out: list[str] = []
    for part in re.split(r"[,\|\n\r\t ]+", s):
        part = _norm(part)
        if part:
            out.append(part)
    return out


def _split_path_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                path = _norm(item.get("image_path") or item.get("png_path") or item.get("path"))
                if path:
                    out.append(path)
                continue
            s = _norm(item)
            if s:
                out.append(s)
        return out
    s = _norm(raw)
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            loaded = json.loads(s)
        except Exception:
            loaded = None
        if isinstance(loaded, list):
            return _split_path_list(loaded)
    out: list[str] = []
    for part in re.split(r"[\|\n\r\t]+", s):
        part = _norm(part)
        if part:
            out.append(part)
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [{k: _norm(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_structured_pages(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        return []
    if not all(isinstance(item, dict) for item in raw):
        return []

    pages: list[dict[str, str]] = []
    for idx, item in enumerate(raw, start=1):
        page_id = _norm(item.get("page_id"))
        if not page_id:
            raise SystemExit(f"Manifest pages[{idx}] is missing page_id")
        pages.append(
            {
                "page_id": page_id,
                "page_num": _norm(item.get("page_num")) or str(idx),
                "text_source": _norm(item.get("text_source")),
                "materialized_page_text_path": _norm(
                    item.get("materialized_page_text_path") or item.get("page_text_path")
                ),
                "image_path": _norm(item.get("png_path") or item.get("image_path") or item.get("path")),
            }
        )
    return pages


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Transcribe gated zoning-ordinance issues using full OCR plus selected page images.")
    ap.add_argument(
        "--issue-txt-dir",
        default="",
        help="Optional directory containing <issue_id>.txt transcripts. Not required when manifest rows include transcript_path.",
    )
    ap.add_argument(
        "--manifest-path",
        default="",
        help="Optional CSV, JSONL, or TXT manifest of issue_ids to transcribe. CSV/JSONL may include page_ids, image_paths, or dataset-style pages[].",
    )
    ap.add_argument(
        "--prompt-path",
        default=str(repo_root / "prompts" / "transcription_v13_issue_selected_images_postgate.txt"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(repo_root / "artifacts/scratch" / "postgate_issue_transcription" / "run_001"),
    )
    ap.add_argument("--env-file", default=str(repo_root / ".env"))
    ap.add_argument(
        "--request-model",
        default="gpt-5.4",
        help="Model name embedded in the request body.",
    )
    ap.add_argument(
        "--gateway-model",
        default="openai:gpt-5.4",
        help="Gateway provider:model to execute.",
    )
    ap.add_argument("--max-concurrency", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=21600.0)
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--fetch-images", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--local-png-root",
        default="",
        help="Optional local root that contains page PNGs. Exact path or recursive search by <page_id>.png is supported.",
    )
    ap.add_argument("--torch-host", default="torch")
    ap.add_argument("--torch-png-root", default="/scratch/sxr203/newspaper-downloads/dedupe-webp/unique_png")
    ap.add_argument("--gateway-runner", default=str(repo_root / "scripts" / "run_openai_requests_via_gateway.py"))
    ap.add_argument("--gateway-pythonpath", default=str(repo_root / "vendor/agent-gateway" / "src"))
    ap.add_argument("--gateway-url", default="", help="Optional existing gateway URL to use instead of starting a local one.")
    ap.add_argument("--gateway-port", type=int, default=0, help="Optional local gateway port to start per worker.")
    return ap.parse_args()


def _discover_manifest(issue_txt_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(issue_txt_dir.glob("*.txt")):
        rows.append({"request_id": path.stem, "issue_id": path.stem, "transcript_path": "", "page_ids": [], "image_paths": []})
    if not rows:
        raise SystemExit(f"No issue transcripts found under {issue_txt_dir}")
    return rows


def _normalize_manifest_row(row: dict[str, Any]) -> dict[str, Any] | None:
    transcript_path = _norm(row.get("transcript_path") or row.get("issue_transcript_path"))
    issue_id = _norm(row.get("issue_id"))
    if not issue_id and transcript_path:
        issue_id = Path(transcript_path).stem
    if not issue_id:
        return None
    request_id = _norm(row.get("request_id")) or issue_id

    structured_pages = _normalize_structured_pages(row.get("pages"))
    images_raw = row.get("images")
    page_ids_raw = row.get("page_ids") or row.get("page_id_list")
    image_paths_raw = row.get("image_paths") or row.get("image_path_list")

    image_paths: list[str] = []
    image_page_ids: list[str] = []
    if isinstance(images_raw, list):
        for item in images_raw:
            if isinstance(item, dict):
                path = _norm(item.get("image_path") or item.get("png_path") or item.get("path"))
                if path:
                    image_paths.append(path)
                    image_page_ids.append(_norm(item.get("page_id")))
            else:
                path = _norm(item)
                if path:
                    image_paths.append(path)
                    image_page_ids.append("")
    else:
        image_paths = _split_path_list(image_paths_raw or images_raw)

    page_ids = _split_page_ids(page_ids_raw)
    if not page_ids and structured_pages:
        page_ids = [page["page_id"] for page in structured_pages]
    if not page_ids and any(pid for pid in image_page_ids):
        page_ids = [pid for pid in image_page_ids if pid]
    if not image_paths and structured_pages:
        image_paths = [page["image_path"] for page in structured_pages if page["image_path"]]

    return {
        "request_id": request_id,
        "issue_id": issue_id,
        "transcript_path": transcript_path,
        "page_ids": page_ids,
        "image_paths": image_paths,
        "pages": structured_pages,
    }


def _load_manifest(path: Path, *, issue_txt_dir: Path | None) -> list[dict[str, Any]]:
    if not path.name:
        if issue_txt_dir is None:
            raise SystemExit("issue_txt_dir is required when manifest is not provided")
        return _discover_manifest(issue_txt_dir)
    if not path.is_file():
        raise SystemExit(f"Missing manifest: {path}")

    rows: list[dict[str, Any]] = []
    suffix = path.suffix.lower()
    if suffix == ".csv":
        for row in _read_csv(path):
            normalized = _normalize_manifest_row(row)
            if normalized is not None:
                rows.append(normalized)
    elif suffix == ".jsonl":
        for row in iter_jsonl(path):
            normalized = _normalize_manifest_row(row)
            if normalized is not None:
                rows.append(normalized)
    else:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            issue_id = _norm(raw)
            if issue_id:
                rows.append({"request_id": issue_id, "issue_id": issue_id, "transcript_path": "", "page_ids": [], "image_paths": []})

    if not rows:
        raise SystemExit(f"No usable manifest rows found in {path}")
    return rows


def _synthesize_issue_pages_from_manifest_pages(raw_pages: list[dict[str, str]]) -> list[dict[str, str]]:
    pages: list[dict[str, str]] = []
    for idx, page in enumerate(raw_pages, start=1):
        text_path_raw = _norm(page.get("materialized_page_text_path"))
        if not text_path_raw:
            raise SystemExit(f"Structured manifest page is missing materialized_page_text_path for page_id={page['page_id']}")
        text_path = Path(text_path_raw).expanduser().resolve()
        if not text_path.is_file():
            raise SystemExit(f"Structured manifest page text does not exist for page_id={page['page_id']}: {text_path}")
        pages.append(
            {
                "page_id": page["page_id"],
                "page_num": _norm(page.get("page_num")) or str(idx),
                "text_source": _norm(page.get("text_source")),
                "page_text": _norm(text_path.read_text(encoding="utf-8", errors="replace")),
            }
        )
    if not pages:
        raise SystemExit("Structured manifest pages produced no usable pages")
    return pages


def _extract_issue_pages(issue_transcript: str, *, fallback_page_ids: list[str] | None = None) -> list[dict[str, str]]:
    pattern = re.compile(
        r"(?ms)^=== PAGE START ===\npage_id=(?P<page_id>[^\n]+)\npage_num=(?P<page_num>[^\n]*)\ntext_source=(?P<text_source>[^\n]*)\npage_text:\n(?P<page_text>.*?)(?=^=== PAGE START ===|\Z)"
    )
    pages: list[dict[str, str]] = []
    for m in pattern.finditer(issue_transcript):
        pages.append(
            {
                "page_id": _norm(m.group("page_id")),
                "page_num": _norm(m.group("page_num")),
                "text_source": _norm(m.group("text_source")),
                "page_text": _norm(m.group("page_text")),
            }
        )
    if pages:
        return pages

    fallback_ids = [pid for pid in (fallback_page_ids or []) if _norm(pid)]
    if fallback_ids:
        parts = issue_transcript.split(ISSUE_PAGE_DELIMITER)
        if len(parts) != len(fallback_ids):
            raise SystemExit(
                "Issue transcript is not in wrapped page format, and plain page-break parsing did not match the "
                f"requested pages (segments={len(parts)} page_ids={len(fallback_ids)})"
            )
        return [
            {
                "page_id": page_id,
                "page_num": str(idx),
                "text_source": "",
                "page_text": _norm(part),
            }
            for idx, (page_id, part) in enumerate(zip(fallback_ids, parts, strict=True), start=1)
        ]

    raise SystemExit(
        "No pages found in issue transcript. Expected either wrapped page blocks or manifest page_ids aligned "
        "to the plain page-break transcript format."
    )


def _select_pages(issue_pages: list[dict[str, str]], requested_page_ids: list[str]) -> list[dict[str, str]]:
    if not requested_page_ids:
        return issue_pages
    requested_set = {pid for pid in requested_page_ids if pid}
    selected = [page for page in issue_pages if page["page_id"] in requested_set]
    found = {page["page_id"] for page in selected}
    missing = [pid for pid in requested_page_ids if pid not in found]
    if missing:
        raise SystemExit(f"Requested page_ids not found in transcript: {missing}")
    if not selected:
        raise SystemExit("Requested page_ids produced an empty page selection")
    return selected


def _select_pages_for_manifest_row(
    *,
    issue_pages: list[dict[str, str]],
    requested_page_ids: list[str],
    image_paths: list[str],
) -> tuple[list[dict[str, str]], list[str]]:
    derived_page_ids = list(requested_page_ids)
    if image_paths and not derived_page_ids:
        derived_page_ids = [Path(p).stem for p in image_paths]
    selected_pages = _select_pages(issue_pages, derived_page_ids)
    return selected_pages, derived_page_ids


def _page_image_order_block(pages: list[dict[str, str]]) -> str:
    lines = []
    for idx, page in enumerate(pages, start=1):
        lines.append(f"{idx}. page_id={page['page_id']} page_num={page['page_num']} text_source={page['text_source']}")
    return "\n".join(lines)


def _fill_prompt(template: str, *, source_name: str, issue_transcript: str, pages: list[dict[str, str]]) -> str:
    required = ("{SOURCE_NAME}", "{ISSUE_TRANSCRIPT}", "{PAGE_IMAGE_ORDER}")
    for token in required:
        if token not in template:
            raise SystemExit(f"Prompt template missing placeholder {token}")
    return (
        template.replace("{SOURCE_NAME}", source_name)
        .replace("{ISSUE_TRANSCRIPT}", issue_transcript)
        .replace("{PAGE_IMAGE_ORDER}", _page_image_order_block(pages))
    )


def _image_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type or 'image/png'};base64,{payload}"


def _resolve_local_image_path(local_png_root: Path, page_id: str) -> Path | None:
    direct = local_png_root / f"{page_id}.png"
    if direct.is_file():
        return direct
    matches = list(local_png_root.rglob(f"{page_id}.png"))
    if matches:
        return matches[0]
    return None


def _resolve_remote_image_path(torch_host: str, torch_png_root: str, page_id: str) -> str:
    root = _norm(torch_png_root).rstrip("/")
    name = f"{page_id}.png"
    direct = f"{root}/{name}"
    quoted_root = shlex.quote(root)
    quoted_name = shlex.quote(name)
    quoted_direct = shlex.quote(direct)
    probe = (
        f"if [ -f {quoted_direct} ]; then "
        f"printf '%s\\n' {quoted_direct}; "
        f"else find {quoted_root} -name {quoted_name} -print -quit; fi"
    )
    proc = subprocess.run(
        ["ssh", str(torch_host), probe],
        check=True,
        capture_output=True,
        text=True,
    )
    remote = _norm(proc.stdout)
    if not remote:
        raise SystemExit(f"Could not resolve remote image path for {page_id} under {torch_png_root}")
    return remote


def _fetch_remote_image(torch_host: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.is_file():
        return
    subprocess.run(["scp", f"{torch_host}:{remote_path}", str(local_path)], check=True)


def _ensure_local_image(
    *,
    page_id: str,
    cache_path: Path,
    local_png_root: Path | None,
    fetch_images: bool,
    torch_host: str,
    torch_png_root: str,
) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file():
        return cache_path
    if local_png_root is not None:
        local_found = _resolve_local_image_path(local_png_root, page_id)
        if local_found is not None:
            subprocess.run(["cp", str(local_found), str(cache_path)], check=True)
            return cache_path
    if not fetch_images:
        raise SystemExit(f"Missing local image for {page_id}: {cache_path}")
    remote_path = _resolve_remote_image_path(torch_host, torch_png_root, page_id)
    _fetch_remote_image(torch_host, remote_path, cache_path)
    return cache_path


def _resolve_direct_image_paths(*, image_paths: list[str], page_ids: list[str]) -> list[Path]:
    if image_paths and page_ids and len(image_paths) != len(page_ids):
        raise SystemExit(
            f"Manifest row has mismatched image_paths ({len(image_paths)}) and page_ids ({len(page_ids)})"
        )
    resolved: list[Path] = []
    for raw in image_paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Manifest image_path does not exist: {path}")
        resolved.append(path)
    return resolved


def _provider_name(model: str) -> str:
    if ":" in model:
        return model.split(":", 1)[0].strip().lower()
    return str(model).strip().lower()


def _jpeg_ready_image(image: Image.Image) -> Image.Image:
    if image.mode == "L":
        return image
    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA"}:
        base = Image.new("RGB", image.size, "white")
        alpha = image.getchannel("A")
        rgb = image.convert("RGB")
        base.paste(rgb, mask=alpha)
        return base
    return image.convert("RGB")


def _prepare_request_image(*, image_path: Path, cache_dir: Path, gateway_model: str) -> Path:
    if _provider_name(gateway_model) != "gemini":
        return image_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    optimized_path = cache_dir / f"{image_path.stem}__gemini.jpg"
    if optimized_path.is_file() and optimized_path.stat().st_mtime >= image_path.stat().st_mtime:
        return optimized_path

    with Image.open(image_path) as opened:
        working = _jpeg_ready_image(opened.copy())
        working.thumbnail((GEMINI_IMAGE_MAX_EDGE, GEMINI_IMAGE_MAX_EDGE), Image.Resampling.LANCZOS)
        working.save(
            optimized_path,
            format="JPEG",
            quality=GEMINI_IMAGE_JPEG_QUALITY,
            optimize=True,
        )
    return optimized_path


def _build_request_line(
    *,
    custom_id: str,
    request_model: str,
    prompt: str,
    image_paths: list[Path],
    pages: list[dict[str, str]],
    reasoning_effort: str,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for idx, (page, image_path) in enumerate(zip(pages, image_paths, strict=True), start=1):
        content.append(
            {
                "type": "input_text",
                "text": f"Attached issue page image {idx} of {len(image_paths)} corresponds to page_id={page['page_id']} page_num={page['page_num']}.",
            }
        )
        content.append({"type": "input_image", "image_url": _image_data_url(image_path)})

    body: dict[str, Any] = {
        "model": str(request_model),
        "input": [{"role": "user", "content": content}],
        "stream": False,
    }
    effort = _norm(reasoning_effort).lower()
    if effort:
        body["reasoning"] = {"effort": effort}
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def _parse_contract(text: str) -> dict[str, Any]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    m = re.search(
        r"^\s*SOURCE:\s*(?P<source>.*?)\n\s*=== ORDINANCE TEXT ===\s*\n(?P<ord>.*?)\n\s*--- NOTES ---\s*\n(?P<notes>.*)\Z",
        raw,
        flags=re.S,
    )
    if not m:
        return {"contract_ok": 0, "source_line": "", "ordinance_text": "", "notes_text": ""}
    return {
        "contract_ok": 1,
        "source_line": _norm(m.group("source")),
        "ordinance_text": _norm(m.group("ord")),
        "notes_text": _norm(m.group("notes")),
    }


def _render_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    issue_txt_dir = Path(args.issue_txt_dir).expanduser().resolve() if _norm(args.issue_txt_dir) else None
    prompt_path = Path(args.prompt_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    env_file = Path(args.env_file).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve() if _norm(args.manifest_path) else Path("")
    gateway_runner = Path(args.gateway_runner).expanduser().resolve()
    gateway_pythonpath = Path(args.gateway_pythonpath).expanduser().resolve()
    local_png_root = Path(args.local_png_root).expanduser().resolve() if _norm(args.local_png_root) else None

    if issue_txt_dir is not None and not issue_txt_dir.is_dir():
        raise SystemExit(f"Missing issue transcript dir: {issue_txt_dir}")
    if issue_txt_dir is None and not manifest_path.name:
        raise SystemExit("issue_txt_dir is required when manifest_path is not provided")
    if not prompt_path.is_file():
        raise SystemExit(f"Missing prompt: {prompt_path}")
    if not env_file.is_file():
        raise SystemExit(f"Missing env file: {env_file}")
    if local_png_root is not None and not local_png_root.exists():
        raise SystemExit(f"Missing local png root: {local_png_root}")

    prompt_template = prompt_path.read_text(encoding="utf-8")
    manifest_rows = _load_manifest(manifest_path, issue_txt_dir=issue_txt_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    requests_root = output_dir / "requests"
    results_root = output_dir / "results"
    rendered_root = output_dir / "rendered_outputs"
    images_root = output_dir / "images"
    inputs_root = output_dir / "inputs"

    build_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []

    for row in manifest_rows:
        request_id = _norm(row.get("request_id")) or _norm(row["issue_id"])
        issue_id = row["issue_id"]
        manifest_transcript_path = _norm(row.get("transcript_path"))
        requested_page_ids = list(row["page_ids"])
        direct_image_paths = list(row.get("image_paths") or [])
        structured_pages = list(row.get("pages") or [])
        parse_page_ids = list(requested_page_ids) or [Path(p).stem for p in direct_image_paths]
        if manifest_transcript_path:
            transcript_path = Path(manifest_transcript_path).expanduser().resolve()
        else:
            if structured_pages:
                transcript_path = inputs_root / "issue_txt" / f"{issue_id}.txt"
            elif issue_txt_dir is None:
                raise SystemExit(
                    f"Manifest row for issue_id={issue_id} omitted transcript_path, and issue_txt_dir was not provided"
                )
            else:
                transcript_path = issue_txt_dir / f"{issue_id}.txt"
        if manifest_transcript_path and not transcript_path.is_file():
            raise SystemExit(f"Missing issue transcript for issue_id={issue_id}: {transcript_path}")
        if manifest_transcript_path:
            issue_transcript = transcript_path.read_text(encoding="utf-8", errors="replace")
        else:
            issue_transcript = ""

        if structured_pages:
            issue_pages = _synthesize_issue_pages_from_manifest_pages(structured_pages)
            if not issue_transcript:
                issue_transcript = ISSUE_PAGE_DELIMITER.join(page["page_text"] for page in issue_pages)
        else:
            issue_pages = _extract_issue_pages(issue_transcript, fallback_page_ids=parse_page_ids)
        selected_pages, resolved_page_ids = _select_pages_for_manifest_row(
            issue_pages=issue_pages,
            requested_page_ids=requested_page_ids,
            image_paths=direct_image_paths,
        )

        if direct_image_paths:
            local_image_paths = _resolve_direct_image_paths(image_paths=direct_image_paths, page_ids=resolved_page_ids)
        else:
            local_image_paths = []
            for page in selected_pages:
                local_image_paths.append(
                    _ensure_local_image(
                        page_id=page["page_id"],
                        cache_path=images_root / issue_id / f"{page['page_id']}.png",
                        local_png_root=local_png_root,
                        fetch_images=bool(args.fetch_images),
                        torch_host=str(args.torch_host),
                        torch_png_root=str(args.torch_png_root),
                    )
                )
        request_image_paths = [
            _prepare_request_image(
                image_path=path,
                cache_dir=images_root / issue_id / "_request_ready",
                gateway_model=str(args.gateway_model),
            )
            for path in local_image_paths
        ]

        _render_text(inputs_root / "issue_txt" / f"{issue_id}.txt", issue_transcript)
        prompt = _fill_prompt(
            prompt_template,
            source_name=issue_id,
            issue_transcript=issue_transcript,
            pages=selected_pages,
        )
        request_rows.append(
            _build_request_line(
                custom_id=request_id,
                request_model=str(args.request_model),
                prompt=prompt,
                image_paths=request_image_paths,
                pages=selected_pages,
                reasoning_effort=str(args.reasoning_effort),
            )
        )
        build_rows.append(
            {
                "request_id": request_id,
                "issue_id": issue_id,
                "transcript_path": str(transcript_path),
                "issue_page_count": len(issue_pages),
                "attached_page_count": len(selected_pages),
                "attached_page_ids": "|".join(page["page_id"] for page in selected_pages),
                "requested_page_ids": "|".join(resolved_page_ids),
                "direct_image_paths": "|".join(str(p) for p in request_image_paths if direct_image_paths),
                "ocr_chars": len(issue_transcript),
                "image_bytes_total": sum(path.stat().st_size for path in request_image_paths),
            }
        )

    _write_csv(
        output_dir / "build_manifest.csv",
        build_rows,
        fieldnames=[
            "issue_id",
            "request_id",
            "transcript_path",
            "issue_page_count",
            "attached_page_count",
            "attached_page_ids",
            "requested_page_ids",
            "direct_image_paths",
            "ocr_chars",
            "image_bytes_total",
        ],
    )
    (output_dir / "prompt_used.txt").write_text(prompt_template, encoding="utf-8")
    _write_jsonl(requests_root / "openai_requests_shard000.jsonl", request_rows)

    if bool(args.dry_run):
        print(json.dumps({"output_dir": str(output_dir), "issues": len(request_rows), "dry_run": True}, indent=2))
        return

    run_gateway_requests(
        request_dir=requests_root,
        output_dir=results_root,
        runner_path=gateway_runner,
        model=str(args.gateway_model),
        gateway_pythonpath=gateway_pythonpath,
        timeout_s=float(args.timeout),
        concurrency=int(args.max_concurrency),
        gov_env_path=env_file,
        skip_existing=bool(args.skip_existing),
        dry_run=False,
        gateway_url=str(args.gateway_url),
        port=int(args.gateway_port),
    )

    result_map: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(results_root / "openai_results_shard000.jsonl"):
        request_id = _norm(row.get("custom_id"))
        body = ((row.get("response") or {}).get("body") or {})
        text = extract_openai_output_text(body)
        parsed = _parse_contract(text)
        result_map[request_id] = {
            "raw_text": text,
            **parsed,
            "ordinance_chars": len(parsed["ordinance_text"]),
        }

    transcription_rows: list[dict[str, Any]] = []
    blank_count = 0
    contract_failures = 0
    for build in build_rows:
        request_id = build["request_id"]
        issue_id = build["issue_id"]
        parsed = result_map.get(request_id)
        if parsed is None:
            raise SystemExit(f"Missing result for request_id={request_id}")
        if not parsed["contract_ok"]:
            contract_failures += 1
        if not parsed["ordinance_text"]:
            blank_count += 1

        _render_text(rendered_root / "raw" / f"{issue_id}.txt", parsed["raw_text"])
        _render_text(rendered_root / "ordinance_text" / f"{issue_id}.txt", parsed["ordinance_text"])
        _render_text(rendered_root / "notes" / f"{issue_id}.txt", parsed["notes_text"])

        transcription_rows.append(
            {
                "issue_id": issue_id,
                "request_id": request_id,
                "transcript_path": build["transcript_path"],
                "issue_page_count": build["issue_page_count"],
                "attached_page_count": build["attached_page_count"],
                "attached_page_ids": build["attached_page_ids"],
                "requested_page_ids": build["requested_page_ids"],
                "direct_image_paths": build["direct_image_paths"],
                "contract_ok": parsed["contract_ok"],
                "ordinance_chars": parsed["ordinance_chars"],
                "source_line": parsed["source_line"],
                "notes_text": parsed["notes_text"],
                "ordinance_text": parsed["ordinance_text"],
            }
        )

    _write_csv(
        output_dir / "transcriptions.csv",
        transcription_rows,
        fieldnames=[
            "issue_id",
            "request_id",
            "transcript_path",
            "issue_page_count",
            "attached_page_count",
            "attached_page_ids",
            "requested_page_ids",
            "direct_image_paths",
            "contract_ok",
            "ordinance_chars",
            "source_line",
            "notes_text",
            "ordinance_text",
        ],
    )
    _write_jsonl(output_dir / "transcriptions.jsonl", transcription_rows)
    summary = {
        "issues": len(transcription_rows),
        "blank_outputs": blank_count,
        "contract_failures": contract_failures,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

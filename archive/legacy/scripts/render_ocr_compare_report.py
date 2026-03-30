#!/usr/bin/env python3
"""
Render a human-friendly OCR comparison report for one page:
  - Draws bbox overlays on the page image
  - Produces per-box crop thumbnails
  - Shows Gemini vs OpenAI transcripts side-by-side in HTML

Intended use: quickly eyeball OCR quality differences between providers.
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
from collections import Counter
from os.path import expanduser
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render an OCR comparison HTML report for one page.")
    p.add_argument("--gemini-page", required=True, help="Path to Gemini per-page *.vlm.json")
    p.add_argument("--openai-page", required=True, help="Path to OpenAI per-page *.vlm.json")
    p.add_argument("--output-dir", required=True, help="Directory to write report artifacts")
    p.add_argument(
        "--max-page-dim",
        type=int,
        default=2000,
        help="Max dimension (px) for rendered overlay page image",
    )
    p.add_argument(
        "--thumb-max-dim",
        type=int,
        default=240,
        help="Max dimension (px) for per-box crop thumbnails",
    )
    p.add_argument(
        "--skip-thumbs",
        action="store_true",
        help="Do not render per-box thumbnails (faster, less disk usage).",
    )
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _box_sort_key(b: dict) -> tuple[int, int, int]:
    bbox = b.get("bbox") or {}
    return (int(bbox.get("y0", 0)), int(bbox.get("x0", 0)), int(b.get("id", 0)))


def _norm_text(t: str | None) -> str:
    if not t:
        return ""
    return " ".join(t.split()).strip()


def _ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _resize_to_max(im: Image.Image, max_dim: int) -> tuple[Image.Image, float]:
    max_dim = max(1, int(max_dim))
    w, h = im.size
    scale = min(max_dim / max(w, h), 1.0)
    if scale >= 1.0:
        return im.copy(), 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS), scale


def _safe_crop(im: Image.Image, bbox: dict) -> Image.Image:
    w, h = im.size
    x0 = max(0, min(w, int(bbox.get("x0", 0))))
    y0 = max(0, min(h, int(bbox.get("y0", 0))))
    x1 = max(0, min(w, int(bbox.get("x1", 0))))
    y1 = max(0, min(h, int(bbox.get("y1", 0))))
    if x1 <= x0 or y1 <= y0:
        # Return a 1x1 placeholder to keep the pipeline moving.
        return Image.new("RGB", (1, 1), color=(255, 255, 255))
    return im.crop((x0, y0, x1, y1))


def page_text_from_boxes(page: dict) -> str:
    boxes = page.get("boxes") or []
    if not isinstance(boxes, list):
        return ""
    boxes = sorted([b for b in boxes if isinstance(b, dict)], key=_box_sort_key)
    parts: list[str] = []
    for b in boxes:
        if b.get("status") != "ok":
            continue
        t = (b.get("transcript") or "").strip()
        if not t:
            continue
        parts.append(t)
    return "\n\n".join(parts).strip()


def main() -> None:
    args = parse_args()

    gemini_path = Path(expanduser(args.gemini_page))
    openai_path = Path(expanduser(args.openai_page))
    out_dir = Path(expanduser(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    gemini = load_json(gemini_path)
    openai = load_json(openai_path)

    page_id = gemini.get("page_id") or gemini_path.stem.replace(".vlm", "")
    if openai.get("page_id") and openai.get("page_id") != page_id:
        raise SystemExit(f"Page id mismatch: gemini={page_id} openai={openai.get('page_id')}")

    gemini_model = gemini.get("model")
    openai_model = openai.get("model")

    png_path = gemini.get("png_path") or openai.get("png_path")
    if not png_path:
        raise SystemExit("Could not find png_path in either page JSON.")
    png_path = Path(expanduser(png_path))
    if not png_path.is_file():
        raise SystemExit(f"PNG not found: {png_path}")

    gemini_boxes = {b.get("id"): b for b in (gemini.get("boxes") or []) if isinstance(b, dict) and "id" in b}
    openai_boxes = {b.get("id"): b for b in (openai.get("boxes") or []) if isinstance(b, dict) and "id" in b}

    # Reading order based on Gemini (same layout expected).
    ordered_boxes = sorted([b for b in (gemini.get("boxes") or []) if isinstance(b, dict)], key=_box_sort_key)

    # Render page overlay image (scaled).
    with Image.open(png_path) as im:
        page_im = im.convert("RGB")

    overlay_im, scale = _resize_to_max(page_im, args.max_page_dim)
    draw = ImageDraw.Draw(overlay_im)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None

    summary_counts = Counter()
    diffs: list[float] = []

    for b in ordered_boxes:
        box_id = b.get("id")
        bbox = b.get("bbox") or {}
        x0 = int(round(int(bbox.get("x0", 0)) * scale))
        y0 = int(round(int(bbox.get("y0", 0)) * scale))
        x1 = int(round(int(bbox.get("x1", 0)) * scale))
        y1 = int(round(int(bbox.get("y1", 0)) * scale))

        gb = gemini_boxes.get(box_id, {})
        ob = openai_boxes.get(box_id, {})
        gs = gb.get("status")
        os = ob.get("status")
        gt = _norm_text(gb.get("transcript"))
        ot = _norm_text(ob.get("transcript"))

        both_ok = gs == "ok" and os == "ok"
        if both_ok:
            r = _ratio(gt, ot)
            diffs.append(r)
            if gt == ot:
                kind = "match"
            else:
                kind = "diff"
        elif gs == "ok" and os != "ok":
            kind = "gemini_only"
        elif gs != "ok" and os == "ok":
            kind = "openai_only"
        else:
            kind = "neither_ok"

        summary_counts[kind] += 1

        # Color code:
        # - match: green
        # - diff: yellow
        # - one ok: orange/blue
        # - neither: red
        if kind == "match":
            color = (0, 200, 0)
        elif kind == "diff":
            color = (255, 215, 0)
        elif kind == "gemini_only":
            color = (255, 140, 0)
        elif kind == "openai_only":
            color = (0, 140, 255)
        else:
            color = (220, 0, 0)

        # Draw rectangle
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

        # Draw small label
        label = str(box_id)
        pad = 2
        if font is not None:
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        else:
            tw, th = (len(label) * 6, 10)
        lx0, ly0 = x0, max(0, y0 - (th + 2 * pad))
        lx1, ly1 = x0 + tw + 2 * pad, ly0 + th + 2 * pad
        draw.rectangle([lx0, ly0, lx1, ly1], fill=(0, 0, 0))
        draw.text((lx0 + pad, ly0 + pad), label, fill=(255, 255, 255), font=font)

    overlay_path = out_dir / f"{page_id}.overlay.jpg"
    overlay_im.save(overlay_path, quality=85)

    # Optional thumbnails.
    crops_dir = out_dir / "crops"
    thumb_paths: dict[int, str] = {}
    if not args.skip_thumbs:
        crops_dir.mkdir(parents=True, exist_ok=True)
        for b in ordered_boxes:
            box_id = b.get("id")
            if not isinstance(box_id, int):
                continue
            bbox = b.get("bbox") or {}
            crop = _safe_crop(page_im, bbox)
            thumb, _ = _resize_to_max(crop, args.thumb_max_dim)
            out = crops_dir / f"box_{box_id:04d}.jpg"
            thumb.save(out, quality=85)
            thumb_paths[box_id] = str(out.relative_to(out_dir))

    gemini_text = page_text_from_boxes(gemini)
    openai_text = page_text_from_boxes(openai)
    text_ratio = _ratio(_norm_text(gemini_text), _norm_text(openai_text))

    # Build HTML report.
    rows_html: list[str] = []
    for b in ordered_boxes:
        box_id = b.get("id")
        if not isinstance(box_id, int):
            continue
        gb = gemini_boxes.get(box_id, {})
        ob = openai_boxes.get(box_id, {})

        gs = gb.get("status")
        os = ob.get("status")
        gt = (gb.get("transcript") or "").strip()
        ot = (ob.get("transcript") or "").strip()

        gt_norm = _norm_text(gt)
        ot_norm = _norm_text(ot)
        sim = _ratio(gt_norm, ot_norm)

        if gs == "ok" and os == "ok":
            row_class = "match" if gt_norm == ot_norm else "diff"
        elif gs == "ok" and os != "ok":
            row_class = "gemini_only"
        elif gs != "ok" and os == "ok":
            row_class = "openai_only"
        else:
            row_class = "neither_ok"

        thumb_rel = thumb_paths.get(box_id)
        thumb_html = f'<img class="thumb" src="{html.escape(thumb_rel)}" />' if thumb_rel else ""

        rows_html.append(
            "\n".join(
                [
                    f'<tr class="{row_class}" id="box-{box_id}">',
                    f"<td>{box_id}</td>",
                    f"<td>{html.escape(str(b.get('class') or b.get('cls') or ''))}</td>",
                    f"<td>{html.escape(json.dumps(b.get('bbox') or {}, ensure_ascii=False))}</td>",
                    f"<td>{thumb_html}</td>",
                    f"<td>{html.escape(str(gs))}</td>",
                    f"<td>{html.escape(str(os))}</td>",
                    f"<td>{sim:.3f}</td>",
                    f"<td><pre>{html.escape(gt)}</pre></td>",
                    f"<td><pre>{html.escape(ot)}</pre></td>",
                    "</tr>",
                ]
            )
        )

    report_path = out_dir / "report.html"
    report_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                '<meta charset="utf-8" />',
                f"<title>OCR Compare: {html.escape(page_id)}</title>",
                "<style>",
                "body { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 20px; }",
                ".meta { color: #333; margin-bottom: 12px; }",
                ".summary { margin: 10px 0 18px; padding: 10px; background: #f6f7f8; border: 1px solid #ddd; }",
                ".grid { display: grid; grid-template-columns: 1fr; gap: 12px; }",
                "img.page { max-width: 100%; height: auto; border: 1px solid #ddd; }",
                "table { border-collapse: collapse; width: 100%; }",
                "th, td { border: 1px solid #ddd; padding: 6px; vertical-align: top; }",
                "th { background: #fafafa; position: sticky; top: 0; z-index: 1; }",
                "pre { margin: 0; white-space: pre-wrap; word-wrap: break-word; max-width: 520px; }",
                ".thumb { width: 160px; height: auto; border: 1px solid #ccc; }",
                "tr.match { background: #eaffea; }",
                "tr.diff { background: #fff8d6; }",
                "tr.gemini_only { background: #ffe6cc; }",
                "tr.openai_only { background: #d6ecff; }",
                "tr.neither_ok { background: #ffe0e0; }",
                "details pre { max-width: none; }",
                "</style>",
                "</head>",
                "<body>",
                f"<h1>OCR Compare: {html.escape(page_id)}</h1>",
                '<div class="meta">',
                f"<div><b>Gemini page:</b> {html.escape(str(gemini_path))}</div>",
                f"<div><b>OpenAI page:</b> {html.escape(str(openai_path))}</div>",
                f"<div><b>PNG:</b> {html.escape(str(png_path))}</div>",
                f"<div><b>Gemini model:</b> {html.escape(str(gemini_model))}</div>",
                f"<div><b>OpenAI model:</b> {html.escape(str(openai_model))}</div>",
                "</div>",
                '<div class="summary">',
                "<div><b>Box summary</b></div>",
                "<ul>",
                f"<li>match (both ok, identical text): {summary_counts['match']}</li>",
                f"<li>diff (both ok, different text): {summary_counts['diff']}</li>",
                f"<li>gemini_only (gemini ok, openai not): {summary_counts['gemini_only']}</li>",
                f"<li>openai_only (openai ok, gemini not): {summary_counts['openai_only']}</li>",
                f"<li>neither_ok: {summary_counts['neither_ok']}</li>",
                "</ul>",
                f"<div><b>Avg box similarity (both ok):</b> {sum(diffs)/len(diffs):.3f}</div>"
                if diffs
                else "<div><b>Avg box similarity (both ok):</b> n/a</div>",
                f"<div><b>Page-text similarity:</b> {text_ratio:.3f} (flattened ok transcripts)</div>",
                "</div>",
                '<div class="grid">',
                f'<img class="page" src="{html.escape(overlay_path.name)}" alt="page overlay" />',
                "</div>",
                "<details>",
                "<summary><b>Flattened page text (Gemini vs OpenAI)</b></summary>",
                "<h3>Gemini page text</h3>",
                f"<pre>{html.escape(gemini_text[:200000])}</pre>",
                "<h3>OpenAI page text</h3>",
                f"<pre>{html.escape(openai_text[:200000])}</pre>",
                "</details>",
                "<h2>Per-box comparison</h2>",
                "<table>",
                "<thead>",
                "<tr>",
                "<th>id</th><th>class</th><th>bbox</th><th>crop</th><th>gemini status</th><th>openai status</th><th>sim</th><th>gemini transcript</th><th>openai transcript</th>",
                "</tr>",
                "</thead>",
                "<tbody>",
                "\n".join(rows_html),
                "</tbody>",
                "</table>",
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )

    meta_path = out_dir / "report.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "page_id": page_id,
                "gemini_page": str(gemini_path),
                "openai_page": str(openai_path),
                "png_path": str(png_path),
                "gemini_model": gemini_model,
                "openai_model": openai_model,
                "overlay_image": str(overlay_path),
                "report_html": str(report_path),
                "box_summary": dict(summary_counts),
                "page_text_similarity": text_ratio,
                "avg_box_similarity_both_ok": (sum(diffs) / len(diffs)) if diffs else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {report_path}")
    print(f"Wrote {overlay_path}")
    if not args.skip_thumbs:
        print(f"Wrote crops under {crops_dir}")


if __name__ == "__main__":
    main()


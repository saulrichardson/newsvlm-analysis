#!/usr/bin/env python3
"""Download one Newspapers.com page image from a real subscribed Chrome session.

This is a proof-of-viability tool, not a bulk scraper. It assumes:

1. Google Chrome is already running with a real user profile and a CDP port.
2. That Chrome profile is signed into a paid Newspapers.com account.
3. The target `/image/<id>/` page can be opened in that real session.

The script extracts the signed image token from the live page DOM, constructs the
full-size `img.newspapers.com` URL, and downloads the JPEG outside the browser.
It fails loudly when the page is not actually viewable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import websockets


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.7680.80 Safari/537.36"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use a real subscribed Chrome session to extract a signed Newspapers.com "
            "image URL and download one full page JPEG."
        )
    )
    parser.add_argument(
        "--image-page-url",
        required=True,
        help="Full Newspapers.com /image/<id>/ URL to probe.",
    )
    parser.add_argument(
        "--chrome-debug-base",
        default="http://127.0.0.1:9223",
        help="Base URL for the real Chrome CDP instance.",
    )
    parser.add_argument(
        "--chrome-app-name",
        default="Google Chrome",
        help="macOS app name used by AppleScript for front-tab navigation.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=6.0,
        help="Seconds to wait after front-tab navigation before probing CDP.",
    )
    parser.add_argument(
        "--no-navigate",
        action="store_true",
        help="Do not navigate the front Chrome tab; require the target page to already be open.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the JPEG and probe metadata should be written.",
    )
    return parser.parse_args()


def navigate_front_tab(app_name: str, target_url: str) -> None:
    activate_script = f'tell application "{app_name}" to activate'
    navigate_script = f'''
tell application "{app_name}"
  if (count of windows) = 0 then
    make new window
  end if
  set URL of active tab of front window to "{target_url}"
end tell
'''
    subprocess.run(
        ["osascript", "-e", activate_script, "-e", navigate_script],
        check=True,
        capture_output=True,
        text=True,
    )


def chrome_json(debug_base: str, path: str) -> Any:
    with urlopen(f"{debug_base.rstrip('/')}{path}", timeout=30) as response:
        return json.load(response)


def find_page_ws_url(debug_base: str, target_url: str) -> str:
    pages = chrome_json(debug_base, "/json/list")
    exact_match = None
    for page in pages:
        if page.get("type") != "page":
            continue
        page_url = page.get("url", "")
        if page_url == target_url:
            exact_match = page
            break
    if exact_match is None:
        partial_matches = [
            page
            for page in pages
            if page.get("type") == "page" and target_url in page.get("url", "")
        ]
        if not partial_matches:
            raise RuntimeError(f"Could not find an open Chrome tab for {target_url}")
        exact_match = partial_matches[0]
    ws_url = exact_match.get("webSocketDebuggerUrl")
    if not ws_url:
        raise RuntimeError(f"No webSocketDebuggerUrl found for {target_url}")
    return ws_url


async def cdp_evaluate_json(ws_url: str, expression: str) -> Any:
    async with websockets.connect(ws_url, max_size=2**27) as ws:
        await ws.send(
            json.dumps(
                {
                    "id": 1,
                    "method": "Runtime.evaluate",
                    "params": {
                        "expression": expression,
                        "returnByValue": True,
                    },
                }
            )
        )
        while True:
            message = json.loads(await ws.recv())
            if message.get("id") != 1:
                continue
            result = message.get("result", {}).get("result", {})
            if "value" not in result:
                raise RuntimeError(
                    f"CDP evaluation returned no value: {json.dumps(message)[:500]}"
                )
            return json.loads(result["value"])


def applescript_evaluate_json(app_name: str, expression: str) -> Any:
    script = f'''
tell application "{app_name}"
  return execute active tab of front window javascript {json.dumps(expression)}
end tell
'''
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "Allow JavaScript from Apple Events" in stderr:
            raise RuntimeError(
                "Google Chrome is not allowing JavaScript from Apple Events. "
                "Enable View > Developer > Allow JavaScript from Apple Events."
            )
        raise RuntimeError(stderr or "AppleScript JavaScript evaluation failed")
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError("AppleScript JavaScript evaluation returned no output")
    return json.loads(stdout)


def evaluate_live_probe(
    *,
    chrome_debug_base: str,
    chrome_app_name: str,
    target_url: str,
) -> dict[str, Any]:
    try:
        ws_url = find_page_ws_url(chrome_debug_base, target_url)
        return asyncio.run(cdp_evaluate_json(ws_url, page_probe_expression()))
    except (RuntimeError, URLError, OSError):
        return applescript_evaluate_json(chrome_app_name, page_probe_expression())


def page_probe_expression() -> str:
    return r"""JSON.stringify((() => {
  const toHref = (el) => el.getAttribute('href') || el.getAttributeNS('http://www.w3.org/1999/xlink', 'href') || '';
  const images = Array.from(document.querySelectorAll('image')).map((img) => ({
    href: toHref(img),
    width: img.getAttribute('width') || '',
    height: img.getAttribute('height') || '',
    x: img.getAttribute('x') || '',
    y: img.getAttribute('y') || '',
  }));
  const tile = images.find((img) => img.href.includes('https://img.newspapers.com/img/img?')) || null;
  const thumbnail = images.find((img) => img.href.includes('/img/thumbnail/')) || null;
  return {
    url: location.href,
    title: document.title,
    bodySnippet: (document.body?.innerText || '').slice(0, 1200),
    tile,
    thumbnail,
    imageElementCount: images.length,
  };
})())"""


def build_full_image_url(probe: dict[str, Any]) -> str:
    tile = probe.get("tile") or {}
    thumbnail = probe.get("thumbnail") or {}
    tile_href = str(tile.get("href", "")).strip()
    if not tile_href:
        raise RuntimeError("No signed tile href found in the live page DOM")
    full_width = str(thumbnail.get("width", "")).strip()
    full_height = str(thumbnail.get("height", "")).strip()
    if not full_width.isdigit() or not full_height.isdigit():
        raise RuntimeError(
            "Could not extract full page width/height from the page thumbnail element"
        )

    parsed = urlparse(tile_href)
    query = parse_qs(parsed.query)
    required = ("id", "user", "iat")
    missing = [key for key in required if not query.get(key)]
    if missing:
        raise RuntimeError(
            f"Signed tile URL is missing required params {missing}: {tile_href}"
        )

    full_query = {
        "id": query["id"][0],
        "user": query["user"][0],
        "iat": query["iat"][0],
        "brightness": query.get("brightness", ["0"])[0],
        "contrast": query.get("contrast", ["0"])[0],
        "invert": query.get("invert", ["0"])[0],
        "width": full_width,
        "height": full_height,
    }
    return f"https://img.newspapers.com/img/img?{urlencode(full_query)}"


def download_binary(url: str, output_path: Path) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    context = ssl.create_default_context()
    with urlopen(request, context=context, timeout=60) as response:
        data = response.read()
        content_type = response.headers.get("Content-Type", "")
        output_path.write_bytes(data)
        return {
            "status": response.status,
            "content_type": content_type,
            "byte_count": len(data),
        }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_navigate:
        navigate_front_tab(args.chrome_app_name, args.image_page_url)
        time.sleep(args.wait_seconds)

    probe = evaluate_live_probe(
        chrome_debug_base=args.chrome_debug_base,
        chrome_app_name=args.chrome_app_name,
        target_url=args.image_page_url,
    )

    title = probe.get("title", "")
    body_snippet = probe.get("bodySnippet", "")
    if "Access denied" in title or "Cloudflare" in title:
        raise RuntimeError(f"Cloudflare challenge page detected: {title}")
    if "Sign in to Newspapers.com" in title or "Sign in to Newspapers.com" in body_snippet:
        raise RuntimeError("Chrome session is not signed into a subscribed Newspapers.com account")
    if "/image/" not in str(probe.get("url", "")):
        raise RuntimeError(
            "The target page did not resolve to a normal Newspapers.com image page: "
            f"{title}"
        )

    full_image_url = build_full_image_url(probe)
    image_id = urlparse(args.image_page_url).path.rstrip("/").split("/")[-1]
    output_path = output_dir / f"{image_id}.jpg"
    download_meta = download_binary(full_image_url, output_path)
    if download_meta["content_type"] != "image/jpeg":
        raise RuntimeError(
            f"Expected image/jpeg from signed page URL, got {download_meta['content_type']!r}"
        )

    summary = {
        "image_page_url": args.image_page_url,
        "page_title": title,
        "full_image_url": full_image_url,
        "output_path": str(output_path),
        "download": download_meta,
        "probe": probe,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

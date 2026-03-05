#!/usr/bin/env python3
"""
Run OpenAI-Batch-style request JSONLs *synchronously* through the local agent-gateway.

This is a convenience runner for fast iteration when you already have exported
`openai_requests_shard*.jsonl` files (POST /v1/responses) but don't want to wait
for provider Batch APIs.

It:
  - optionally starts a local gateway server (FastAPI/uvicorn)
  - runs many requests concurrently against that gateway
  - writes OpenAI-batch-shaped `openai_results_shard*.jsonl` + `openai_errors_shard*.jsonl`

This repo includes `agent-gateway/` as a git submodule. By default this script
will add `agent-gateway/src` to both `sys.path` (for imports) and `PYTHONPATH`
(for the gateway subprocess), so you usually do not need to manage PYTHONPATH.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMODULE_SRC = REPO_ROOT / "agent-gateway" / "src"

# Make imports work when running this script directly (without pip install -e agent-gateway).
if SUBMODULE_SRC.is_dir():
    sys.path.insert(0, str(SUBMODULE_SRC))

try:
    from gateway.client import GatewayAgentClient, build_user_message
except ModuleNotFoundError as exc:
    hint = ""
    if not SUBMODULE_SRC.exists():
        hint = (
            "\nHint: the `agent-gateway/` submodule is not present. Run:\n"
            "  git submodule update --init --recursive\n"
        )
    elif not SUBMODULE_SRC.is_dir():
        hint = (
            "\nHint: the `agent-gateway/` submodule exists but does not look initialized.\n"
            "Run:\n"
            "  git submodule update --init --recursive\n"
        )
    raise SystemExit(
        "Could not import `gateway` (agent-gateway).\n"
        "Make sure dependencies are installed (e.g. `pip install -e agent-gateway`) "
        "or that `agent-gateway/src` is on PYTHONPATH."
        + hint
    ) from exc


REAL_PROVIDERS = {"openai", "gemini", "claude"}


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _parse_shard_index(name: str, *, prefix: str, suffix: str) -> int:
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Not a shard filename: {name} (expected {prefix}*{suffix})")
    mid = name[len(prefix) : -len(suffix)]
    if not mid.isdigit():
        raise ValueError(f"Shard index is not digits: {name}")
    return int(mid)


def _validate_model_provider(model: str, *, allow_test_providers: bool) -> None:
    if ":" not in model:
        if allow_test_providers:
            return
        raise SystemExit(f"Model '{model}' must be prefixed with a provider (one of {sorted(REAL_PROVIDERS)}).")
    provider = model.split(":", 1)[0].lower()
    if provider not in REAL_PROVIDERS and not allow_test_providers:
        raise SystemExit(
            f"Provider '{provider}' is not allowed. Use one of {sorted(REAL_PROVIDERS)}, "
            "or pass --allow-test-providers."
        )


def _gateway_health_ok(url: str) -> bool:
    health_url = url.rstrip("/") + "/healthz"
    try:
        with urllib.request.urlopen(health_url, timeout=1.0) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def wait_for_gateway(url: str, timeout_s: float = 45.0, proc: subprocess.Popen | None = None) -> None:
    deadline = time.time() + timeout_s
    health_url = url.rstrip("/") + "/healthz"
    last_err: Exception | None = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError("Gateway process exited before becoming healthy.")
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as r:
                if r.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(0.5)
    raise RuntimeError(f"Gateway did not become healthy at {health_url}: {last_err}")


def _extract_openai_prompt_text(req_obj: dict[str, Any]) -> str:
    """Extract the exact prompt text embedded in an OpenAI /v1/responses request object."""
    body = req_obj.get("body")
    if not isinstance(body, dict):
        raise ValueError("OpenAI request JSON missing body object")
    inp = body.get("input")
    if not isinstance(inp, list) or not inp:
        raise ValueError("OpenAI request JSON missing body.input[]")
    msg0 = inp[0]
    if not isinstance(msg0, dict):
        raise ValueError("OpenAI request JSON body.input[0] is not an object")
    content = msg0.get("content")
    # gateway.client.build_user_message may serialize content as a plain string.
    if isinstance(content, str) and content.strip():
        return content
    if not isinstance(content, list) or not content:
        raise ValueError("OpenAI request JSON body.input[0].content missing")
    # Typical export format: [{"type":"input_text","text": "..."}]
    for c in content:
        if isinstance(c, dict) and c.get("type") == "input_text":
            text = c.get("text")
            if isinstance(text, str) and text.strip():
                return text
    raise ValueError("OpenAI request JSON missing input_text.text")


@dataclass(frozen=True)
class _WorkItem:
    shard: int
    custom_id: str
    prompt_text: str


def _stable_fake_id(custom_id: str, *, model: str) -> str:
    h = hashlib.sha256(f"{model}::{custom_id}".encode("utf-8")).hexdigest()[:24]
    return f"gw_req_{h}"


def _openai_result_line(*, custom_id: str, text: str, model: str) -> dict[str, Any]:
    # Minimal shape needed by the downstream _extract_openai_output_text helpers.
    return {
        "id": _stable_fake_id(custom_id, model=model),
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": "",
            "body": {
                "status": "completed",
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": text,
                            }
                        ]
                    }
                ],
            },
        },
        "error": None,
    }


def _openai_error_line(*, custom_id: str, status_code: int, message: str, model: str) -> dict[str, Any]:
    return {
        "id": _stable_fake_id(custom_id, model=model),
        "custom_id": custom_id,
        "response": {
            "status_code": int(status_code),
            "request_id": "",
            "body": {"error": {"message": str(message), "type": "gateway_error"}},
        },
        "error": None,
    }


def _collect_work(request_dir: Path) -> list[_WorkItem]:
    req_paths = sorted(request_dir.glob("openai_requests_shard*.jsonl"))
    if not req_paths:
        raise SystemExit(f"No openai_requests_shard*.jsonl found in {request_dir}")

    out: list[_WorkItem] = []
    for rp in req_paths:
        shard = _parse_shard_index(rp.name, prefix="openai_requests_shard", suffix=".jsonl")
        for raw in rp.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id")
            if not isinstance(cid, str) or not cid:
                raise SystemExit(f"OpenAI request line missing custom_id in {rp}")
            prompt_text = _extract_openai_prompt_text(obj)
            out.append(_WorkItem(shard=shard, custom_id=cid, prompt_text=prompt_text))
    return out


def _read_existing_custom_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = obj.get("custom_id")
        if isinstance(cid, str) and cid:
            out.add(cid)
    return out


def _write_jsonl_line(f, obj: dict[str, Any]) -> None:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run openai_requests_shard*.jsonl through agent-gateway synchronously.")
    ap.add_argument("--request-dir", required=True, help="Directory containing openai_requests_shard*.jsonl")
    ap.add_argument("--output-dir", required=True, help="Directory to write openai_results_shard*.jsonl into")
    ap.add_argument(
        "--model",
        required=True,
        help="Gateway model to use for completing each prompt (e.g. gemini:gemini-2.5-flash)",
    )
    ap.add_argument("--max-concurrency", type=int, default=4, help="How many requests to run concurrently")
    ap.add_argument("--timeout", type=float, default=180.0, help="Gateway timeout seconds per request")
    ap.add_argument("--max-retries", type=int, default=2, help="Retries per request on transport/parse errors")
    ap.add_argument("--skip-existing", action="store_true", help="Skip custom_ids already present in output results JSONLs")
    ap.add_argument("--dry-run", action="store_true", help="Print what would run, but do not call gateway")
    ap.add_argument("--allow-test-providers", action="store_true", help="Allow non-real providers for debugging")
    ap.add_argument(
        "--gateway-url",
        default=None,
        help="If provided, use this existing gateway instead of starting one",
    )
    ap.add_argument("--port", type=int, default=8000, help="Port to start gateway on (if auto-starting)")
    return ap.parse_args()


async def _run_work(
    *,
    items: list[_WorkItem],
    out_dir: Path,
    model: str,
    gateway_url: str,
    max_concurrency: int,
    timeout: float,
    max_retries: int,
    skip_existing: bool,
    dry_run: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open shard output files once.
    shards = sorted({it.shard for it in items})
    res_files = {s: (out_dir / f"openai_results_shard{s:03d}.jsonl") for s in shards}
    err_files = {s: (out_dir / f"openai_errors_shard{s:03d}.jsonl") for s in shards}

    existing_by_shard: dict[int, set[str]] = {}
    if skip_existing:
        for s, p in res_files.items():
            existing_by_shard[s] = _read_existing_custom_ids(p)

    # Ensure error files exist even if no errors occur (keeps downstream globs stable).
    for s, ep in err_files.items():
        ep.parent.mkdir(parents=True, exist_ok=True)
        if not ep.exists():
            ep.write_text("", encoding="utf-8")

    queue: asyncio.Queue[_WorkItem | None] = asyncio.Queue()
    lock = asyncio.Lock()

    stats = {
        "model": model,
        "gateway_url": gateway_url,
        "items_total": len(items),
        "items_skipped_existing": 0,
        "items_written_ok": 0,
        "items_written_error": 0,
    }

    async with GatewayAgentClient(base_url=gateway_url, timeout=timeout) as client:

        async def one(it: _WorkItem) -> None:
            if skip_existing and it.custom_id in existing_by_shard.get(it.shard, set()):
                async with lock:
                    stats["items_skipped_existing"] += 1
                return

            if dry_run:
                _eprint(f"[dry-run] would run custom_id={it.custom_id} shard={it.shard}")
                return

            attempts = 0
            last_err: Exception | None = None
            while attempts < max_retries + 1:
                attempts += 1
                try:
                    msg = build_user_message(it.prompt_text)
                    resp = await client.complete_response(model=model, input_messages=[msg])
                    text = resp.get("text") or ""
                    line = _openai_result_line(custom_id=it.custom_id, text=str(text), model=model)
                    async with lock:
                        with res_files[it.shard].open("a", encoding="utf-8") as f:
                            _write_jsonl_line(f, line)
                        stats["items_written_ok"] += 1
                    return
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    if attempts >= max_retries + 1:
                        break
                    await asyncio.sleep(0.5 * attempts)

            err_line = _openai_error_line(
                custom_id=it.custom_id,
                status_code=0,
                message=str(last_err) if last_err else "unknown error",
                model=model,
            )
            async with lock:
                with err_files[it.shard].open("a", encoding="utf-8") as f:
                    _write_jsonl_line(f, err_line)
                stats["items_written_error"] += 1

        async def worker() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    await one(item)
                finally:
                    queue.task_done()

        max_concurrency = max(1, int(max_concurrency))
        workers = [asyncio.create_task(worker()) for _ in range(max_concurrency)]
        try:
            for it in items:
                await queue.put(it)
            for _ in range(max_concurrency):
                await queue.put(None)
            await queue.join()
        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    return stats


def main() -> None:
    args = parse_args()

    request_dir = Path(args.request_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    model = str(args.model).strip()
    _validate_model_provider(model, allow_test_providers=bool(args.allow_test_providers))

    if not request_dir.is_dir():
        raise SystemExit(f"--request-dir is not a directory: {request_dir}")

    gateway_proc: subprocess.Popen | None = None
    gateway_log_f = None
    gateway_url = args.gateway_url
    if gateway_url is None:
        port = int(args.port)
        gateway_url = f"http://127.0.0.1:{port}"

        if _gateway_health_ok(gateway_url):
            _eprint(f"Gateway already running at {gateway_url}; reusing it.")
        else:
            # Ensure output dir exists so we can safely log gateway stdout/stderr without
            # leaking provider keys into the user's terminal.
            out_dir.mkdir(parents=True, exist_ok=True)
            gateway_log_path = out_dir / "gateway.log"
            gateway_log_f = gateway_log_path.open("a", encoding="utf-8")

            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "gateway.app:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "error",
            ]
            env = dict(os.environ)
            # Ensure the gateway subprocess can import from the submodule checkout without
            # requiring a separate `pip install -e agent-gateway`.
            if SUBMODULE_SRC.is_dir():
                prior = env.get("PYTHONPATH", "").strip()
                env["PYTHONPATH"] = str(SUBMODULE_SRC) if not prior else f"{str(SUBMODULE_SRC)}:{prior}"

            # The gateway reads .env, but the env var wins; we need longer than the repo-default
            # 30s for some zoning/questionnaire prompts.
            env["GATEWAY_TIMEOUT_SECONDS"] = str(int(max(60.0, float(args.timeout))))

            gateway_proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=gateway_log_f,
                stderr=gateway_log_f,
                env=env,
            )
            try:
                wait_for_gateway(gateway_url, timeout_s=60.0, proc=gateway_proc)
            except Exception:
                gateway_proc.terminate()
                gateway_proc.wait(timeout=5)
                raise

    try:
        items = _collect_work(request_dir)
        _eprint(f"Loaded {len(items)} requests from {request_dir}")
        stats = asyncio.run(
            _run_work(
                items=items,
                out_dir=out_dir,
                model=model,
                gateway_url=gateway_url,
                max_concurrency=int(args.max_concurrency),
                timeout=float(args.timeout),
                max_retries=int(args.max_retries),
                skip_existing=bool(args.skip_existing),
                dry_run=bool(args.dry_run),
            )
        )
        print(json.dumps(stats, indent=2))
    finally:
        if gateway_proc is not None:
            gateway_proc.terminate()
            try:
                gateway_proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                gateway_proc.kill()
        if gateway_log_f is not None:
            gateway_log_f.close()


if __name__ == "__main__":
    main()


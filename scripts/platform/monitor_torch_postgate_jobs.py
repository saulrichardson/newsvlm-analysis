#!/usr/bin/env python3
"""
Snapshot Torch Slurm and result-file state for post-gate transcription jobs.

This is a lightweight operational monitor. It does not modify jobs; it only
collects state from Torch at a fixed interval and writes JSON snapshots locally.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


def _run_ssh(target: str, remote_script: str) -> dict[str, Any]:
    cmd = [
        "ssh",
        "-o",
        "IdentitiesOnly=yes",
        target,
        f"python - <<'PY'\n{remote_script}\nPY",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ssh failed rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError("ssh command returned empty stdout")
    return json.loads(stdout)


def _remote_snapshot_script(job_ids: list[str], run_roots: dict[str, str]) -> str:
    jobs_json = json.dumps(job_ids)
    roots_json = json.dumps(run_roots)
    return f"""
import json
import re
import subprocess
from pathlib import Path

JOB_IDS = json.loads({jobs_json!r})
RUN_ROOTS = json.loads({roots_json!r})
WARNING_PATTERNS = [
    ("RESOURCE_EXHAUSTED", re.compile(r"RESOURCE_EXHAUSTED")),
    ("429", re.compile(r"\\b429\\b")),
    ("Too Many Requests", re.compile(r"Too Many Requests")),
    ("Deadline expired", re.compile(r"Deadline expired")),
    ("OUT_OF_MEMORY", re.compile(r"OUT_OF_MEMORY")),
]

def run_text(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {{
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }}

def count_jsonl_rows(path):
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())

def collect_run(root_str):
    root = Path(root_str)
    out = {{
        "exists": root.exists(),
        "provider_result_rows": 0,
        "provider_error_rows": 0,
        "parsed_transcription_rows": 0,
        "workers": {{}},
        "gateway_warnings": [],
    }}
    if not root.exists():
        return out
    results_root = root / "results"
    for worker in sorted(results_root.glob("worker*")):
        if not worker.is_dir():
            continue
        wname = worker.name
        result_rows = count_jsonl_rows(worker / "results" / "openai_results_shard000.jsonl")
        error_rows = count_jsonl_rows(worker / "results" / "openai_errors_shard000.jsonl")
        parsed_rows = count_jsonl_rows(worker / "transcriptions.jsonl")
        out["provider_result_rows"] += result_rows
        out["provider_error_rows"] += error_rows
        out["parsed_transcription_rows"] += parsed_rows
        out["workers"][wname] = {{
            "provider_result_rows": result_rows,
            "provider_error_rows": error_rows,
            "parsed_transcription_rows": parsed_rows,
        }}
        glog = worker / "results" / "gateway.log"
        if glog.is_file():
            text = glog.read_text(encoding="utf-8", errors="replace")
            for label, pattern in WARNING_PATTERNS:
                if pattern.search(text):
                    out["gateway_warnings"].append({{"worker": wname, "needle": label}})
                    break
    return out

squeue = run_text([
    "squeue",
    "-j",
    ",".join(JOB_IDS),
    "-o",
    "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R",
])
sacct = run_text([
    "sacct",
    "-j",
    ",".join(JOB_IDS),
    "--format=JobID,State,Elapsed,ExitCode",
    "-n",
])

payload = {{
    "captured_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "job_ids": JOB_IDS,
    "squeue": squeue,
    "sacct": sacct,
    "runs": {{name: collect_run(path) for name, path in RUN_ROOTS.items()}},
}}
print(json.dumps(payload, ensure_ascii=False))
"""


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ssh-target", default="torch")
    ap.add_argument("--job-id", action="append", required=True, help="Slurm job id to monitor")
    ap.add_argument(
        "--run-root",
        action="append",
        required=True,
        help="Named run root in the form name=/scratch/.../prepared",
    )
    ap.add_argument("--interval-seconds", type=int, default=900)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--output-dir", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_roots: dict[str, str] = {}
    for raw in args.run_root:
        if "=" not in raw:
            raise SystemExit(f"--run-root must be name=path, got: {raw}")
        name, path = raw.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise SystemExit(f"--run-root must be name=path, got: {raw}")
        run_roots[name] = path

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_path = args.output_dir / "snapshots.jsonl"
    latest_path = args.output_dir / "latest.json"
    summary_path = args.output_dir / "README.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"started_at_local={dt.datetime.now().astimezone().isoformat()}",
                f"ssh_target={args.ssh_target}",
                f"job_ids={','.join(args.job_id)}",
                f"interval_seconds={args.interval_seconds}",
                f"iterations={args.iterations}",
                f"run_roots={json.dumps(run_roots, ensure_ascii=False)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    remote_script = _remote_snapshot_script(args.job_id, run_roots)

    for idx in range(args.iterations):
        payload = {
            "iteration": idx + 1,
            "iterations_total": args.iterations,
            "captured_at_local": dt.datetime.now().astimezone().isoformat(),
        }
        try:
            payload["snapshot"] = _run_ssh(args.ssh_target, remote_script)
            payload["ok"] = True
        except Exception as exc:  # noqa: BLE001
            payload["ok"] = False
            payload["error"] = str(exc)

        with snapshots_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        latest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        if idx + 1 < args.iterations:
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()

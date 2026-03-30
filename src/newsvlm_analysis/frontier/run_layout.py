from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .contracts import (
    DEFAULT_RUNS_BASE,
    RUN_SUBDIRS,
    FrontierExperiment,
    FrontierManifest,
    FrontierPaths,
    default_experiments,
    utc_now_iso,
)
from .io_utils import write_json


def make_run_id(prefix: str = "frontier_fullcode_lab_v1") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def ensure_run_layout(base_dir: Path | str | None = None, run_id: str | None = None) -> dict[str, Path]:
    base = Path(base_dir or DEFAULT_RUNS_BASE).expanduser().resolve()
    rid = run_id or make_run_id()
    run_root = base / rid
    run_root.mkdir(parents=True, exist_ok=True)
    layout = {"run_root": run_root}
    for name in RUN_SUBDIRS:
        path = run_root / name
        path.mkdir(parents=True, exist_ok=True)
        layout[name] = path
    return layout


def layout_from_run_root(run_root: Path | str) -> dict[str, Path]:
    root = Path(run_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Run root does not exist: {root}")
    layout = {"run_root": root}
    for name in RUN_SUBDIRS:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        layout[name] = path
    return layout


def initialize_run_manifest(
    *,
    layout: dict[str, Path],
    paths: FrontierPaths | None = None,
    experiments: list[FrontierExperiment] | None = None,
    scope_label: str = "code_publication_full_issue",
    paper_framing: str = "dual_track_descriptive_first",
) -> FrontierManifest:
    run_root = layout["run_root"]
    manifest = FrontierManifest(
        schema_version="frontier_fullcode_lab_v1",
        run_id=run_root.name,
        created_at_utc=utc_now_iso(),
        run_root=str(run_root),
        scope_label=scope_label,
        paper_framing=paper_framing,
        paths=paths or FrontierPaths(),
        experiments=experiments or default_experiments(),
    )
    write_json(layout["manifests"] / "run_manifest.json", manifest.to_dict())
    write_json(
        layout["manifests"] / "experiment_registry.json",
        {"schema_version": "frontier_fullcode_lab_v1", "experiments": [exp.to_dict() for exp in manifest.experiments]},
    )
    return manifest


def write_stage_summary(layout: dict[str, Path], stage: str, payload: dict[str, Any]) -> None:
    write_json(layout["manifests"] / f"{stage}_summary.json", payload)

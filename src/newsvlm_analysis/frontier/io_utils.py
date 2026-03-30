from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

csv.field_size_limit(sys.maxsize)

STATE_TO_REGION = {
    "ct": "northeast",
    "me": "northeast",
    "ma": "northeast",
    "nh": "northeast",
    "ri": "northeast",
    "vt": "northeast",
    "nj": "northeast",
    "ny": "northeast",
    "pa": "northeast",
    "il": "midwest",
    "in": "midwest",
    "mi": "midwest",
    "oh": "midwest",
    "wi": "midwest",
    "ia": "midwest",
    "ks": "midwest",
    "mn": "midwest",
    "mo": "midwest",
    "ne": "midwest",
    "nd": "midwest",
    "sd": "midwest",
    "de": "south",
    "fl": "south",
    "ga": "south",
    "md": "south",
    "nc": "south",
    "sc": "south",
    "va": "south",
    "dc": "south",
    "wv": "south",
    "al": "south",
    "ky": "south",
    "ms": "south",
    "tn": "south",
    "ar": "south",
    "la": "south",
    "ok": "south",
    "tx": "south",
    "az": "west",
    "co": "west",
    "id": "west",
    "mt": "west",
    "nv": "west",
    "nm": "west",
    "ut": "west",
    "wy": "west",
    "ak": "west",
    "ca": "west",
    "hi": "west",
    "or": "west",
    "wa": "west",
}

JURISDICTION_PREFIX_RE = re.compile(
    r"^(the\s+)?(county of|city of|village of|township of|town of|borough of|parish of|county|city|village|township|town|borough|parish)\s+",
    re.IGNORECASE,
)
JURISDICTION_SUFFIX_RE = re.compile(
    r"\s+(county|city|village|township|town|borough|parish)$",
    re.IGNORECASE,
)
LEADING_OF_RE = re.compile(r"^of\s+", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")
SLUG_RE = re.compile(r"[^a-z0-9]+")


def norm_str(value: Any) -> str:
    return str(value or "").strip()


def clean_optional_str(value: Any) -> str:
    out = norm_str(value)
    if out.lower() in {"", "nan", "none", "null"}:
        return ""
    return out


def slugify(text: str) -> str:
    s = clean_optional_str(text).lower()
    s = SLUG_RE.sub("-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "unknown"


def normalize_city_name(text: str) -> str:
    s = clean_optional_str(text)
    while True:
        prior = s
        s = JURISDICTION_PREFIX_RE.sub("", s)
        s = LEADING_OF_RE.sub("", s)
        s = JURISDICTION_SUFFIX_RE.sub("", s)
        s = MULTISPACE_RE.sub(" ", s).strip(" ,.-")
        if s == prior:
            break
    s = s.replace("&", " and ")
    s = MULTISPACE_RE.sub(" ", s).strip(" ,.-")
    return s


def make_city_key(city_name: str, state_abbr: str) -> str:
    city = normalize_city_name(city_name)
    state = clean_optional_str(state_abbr).lower()
    if not city or not state:
        return ""
    return f"{slugify(city)}__{state}"


def region_for_state(state_abbr: str) -> str:
    return STATE_TO_REGION.get(clean_optional_str(state_abbr).lower(), "unknown")


def collapse_text(text: str) -> str:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def normalize_for_fingerprint(text: str) -> str:
    s = collapse_text(text).lower()
    s = re.sub(r"[\u2018\u2019]", "'", s)
    s = re.sub(r"[\u201c\u201d]", '"', s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = MULTISPACE_RE.sub(" ", s)
    return s.strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def simhash64(text: str) -> str:
    tokens = normalize_for_fingerprint(text).split()
    if not tokens:
        return "0" * 16
    weights = [0] * 64
    for token in tokens:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(h, "big", signed=False)
        for i in range(64):
            bit = 1 if ((value >> i) & 1) else -1
            weights[i] += bit
    out = 0
    for i, weight in enumerate(weights):
        if weight >= 0:
            out |= 1 << i
    return f"{out:016x}"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def relative_to_or_empty(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return ""


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default

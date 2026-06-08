"""Artifact IO helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


def write_jsonl(path: str | Path, rows: Iterable[Any], append: bool = False) -> Path:
    p = ensure_parent(path)
    mode = "a" if append else "w"
    with p.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_plain(row), sort_keys=True) + "\n")
    return p


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    materialized = [dict(row) for row in rows]
    p = ensure_parent(path)
    if not materialized:
        p.write_text("", encoding="utf-8")
        return p
    fieldnames = sorted({key for row in materialized for key in row})
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
    return p

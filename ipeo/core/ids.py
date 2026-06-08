"""Stable ID helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(payload: Any, prefix: str = "", length: int = 12) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:length]
    return f"{prefix}{digest}" if prefix else digest

"""Optional GEPA/MIPROv2/CAPO wrappers.

The dry-run MVP does not require these packages. These wrappers make the
availability state explicit in benchmark artifacts instead of failing imports at
module import time.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class OptionalBaselineStatus:
    name: str
    available: bool
    package: str
    reason: str | None = None


def _status(name: str, package: str) -> OptionalBaselineStatus:
    if importlib.util.find_spec(package) is None:
        return OptionalBaselineStatus(name=name, available=False, package=package, reason=f"Package '{package}' is not installed")
    return OptionalBaselineStatus(name=name, available=True, package=package)


def gepa_status() -> OptionalBaselineStatus:
    return _status("gepa", "dspy")


def miprov2_status() -> OptionalBaselineStatus:
    return _status("miprov2", "dspy")


def capo_status() -> OptionalBaselineStatus:
    return _status("capo", "promptolution")


def optional_baseline_statuses() -> list[OptionalBaselineStatus]:
    return [gepa_status(), miprov2_status(), capo_status()]

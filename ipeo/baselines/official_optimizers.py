"""Official optimizer integration status records.

The offline MVP records these baselines every run. Actual optimizer execution is
reserved for a live/API runner because GEPA, MIPROv2, and CAPO require their
own package-level task/program abstractions and usually paid or local LLM
backends.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipeo.baselines.optional_wrappers import OptionalBaselineStatus, optional_baseline_statuses


@dataclass(frozen=True)
class OfficialOptimizerRecord:
    name: str
    status: str
    package: str
    reason: str


def official_optimizer_records() -> list[OfficialOptimizerRecord]:
    records: list[OfficialOptimizerRecord] = []
    for status in optional_baseline_statuses():
        records.append(_record_from_status(status))
    return records


def _record_from_status(status: OptionalBaselineStatus) -> OfficialOptimizerRecord:
    if not status.available:
        return OfficialOptimizerRecord(
            name=status.name,
            status="skipped",
            package=status.package,
            reason=status.reason or "optional package unavailable",
        )
    return OfficialOptimizerRecord(
        name=status.name,
        status="available_not_run",
        package=status.package,
        reason="offline dry run records availability; live official optimizer execution belongs in an API-backed runner",
    )

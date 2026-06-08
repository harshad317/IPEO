"""Official optimizer integration status records."""

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
    if status.name in {"gepa", "miprov2"}:
        return OfficialOptimizerRecord(
            name=status.name,
            status="ready",
            package=status.package,
            reason="runner can execute this optimizer through DSPy when OPENAI_API_KEY is set",
        )
    return OfficialOptimizerRecord(
        name=status.name,
        status="not_implemented",
        package=status.package,
        reason="package is installed, but this runner does not yet execute CAPO",
    )

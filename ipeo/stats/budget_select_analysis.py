"""Analyze source-only budget selector decisions."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ipeo.core.io import read_jsonl

BUDGET_SELECT_METHOD = "ipeo_budget_select"
BUDGET_SELECT_SOURCE_VAL_METHOD = "ipeo_budget_select_source_val"
BUDGET_SELECTOR_METHODS = (BUDGET_SELECT_METHOD, BUDGET_SELECT_SOURCE_VAL_METHOD)


def budget_select_decision_rows(
    artifact_dir: str | Path,
    transfer_rows: list[dict[str, Any]],
    *,
    focus_task: str | None = None,
    run_label: str | None = None,
) -> list[dict[str, Any]]:
    """Join budget-selector audit JSONL with realized transfer rows."""

    stats_dir = _stats_dir(artifact_dir)
    transfer_by_task_method = {
        (str(row.get("task_id", "")), str(row.get("method", ""))): row
        for row in transfer_rows
    }
    outputs: list[dict[str, Any]] = []
    for selector_method in BUDGET_SELECTOR_METHODS:
        suffix = f"_{selector_method}.jsonl"
        for path in sorted(stats_dir.glob(f"*{suffix}")):
            task_id = path.name[: -len(suffix)]
            if focus_task is not None and task_id != focus_task:
                continue
            for audit in read_jsonl(path):
                row = _decision_row(
                    task_id=task_id,
                    selector_method=selector_method,
                    audit=audit,
                    transfer_by_task_method=transfer_by_task_method,
                )
                if run_label is not None:
                    row["run_label"] = run_label
                outputs.append(row)
    return outputs


def summarize_budget_select_decisions(
    rows: list[dict[str, Any]],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    confidence_level: float = 0.95,
    tolerance: float = 1e-9,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("task_id", "")), str(row.get("method", "")))].append(row)

    outputs: list[dict[str, Any]] = []
    for index, ((task_id, method), task_rows) in enumerate(sorted(grouped.items())):
        known = [row for row in task_rows if _has_number(row.get("oracle_budget_target_score"))]
        regrets = [_number(row.get("budget_selector_regret")) for row in known]
        chosen_scores = [_number(row.get("selected_target_score")) for row in known]
        oracle_scores = [_number(row.get("oracle_budget_target_score")) for row in known]
        chosen_calls = [_number(row.get("chosen_source_calls")) for row in task_rows if _has_number(row.get("chosen_source_calls"))]
        oracle_calls = [_number(row.get("oracle_budget_source_calls")) for row in known if _has_number(row.get("oracle_budget_source_calls"))]
        regret_ci = _ci(_bootstrap_mean_distribution(regrets, n_bootstrap=n_bootstrap, seed=seed + index), confidence_level)
        outputs.append(
            {
                "task_id": task_id,
                "method": method,
                "num_runs": len(task_rows),
                "num_known_oracle_runs": len(known),
                "selection_accuracy": _mean(1.0 if row.get("chosen_method") == row.get("oracle_budget_method") else 0.0 for row in known),
                "regret_free_rate": _mean(1.0 if _number(row.get("budget_selector_regret")) <= tolerance else 0.0 for row in known),
                "mean_selected_target_score": _mean(chosen_scores),
                "mean_oracle_budget_target_score": _mean(oracle_scores),
                "mean_budget_selector_regret": _mean(regrets),
                "budget_selector_regret_ci_low": regret_ci[0],
                "budget_selector_regret_ci_high": regret_ci[1],
                "mean_chosen_source_calls": _mean(chosen_calls),
                "mean_oracle_budget_source_calls": _mean(oracle_calls),
                "chosen_method_counts": _counts(row.get("chosen_method") for row in task_rows),
                "oracle_budget_method_counts": _counts(row.get("oracle_budget_method") for row in known),
            }
        )
    return outputs


def _decision_row(
    *,
    task_id: str,
    selector_method: str,
    audit: dict[str, Any],
    transfer_by_task_method: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    candidate_scores = _candidate_scores(audit)
    chosen_method = str(audit.get("chosen_method", ""))
    selected_transfer = transfer_by_task_method.get((task_id, selector_method), {})
    chosen_transfer = transfer_by_task_method.get((task_id, chosen_method), {})
    candidate_transfers = {
        method: transfer_by_task_method.get((task_id, method), {})
        for method in [str(row.get("method", "")) for row in candidate_scores]
    }
    scored_candidate_transfers = {
        method: row
        for method, row in candidate_transfers.items()
        if _has_number(row.get("target_score"))
    }
    oracle = _oracle_budget_row(scored_candidate_transfers)
    selected_target_score = _optional_number(selected_transfer.get("target_score"))
    oracle_score = _optional_number(oracle.get("target_score"))
    regret = None if selected_target_score is None or oracle_score is None else max(0.0, oracle_score - selected_target_score)
    source_rank = _rank_by_method(
        candidate_scores,
        method_key="method",
        key=lambda row: (-_number(row.get("source_score")), _number(row.get("source_calls")), str(row.get("method", ""))),
    )
    target_rank = _rank_by_method(
        [
            {"method": method, **row}
            for method, row in scored_candidate_transfers.items()
        ],
        method_key="method",
        key=lambda row: (-_number(row.get("target_score")), _total_calls(row), str(row.get("method", ""))),
    )
    return {
        "task_id": task_id,
        "method": selector_method,
        "chosen_method": chosen_method,
        "chosen_requested_budget": _optional_int(audit.get("requested_budget")),
        "chosen_source_calls": _optional_int(audit.get("source_calls")),
        "chosen_source_score": _optional_number(audit.get("source_score")),
        "chosen_prompt_id": audit.get("prompt_id", ""),
        "selected_target_score": selected_target_score,
        "chosen_candidate_target_score": _optional_number(chosen_transfer.get("target_score")),
        "oracle_budget_method": oracle.get("method", ""),
        "oracle_budget_target_score": oracle_score,
        "oracle_budget_source_calls": _optional_int(oracle.get("source_calls")),
        "oracle_budget_total_calls": _total_calls(oracle) if oracle else None,
        "oracle_budget_total_dollars": _optional_number(oracle.get("total_dollars")),
        "budget_selector_regret": regret,
        "budget_selection_outcome": _selection_outcome(regret),
        "source_score_rank_of_chosen": source_rank.get(chosen_method),
        "source_score_rank_of_oracle": source_rank.get(str(oracle.get("method", ""))),
        "target_rank_of_chosen": target_rank.get(chosen_method),
        "num_candidates": len(candidate_scores),
        "num_candidate_target_scores": len(scored_candidate_transfers),
        "candidate_score_summary": _candidate_summary(candidate_scores, scored_candidate_transfers),
    }


def _candidate_scores(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = audit.get("candidate_scores", [])
    return [dict(row) for row in rows if isinstance(row, dict)]


def _oracle_budget_row(candidate_transfers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not candidate_transfers:
        return {}
    rows = [dict(row, method=method) for method, row in candidate_transfers.items()]
    return max(rows, key=lambda row: (_number(row.get("target_score")), -_total_calls(row), -_number(row.get("total_dollars")), str(row.get("method", ""))))


def _rank_by_method(rows: list[dict[str, Any]], *, method_key: str, key: Any) -> dict[str, int]:
    ranked = sorted(rows, key=key)
    return {str(row.get(method_key, "")): index + 1 for index, row in enumerate(ranked)}


def _candidate_summary(candidate_scores: list[dict[str, Any]], candidate_transfers: dict[str, dict[str, Any]]) -> str:
    parts = []
    for row in sorted(candidate_scores, key=lambda item: _optional_int(item.get("requested_budget")) or 0):
        method = str(row.get("method", ""))
        target_score = _optional_number(candidate_transfers.get(method, {}).get("target_score"))
        target_text = "-" if target_score is None else f"{target_score:.3f}"
        parts.append(
            f"{method}:source={_number(row.get('source_score')):.3f},test={target_text},calls={_optional_int(row.get('source_calls')) or 0}"
        )
    return " | ".join(parts)


def _selection_outcome(regret: float | None, tolerance: float = 1e-9) -> str:
    if regret is None:
        return "unknown"
    return "oracle" if regret <= tolerance else "miss"


def _stats_dir(path: str | Path) -> Path:
    p = Path(path)
    return p if p.name == "stats" else p / "stats"


def _counts(values: Any) -> str:
    counter = Counter(str(value) for value in values if value not in (None, ""))
    return ",".join(f"{key}:{counter[key]}" for key in sorted(counter))


def _bootstrap_mean_distribution(values: list[float], *, n_bootstrap: int, seed: int) -> list[float]:
    if not values:
        return [0.0]
    if n_bootstrap <= 0:
        return [_mean(values)]
    rng = random.Random(seed)
    distribution = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(len(values))] for _ in values]
        distribution.append(_mean(sample))
    return distribution


def _ci(values: list[float], confidence_level: float) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    clipped = min(max(confidence_level, 0.0), 1.0)
    alpha = (1.0 - clipped) / 2.0
    ordered = sorted(values)
    lo_index = int(alpha * (len(ordered) - 1))
    hi_index = int((1.0 - alpha) * (len(ordered) - 1))
    return (ordered[lo_index], ordered[hi_index])


def _mean(values: Any) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else 0.0


def _total_calls(row: dict[str, Any]) -> int:
    return _int(row.get("source_calls")) + _int(row.get("target_calls"))


def _has_number(value: Any) -> bool:
    return _optional_number(value) is not None


def _optional_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float:
    parsed = _optional_number(value)
    return parsed if parsed is not None else 0.0


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    parsed = _optional_int(value)
    return parsed if parsed is not None else 0

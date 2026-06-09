"""Post-run benchmark analysis from transfer-regret artifacts."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ipeo.core.io import read_csv, write_csv
from ipeo.stats.budget_select_analysis import budget_select_decision_rows, summarize_budget_select_decisions

FLOAT_FIELDS = {
    "deployment_cost",
    "fixed_pool_oracle_score",
    "fixed_pool_regret",
    "p_value",
    "regret_reduction_vs_source_average",
    "target_score",
    "total_dollars",
}
INT_FIELDS = {
    "final_target_test_calls",
    "source_calls",
    "source_train_calls",
    "source_validation_calls",
    "target_calls",
    "target_optimization_calls",
    "target_validation_calls",
}
BOOL_FIELDS = {"uses_target_test_for_selection", "uses_target_train", "uses_target_validation"}
DEFAULT_BASELINES = [
    "miprov2",
    "gepa",
    "source_average",
    "pooled_source",
    "worst_source_robust",
    "asha_fixed_pool",
    "target_only_bo_fixed_pool",
    "best_source_transfer",
]
DEFAULT_IPEO_METHODS = [
    "ipeo_budget_select",
    "ipeo_budget_select_source_val",
    "ipeo_budget_200",
    "ipeo_budget_500",
    "ipeo_budget_1000",
    "ipeo_select_existing",
    "ipeo_composed_vs_existing",
    "ipeo_zero",
]


def analyze_artifact_dir(
    artifact_dir: str | Path,
    *,
    focus_task: str | None = None,
    ipeo_methods: list[str] | None = None,
    baseline_methods: list[str] | None = None,
    best_score_tolerance: float = 1e-9,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
    confidence_level: float = 0.95,
) -> dict[str, list[dict[str, Any]]]:
    artifact_path = Path(artifact_dir)
    transfer_path = _resolve_transfer_path(artifact_path)
    all_rows = load_transfer_rows(transfer_path)
    rows = all_rows
    if focus_task is not None:
        rows = [row for row in rows if row.get("task_id") == focus_task]
    if not rows:
        raise ValueError(_transfer_rows_error(artifact_path, transfer_path, all_rows, focus_task))

    ipeo_methods = ipeo_methods or DEFAULT_IPEO_METHODS
    baseline_methods = baseline_methods or DEFAULT_BASELINES
    budget_select_rows = budget_select_decision_rows(
        transfer_path.parent,
        rows,
        focus_task=focus_task,
    )
    outputs = {
        "per_task_winners": per_task_winners(rows, best_score_tolerance=best_score_tolerance),
        "track_summary": track_summary(rows),
        "method_task_summary": method_task_summary(rows),
        "ipeo_vs_baselines": ipeo_vs_baselines(rows, ipeo_methods=ipeo_methods, baseline_methods=baseline_methods),
        "bootstrap_comparisons": bootstrap_comparisons(
            rows,
            ipeo_methods=ipeo_methods,
            baseline_methods=baseline_methods,
            n_bootstrap=bootstrap_samples,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
        ),
        "cost_frontier": cost_frontier(rows),
        "budget_select_decisions": budget_select_rows,
        "budget_select_summary": summarize_budget_select_decisions(
            budget_select_rows,
            n_bootstrap=bootstrap_samples,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
        ),
    }
    stats_dir = transfer_path.parent
    for name, output_rows in outputs.items():
        suffix = f"_{focus_task}" if focus_task else ""
        write_csv(stats_dir / f"analysis_{name}{suffix}.csv", output_rows)
    return outputs


def _resolve_transfer_path(artifact_path: Path) -> Path:
    default_path = artifact_path / "stats" / "transfer_regret.csv"
    if default_path.exists():
        return default_path
    direct_stats_path = artifact_path / "transfer_regret.csv"
    if direct_stats_path.exists():
        return direct_stats_path
    return default_path


def load_transfer_rows(path: str | Path) -> list[dict[str, Any]]:
    return [_coerce_row(row) for row in read_csv(path)]


def _transfer_rows_error(
    artifact_path: Path,
    transfer_path: Path,
    loaded_rows: list[dict[str, Any]],
    focus_task: str | None,
) -> str:
    if focus_task is not None and loaded_rows:
        available_tasks = ", ".join(sorted({str(row.get("task_id", "")) for row in loaded_rows}))
        return (
            f"No transfer rows for --focus_task {focus_task!r} in {transfer_path}. "
            f"Available tasks: {available_tasks or 'none'}."
        )

    reason = "missing" if not transfer_path.exists() else "empty"
    message = [
        f"No transfer rows found in {transfer_path} ({reason}).",
        "Point --artifact_dir at a completed run directory that contains stats/transfer_regret.csv,",
        "or at the stats directory itself.",
    ]
    candidates = _nearby_transfer_paths(artifact_path)
    if candidates:
        message.append("Nearby transfer_regret.csv files:")
        for candidate in candidates[:10]:
            row_count = len(read_csv(candidate))
            message.append(f"  - {candidate.parent.parent} ({row_count} rows)")
    else:
        message.append("No nearby transfer_regret.csv files were found.")
    return "\n".join(message)


def _nearby_transfer_paths(artifact_path: Path) -> list[Path]:
    roots = []
    if artifact_path.exists() and artifact_path.is_dir():
        roots.append(artifact_path)
    if artifact_path.parent.exists() and artifact_path.parent.is_dir():
        roots.append(artifact_path.parent)
    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in roots:
        for candidate in sorted(root.glob("**/stats/transfer_regret.csv")):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(candidate)
    return candidates


def per_task_winners(rows: list[dict[str, Any]], *, best_score_tolerance: float = 1e-9) -> list[dict[str, Any]]:
    grouped = _group_by(rows, "task_id")
    outputs: list[dict[str, Any]] = []
    for task_id, task_rows in sorted(grouped.items()):
        best_score = max(_float(row, "target_score") for row in task_rows)
        best_rows = [row for row in task_rows if _float(row, "target_score") >= best_score - best_score_tolerance]
        cheapest_best = min(best_rows, key=_total_calls_then_dollars)
        best_ipeo = _best_matching(task_rows, lambda row: str(row.get("method", "")).startswith("ipeo_"))
        best_target = _best_matching(task_rows, lambda row: row.get("benchmark_track") == "target_optimization")
        best_source = _best_matching(task_rows, lambda row: row.get("benchmark_track") == "source_transfer")
        outputs.append(
            {
                "task_id": task_id,
                "best_score": best_score,
                "best_methods": ",".join(sorted(str(row["method"]) for row in best_rows)),
                "cheapest_best_method": cheapest_best["method"],
                "cheapest_best_total_calls": _total_calls(cheapest_best),
                "cheapest_best_total_dollars": _float(cheapest_best, "total_dollars"),
                "best_ipeo_method": best_ipeo.get("method") if best_ipeo else "",
                "best_ipeo_score": _float(best_ipeo, "target_score") if best_ipeo else None,
                "best_target_optimization_method": best_target.get("method") if best_target else "",
                "best_target_optimization_score": _float(best_target, "target_score") if best_target else None,
                "best_source_transfer_method": best_source.get("method") if best_source else "",
                "best_source_transfer_score": _float(best_source, "target_score") if best_source else None,
            }
        )
    return outputs


def track_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_by(rows, "benchmark_track")
    outputs: list[dict[str, Any]] = []
    for track, track_rows in sorted(grouped.items()):
        outputs.append(
            {
                "benchmark_track": track,
                "num_rows": len(track_rows),
                "mean_target_score": _mean(_float(row, "target_score") for row in track_rows),
                "mean_fixed_pool_regret": _mean(_float(row, "fixed_pool_regret") for row in track_rows),
                "mean_total_calls": _mean(_total_calls(row) for row in track_rows),
                "mean_total_dollars": _mean(_float(row, "total_dollars") for row in track_rows),
                "best_score": max(_float(row, "target_score") for row in track_rows),
                "best_method": max(track_rows, key=lambda row: (_float(row, "target_score"), -_total_calls(row)))["method"],
            }
        )
    return outputs


def method_task_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (str(item.get("task_id", "")), -_float(item, "target_score"), _total_calls(item), str(item.get("method", "")))):
        outputs.append(
            {
                "task_id": row.get("task_id"),
                "method": row.get("method"),
                "benchmark_track": row.get("benchmark_track", ""),
                "target_score": _float(row, "target_score"),
                "fixed_pool_regret": _float(row, "fixed_pool_regret"),
                "total_calls": _total_calls(row),
                "source_calls": _int(row, "source_calls"),
                "target_calls": _int(row, "target_calls"),
                "total_dollars": _float(row, "total_dollars"),
                "selection_access": row.get("selection_access", ""),
                "uses_target_validation": row.get("uses_target_validation", False),
                "uses_target_test_for_selection": row.get("uses_target_test_for_selection", False),
            }
        )
    return outputs


def ipeo_vs_baselines(
    rows: list[dict[str, Any]],
    *,
    ipeo_methods: list[str],
    baseline_methods: list[str],
) -> list[dict[str, Any]]:
    grouped = _group_by(rows, "task_id")
    outputs: list[dict[str, Any]] = []
    for task_id, task_rows in sorted(grouped.items()):
        for ipeo_method in ipeo_methods:
            ipeo = next((row for row in task_rows if row.get("method") == ipeo_method), None)
            if ipeo is None:
                continue
            for baseline in _baseline_rows(task_rows, baseline_methods):
                baseline_method = str(baseline.get("method", ""))
                score_delta = _float(ipeo, "target_score") - _float(baseline, "target_score")
                call_delta = _total_calls(ipeo) - _total_calls(baseline)
                dollar_delta = _float(ipeo, "total_dollars") - _float(baseline, "total_dollars")
                outputs.append(
                    {
                        "task_id": task_id,
                        "ipeo_method": ipeo_method,
                        "baseline_method": baseline_method,
                        "ipeo_score": _float(ipeo, "target_score"),
                        "baseline_score": _float(baseline, "target_score"),
                        "score_delta": score_delta,
                        "ipeo_total_calls": _total_calls(ipeo),
                        "baseline_total_calls": _total_calls(baseline),
                        "call_delta": call_delta,
                        "ipeo_total_dollars": _float(ipeo, "total_dollars"),
                        "baseline_total_dollars": _float(baseline, "total_dollars"),
                        "dollar_delta": dollar_delta,
                        "winner": _winner(score_delta),
                    }
                )
    return outputs


def bootstrap_comparisons(
    rows: list[dict[str, Any]],
    *,
    ipeo_methods: list[str],
    baseline_methods: list[str],
    n_bootstrap: int = 1000,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> list[dict[str, Any]]:
    grouped = _group_by(rows, "task_id")
    paired: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task_id, task_rows in sorted(grouped.items()):
        baselines = _baseline_rows(task_rows, baseline_methods)
        for ipeo_method in ipeo_methods:
            ipeo = next((row for row in task_rows if row.get("method") == ipeo_method), None)
            if ipeo is None:
                continue
            for baseline in baselines:
                baseline_method = str(baseline.get("method", ""))
                paired[(ipeo_method, baseline_method)].append(
                    {
                        "task_id": task_id,
                        "ipeo_score": _float(ipeo, "target_score"),
                        "baseline_score": _float(baseline, "target_score"),
                        "score_delta": _float(ipeo, "target_score") - _float(baseline, "target_score"),
                        "call_delta": _total_calls(ipeo) - _total_calls(baseline),
                        "dollar_delta": _float(ipeo, "total_dollars") - _float(baseline, "total_dollars"),
                    }
                )

    outputs: list[dict[str, Any]] = []
    for pair_index, ((ipeo_method, baseline_method), pair_rows) in enumerate(sorted(paired.items())):
        score_deltas = [float(row["score_delta"]) for row in pair_rows]
        call_deltas = [float(row["call_delta"]) for row in pair_rows]
        dollar_deltas = [float(row["dollar_delta"]) for row in pair_rows]
        score_dist = _bootstrap_mean_distribution(score_deltas, n_bootstrap=n_bootstrap, seed=seed + pair_index * 3)
        call_dist = _bootstrap_mean_distribution(call_deltas, n_bootstrap=n_bootstrap, seed=seed + pair_index * 3 + 1)
        dollar_dist = _bootstrap_mean_distribution(dollar_deltas, n_bootstrap=n_bootstrap, seed=seed + pair_index * 3 + 2)
        score_ci = _ci(score_dist, confidence_level)
        call_ci = _ci(call_dist, confidence_level)
        dollar_ci = _ci(dollar_dist, confidence_level)
        outputs.append(
            {
                "ipeo_method": ipeo_method,
                "baseline_method": baseline_method,
                "num_tasks": len(pair_rows),
                "tasks": ",".join(str(row["task_id"]) for row in pair_rows),
                "mean_ipeo_score": _mean(row["ipeo_score"] for row in pair_rows),
                "mean_baseline_score": _mean(row["baseline_score"] for row in pair_rows),
                "mean_score_delta": _mean(score_deltas),
                "score_delta_ci_low": score_ci[0],
                "score_delta_ci_high": score_ci[1],
                "probability_ipeo_score_better": _probability(score_dist, lambda value: value > 0.0),
                "score_outcome": _ci_outcome(score_ci),
                "mean_call_delta": _mean(call_deltas),
                "call_delta_ci_low": call_ci[0],
                "call_delta_ci_high": call_ci[1],
                "probability_ipeo_fewer_calls": _probability(call_dist, lambda value: value < 0.0),
                "mean_dollar_delta": _mean(dollar_deltas),
                "dollar_delta_ci_low": dollar_ci[0],
                "dollar_delta_ci_high": dollar_ci[1],
                "probability_ipeo_cheaper": _probability(dollar_dist, lambda value: value < 0.0),
                "cost_outcome": _cost_ci_outcome(dollar_ci),
            }
        )
    return sorted(outputs, key=lambda row: (str(row["ipeo_method"]), str(row["baseline_method"])))


def _baseline_rows(rows: list[dict[str, Any]], baseline_methods: list[str]) -> list[dict[str, Any]]:
    methods = set(baseline_methods)
    if "all" in methods:
        return [row for row in rows if not str(row.get("method", "")).startswith("ipeo_")]
    selected: list[dict[str, Any]] = []
    for row in rows:
        method = str(row.get("method", ""))
        if method in methods:
            selected.append(row)
        elif "best_source_transfer" in methods and method.startswith("best_source_transfer:"):
            selected.append(row)
    return selected


def cost_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_by(rows, "task_id")
    outputs: list[dict[str, Any]] = []
    for task_id, task_rows in sorted(grouped.items()):
        for row in task_rows:
            if _is_pareto_efficient(row, task_rows):
                outputs.append(
                    {
                        "task_id": task_id,
                        "method": row.get("method"),
                        "benchmark_track": row.get("benchmark_track", ""),
                        "target_score": _float(row, "target_score"),
                        "fixed_pool_regret": _float(row, "fixed_pool_regret"),
                        "total_calls": _total_calls(row),
                        "total_dollars": _float(row, "total_dollars"),
                    }
                )
    return sorted(outputs, key=lambda item: (str(item["task_id"]), _int(item, "total_calls"), -_float(item, "target_score")))


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)
    for field in FLOAT_FIELDS:
        if field in out:
            out[field] = _parse_float(out[field])
    for field in INT_FIELDS:
        if field in out:
            out[field] = _parse_int(out[field])
    for field in BOOL_FIELDS:
        if field in out:
            out[field] = str(out[field]).strip().lower() == "true"
    if "benchmark_track" not in out:
        out["benchmark_track"] = ""
    return out


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return grouped


def _best_matching(rows: list[dict[str, Any]], predicate: Any) -> dict[str, Any] | None:
    matched = [row for row in rows if predicate(row)]
    if not matched:
        return None
    return max(matched, key=lambda row: (_float(row, "target_score"), -_total_calls(row), -_float(row, "total_dollars")))


def _is_pareto_efficient(row: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    score = _float(row, "target_score")
    calls = _total_calls(row)
    dollars = _float(row, "total_dollars")
    for other in rows:
        if other is row:
            continue
        other_score = _float(other, "target_score")
        other_calls = _total_calls(other)
        other_dollars = _float(other, "total_dollars")
        no_worse = other_score >= score and other_calls <= calls and other_dollars <= dollars
        strictly_better = other_score > score or other_calls < calls or other_dollars < dollars
        if no_worse and strictly_better:
            return False
    return True


def _winner(score_delta: float, tolerance: float = 1e-9) -> str:
    if score_delta > tolerance:
        return "ipeo"
    if score_delta < -tolerance:
        return "baseline"
    return "tie"


def _ci_outcome(ci: tuple[float, float], tolerance: float = 1e-9) -> str:
    if ci[0] > tolerance:
        return "ipeo"
    if ci[1] < -tolerance:
        return "baseline"
    return "uncertain"


def _cost_ci_outcome(ci: tuple[float, float], tolerance: float = 1e-12) -> str:
    if ci[1] < -tolerance:
        return "ipeo_cheaper"
    if ci[0] > tolerance:
        return "baseline_cheaper"
    return "uncertain"


def _bootstrap_mean_distribution(values: list[float], *, n_bootstrap: int, seed: int) -> list[float]:
    if not values:
        return [0.0]
    if n_bootstrap <= 0:
        return [_mean(values)]
    rng = random.Random(seed)
    distribution: list[float] = []
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


def _probability(values: list[float], predicate: Any) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if predicate(value)) / len(values)


def _mean(values: Any) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else 0.0


def _total_calls(row: dict[str, Any]) -> int:
    return _int(row, "source_calls") + _int(row, "target_calls")


def _total_calls_then_dollars(row: dict[str, Any]) -> tuple[int, float, str]:
    return (_total_calls(row), _float(row, "total_dollars"), str(row.get("method", "")))


def _float(row: dict[str, Any] | None, key: str) -> float:
    if row is None:
        return 0.0
    return _parse_float(row.get(key, 0.0))


def _int(row: dict[str, Any] | None, key: str) -> int:
    if row is None:
        return 0
    return _parse_int(row.get(key, 0))


def _parse_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0

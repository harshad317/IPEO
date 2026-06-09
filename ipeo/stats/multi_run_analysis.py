"""Aggregate benchmark analysis across multiple completed runs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ipeo.core.io import write_csv
from ipeo.stats.benchmark_analysis import (
    DEFAULT_BASELINES,
    DEFAULT_IPEO_METHODS,
    _baseline_rows,
    _bootstrap_mean_distribution,
    _ci,
    _ci_outcome,
    _cost_ci_outcome,
    _float,
    _parse_float,
    _probability,
    _resolve_transfer_path,
    _total_calls,
    load_transfer_rows,
)


def analyze_many_artifact_dirs(
    artifact_dirs: list[str | Path],
    *,
    output_dir: str | Path,
    focus_task: str | None = None,
    ipeo_methods: list[str] | None = None,
    baseline_methods: list[str] | None = None,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
    confidence_level: float = 0.95,
    best_score_tolerance: float = 1e-9,
) -> dict[str, list[dict[str, Any]]]:
    rows, input_runs = load_multi_run_transfer_rows(artifact_dirs, focus_task=focus_task)
    if not rows:
        raise ValueError("No transfer rows found across the requested artifact directories.")

    ipeo_methods = ipeo_methods or DEFAULT_IPEO_METHODS
    baseline_methods = baseline_methods or DEFAULT_BASELINES
    method_rows = multi_run_method_summary(
        rows,
        n_bootstrap=bootstrap_samples,
        seed=bootstrap_seed,
        confidence_level=confidence_level,
        best_score_tolerance=best_score_tolerance,
    )
    comparison_rows = multi_run_ipeo_vs_baselines(
        rows,
        ipeo_methods=ipeo_methods,
        baseline_methods=baseline_methods,
        n_bootstrap=bootstrap_samples,
        seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    frontier_rows = multi_run_cost_frontier(method_rows)
    outputs = {
        "input_runs": input_runs,
        "method_summary": method_rows,
        "ipeo_vs_baselines": comparison_rows,
        "cost_frontier": frontier_rows,
        "combined_transfer_rows": rows,
    }

    stats_dir = Path(output_dir) / "stats"
    for name, output_rows in outputs.items():
        suffix = f"_{focus_task}" if focus_task else ""
        write_csv(stats_dir / f"multi_run_{name}{suffix}.csv", output_rows)
    return outputs


def load_multi_run_transfer_rows(
    artifact_dirs: list[str | Path],
    *,
    focus_task: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = _run_labels([Path(path) for path in artifact_dirs])
    rows: list[dict[str, Any]] = []
    input_runs: list[dict[str, Any]] = []
    for artifact_dir, run_label in zip([Path(path) for path in artifact_dirs], labels):
        transfer_path = _resolve_transfer_path(artifact_dir)
        run_rows = load_transfer_rows(transfer_path)
        if focus_task is not None:
            run_rows = [row for row in run_rows if row.get("task_id") == focus_task]
        for row in run_rows:
            enriched = dict(row)
            enriched["run_label"] = run_label
            enriched["artifact_dir"] = str(transfer_path.parent.parent)
            enriched["transfer_path"] = str(transfer_path)
            rows.append(enriched)
        input_runs.append(
            {
                "run_label": run_label,
                "artifact_dir": str(artifact_dir),
                "transfer_path": str(transfer_path),
                "row_count": len(run_rows),
            }
        )
    return rows, input_runs


def multi_run_method_summary(
    rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
    confidence_level: float,
    best_score_tolerance: float = 1e-9,
) -> list[dict[str, Any]]:
    best_by_context = _best_by_context(rows, best_score_tolerance=best_score_tolerance)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("task_id", "")), str(row.get("method", "")))].append(row)

    outputs: list[dict[str, Any]] = []
    for index, ((task_id, method), group_rows) in enumerate(sorted(grouped.items())):
        scores = [_float(row, "target_score") for row in group_rows]
        calls = [_total_calls(row) for row in group_rows]
        dollars = [_float(row, "total_dollars") for row in group_rows]
        score_ci = _ci(_bootstrap_means(scores, n_bootstrap=n_bootstrap, seed=seed + index * 3), confidence_level)
        call_ci = _ci(_bootstrap_means(calls, n_bootstrap=n_bootstrap, seed=seed + index * 3 + 1), confidence_level)
        dollar_ci = _ci(_bootstrap_means(dollars, n_bootstrap=n_bootstrap, seed=seed + index * 3 + 2), confidence_level)
        best_wins = 0
        cheapest_best_wins = 0
        for row in group_rows:
            context = (str(row.get("run_label", "")), str(row.get("task_id", "")))
            best = best_by_context.get(context, {})
            best_methods = set(best.get("best_methods", []))
            if method in best_methods:
                best_wins += 1
            if method == best.get("cheapest_best_method"):
                cheapest_best_wins += 1
        outputs.append(
            {
                "task_id": task_id,
                "method": method,
                "benchmark_track": group_rows[0].get("benchmark_track", ""),
                "num_run_task_rows": len(group_rows),
                "num_runs": len({str(row.get("run_label", "")) for row in group_rows}),
                "mean_target_score": _mean(scores),
                "score_ci_low": score_ci[0],
                "score_ci_high": score_ci[1],
                "mean_total_calls": _mean(calls),
                "calls_ci_low": call_ci[0],
                "calls_ci_high": call_ci[1],
                "mean_total_dollars": _mean(dollars),
                "dollars_ci_low": dollar_ci[0],
                "dollars_ci_high": dollar_ci[1],
                "best_score_win_rate": best_wins / len(group_rows) if group_rows else 0.0,
                "cheapest_best_win_rate": cheapest_best_wins / len(group_rows) if group_rows else 0.0,
            }
        )
    return sorted(outputs, key=lambda row: (str(row["task_id"]), -_parse_float(row["mean_target_score"]), _parse_float(row["mean_total_calls"]), str(row["method"])))


def multi_run_ipeo_vs_baselines(
    rows: list[dict[str, Any]],
    *,
    ipeo_methods: list[str],
    baseline_methods: list[str],
    n_bootstrap: int,
    seed: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    contexts = _group_by_context(rows)
    paired: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (run_label, task_id), context_rows in sorted(contexts.items()):
        baselines = _baseline_rows(context_rows, baseline_methods)
        for ipeo_method in ipeo_methods:
            ipeo = next((row for row in context_rows if row.get("method") == ipeo_method), None)
            if ipeo is None:
                continue
            for baseline in baselines:
                baseline_method = str(baseline.get("method", ""))
                paired[(ipeo_method, baseline_method)].append(
                    {
                        "run_label": run_label,
                        "task_id": task_id,
                        "ipeo_score": _float(ipeo, "target_score"),
                        "baseline_score": _float(baseline, "target_score"),
                        "score_delta": _float(ipeo, "target_score") - _float(baseline, "target_score"),
                        "call_delta": _total_calls(ipeo) - _total_calls(baseline),
                        "dollar_delta": _float(ipeo, "total_dollars") - _float(baseline, "total_dollars"),
                    }
                )

    outputs: list[dict[str, Any]] = []
    for index, ((ipeo_method, baseline_method), pair_rows) in enumerate(sorted(paired.items())):
        score_deltas = [float(row["score_delta"]) for row in pair_rows]
        call_deltas = [float(row["call_delta"]) for row in pair_rows]
        dollar_deltas = [float(row["dollar_delta"]) for row in pair_rows]
        score_dist = _bootstrap_means(score_deltas, n_bootstrap=n_bootstrap, seed=seed + index * 3)
        call_dist = _bootstrap_means(call_deltas, n_bootstrap=n_bootstrap, seed=seed + index * 3 + 1)
        dollar_dist = _bootstrap_means(dollar_deltas, n_bootstrap=n_bootstrap, seed=seed + index * 3 + 2)
        score_ci = _ci(score_dist, confidence_level)
        call_ci = _ci(call_dist, confidence_level)
        dollar_ci = _ci(dollar_dist, confidence_level)
        wins = sum(1 for value in score_deltas if value > 1e-9)
        ties = sum(1 for value in score_deltas if abs(value) <= 1e-9)
        losses = sum(1 for value in score_deltas if value < -1e-9)
        outputs.append(
            {
                "ipeo_method": ipeo_method,
                "baseline_method": baseline_method,
                "num_pairs": len(pair_rows),
                "num_runs": len({str(row["run_label"]) for row in pair_rows}),
                "tasks": ",".join(sorted({str(row["task_id"]) for row in pair_rows})),
                "mean_ipeo_score": _mean(row["ipeo_score"] for row in pair_rows),
                "mean_baseline_score": _mean(row["baseline_score"] for row in pair_rows),
                "mean_score_delta": _mean(score_deltas),
                "score_delta_ci_low": score_ci[0],
                "score_delta_ci_high": score_ci[1],
                "score_win_rate": wins / len(pair_rows) if pair_rows else 0.0,
                "score_tie_rate": ties / len(pair_rows) if pair_rows else 0.0,
                "score_loss_rate": losses / len(pair_rows) if pair_rows else 0.0,
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


def multi_run_cost_frontier(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in method_rows:
        grouped[str(row.get("task_id", ""))].append(row)

    outputs: list[dict[str, Any]] = []
    for task_id, task_rows in sorted(grouped.items()):
        for row in task_rows:
            if _aggregate_is_pareto_efficient(row, task_rows):
                outputs.append(
                    {
                        "task_id": task_id,
                        "method": row.get("method"),
                        "benchmark_track": row.get("benchmark_track", ""),
                        "mean_target_score": _parse_float(row.get("mean_target_score")),
                        "score_ci_low": _parse_float(row.get("score_ci_low")),
                        "score_ci_high": _parse_float(row.get("score_ci_high")),
                        "mean_total_calls": _parse_float(row.get("mean_total_calls")),
                        "mean_total_dollars": _parse_float(row.get("mean_total_dollars")),
                        "best_score_win_rate": _parse_float(row.get("best_score_win_rate")),
                        "cheapest_best_win_rate": _parse_float(row.get("cheapest_best_win_rate")),
                    }
                )
    return sorted(outputs, key=lambda row: (str(row["task_id"]), _parse_float(row["mean_total_calls"]), -_parse_float(row["mean_target_score"])))


def _best_by_context(rows: list[dict[str, Any]], *, best_score_tolerance: float) -> dict[tuple[str, str], dict[str, Any]]:
    outputs: dict[tuple[str, str], dict[str, Any]] = {}
    for context, context_rows in _group_by_context(rows).items():
        best_score = max(_float(row, "target_score") for row in context_rows)
        best_rows = [row for row in context_rows if _float(row, "target_score") >= best_score - best_score_tolerance]
        cheapest_best = min(best_rows, key=lambda row: (_total_calls(row), _float(row, "total_dollars"), str(row.get("method", ""))))
        outputs[context] = {
            "best_score": best_score,
            "best_methods": [str(row.get("method", "")) for row in best_rows],
            "cheapest_best_method": str(cheapest_best.get("method", "")),
        }
    return outputs


def _group_by_context(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("run_label", "")), str(row.get("task_id", "")))].append(row)
    return grouped


def _aggregate_is_pareto_efficient(row: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    score = _parse_float(row.get("mean_target_score"))
    calls = _parse_float(row.get("mean_total_calls"))
    dollars = _parse_float(row.get("mean_total_dollars"))
    for other in rows:
        if other is row:
            continue
        other_score = _parse_float(other.get("mean_target_score"))
        other_calls = _parse_float(other.get("mean_total_calls"))
        other_dollars = _parse_float(other.get("mean_total_dollars"))
        no_worse = other_score >= score and other_calls <= calls and other_dollars <= dollars
        strictly_better = other_score > score or other_calls < calls or other_dollars < dollars
        if no_worse and strictly_better:
            return False
    return True


def _run_labels(paths: list[Path]) -> list[str]:
    raw_labels = []
    for path in paths:
        label_path = path.parent if path.name == "stats" else path
        raw_labels.append(label_path.name or str(label_path))
    counts: dict[str, int] = defaultdict(int)
    labels: list[str] = []
    for raw_label in raw_labels:
        counts[raw_label] += 1
        labels.append(raw_label if counts[raw_label] == 1 else f"{raw_label}_{counts[raw_label]}")
    return labels


def _bootstrap_means(values: list[float] | list[int], *, n_bootstrap: int, seed: int) -> list[float]:
    return _bootstrap_mean_distribution([float(value) for value in values], n_bootstrap=n_bootstrap, seed=seed)


def _mean(values: Any) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else 0.0

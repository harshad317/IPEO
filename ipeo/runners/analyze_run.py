"""Analyze an existing IPEO benchmark artifact directory."""

from __future__ import annotations

import argparse
from typing import Any

from rich.console import Console
from rich.table import Table

from ipeo.stats.benchmark_analysis import DEFAULT_BASELINES, DEFAULT_IPEO_METHODS, analyze_artifact_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze IPEO benchmark results")
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--focus_task", default=None)
    parser.add_argument("--ipeo_methods", nargs="+", default=DEFAULT_IPEO_METHODS)
    parser.add_argument("--baseline_methods", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument("--confidence_level", type=float, default=0.95)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_color", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    outputs = analyze_artifact_dir(
        args.artifact_dir,
        focus_task=args.focus_task,
        ipeo_methods=_expand_values(args.ipeo_methods),
        baseline_methods=_expand_values(args.baseline_methods),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    console = Console(no_color=args.no_color, quiet=args.quiet)
    if not args.quiet:
        _print_per_task_winners(console, outputs["per_task_winners"])
        _print_track_summary(console, outputs["track_summary"])
        _print_cost_frontier(console, outputs["cost_frontier"])
        _print_budget_select_summary(console, outputs["budget_select_summary"])
        _print_budget_select_decisions(console, outputs["budget_select_decisions"])
        _print_bootstrap_comparisons(console, outputs["bootstrap_comparisons"])
        _print_ipeo_deltas(console, outputs["ipeo_vs_baselines"])
        suffix = f" for {args.focus_task}" if args.focus_task else ""
        console.print(f"[bold cyan]Wrote analysis CSVs under {args.artifact_dir}/stats{suffix}.[/bold cyan]")
    return outputs


def _expand_values(values: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        expanded.extend(part.strip() for part in value.split(",") if part.strip())
    return expanded


def _print_per_task_winners(console: Console, rows: list[dict[str, Any]]) -> None:
    table = Table(title="Per-task winners")
    for column in ["task_id", "best_score", "cheapest_best_method", "best_ipeo_method", "best_target_optimization_method", "best_source_transfer_method"]:
        table.add_column(column)
    for row in rows:
        table.add_row(
            str(row.get("task_id", "")),
            _fmt(row.get("best_score")),
            str(row.get("cheapest_best_method", "")),
            _method_score(row, "best_ipeo_method", "best_ipeo_score"),
            _method_score(row, "best_target_optimization_method", "best_target_optimization_score"),
            _method_score(row, "best_source_transfer_method", "best_source_transfer_score"),
        )
    console.print(table)


def _print_track_summary(console: Console, rows: list[dict[str, Any]]) -> None:
    table = Table(title="Benchmark tracks")
    for column in ["benchmark_track", "mean_target_score", "mean_total_calls", "mean_total_dollars", "best_score", "best_method"]:
        table.add_column(column)
    for row in rows:
        table.add_row(
            str(row.get("benchmark_track", "")),
            _fmt(row.get("mean_target_score")),
            _fmt(row.get("mean_total_calls"), digits=1),
            _fmt(row.get("mean_total_dollars"), digits=6),
            _fmt(row.get("best_score")),
            str(row.get("best_method", "")),
        )
    console.print(table)


def _print_cost_frontier(console: Console, rows: list[dict[str, Any]]) -> None:
    table = Table(title="Cost/performance frontier")
    for column in ["task_id", "method", "track", "score", "calls", "dollars"]:
        table.add_column(column)
    for row in rows[:30]:
        table.add_row(
            str(row.get("task_id", "")),
            str(row.get("method", "")),
            str(row.get("benchmark_track", "")),
            _fmt(row.get("target_score")),
            str(int(float(row.get("total_calls", 0) or 0))),
            _fmt(row.get("total_dollars"), digits=6),
        )
    console.print(table)


def _print_budget_select_summary(console: Console, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    table = Table(title="Budget selector summary")
    for column in ["task", "selector", "runs", "exact", "regret_free", "regret_ci", "chosen", "oracle"]:
        table.add_column(column)
    for row in rows:
        regret_ci = f"{_fmt(row.get('mean_budget_selector_regret'))} [{_fmt(row.get('budget_selector_regret_ci_low'))}, {_fmt(row.get('budget_selector_regret_ci_high'))}]"
        table.add_row(
            str(row.get("task_id", "")),
            str(row.get("method", "")),
            str(row.get("num_runs", "")),
            _fmt(row.get("selection_accuracy"), digits=2),
            _fmt(row.get("regret_free_rate"), digits=2),
            regret_ci,
            str(row.get("chosen_method_counts", "")),
            str(row.get("oracle_budget_method_counts", "")),
        )
    console.print(table)


def _print_budget_select_decisions(console: Console, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    table = Table(title="Budget selector decisions")
    for column in ["task", "selector", "chosen", "score", "oracle", "oracle_score", "regret", "outcome"]:
        table.add_column(column)
    for row in rows[:20]:
        table.add_row(
            str(row.get("task_id", "")),
            str(row.get("method", "")),
            str(row.get("chosen_method", "")),
            _fmt(row.get("selected_target_score")),
            str(row.get("oracle_budget_method", "")),
            _fmt(row.get("oracle_budget_target_score")),
            _fmt(row.get("budget_selector_regret")),
            str(row.get("budget_selection_outcome", "")),
        )
    console.print(table)


def _print_bootstrap_comparisons(console: Console, rows: list[dict[str, Any]]) -> None:
    table = Table(title="Bootstrap IPEO comparisons")
    for column in ["ipeo", "baseline", "tasks", "score_delta_ci", "call_delta", "dollar_delta", "score_outcome"]:
        table.add_column(column)
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("ipeo_method", "")),
            str(row.get("baseline_method", "")),
        ),
    )
    for row in sorted_rows[:30]:
        score_ci = f"{_fmt(row.get('mean_score_delta'))} [{_fmt(row.get('score_delta_ci_low'))}, {_fmt(row.get('score_delta_ci_high'))}]"
        table.add_row(
            str(row.get("ipeo_method", "")),
            str(row.get("baseline_method", "")),
            str(row.get("num_tasks", "")),
            score_ci,
            _fmt(row.get("mean_call_delta"), digits=1),
            _fmt(row.get("mean_dollar_delta"), digits=6),
            str(row.get("score_outcome", "")),
        )
    console.print(table)


def _print_ipeo_deltas(console: Console, rows: list[dict[str, Any]]) -> None:
    table = Table(title="IPEO vs baselines")
    for column in ["task_id", "ipeo", "baseline", "score_delta", "call_delta", "winner"]:
        table.add_column(column)
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("task_id", "")),
            str(row.get("ipeo_method", "")),
            -abs(float(row.get("score_delta", 0.0) or 0.0)),
            str(row.get("baseline_method", "")),
        ),
    )
    for row in sorted_rows[:60]:
        table.add_row(
            str(row.get("task_id", "")),
            str(row.get("ipeo_method", "")),
            str(row.get("baseline_method", "")),
            _fmt(row.get("score_delta")),
            str(int(float(row.get("call_delta", 0) or 0))),
            str(row.get("winner", "")),
        )
    console.print(table)


def _method_score(row: dict[str, Any], method_key: str, score_key: str) -> str:
    method = str(row.get(method_key, ""))
    if not method:
        return ""
    return f"{method} ({_fmt(row.get(score_key))})"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

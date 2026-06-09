"""Analyze multiple completed IPEO benchmark runs."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from ipeo.stats.benchmark_analysis import DEFAULT_BASELINES, DEFAULT_IPEO_METHODS
from ipeo.stats.multi_run_analysis import analyze_many_artifact_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate IPEO benchmark results across runs or seeds")
    parser.add_argument("--artifact_dirs", nargs="*", default=[])
    parser.add_argument("--artifact_glob", action="append", default=[])
    parser.add_argument("--output_dir", required=True)
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
    artifact_dirs = _artifact_dirs(args.artifact_dirs, args.artifact_glob)
    if not artifact_dirs:
        raise ValueError("Provide at least one --artifact_dirs path or --artifact_glob pattern.")
    outputs = analyze_many_artifact_dirs(
        artifact_dirs,
        output_dir=args.output_dir,
        focus_task=args.focus_task,
        ipeo_methods=_expand_values(args.ipeo_methods),
        baseline_methods=_expand_values(args.baseline_methods),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    console = Console(no_color=args.no_color, quiet=args.quiet)
    if not args.quiet:
        _print_input_runs(console, outputs["input_runs"])
        _print_method_summary(console, outputs["method_summary"])
        _print_cost_frontier(console, outputs["cost_frontier"])
        _print_budget_select_summary(console, outputs["budget_select_summary"])
        _print_budget_select_decisions(console, outputs["budget_select_decisions"])
        _print_ipeo_comparisons(console, outputs["ipeo_vs_baselines"])
        suffix = f" for {args.focus_task}" if args.focus_task else ""
        console.print(f"[bold cyan]Wrote multi-run analysis CSVs under {args.output_dir}/stats{suffix}.[/bold cyan]")
    return outputs


def _artifact_dirs(values: list[str], patterns: list[str]) -> list[str]:
    dirs = [str(Path(value)) for value in values]
    for pattern in patterns:
        dirs.extend(sorted(glob.glob(pattern)))
    seen: set[str] = set()
    unique: list[str] = []
    for value in dirs:
        normalized = str(Path(value))
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _expand_values(values: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        expanded.extend(part.strip() for part in value.split(",") if part.strip())
    return expanded


def _print_input_runs(console: Console, rows: list[dict[str, Any]]) -> None:
    table = Table(title="Input runs")
    for column in ["run", "rows", "artifact_dir"]:
        table.add_column(column)
    for row in rows:
        table.add_row(str(row.get("run_label", "")), str(row.get("row_count", "")), str(row.get("artifact_dir", "")))
    console.print(table)


def _print_method_summary(console: Console, rows: list[dict[str, Any]]) -> None:
    table = Table(title="Multi-run method summary")
    for column in ["task", "method", "runs", "score_ci", "calls", "dollars", "best_win"]:
        table.add_column(column)
    for row in rows[:30]:
        score_ci = f"{_fmt(row.get('mean_target_score'))} [{_fmt(row.get('score_ci_low'))}, {_fmt(row.get('score_ci_high'))}]"
        table.add_row(
            str(row.get("task_id", "")),
            str(row.get("method", "")),
            str(row.get("num_runs", "")),
            score_ci,
            _fmt(row.get("mean_total_calls"), digits=1),
            _fmt(row.get("mean_total_dollars"), digits=6),
            _fmt(row.get("best_score_win_rate"), digits=2),
        )
    console.print(table)


def _print_cost_frontier(console: Console, rows: list[dict[str, Any]]) -> None:
    table = Table(title="Multi-run cost frontier")
    for column in ["task", "method", "score", "calls", "dollars", "best_win"]:
        table.add_column(column)
    for row in rows[:30]:
        table.add_row(
            str(row.get("task_id", "")),
            str(row.get("method", "")),
            _fmt(row.get("mean_target_score")),
            _fmt(row.get("mean_total_calls"), digits=1),
            _fmt(row.get("mean_total_dollars"), digits=6),
            _fmt(row.get("best_score_win_rate"), digits=2),
        )
    console.print(table)


def _print_budget_select_summary(console: Console, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    table = Table(title="Multi-run budget selector summary")
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
    table = Table(title="Multi-run budget selector decisions")
    for column in ["run", "task", "selector", "chosen", "score", "oracle", "oracle_score", "regret", "outcome"]:
        table.add_column(column)
    for row in rows[:30]:
        table.add_row(
            str(row.get("run_label", "")),
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


def _print_ipeo_comparisons(console: Console, rows: list[dict[str, Any]]) -> None:
    table = Table(title="Multi-run IPEO vs baselines")
    for column in ["ipeo", "baseline", "pairs", "score_ci", "win/tie/loss", "call_delta", "dollar_delta", "outcome"]:
        table.add_column(column)
    for row in rows[:40]:
        score_ci = f"{_fmt(row.get('mean_score_delta'))} [{_fmt(row.get('score_delta_ci_low'))}, {_fmt(row.get('score_delta_ci_high'))}]"
        win_tie_loss = "/".join(
            [
                _fmt(row.get("score_win_rate"), digits=2),
                _fmt(row.get("score_tie_rate"), digits=2),
                _fmt(row.get("score_loss_rate"), digits=2),
            ]
        )
        table.add_row(
            str(row.get("ipeo_method", "")),
            str(row.get("baseline_method", "")),
            str(row.get("num_pairs", "")),
            score_ci,
            win_tie_loss,
            _fmt(row.get("mean_call_delta"), digits=1),
            _fmt(row.get("mean_dollar_delta"), digits=6),
            str(row.get("score_outcome", "")),
        )
    console.print(table)


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

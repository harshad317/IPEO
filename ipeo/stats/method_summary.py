"""End-of-run method summary statistics."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from ipeo.core.io import read_jsonl
from ipeo.core.schemas import EvalResult, MethodSelection


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _stddev(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.pstdev(values)


def _score_stats(results: list[EvalResult]) -> tuple[float | None, float | None]:
    scores = [float(row.score) for row in results]
    return _mean(scores), _stddev(scores)


def _cost_rows(
    *,
    cost_rows: list[dict[str, Any]],
    run_id: str,
    task_id: str,
    prompt_id: str,
    phase: str,
    model_ids: set[str],
    method: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in cost_rows:
        if row.get("run_id") != run_id:
            continue
        if row.get("task_id") != task_id:
            continue
        if row.get("prompt_id") != prompt_id:
            continue
        if row.get("phase") != phase:
            continue
        if row.get("model_id") not in model_ids:
            continue
        if method is not None and row.get("method") != method:
            continue
        rows.append(row)
    return rows


def _avg_tokens(rows: list[dict[str, Any]]) -> float | None:
    values = [float(row.get("total_tokens", 0.0)) for row in rows]
    return _mean(values)


def _api_calls(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if not row.get("cache_hit", False))


def build_method_summary_rows(
    *,
    run_id: str,
    task_id: str,
    target_model: str,
    source_models: list[str],
    selections: list[MethodSelection],
    transfer_rows: list[dict[str, object]],
    pool_results: list[EvalResult],
    final_results: list[EvalResult],
    cost_log_path: str | Path,
) -> list[dict[str, Any]]:
    cost_rows = read_jsonl(cost_log_path)
    transfer_by_method = {str(row["method"]): row for row in transfer_rows}
    source_model_set = set(source_models)
    target_model_set = {target_model}
    rows: list[dict[str, Any]] = []

    for selection in selections:
        method = selection.method
        prompt_id = selection.prompt_id
        train_results = [
            row
            for row in pool_results
            if row.prompt_id == prompt_id and row.model_id in source_model_set and row.split == "val"
        ]
        val_results = [
            row
            for row in pool_results
            if row.prompt_id == prompt_id and row.model_id == target_model and row.split == "val"
        ]
        test_results = [
            row
            for row in final_results
            if row.prompt_id == prompt_id and row.model_id == target_model and row.split == "test"
        ]
        split_specs = [
            (
                "train",
                train_results,
                _cost_rows(
                    cost_rows=cost_rows,
                    run_id=run_id,
                    task_id=task_id,
                    prompt_id=prompt_id,
                    phase="evaluation",
                    model_ids=source_model_set,
                    method="fixed_pool",
                ),
            ),
            (
                "val",
                val_results,
                _cost_rows(
                    cost_rows=cost_rows,
                    run_id=run_id,
                    task_id=task_id,
                    prompt_id=prompt_id,
                    phase="evaluation",
                    model_ids=target_model_set,
                    method="fixed_pool",
                ),
            ),
            (
                "test",
                test_results,
                _cost_rows(
                    cost_rows=cost_rows,
                    run_id=run_id,
                    task_id=task_id,
                    prompt_id=prompt_id,
                    phase="final_test",
                    model_ids=target_model_set,
                    method="selected_methods",
                ),
            ),
        ]
        for split, split_results, split_cost_rows in split_specs:
            score, stddev = _score_stats(split_results)
            rows.append(
                {
                    "task_id": task_id,
                    "method": method,
                    "split": split,
                    "score": score,
                    "stddev": stddev,
                    "avg_tokens": _avg_tokens(split_cost_rows),
                    "api_calls": _api_calls(split_cost_rows),
                }
            )

        transfer = transfer_by_method.get(method, {})
        rows.append(
            {
                "task_id": task_id,
                "method": method,
                "split": "optimization",
                "score": None,
                "stddev": None,
                "avg_tokens": None,
                "api_calls": int(transfer.get("source_calls", 0) or 0) + int(transfer.get("target_calls", 0) or 0),
            }
        )
    return rows


def aggregate_method_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), str(row["split"])), []).append(row)

    out: list[dict[str, Any]] = []
    for (method, split), group_rows in sorted(grouped.items()):
        scores = [float(row["score"]) for row in group_rows if row.get("score") is not None]
        stddevs = [float(row["stddev"]) for row in group_rows if row.get("stddev") is not None]
        avg_tokens = [float(row["avg_tokens"]) for row in group_rows if row.get("avg_tokens") is not None]
        out.append(
            {
                "task_id": "ALL",
                "method": method,
                "split": split,
                "score": _mean(scores),
                "stddev": _mean(stddevs),
                "avg_tokens": _mean(avg_tokens),
                "api_calls": sum(int(row.get("api_calls", 0) or 0) for row in group_rows),
            }
        )
    split_order = {"train": 0, "val": 1, "test": 2, "optimization": 3}
    return sorted(out, key=lambda row: (str(row["method"]), split_order.get(str(row["split"]), 99)))

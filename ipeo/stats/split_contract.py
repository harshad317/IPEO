"""Benchmark split and data-access contract helpers."""

from __future__ import annotations

from typing import Any

from ipeo.core.schemas import MethodSelection

DSPY_METHODS = {"gepa", "miprov2"}
TARGET_VALIDATION_METHODS = {"target_only_bo_fixed_pool"}
IPEO_METHOD_PREFIXES = ("ipeo_",)
SOURCE_VALIDATION_METHODS = {
    "source_average",
    "pooled_source",
    "worst_source_robust",
    "asha_fixed_pool",
    "promptbridge_emulation",
}
NO_OPTIMIZATION_METHODS = {"original", "random_search"}


def is_ipeo_method(method: str) -> bool:
    return method.startswith(IPEO_METHOD_PREFIXES)


def is_best_source_method(method: str) -> bool:
    return method.startswith("best_source_transfer:")


def benchmark_track(method: str) -> str:
    if method in DSPY_METHODS or method in TARGET_VALIDATION_METHODS:
        return "target_optimization"
    if is_ipeo_method(method):
        return "zero_target_transfer"
    if method in SOURCE_VALIDATION_METHODS or is_best_source_method(method):
        return "source_transfer"
    return "fixed_prompt"


def summary_train_models(method: str, source_models: list[str], target_model: str) -> set[str]:
    if method in DSPY_METHODS:
        return {target_model}
    if method in TARGET_VALIDATION_METHODS or method in NO_OPTIMIZATION_METHODS:
        return set()
    return set(source_models)


def summary_validation_models(method: str, source_models: list[str], target_model: str) -> set[str]:
    if method in DSPY_METHODS or method in TARGET_VALIDATION_METHODS:
        return {target_model}
    if method in NO_OPTIMIZATION_METHODS:
        return set()
    return set(source_models)


def access_row(
    *,
    task_id: str,
    selection: MethodSelection,
    source_train_calls: int,
    source_validation_calls: int,
    target_validation_calls: int,
    target_optimization_calls: int,
    final_target_test_calls: int,
) -> dict[str, Any]:
    method = selection.method
    track = benchmark_track(method)
    source_train = 0
    source_val = 0
    target_val = 0
    target_opt = 0
    train_access = "none"
    validation_access = "none"
    selection_access = "none"

    if is_ipeo_method(method):
        source_train = source_train_calls
        train_access = "source_train"
        selection_access = "source_train"
    elif method in SOURCE_VALIDATION_METHODS:
        source_val = source_validation_calls
        validation_access = "source_validation"
        selection_access = "source_validation"
    elif is_best_source_method(method):
        source_val = max(0, source_validation_calls // max(1, len(selection.source_models)))
        validation_access = "source_validation_single_source"
        selection_access = "source_validation_single_source"
    elif method in TARGET_VALIDATION_METHODS:
        target_val = target_validation_calls
        validation_access = "target_validation"
        selection_access = "target_validation"
    elif method in DSPY_METHODS:
        target_opt = target_optimization_calls
        train_access = "target_train"
        validation_access = "target_validation"
        selection_access = "target_train,target_validation"

    return {
        "task_id": task_id,
        "method": method,
        "benchmark_track": track,
        "train_access": train_access,
        "validation_access": validation_access,
        "selection_access": selection_access,
        "test_access": "target_test_final_only",
        "uses_target_train": train_access == "target_train",
        "uses_target_validation": "target_validation" in validation_access,
        "uses_target_test_for_selection": False,
        "source_train_calls": source_train,
        "source_validation_calls": source_val,
        "target_validation_calls": target_val,
        "target_optimization_calls": target_opt,
        "final_target_test_calls": final_target_test_calls,
    }


def access_rows_by_method(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["method"]): row for row in rows}

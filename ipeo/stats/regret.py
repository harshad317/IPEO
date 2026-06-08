"""Transfer-regret and uncertainty reporting."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from ipeo.core.schemas import EvalResult, MethodSelection, PromptCandidate


def mean_score(results: list[EvalResult], prompt_id: str, model_id: str, split: str) -> float:
    scores = [row.score for row in results if row.prompt_id == prompt_id and row.model_id == model_id and row.split == split]
    return sum(scores) / len(scores) if scores else 0.0


def fixed_pool_oracle_score(pool_results: list[EvalResult], pool: list[PromptCandidate], target_model: str, split: str) -> float:
    if not pool:
        return 0.0
    return max(mean_score(pool_results, prompt.prompt_id, target_model, split) for prompt in pool)


def paired_bootstrap_ci(scores: list[float], n_bootstrap: int = 500, seed: int = 0) -> tuple[float, float]:
    if not scores:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = [scores[rng.randrange(len(scores))] for _ in scores]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return (lo, hi)


def permutation_p_value(differences: list[float], n_permutations: int = 500, seed: int = 0) -> float:
    if not differences:
        return 1.0
    observed = abs(sum(differences) / len(differences))
    rng = random.Random(seed)
    count = 0
    for _ in range(n_permutations):
        flipped = [diff if rng.random() < 0.5 else -diff for diff in differences]
        stat = abs(sum(flipped) / len(flipped))
        if stat >= observed:
            count += 1
    return (count + 1) / (n_permutations + 1)


def method_example_scores(results: list[EvalResult], prompt_id: str, model_id: str, split: str) -> dict[str, float]:
    return {
        row.example_id: row.score
        for row in results
        if row.prompt_id == prompt_id and row.model_id == model_id and row.split == split
    }


def build_transfer_rows(
    *,
    task_id: str,
    fold_id: str,
    target_model: str,
    source_models: list[str],
    selections: list[MethodSelection],
    final_results: list[EvalResult],
    pool_results: list[EvalResult],
    pool: list[PromptCandidate],
    source_average_prompt_id: str,
    source_calls_by_method: dict[str, int] | None = None,
    target_calls_by_method: dict[str, int] | None = None,
    dollars_by_method: dict[str, float] | None = None,
    method_access_by_method: dict[str, dict[str, Any]] | None = None,
    split: str = "test",
) -> list[dict[str, Any]]:
    source_calls_by_method = source_calls_by_method or {}
    target_calls_by_method = target_calls_by_method or {}
    dollars_by_method = dollars_by_method or {}
    method_access_by_method = method_access_by_method or {}
    oracle = fixed_pool_oracle_score(pool_results, pool, target_model, split)
    source_avg_score = mean_score(final_results, source_average_prompt_id, target_model, split)
    source_avg_regret = oracle - source_avg_score
    baseline_examples = method_example_scores(final_results, source_average_prompt_id, target_model, split)
    rows = []
    for selection in selections:
        score = mean_score(final_results, selection.prompt_id, target_model, split)
        regret = oracle - score
        relative_reduction = 0.0
        if abs(source_avg_regret) > 1e-12:
            relative_reduction = (source_avg_regret - regret) / abs(source_avg_regret)
        ex_scores = method_example_scores(final_results, selection.prompt_id, target_model, split)
        ci = paired_bootstrap_ci(list(ex_scores.values()))
        common_ids = sorted(set(ex_scores) & set(baseline_examples))
        p_value = permutation_p_value([ex_scores[ex_id] - baseline_examples[ex_id] for ex_id in common_ids])
        row = {
            "task_id": task_id,
            "fold_id": fold_id,
            "target_model": target_model,
            "source_models": ",".join(source_models),
            "method": selection.method,
            "prompt_id": selection.prompt_id,
            "target_score": score,
            "fixed_pool_oracle_score": oracle,
            "fixed_pool_regret": regret,
            "regret_reduction_vs_source_average": relative_reduction,
            "source_calls": source_calls_by_method.get(selection.method, selection.source_calls),
            "target_calls": target_calls_by_method.get(selection.method, selection.target_calls),
            "total_dollars": dollars_by_method.get(selection.method, selection.total_dollars),
            "deployment_cost": len(selection.prompt_text.split()) * 0.0001 / 1000,
            "p_value": p_value,
            "confidence_interval": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
        }
        row.update(method_access_by_method.get(selection.method, {}))
        rows.append(row)
    return rows


def aggregate_method_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(float(row["target_score"]))
    return {method: sum(values) / len(values) for method, values in grouped.items()}

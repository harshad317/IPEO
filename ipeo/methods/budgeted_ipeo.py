"""Budgeted source-evaluation subsets for IPEO."""

from __future__ import annotations

import random
from dataclasses import dataclass

from ipeo.core.schemas import EvalResult, Example, InvariantEditStats, MethodSelection, PromptCandidate
from ipeo.models.base import count_tokens


@dataclass(frozen=True)
class BudgetedSourcePlan:
    pool: list[PromptCandidate]
    examples: list[Example]
    requested_budget: int
    planned_source_calls: int
    prompt_ids: list[str]
    example_ids: list[str]
    source_model_ids: list[str]


@dataclass(frozen=True)
class BudgetedSourceSubset:
    pool: list[PromptCandidate]
    eval_results: list[EvalResult]
    source_calls: int
    requested_budget: int
    prompt_ids: list[str]
    example_ids: list[str]
    source_model_ids: list[str]


@dataclass(frozen=True)
class BudgetedPromptCandidate:
    method: str
    requested_budget: int
    source_calls: int
    prompt: PromptCandidate
    selection: MethodSelection
    invariant_table: list[InvariantEditStats]


@dataclass(frozen=True)
class BudgetedPromptChoice:
    selection: MethodSelection
    prompt: PromptCandidate
    chosen_method: str
    requested_budget: int
    source_calls: int
    source_score: float
    score_rows: list[dict[str, float | int | str]]


def plan_budgeted_source_subset(
    *,
    pool: list[PromptCandidate],
    train_examples: list[Example],
    source_model_ids: list[str],
    budget: int,
    seed: int,
) -> BudgetedSourcePlan:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not pool:
        raise ValueError("pool must be non-empty")
    if not train_examples:
        raise ValueError("train_examples must be non-empty")
    if not source_model_ids:
        raise ValueError("source_model_ids must be non-empty")

    full_calls = len(pool) * len(train_examples) * len(source_model_ids)
    effective_budget = min(int(budget), full_calls)
    selected_source_model_ids = list(source_model_ids)
    if effective_budget < len(selected_source_model_ids):
        selected_source_model_ids = selected_source_model_ids[:effective_budget]
        effective_budget = len(selected_source_model_ids)

    prompt_count = len(pool)
    source_count = max(1, len(selected_source_model_ids))
    example_count = effective_budget // (source_count * prompt_count)
    if example_count < 1:
        example_count = 1
        prompt_count = max(1, effective_budget // source_count)
    example_count = min(example_count, len(train_examples))
    prompt_count = min(prompt_count, len(pool))

    shuffled_pool = list(pool)
    rng = random.Random(seed)
    rng.shuffle(shuffled_pool)
    selected_prompt_ids = {prompt.prompt_id for prompt in shuffled_pool[:prompt_count]}
    selected_pool = [prompt for prompt in pool if prompt.prompt_id in selected_prompt_ids]

    shuffled_examples = list(train_examples)
    rng.shuffle(shuffled_examples)
    selected_examples = sorted(shuffled_examples[:example_count], key=lambda example: example.example_id)
    prompt_ids = [prompt.prompt_id for prompt in selected_pool]
    example_ids = [example.example_id for example in selected_examples]
    return BudgetedSourcePlan(
        pool=selected_pool,
        examples=selected_examples,
        requested_budget=budget,
        planned_source_calls=len(selected_pool) * len(selected_examples) * len(selected_source_model_ids),
        prompt_ids=prompt_ids,
        example_ids=example_ids,
        source_model_ids=selected_source_model_ids,
    )


def build_budgeted_source_subset(
    *,
    pool: list[PromptCandidate],
    train_examples: list[Example],
    source_model_ids: list[str],
    pool_train_results: list[EvalResult],
    budget: int,
    seed: int,
) -> BudgetedSourceSubset:
    plan = plan_budgeted_source_subset(
        pool=pool,
        train_examples=train_examples,
        source_model_ids=source_model_ids,
        budget=budget,
        seed=seed,
    )
    prompt_id_set = set(plan.prompt_ids)
    example_id_set = set(plan.example_ids)
    model_id_set = set(plan.source_model_ids)
    rows = [
        row
        for row in pool_train_results
        if row.prompt_id in prompt_id_set
        and row.example_id in example_id_set
        and row.model_id in model_id_set
        and row.split == "opt"
    ]
    return BudgetedSourceSubset(
        pool=plan.pool,
        eval_results=rows,
        source_calls=len(rows),
        requested_budget=budget,
        prompt_ids=plan.prompt_ids,
        example_ids=plan.example_ids,
        source_model_ids=plan.source_model_ids,
    )


def budget_candidate_source_score(candidate: BudgetedPromptCandidate) -> dict[str, float | int | str]:
    row_by_edit = {row.edit_id: row for row in candidate.invariant_table}
    selected_rows = [row_by_edit[edit_id] for edit_id in candidate.selection.selected_edit_ids if edit_id in row_by_edit]
    edit_count = len(selected_rows)
    if selected_rows:
        sum_ipeo = sum(row.ipeo_score for row in selected_rows)
        mean_lcb = sum(row.lcb_mean_effect for row in selected_rows) / edit_count
        mean_sign = sum(row.sign_agreement for row in selected_rows) / edit_count
        mean_rank = sum(row.rank_stability for row in selected_rows) / edit_count
        mean_variance = sum(row.effect_variance for row in selected_rows) / edit_count
    else:
        sum_ipeo = 0.0
        mean_lcb = 0.0
        mean_sign = 0.0
        mean_rank = 0.0
        mean_variance = 0.0
    prompt_tokens = count_tokens(candidate.prompt.text)
    source_call_penalty = (candidate.source_calls / 1000.0) ** 0.5
    sample_adequacy_bonus = 0.25 * min(candidate.source_calls / 450.0, 1.0)
    small_sample_penalty = 0.20 if candidate.source_calls < 300 else 0.0
    large_budget_penalty = 0.15 * max(0.0, (candidate.source_calls - 600) / 400.0)
    prompt_length_penalty = prompt_tokens / 1000.0
    score = (
        sum_ipeo
        + 0.5 * mean_lcb
        + 0.25 * mean_sign
        + 0.10 * mean_rank
        - 0.25 * mean_variance
        + sample_adequacy_bonus
        - small_sample_penalty
        - large_budget_penalty
        - 0.05 * source_call_penalty
        - 0.02 * prompt_length_penalty
    )
    return {
        "method": candidate.method,
        "requested_budget": candidate.requested_budget,
        "source_calls": candidate.source_calls,
        "source_score": float(score),
        "sum_ipeo_score": float(sum_ipeo),
        "mean_lcb": float(mean_lcb),
        "mean_sign_agreement": float(mean_sign),
        "mean_rank_stability": float(mean_rank),
        "mean_effect_variance": float(mean_variance),
        "sample_adequacy_bonus": float(sample_adequacy_bonus),
        "small_sample_penalty": float(small_sample_penalty),
        "large_budget_penalty": float(large_budget_penalty),
        "prompt_tokens": prompt_tokens,
        "edit_count": edit_count,
    }


def select_budgeted_prompt(
    *,
    candidates: list[BudgetedPromptCandidate],
    task_id: str,
    fold_id: str,
    target_model: str,
    method_name: str = "ipeo_budget_select",
) -> BudgetedPromptChoice:
    if not candidates:
        raise ValueError("candidates must be non-empty")
    score_rows = [budget_candidate_source_score(candidate) for candidate in candidates]
    candidate_by_method = {candidate.method: candidate for candidate in candidates}
    best_row = max(
        score_rows,
        key=lambda row: (
            float(row["source_score"]),
            -int(row["source_calls"]),
            -int(row["prompt_tokens"]),
            str(row["method"]),
        ),
    )
    chosen = candidate_by_method[str(best_row["method"])]
    selection = MethodSelection(
        method=method_name,
        task_id=task_id,
        fold_id=fold_id,
        target_model=target_model,
        source_models=chosen.selection.source_models,
        prompt_id=chosen.prompt.prompt_id,
        prompt_text=chosen.prompt.text,
        selected_edit_ids=chosen.prompt.edit_ids,
    )
    return BudgetedPromptChoice(
        selection=selection,
        prompt=chosen.prompt,
        chosen_method=chosen.method,
        requested_budget=chosen.requested_budget,
        source_calls=chosen.source_calls,
        source_score=float(best_row["source_score"]),
        score_rows=score_rows,
    )


def select_budgeted_prompt_by_source_validation(
    *,
    candidates: list[BudgetedPromptCandidate],
    validation_results: list[EvalResult],
    task_id: str,
    fold_id: str,
    target_model: str,
    method_name: str = "ipeo_budget_select_source_val",
) -> BudgetedPromptChoice:
    if not candidates:
        raise ValueError("candidates must be non-empty")
    validation_score_by_prompt = _mean_validation_scores(validation_results)
    invariant_rows = {row["method"]: row for row in [budget_candidate_source_score(candidate) for candidate in candidates]}
    score_rows: list[dict[str, float | int | str]] = []
    for candidate in candidates:
        validation_score = validation_score_by_prompt.get(candidate.prompt.prompt_id, 0.0)
        invariant_row = invariant_rows[candidate.method]
        score_rows.append(
            {
                **invariant_row,
                "source_score": float(validation_score),
                "validation_score": float(validation_score),
                "invariant_source_score": float(invariant_row["source_score"]),
                "validation_calls": sum(1 for row in validation_results if row.prompt_id == candidate.prompt.prompt_id),
            }
        )
    candidate_by_method = {candidate.method: candidate for candidate in candidates}
    best_row = max(
        score_rows,
        key=lambda row: (
            float(row["validation_score"]),
            float(row["invariant_source_score"]),
            -int(row["source_calls"]),
            -int(row["prompt_tokens"]),
            str(row["method"]),
        ),
    )
    chosen = candidate_by_method[str(best_row["method"])]
    selection = MethodSelection(
        method=method_name,
        task_id=task_id,
        fold_id=fold_id,
        target_model=target_model,
        source_models=chosen.selection.source_models,
        prompt_id=chosen.prompt.prompt_id,
        prompt_text=chosen.prompt.text,
        selected_edit_ids=chosen.prompt.edit_ids,
    )
    return BudgetedPromptChoice(
        selection=selection,
        prompt=chosen.prompt,
        chosen_method=chosen.method,
        requested_budget=chosen.requested_budget,
        source_calls=chosen.source_calls,
        source_score=float(best_row["source_score"]),
        score_rows=score_rows,
    )


def _mean_validation_scores(results: list[EvalResult]) -> dict[str, float]:
    scores_by_prompt: dict[str, list[float]] = {}
    for row in results:
        if row.split != "val":
            continue
        scores_by_prompt.setdefault(row.prompt_id, []).append(float(row.score))
    return {
        prompt_id: sum(scores) / len(scores)
        for prompt_id, scores in scores_by_prompt.items()
        if scores
    }

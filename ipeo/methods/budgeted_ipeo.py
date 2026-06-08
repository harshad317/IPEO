"""Budgeted source-evaluation subsets for IPEO."""

from __future__ import annotations

import random
from dataclasses import dataclass

from ipeo.core.schemas import EvalResult, Example, PromptCandidate


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

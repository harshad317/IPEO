"""Fixed-pool baseline selectors."""

from __future__ import annotations

import random
from dataclasses import replace

from ipeo.core.schemas import EvalResult, MethodSelection, PromptCandidate


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _scores_by_prompt_model(results: list[EvalResult], split: str) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in results:
        if row.split == split:
            grouped.setdefault((row.prompt_id, row.model_id), []).append(row.score)
    return {key: _mean(values) for key, values in grouped.items()}


def _select_prompt(pool: list[PromptCandidate], prompt_id: str) -> PromptCandidate:
    return next(prompt for prompt in pool if prompt.prompt_id == prompt_id)


def original_prompt(
    *,
    task_id: str,
    fold_id: str,
    target_model: str,
    source_models: list[str],
    pool: list[PromptCandidate],
) -> MethodSelection:
    prompt = pool[0]
    return MethodSelection("original", task_id, fold_id, target_model, source_models, prompt.prompt_id, prompt.text, prompt.edit_ids)


def source_average_selection(
    *,
    task_id: str,
    fold_id: str,
    target_model: str,
    source_models: list[str],
    pool: list[PromptCandidate],
    eval_results: list[EvalResult],
    split: str = "val",
) -> MethodSelection:
    scores = _scores_by_prompt_model(eval_results, split)
    best_prompt = max(
        pool,
        key=lambda prompt: _mean([scores.get((prompt.prompt_id, model_id), 0.0) for model_id in source_models]),
    )
    return MethodSelection(
        "source_average",
        task_id,
        fold_id,
        target_model,
        source_models,
        best_prompt.prompt_id,
        best_prompt.text,
        best_prompt.edit_ids,
    )


def pooled_source_selection(
    *,
    task_id: str,
    fold_id: str,
    target_model: str,
    source_models: list[str],
    pool: list[PromptCandidate],
    eval_results: list[EvalResult],
    split: str = "val",
) -> MethodSelection:
    grouped: dict[str, list[float]] = {prompt.prompt_id: [] for prompt in pool}
    for row in eval_results:
        if row.split == split and row.model_id in source_models:
            grouped.setdefault(row.prompt_id, []).append(row.score)
    best_prompt = max(pool, key=lambda prompt: _mean(grouped.get(prompt.prompt_id, [])))
    return MethodSelection("pooled_source", task_id, fold_id, target_model, source_models, best_prompt.prompt_id, best_prompt.text, best_prompt.edit_ids)


def robust_source_selection(
    *,
    task_id: str,
    fold_id: str,
    target_model: str,
    source_models: list[str],
    pool: list[PromptCandidate],
    eval_results: list[EvalResult],
    split: str = "val",
) -> MethodSelection:
    scores = _scores_by_prompt_model(eval_results, split)
    best_prompt = max(
        pool,
        key=lambda prompt: min(scores.get((prompt.prompt_id, model_id), 0.0) for model_id in source_models),
    )
    return MethodSelection("worst_source_robust", task_id, fold_id, target_model, source_models, best_prompt.prompt_id, best_prompt.text, best_prompt.edit_ids)


def best_source_transfer(
    *,
    task_id: str,
    fold_id: str,
    target_model: str,
    source_models: list[str],
    pool: list[PromptCandidate],
    eval_results: list[EvalResult],
    split: str = "val",
) -> list[MethodSelection]:
    scores = _scores_by_prompt_model(eval_results, split)
    selections: list[MethodSelection] = []
    for source_model in source_models:
        best_prompt = max(pool, key=lambda prompt: scores.get((prompt.prompt_id, source_model), 0.0))
        selections.append(
            MethodSelection(
                f"best_source_transfer:{source_model}",
                task_id,
                fold_id,
                target_model,
                source_models,
                best_prompt.prompt_id,
                best_prompt.text,
                best_prompt.edit_ids,
            )
        )
    return selections


def random_search_selection(
    *,
    task_id: str,
    fold_id: str,
    target_model: str,
    source_models: list[str],
    pool: list[PromptCandidate],
    seed: int = 0,
    budget: int = 8,
) -> MethodSelection:
    rng = random.Random(seed)
    sampled = rng.sample(pool, k=min(budget, len(pool)))
    prompt = sampled[-1]
    return MethodSelection("random_search", task_id, fold_id, target_model, source_models, prompt.prompt_id, prompt.text, prompt.edit_ids)


def target_only_bo_selection(
    *,
    task_id: str,
    fold_id: str,
    target_model: str,
    source_models: list[str],
    pool: list[PromptCandidate],
    eval_results: list[EvalResult],
    split: str = "val",
    budget: int = 8,
) -> MethodSelection:
    scores = _scores_by_prompt_model(eval_results, split)
    candidates = pool[: min(budget, len(pool))]
    best_prompt = max(candidates, key=lambda prompt: scores.get((prompt.prompt_id, target_model), 0.0))
    return MethodSelection(
        "target_only_bo_fixed_pool",
        task_id,
        fold_id,
        target_model,
        source_models,
        best_prompt.prompt_id,
        best_prompt.text,
        best_prompt.edit_ids,
        target_calls=budget,
    )


def asha_selection(
    *,
    task_id: str,
    fold_id: str,
    target_model: str,
    source_models: list[str],
    pool: list[PromptCandidate],
    eval_results: list[EvalResult],
    split: str = "val",
) -> MethodSelection:
    scores = _scores_by_prompt_model(eval_results, split)
    first_round = sorted(
        pool,
        key=lambda prompt: _mean([scores.get((prompt.prompt_id, model_id), 0.0) for model_id in source_models]),
        reverse=True,
    )[: max(1, len(pool) // 2)]
    second_round = sorted(
        first_round,
        key=lambda prompt: min(scores.get((prompt.prompt_id, model_id), 0.0) for model_id in source_models),
        reverse=True,
    )
    best_prompt = second_round[0]
    return MethodSelection("asha_fixed_pool", task_id, fold_id, target_model, source_models, best_prompt.prompt_id, best_prompt.text, best_prompt.edit_ids)


def promptbridge_emulation(
    *,
    task_id: str,
    fold_id: str,
    target_model: str,
    source_models: list[str],
    pool: list[PromptCandidate],
    eval_results: list[EvalResult],
    split: str = "val",
) -> MethodSelection:
    source_avg = source_average_selection(
        task_id=task_id,
        fold_id=fold_id,
        target_model=target_model,
        source_models=source_models,
        pool=pool,
        eval_results=eval_results,
        split=split,
    )
    prompt = _select_prompt(pool, source_avg.prompt_id)
    rewritten = replace(
        prompt,
        prompt_id=f"{prompt.prompt_id}-promptbridge",
        text=prompt.text + "\nTarget rewrite hint: preserve intent but adapt wording for the held-out model family.",
        source_generator="promptbridge_emulation",
        parent_prompt_ids=[prompt.prompt_id],
    )
    return MethodSelection(
        "promptbridge_emulation",
        task_id,
        fold_id,
        target_model,
        source_models,
        rewritten.prompt_id,
        rewritten.text,
        rewritten.edit_ids,
    )

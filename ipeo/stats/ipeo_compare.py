"""Posthoc IPEO selector comparison reports."""

from __future__ import annotations

from typing import Any

from ipeo.core.schemas import EvalResult, MethodSelection, PromptCandidate, InvariantEditStats
from ipeo.methods.ipeo_zero import prompt_invariant_score
from ipeo.stats.regret import mean_score


def build_composed_vs_existing_row(
    *,
    task_id: str,
    fold_id: str,
    target_model: str,
    composed_method: str,
    composed_prompt: PromptCandidate,
    existing_selection: MethodSelection,
    existing_prompt: PromptCandidate,
    comparison_selection: MethodSelection | None,
    invariant_table: list[InvariantEditStats],
    eval_results: list[EvalResult],
    split: str = "test",
) -> dict[str, Any]:
    composed_target_score = mean_score(eval_results, composed_prompt.prompt_id, target_model, split)
    existing_target_score = mean_score(eval_results, existing_prompt.prompt_id, target_model, split)
    if composed_target_score > existing_target_score:
        target_winner = "composed"
    elif existing_target_score > composed_target_score:
        target_winner = "existing"
    else:
        target_winner = "tie"
    invariant_chosen = None
    if comparison_selection is not None:
        invariant_chosen = (
            "existing"
            if comparison_selection.prompt_id == existing_prompt.prompt_id
            else "composed"
            if comparison_selection.prompt_id == composed_prompt.prompt_id
            else "other"
        )
    return {
        "task_id": task_id,
        "fold_id": fold_id,
        "target_model": target_model,
        "composed_method": composed_method,
        "existing_method": existing_selection.method,
        "composed_prompt_id": composed_prompt.prompt_id,
        "existing_prompt_id": existing_prompt.prompt_id,
        "comparison_method": comparison_selection.method if comparison_selection else None,
        "comparison_prompt_id": comparison_selection.prompt_id if comparison_selection else None,
        "invariant_chosen": invariant_chosen,
        "target_winner": target_winner,
        "composed_invariant_score": prompt_invariant_score(composed_prompt, invariant_table),
        "existing_invariant_score": prompt_invariant_score(existing_prompt, invariant_table),
        "composed_target_score": composed_target_score,
        "existing_target_score": existing_target_score,
        "target_score_delta_composed_minus_existing": composed_target_score - existing_target_score,
        "composed_edit_ids": ",".join(composed_prompt.edit_ids),
        "existing_edit_ids": ",".join(existing_prompt.edit_ids),
    }

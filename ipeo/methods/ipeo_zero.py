"""Zero-target IPEO prompt selection."""

from __future__ import annotations

from ipeo.core.ids import stable_hash
from ipeo.core.schemas import AtomicEdit, InvariantEditStats, MethodSelection, PromptCandidate
from ipeo.models.base import count_tokens
from ipeo.prompts.composer import compose_text, has_conflict


def invariant_score_by_edit_id(invariant_table: list[InvariantEditStats]) -> dict[str, float]:
    return {row.edit_id: row.ipeo_score for row in invariant_table}


def prompt_invariant_score(prompt: PromptCandidate, invariant_table: list[InvariantEditStats]) -> float:
    edit_scores = invariant_score_by_edit_id(invariant_table)
    return sum(edit_scores.get(edit_id, 0.0) for edit_id in prompt.edit_ids)


def select_zero_target_prompt(
    *,
    task_id: str,
    seed_prompt: PromptCandidate,
    edits: list[AtomicEdit],
    invariant_table: list[InvariantEditStats],
    fold_id: str,
    target_model: str,
    source_models: list[str],
    max_edits_per_prompt: int = 5,
    max_prompt_tokens: int | None = None,
    min_sign_agreement: float = 1.0,
    min_lcb: float = 0.0,
    exclude_generic: bool = False,
    exclude_edit_types: set[str] | None = None,
    method_name: str = "ipeo_zero",
) -> tuple[PromptCandidate, MethodSelection]:
    edit_by_id = {edit.edit_id: edit for edit in edits}
    exclude_edit_types = exclude_edit_types or set()
    selected: list[AtomicEdit] = []
    token_budget = max_prompt_tokens or int(count_tokens(seed_prompt.text) * 1.5) + 32
    for row in invariant_table:
        edit = edit_by_id[row.edit_id]
        if row.is_placebo:
            continue
        if row.lcb_mean_effect < min_lcb:
            continue
        if row.sign_agreement < min_sign_agreement:
            continue
        if row.is_generic and exclude_generic:
            continue
        if row.edit_type in exclude_edit_types:
            continue
        if has_conflict(edit, selected):
            continue
        proposed = selected + [edit]
        if count_tokens(compose_text(seed_prompt.text, proposed)) > token_budget:
            continue
        selected = proposed
        if len(selected) >= max_edits_per_prompt:
            break

    text = compose_text(seed_prompt.text, selected)
    prompt_id = stable_hash(
        {"method": method_name, "task": task_id, "fold": fold_id, "text": text},
        prefix="p-ipeo-",
    )
    edit_ids = [edit.edit_id for edit in selected]
    vector = [1 if edit.edit_id in edit_ids else 0 for edit in edits]
    prompt = PromptCandidate(
        prompt_id=prompt_id,
        task_id=task_id,
        text=text,
        edit_ids=edit_ids,
        edit_vector=vector,
        source_generator="ipeo_composed",
        parent_prompt_ids=[seed_prompt.prompt_id],
        prompt_tokens_by_model={"mock": count_tokens(text)},
        estimated_deployment_cost={"mock": count_tokens(text) * 0.0001 / 1000},
        coherence_repaired=False,
        frozen_pool_version="mvp-v1",
    )
    selection = MethodSelection(
        method=method_name,
        task_id=task_id,
        fold_id=fold_id,
        target_model=target_model,
        source_models=source_models,
        prompt_id=prompt.prompt_id,
        prompt_text=prompt.text,
        selected_edit_ids=edit_ids,
    )
    return prompt, selection


def select_existing_prompt_by_invariant_score(
    *,
    task_id: str,
    pool: list[PromptCandidate],
    invariant_table: list[InvariantEditStats],
    fold_id: str,
    target_model: str,
    source_models: list[str],
    method_name: str = "ipeo_select_existing",
) -> MethodSelection:
    best_prompt = max(
        pool,
        key=lambda prompt: (
            prompt_invariant_score(prompt, invariant_table),
            -count_tokens(prompt.text),
            prompt.prompt_id,
        ),
    )
    return MethodSelection(
        method=method_name,
        task_id=task_id,
        fold_id=fold_id,
        target_model=target_model,
        source_models=source_models,
        prompt_id=best_prompt.prompt_id,
        prompt_text=best_prompt.text,
        selected_edit_ids=best_prompt.edit_ids,
    )


def select_composed_vs_existing_prompt(
    *,
    task_id: str,
    composed_prompt: PromptCandidate,
    existing_selection: MethodSelection,
    pool: list[PromptCandidate],
    invariant_table: list[InvariantEditStats],
    fold_id: str,
    target_model: str,
    source_models: list[str],
    method_name: str = "ipeo_composed_vs_existing",
) -> MethodSelection:
    prompt_by_id = {prompt.prompt_id: prompt for prompt in pool}
    existing_prompt = prompt_by_id[existing_selection.prompt_id]
    composed_score = prompt_invariant_score(composed_prompt, invariant_table)
    existing_score = prompt_invariant_score(existing_prompt, invariant_table)
    if existing_score >= composed_score:
        chosen = existing_prompt
    else:
        chosen = composed_prompt
    return MethodSelection(
        method=method_name,
        task_id=task_id,
        fold_id=fold_id,
        target_model=target_model,
        source_models=source_models,
        prompt_id=chosen.prompt_id,
        prompt_text=chosen.text,
        selected_edit_ids=chosen.edit_ids,
    )

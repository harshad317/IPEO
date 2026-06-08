"""Few-target IPEO calibration."""

from __future__ import annotations

from ipeo.core.schemas import AtomicEdit, InvariantEditStats, PromptCandidate
from ipeo.effects.invariant_scorer import InvariantScorerConfig, estimate_invariant_effects
from ipeo.evaluation.evaluator import EvalResult
from ipeo.methods.ipeo_zero import select_zero_target_prompt


def shrinkage_scores(
    *,
    source_rows: list[InvariantEditStats],
    target_rows: list[InvariantEditStats],
    lambda_cost: float = 0.01,
) -> dict[str, float]:
    target_by_id = {row.edit_id: row for row in target_rows}
    scores: dict[str, float] = {}
    for src in source_rows:
        tgt = target_by_id.get(src.edit_id)
        src_var = max(src.effect_variance, 1e-6)
        if tgt is None:
            scores[src.edit_id] = src.mean_effect - lambda_cost * src.token_delta
            continue
        tgt_var = max(tgt.effect_variance, 1e-6)
        weight_src = (1.0 / src_var) / ((1.0 / src_var) + (1.0 / tgt_var))
        beta_tilde = weight_src * src.mean_effect + (1.0 - weight_src) * tgt.mean_effect
        scores[src.edit_id] = beta_tilde - lambda_cost * src.token_delta
    return scores


def calibrate_few_target_prompt(
    *,
    task_id: str,
    target_model_id: str,
    source_models: list[str],
    seed_prompt: PromptCandidate,
    pool: list[PromptCandidate],
    edits: list[AtomicEdit],
    source_invariant_table: list[InvariantEditStats],
    calibration_results: list[EvalResult],
    fold_id: str,
    config: InvariantScorerConfig,
) -> PromptCandidate:
    target_rows = estimate_invariant_effects(
        task_id=task_id,
        source_model_ids=[target_model_id],
        pool=pool,
        edits=edits,
        eval_results=calibration_results,
        config=config,
        split="calibration",
    )
    scores = shrinkage_scores(
        source_rows=source_invariant_table,
        target_rows=target_rows,
        lambda_cost=config.lambda_cost,
    )
    adjusted = sorted(
        (
            row.__class__(
                **{
                    **row.__dict__,
                    "ipeo_score": scores.get(row.edit_id, row.ipeo_score),
                    "lcb_mean_effect": min(row.lcb_mean_effect, scores.get(row.edit_id, row.ipeo_score)),
                }
            )
            for row in source_invariant_table
        ),
        key=lambda row: row.ipeo_score,
        reverse=True,
    )
    prompt, _ = select_zero_target_prompt(
        task_id=task_id,
        seed_prompt=seed_prompt,
        edits=edits,
        invariant_table=adjusted,
        fold_id=f"{fold_id}-few_target",
        target_model=target_model_id,
        source_models=source_models,
        max_edits_per_prompt=config.max_edits_per_prompt,
        min_sign_agreement=0.0,
        min_lcb=-1.0,
        exclude_generic=config.exclude_generic,
    )
    return prompt

"""Invariant edit-effect estimation and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ipeo.core.schemas import AtomicEdit, EvalResult, InvariantEditStats, PromptCandidate
from ipeo.effects.bootstrap import bootstrap_effect_se, prompt_scores_for_model
from ipeo.effects.design_matrix import build_edit_matrix
from ipeo.effects.ridge_estimator import RidgeEffectEstimator


@dataclass(frozen=True)
class InvariantScorerConfig:
    ridge_alpha: float = 1.0
    n_bootstrap: int = 30
    lambda_var: float = 1.0
    lambda_sign: float = 0.5
    lambda_rank: float = 0.25
    lambda_cost: float = 0.01
    lambda_lcb: float = 0.5
    min_sign_agreement: float = 1.0
    min_lcb: float = 0.0
    max_edits_per_prompt: int = 5
    max_prompt_tokens_multiplier: float = 1.5
    exclude_generic: bool = False


def _rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return order.astype(float)


def _rank_stability_by_edit(effect_matrix: np.ndarray) -> np.ndarray:
    if effect_matrix.shape[0] <= 1 or effect_matrix.shape[1] <= 1:
        return np.ones(effect_matrix.shape[1])
    ranks = np.vstack([_rank_values(row) for row in effect_matrix])
    max_std = max(1.0, (effect_matrix.shape[1] - 1) / 2)
    stability = 1.0 - np.std(ranks, axis=0) / max_std
    return np.clip(stability, 0.0, 1.0)


def estimate_invariant_effects(
    *,
    task_id: str,
    source_model_ids: list[str],
    pool: list[PromptCandidate],
    edits: list[AtomicEdit],
    eval_results: list[EvalResult],
    config: InvariantScorerConfig,
    split: str = "val",
    seed: int = 0,
) -> list[InvariantEditStats]:
    x = build_edit_matrix(pool, edits)
    per_model_effects: dict[str, np.ndarray] = {}
    per_model_se: dict[str, np.ndarray] = {}
    for offset, model_id in enumerate(source_model_ids):
        y = prompt_scores_for_model(eval_results, pool, model_id, split)
        estimator = RidgeEffectEstimator(alpha=config.ridge_alpha).fit(x, y)
        per_model_effects[model_id] = estimator.coefficients()
        per_model_se[model_id] = bootstrap_effect_se(
            x=x,
            pool=pool,
            eval_results=eval_results,
            model_id=model_id,
            split=split,
            alpha=config.ridge_alpha,
            n_bootstrap=config.n_bootstrap,
            seed=seed + offset,
        )

    effect_matrix = np.vstack([per_model_effects[model_id] for model_id in source_model_ids])
    se_matrix = np.vstack([per_model_se[model_id] for model_id in source_model_ids])
    rank_stability = _rank_stability_by_edit(effect_matrix)
    rows: list[InvariantEditStats] = []
    for idx, edit in enumerate(edits):
        betas = effect_matrix[:, idx]
        mean_effect = float(np.mean(betas))
        effect_variance = float(np.var(betas))
        mean_sign = np.sign(mean_effect)
        sign_agreement = float(np.mean(np.sign(betas) == mean_sign)) if mean_sign != 0 else 0.0
        se_mean = float(np.sqrt(np.sum(se_matrix[:, idx] ** 2)) / max(1, len(source_model_ids)))
        lcb = mean_effect - 1.96 * se_mean
        sign_disagreement = 1.0 - sign_agreement
        ipeo_score = (
            mean_effect
            - config.lambda_var * effect_variance
            - config.lambda_sign * sign_disagreement
            + config.lambda_rank * float(rank_stability[idx])
            - config.lambda_cost * edit.estimated_token_delta
            + config.lambda_lcb * lcb
        )
        rows.append(
            InvariantEditStats(
                task_id=task_id,
                edit_id=edit.edit_id,
                edit_type=edit.edit_type,
                token_delta=edit.estimated_token_delta,
                mean_effect=mean_effect,
                effect_variance=effect_variance,
                sign_agreement=sign_agreement,
                rank_stability=float(rank_stability[idx]),
                lcb_mean_effect=lcb,
                ipeo_score=float(ipeo_score),
                is_generic=edit.is_generic,
                is_placebo=edit.is_placebo,
                per_model_effects={model_id: float(per_model_effects[model_id][idx]) for model_id in source_model_ids},
            )
        )
    return sorted(rows, key=lambda row: row.ipeo_score, reverse=True)


def anti_invariant_pairs(rows: list[InvariantEditStats]) -> list[tuple[str, str]]:
    pairs = []
    for left, right in combinations(rows, 2):
        if left.edit_type == right.edit_type and left.mean_effect * right.mean_effect < 0:
            pairs.append((left.edit_id, right.edit_id))
    return pairs

"""Bootstrap uncertainty for edit effects."""

from __future__ import annotations

import numpy as np

from ipeo.core.schemas import EvalResult, PromptCandidate
from ipeo.effects.ridge_estimator import RidgeEffectEstimator


def prompt_scores_for_model(results: list[EvalResult], pool: list[PromptCandidate], model_id: str, split: str) -> np.ndarray:
    by_prompt: dict[str, list[float]] = {prompt.prompt_id: [] for prompt in pool}
    for row in results:
        if row.model_id == model_id and row.split == split:
            by_prompt.setdefault(row.prompt_id, []).append(row.score)
    return np.array([np.mean(by_prompt[prompt.prompt_id]) if by_prompt[prompt.prompt_id] else 0.0 for prompt in pool])


def bootstrap_effect_se(
    *,
    x: np.ndarray,
    pool: list[PromptCandidate],
    eval_results: list[EvalResult],
    model_id: str,
    split: str,
    alpha: float,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    example_ids = sorted({row.example_id for row in eval_results if row.model_id == model_id and row.split == split})
    if not example_ids or n_bootstrap <= 1:
        return np.zeros(x.shape[1])

    score_lookup: dict[tuple[str, str], float] = {}
    for row in eval_results:
        if row.model_id == model_id and row.split == split:
            score_lookup[(row.prompt_id, row.example_id)] = row.score

    rng = np.random.default_rng(seed)
    coefs: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(example_ids, size=len(example_ids), replace=True)
        y = []
        for prompt in pool:
            scores = [score_lookup.get((prompt.prompt_id, ex_id), 0.0) for ex_id in sampled]
            y.append(float(np.mean(scores)))
        estimator = RidgeEffectEstimator(alpha=alpha).fit(x, np.array(y))
        coefs.append(estimator.coefficients())
    if len(coefs) <= 1:
        return np.zeros(x.shape[1])
    return np.std(np.vstack(coefs), axis=0, ddof=1)

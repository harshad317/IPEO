from __future__ import annotations

from ipeo.core.schemas import EvalResult, InvariantEditStats
from ipeo.effects.invariant_scorer import InvariantScorerConfig, estimate_invariant_effects
from ipeo.methods.ipeo_few_target import shrinkage_scores
from ipeo.prompts.pool_builder import build_frozen_pool


def _eval(run_id: str, task: str, model: str, prompt: str, example: str, split: str, score: float) -> EvalResult:
    return EvalResult(
        run_id=run_id,
        task_id=task,
        model_id=model,
        prompt_id=prompt,
        example_id=example,
        split=split,  # type: ignore[arg-type]
        raw_output_path="",
        parsed_output={"answer": ""},
        score=score,
        parse_success=True,
    )


def test_invariant_effects_use_only_source_models(tmp_path) -> None:
    pool, edits = build_frozen_pool("classification", num_prompts=6, artifact_dir=tmp_path)
    rows = []
    for model in ["source_a", "source_b", "target"]:
        for prompt in pool:
            for idx in range(4):
                base = 0.5
                boost = 0.3 if edits[0].edit_id in prompt.edit_ids and model != "target" else 0.0
                target_boost = 0.9 if model == "target" else 0.0
                rows.append(_eval("run", "classification", model, prompt.prompt_id, f"ex-{idx}", "val", base + boost + target_boost))
    table = estimate_invariant_effects(
        task_id="classification",
        source_model_ids=["source_a", "source_b"],
        pool=pool,
        edits=edits,
        eval_results=rows,
        config=InvariantScorerConfig(n_bootstrap=4),
        split="val",
    )
    assert table
    assert all("target" not in row.per_model_effects for row in table)


def test_shrinkage_scores_prefer_lower_uncertainty() -> None:
    src = InvariantEditStats("task", "e1", "output_format", 5, 0.2, 0.01, 1.0, 1.0, 0.1, 0.2, False, False, {})
    tgt = InvariantEditStats("task", "e1", "output_format", 5, 0.8, 100.0, 1.0, 1.0, 0.1, 0.8, False, False, {})
    score = shrinkage_scores(source_rows=[src], target_rows=[tgt], lambda_cost=0.0)["e1"]
    assert abs(score - 0.2) < 0.01


def test_invariant_score_contains_positive_effects(tmp_path) -> None:
    pool, edits = build_frozen_pool("gsm8k", num_prompts=8, artifact_dir=tmp_path)
    positive_edit = next(edit for edit in edits if edit.edit_type == "reasoning_strategy")
    rows = []
    for model in ["m1", "m2", "m3"]:
        for prompt in pool:
            for idx in range(5):
                score = 0.4 + (0.4 if positive_edit.edit_id in prompt.edit_ids else 0.0)
                rows.append(_eval("run", "gsm8k", model, prompt.prompt_id, f"ex-{idx}", "val", score))
    table = estimate_invariant_effects(
        task_id="gsm8k",
        source_model_ids=["m1", "m2", "m3"],
        pool=pool,
        edits=edits,
        eval_results=rows,
        config=InvariantScorerConfig(n_bootstrap=4, min_lcb=-1),
    )
    top = table[0]
    assert top.mean_effect > 0
    assert top.sign_agreement == 1.0

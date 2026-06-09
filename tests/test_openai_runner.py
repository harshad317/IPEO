from __future__ import annotations

from pathlib import Path

import pytest

from ipeo.baselines.dspy_optimizers import DspyOptimizerConfig, run_dspy_optimizer
from ipeo.core.schemas import EvalResult, MethodSelection
from ipeo.core.schemas import GenerationConfig
from ipeo.models.openai_adapter import OpenAIResponsesAdapter, build_openai_environments, clamp_openai_max_output_tokens
from ipeo.runners.run_openai import normalize_methods
from ipeo.stats.method_summary import build_method_summary_rows
from ipeo.tasks.adapters import get_task


def test_normalize_methods_accepts_commas_and_aliases() -> None:
    fixed, official = normalize_methods(["ipeo_zero,ipeo_no_generic,source_average", "gepa", "mipro", "capo"])
    assert fixed == {"ipeo_zero", "ipeo_no_generic", "source_average"}
    assert official == {"gepa", "miprov2", "capo"}


def test_normalize_methods_all_expands() -> None:
    fixed, official = normalize_methods(["all"])
    assert "ipeo_zero" in fixed
    assert {"ipeo_no_generic", "ipeo_no_cost", "ipeo_no_generic_no_cost"} <= fixed
    assert {"ipeo_budget_200", "ipeo_budget_500", "ipeo_budget_1000", "ipeo_budget_select", "ipeo_budget_select_source_val"} <= fixed
    assert {"ipeo_select_existing", "ipeo_composed_vs_existing"} <= fixed
    assert {"gepa", "miprov2", "capo"} <= official


def test_openai_environments_use_single_api_model() -> None:
    envs = build_openai_environments("gpt-4.1-mini", count=4, timeout_seconds=300, max_retries=7)
    assert len(envs) == 4
    assert {env.api_model for env in envs} == {"gpt-4.1-mini"}
    assert len({env.model_id for env in envs}) == 4
    assert {env.timeout_seconds for env in envs} == {300}
    assert {env.max_retries for env in envs} == {7}


def test_openai_max_output_tokens_are_clamped_to_provider_minimum() -> None:
    assert clamp_openai_max_output_tokens(12) == 16
    assert clamp_openai_max_output_tokens(16) == 16
    assert clamp_openai_max_output_tokens(96) == 96


def test_openai_adapter_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = OpenAIResponsesAdapter.from_model("gpt-4.1-mini", "gpt-4.1-mini:target")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        adapter.generate("Prompt", "Input", GenerationConfig())


def test_dspy_optimizer_skips_without_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    task = get_task("gsm8k")
    result = run_dspy_optimizer(
        method="gepa",
        run_id="run-test",
        task=task,
        train_examples=task.load_split("opt", 2),
        val_examples=task.load_split("val", 2),
        test_examples=task.load_split("test", 2),
        artifact_dir=tmp_path,
        config=DspyOptimizerConfig(
            api_model="gpt-4.1-mini",
            fold_id="target-gpt-4.1-mini:target",
            target_model="gpt-4.1-mini:target",
            source_models=[],
        ),
    )
    assert result.status == "skipped"
    assert "OPENAI_API_KEY" in (result.reason or "")


def test_method_summary_uses_compiled_program_opt_val_test_rows(tmp_path: Path) -> None:
    selection = MethodSelection(
        method="gepa",
        task_id="gsm8k",
        fold_id="target-model",
        target_model="model:target",
        source_models=[],
        prompt_id="gepa-prompt",
        prompt_text="optimized",
        selected_edit_ids=[],
        target_calls=7,
    )
    final_results = [
        EvalResult("run", "gsm8k", "model:target", "gepa-prompt", "ex-opt", "opt", "raw", {"raw": "1"}, 0.5, True),
        EvalResult("run", "gsm8k", "model:target", "gepa-prompt", "ex-val", "val", "raw", {"raw": "2"}, 0.75, True),
        EvalResult("run", "gsm8k", "model:target", "gepa-prompt", "ex-test", "test", "raw", {"raw": "3"}, 1.0, True),
    ]
    rows = build_method_summary_rows(
        run_id="run",
        task_id="gsm8k",
        target_model="model:target",
        source_models=[],
        selections=[selection],
        transfer_rows=[{"method": "gepa", "source_calls": 0, "target_calls": 7}],
        pool_results=[],
        final_results=final_results,
        cost_log_path=tmp_path / "costs.jsonl",
    )
    by_split = {row["split"]: row for row in rows}
    assert by_split["train"]["score"] == 0.5
    assert by_split["val"]["score"] == 0.75
    assert by_split["test"]["score"] == 1.0
    assert by_split["test"]["avg_tokens"] is not None
    assert by_split["test"]["api_calls"] == 1
    assert by_split["optimization"]["api_calls"] == 7

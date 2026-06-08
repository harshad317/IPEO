from __future__ import annotations

import pytest

from ipeo.core.schemas import GenerationConfig
from ipeo.models.openai_adapter import OpenAIResponsesAdapter, build_openai_environments
from ipeo.runners.run_openai import normalize_methods


def test_normalize_methods_accepts_commas_and_aliases() -> None:
    fixed, official = normalize_methods(["ipeo_zero,source_average", "gepa", "mipro", "capo"])
    assert fixed == {"ipeo_zero", "source_average"}
    assert official == {"gepa", "miprov2", "capo"}


def test_normalize_methods_all_expands() -> None:
    fixed, official = normalize_methods(["all"])
    assert "ipeo_zero" in fixed
    assert {"gepa", "miprov2", "capo"} <= official


def test_openai_environments_use_single_api_model() -> None:
    envs = build_openai_environments("gpt-4.1-mini", count=4)
    assert len(envs) == 4
    assert {env.api_model for env in envs} == {"gpt-4.1-mini"}
    assert len({env.model_id for env in envs}) == 4


def test_openai_adapter_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = OpenAIResponsesAdapter.from_model("gpt-4.1-mini", "gpt-4.1-mini:target")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        adapter.generate("Prompt", "Input", GenerationConfig())

from __future__ import annotations

from pathlib import Path

from ipeo.core.ids import stable_hash
from ipeo.core.schemas import GenerationConfig
from ipeo.evaluation.cache import make_cache_key
from ipeo.evaluation.cost_ledger import CostLedger
from ipeo.models.mock import get_mock_model
from ipeo.prompts.pool_builder import build_frozen_pool
from ipeo.tasks.adapters import get_task


def test_stable_hash_is_deterministic() -> None:
    left = stable_hash({"b": 2, "a": 1}, prefix="x-")
    right = stable_hash({"a": 1, "b": 2}, prefix="x-")
    assert left == right
    assert left.startswith("x-")


def test_cache_key_changes_with_prompt_and_generation_config(tmp_path: Path) -> None:
    pool, _ = build_frozen_pool("gsm8k", num_prompts=2, artifact_dir=tmp_path)
    task = get_task("gsm8k")
    example = task.load_split("val", 1)[0]
    model = get_mock_model("mock_openai_a")
    key_a = make_cache_key(model, pool[0], example, GenerationConfig(max_tokens=16))
    key_b = make_cache_key(model, pool[1], example, GenerationConfig(max_tokens=16))
    key_c = make_cache_key(model, pool[0], example, GenerationConfig(max_tokens=32))
    assert key_a != key_b
    assert key_a != key_c


def test_task_parsers_and_metrics() -> None:
    gsm = get_task("gsm8k")
    assert gsm.score(gsm.parse_output("Answer: 42"), "42") == 1.0
    cls = get_task("classification")
    assert cls.score(cls.parse_output("sports"), "sports") == 1.0
    qa = get_task("extraction_qa")
    assert qa.score(qa.parse_output("Answer: Paris"), "Paris") == 1.0


def test_cost_ledger_aggregates(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "costs.jsonl")
    model = get_mock_model("mock_openai_a")
    pool, _ = build_frozen_pool("gsm8k", num_prompts=1, artifact_dir=tmp_path)
    response = model.generate(pool[0].text, "Maya buys 2 packs with 3 stickers each, then finds 1 more.", GenerationConfig())
    ledger.log(
        run_id="run",
        task_id="gsm8k",
        model=model,
        method="unit",
        fold_id="fold",
        seed=0,
        phase="evaluation",
        prompt=pool[0],
        edit_id=None,
        example_id="ex",
        response=response,
        config=GenerationConfig(),
        cache_hit=False,
    )
    agg = ledger.aggregate()
    assert agg["calls"] == 1.0
    assert agg["dollars"] > 0

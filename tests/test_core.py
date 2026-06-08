from __future__ import annotations

import json
from pathlib import Path

from ipeo.core.ids import stable_hash
from ipeo.core.schemas import GenerationConfig, InvariantEditStats
from ipeo.evaluation.cache import ResponseCache, make_cache_key
from ipeo.evaluation.cost_ledger import CostLedger
from ipeo.evaluation.evaluator import evaluate_pool
from ipeo.methods.ipeo_zero import prompt_invariant_score, select_existing_prompt_by_invariant_score
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


def test_corrupt_cache_entry_is_regenerated(tmp_path: Path) -> None:
    pool, _ = build_frozen_pool("gsm8k", num_prompts=1, artifact_dir=tmp_path)
    task = get_task("gsm8k")
    example = task.load_split("val", 1)[0]
    model = get_mock_model("mock_openai_a")
    config = GenerationConfig(max_tokens=16)
    cache = ResponseCache(tmp_path / "cache")
    key = make_cache_key(model, pool[0], example, config)
    cache._path(key).write_text("", encoding="utf-8")
    ledger = CostLedger(tmp_path / "costs.jsonl")
    rows = evaluate_pool(
        run_id="run",
        task=task,
        models=[model],
        pool=[pool[0], pool[0]],
        examples=[example, example],
        cache=cache,
        cost_ledger=ledger,
        generation_config=config,
        method="unit",
        fold_id="fold",
        show_tqdm=False,
        workers=4,
    )
    assert len(rows) == 4
    assert cache.load_or_none(key) is not None


def test_task_parsers_and_metrics() -> None:
    gsm = get_task("gsm8k")
    assert gsm.score(gsm.parse_output("Answer: 42"), "42") == 1.0
    cls = get_task("classification")
    assert cls.score(cls.parse_output("sports"), "sports") == 1.0
    qa = get_task("extraction_qa")
    assert qa.score(qa.parse_output("Answer: Paris"), "Paris") == 1.0
    ifbench = get_task("ifbench")
    ex = ifbench.load_split("val", 1)[0]
    assert ifbench.score(ifbench.parse_output("coral reefs protect coral life."), ex.gold) == 1.0
    hard = get_task("ifbench_hard")
    hard_ex = hard.load_split("val", 1)[0]
    assert hard.score(hard.parse_output('{"summary":"ready","risk":"low","action":"verify"}'), hard_ex.gold) == 1.0
    assert hard.score(hard.parse_output('```json\n{"summary":"ready","risk":"low","action":"verify"}\n```'), hard_ex.gold) == 0.0


def test_official_ifbench_adapter_uses_configured_repo(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "IFBench"
    data_dir = repo / "data"
    data_dir.mkdir(parents=True)
    (repo / "evaluation_lib.py").write_text(
        """
import dataclasses

@dataclasses.dataclass
class InputExample:
    key: str
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[dict]

@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]

def test_instruction_following_loose(inp, prompt_to_response):
    response = prompt_to_response[inp.prompt]
    ok = response.strip() == "OK"
    return OutputExample(inp.instruction_id_list, inp.prompt, response, ok, [ok])

def test_instruction_following_strict(inp, prompt_to_response):
    return test_instruction_following_loose(inp, prompt_to_response)
""".strip(),
        encoding="utf-8",
    )
    row = {
        "key": "demo",
        "prompt": "Say OK.",
        "instruction_id_list": ["fake:ok"],
        "kwargs": [{}],
    }
    (data_dir / "IFBench_test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setenv("IFBENCH_REPO", str(repo))

    from ipeo.tasks.ifbench_official import _load_official_eval_lib

    _load_official_eval_lib.cache_clear()
    task = get_task("ifbench_official")
    example = task.load_split("opt", 1)[0]
    assert example.input == "Say OK."
    assert task.score(task.parse_output("OK"), example.gold) == 1.0
    assert task.score(task.parse_output("NO"), example.gold) == 0.0
    _load_official_eval_lib.cache_clear()


def test_invariant_existing_prompt_selector_scores_edit_vectors(tmp_path: Path) -> None:
    pool, edits = build_frozen_pool("gsm8k", num_prompts=12, artifact_dir=tmp_path)
    invariant_rows = [
        InvariantEditStats(
            task_id="gsm8k",
            edit_id=edit.edit_id,
            edit_type=edit.edit_type,
            token_delta=edit.estimated_token_delta,
            mean_effect=0.0,
            effect_variance=0.0,
            sign_agreement=1.0,
            rank_stability=1.0,
            lcb_mean_effect=0.0,
            ipeo_score=float(idx + 1),
            is_generic=edit.is_generic,
            is_placebo=edit.is_placebo,
            per_model_effects={"source": float(idx + 1)},
        )
        for idx, edit in enumerate(edits)
    ]
    selection = select_existing_prompt_by_invariant_score(
        task_id="gsm8k",
        pool=pool,
        invariant_table=invariant_rows,
        fold_id="fold",
        target_model="target",
        source_models=["source"],
    )
    best_score = max(prompt_invariant_score(prompt, invariant_rows) for prompt in pool)
    selected_prompt = next(prompt for prompt in pool if prompt.prompt_id == selection.prompt_id)
    assert selection.method == "ipeo_select_existing"
    assert prompt_invariant_score(selected_prompt, invariant_rows) == best_score


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
        example_id="ex-cache",
        response=response,
        config=GenerationConfig(),
        cache_hit=True,
    )
    ledger.log(
        run_id="other-run",
        task_id="gsm8k",
        model=model,
        method="unit",
        fold_id="fold",
        seed=0,
        phase="evaluation",
        prompt=pool[0],
        edit_id=None,
        example_id="ex-other",
        response=response,
        config=GenerationConfig(),
        cache_hit=False,
    )
    agg = ledger.aggregate()
    assert agg["calls"] == 2.0
    assert agg["dollars"] > 0
    paid_cost = ledger.method_cost("unit", task_id="gsm8k", prompt_ids={pool[0].prompt_id})
    estimated_cost = ledger.method_estimated_cost("unit", task_id="gsm8k", prompt_ids={pool[0].prompt_id})
    assert paid_cost > 0
    assert estimated_cost > paid_cost
    run_estimated_cost = ledger.method_estimated_cost("unit", run_id="run", task_id="gsm8k", prompt_ids={pool[0].prompt_id})
    other_estimated_cost = ledger.method_estimated_cost("unit", run_id="other-run", task_id="gsm8k", prompt_ids={pool[0].prompt_id})
    assert run_estimated_cost > other_estimated_cost
    assert ledger.method_cost("unit", task_id="ifbench") == 0
    assert ledger.method_calls("unit", task_id="gsm8k", prompt_ids={pool[0].prompt_id}) == 2
    assert ledger.method_calls("unit", run_id="run", task_id="gsm8k", prompt_ids={pool[0].prompt_id}) == 1

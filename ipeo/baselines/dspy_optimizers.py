"""Optional DSPy GEPA and MIPROv2 optimizer runners."""

from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ipeo.core.ids import stable_hash
from ipeo.core.io import ensure_parent, write_jsonl
from ipeo.core.schemas import EvalResult, Example, GenerationConfig, MethodSelection
from ipeo.models.base import count_tokens
from ipeo.models.openai_adapter import GPT_41_MINI_INPUT_PER_1K, GPT_41_MINI_OUTPUT_PER_1K
from ipeo.prompts.seed_prompts import seed_prompt_texts
from ipeo.tasks.base import TaskAdapter


@dataclass(frozen=True)
class DspyOptimizerConfig:
    api_model: str
    fold_id: str
    target_model: str
    source_models: list[str]
    seed: int = 0
    auto: str | None = "light"
    program: str = "auto"
    train_examples: int = 16
    val_examples: int = 16
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 4
    max_metric_calls: int | None = None
    num_threads: int = 1
    temperature: float = 0.0
    max_tokens: int = 64
    max_retries: int = 3


@dataclass(frozen=True)
class DspyOptimizerResult:
    method: str
    status: str
    selection: MethodSelection | None = None
    eval_results: list[EvalResult] = field(default_factory=list)
    optimization_calls: int = 0
    eval_calls_by_split: dict[str, int] = field(default_factory=dict)
    total_calls: int = 0
    total_dollars: float = 0.0
    prompt_text: str = ""
    reason: str | None = None


def run_dspy_optimizer(
    *,
    method: str,
    run_id: str,
    task: TaskAdapter,
    train_examples: list[Example],
    val_examples: list[Example],
    test_examples: list[Example],
    artifact_dir: str | Path,
    config: DspyOptimizerConfig,
) -> DspyOptimizerResult:
    """Compile and evaluate a DSPy optimizer if the optional dependency exists."""

    normalized_method = "miprov2" if method in {"mipro", "miprov2"} else method
    if normalized_method not in {"gepa", "miprov2"}:
        return DspyOptimizerResult(
            method=normalized_method,
            status="skipped",
            reason=f"Unsupported DSPy optimizer '{method}'",
        )
    if not os.environ.get("OPENAI_API_KEY"):
        return DspyOptimizerResult(
            method=normalized_method,
            status="skipped",
            reason="OPENAI_API_KEY is required to execute DSPy optimizers",
        )

    try:
        dspy = importlib.import_module("dspy")
    except ModuleNotFoundError:
        return DspyOptimizerResult(
            method=normalized_method,
            status="skipped",
            reason="Package 'dspy' is not installed",
        )

    trainset = _to_dspy_examples(dspy, task, train_examples[: config.train_examples])
    valset = _to_dspy_examples(dspy, task, val_examples[: config.val_examples])
    if not trainset or not valset:
        return DspyOptimizerResult(
            method=normalized_method,
            status="skipped",
            reason="DSPy optimizers require non-empty opt and val splits",
        )

    lm = _build_lm(dspy, config)
    previous_lm = getattr(getattr(dspy, "settings", None), "lm", None)
    dspy.configure(lm=lm)
    program = _build_program(dspy, task, config)
    metric = _build_metric(dspy, task, feedback_mode=normalized_method == "gepa")
    history_start = _history_len(lm)
    status_path = Path(artifact_dir) / "stats" / f"{task.task_id}_{normalized_method}_dspy_status.jsonl"

    try:
        start = time.perf_counter()
        compiled = _compile_optimizer(
            dspy=dspy,
            method=normalized_method,
            program=program,
            metric=metric,
            lm=lm,
            trainset=trainset,
            valset=valset,
            config=config,
            log_dir=Path(artifact_dir) / "dspy" / task.task_id / normalized_method,
        )
        compile_latency_ms = int((time.perf_counter() - start) * 1000)
    except Exception as exc:  # pragma: no cover - depends on optional package/runtime
        result = DspyOptimizerResult(
            method=normalized_method,
            status="failed",
            reason=f"{exc.__class__.__name__}: {exc}",
        )
        write_jsonl(status_path, [result])
        _restore_lm(dspy, previous_lm)
        return result

    history_after_compile = _history_len(lm)
    eval_results: list[EvalResult] = []
    eval_calls_by_split: dict[str, int] = {}
    raw_root = Path(artifact_dir) / "eval_results" / "raw" / task.task_id / normalized_method
    for split_name, examples in [
        ("opt", train_examples[: config.train_examples]),
        ("val", val_examples[: config.val_examples]),
        ("test", test_examples),
    ]:
        before = _history_len(lm)
        split_rows = _evaluate_compiled_program(
            compiled=compiled,
            task=task,
            examples=examples,
            run_id=run_id,
            target_model=config.target_model,
            method=normalized_method,
            raw_root=raw_root,
            split_name=split_name,
        )
        after = _history_len(lm)
        eval_calls_by_split[split_name] = max(after - before, len(split_rows))
        eval_results.extend(split_rows)

    history_end = _history_len(lm)
    history_rows = _history_slice(lm, history_start, history_end)
    prompt_text = _extract_prompt_text(compiled, normalized_method)
    prompt_id = stable_hash(
        {
            "task_id": task.task_id,
            "method": normalized_method,
            "prompt_text": prompt_text,
            "seed": config.seed,
            "model": config.api_model,
        },
        prefix=f"{normalized_method}-",
    )
    eval_results = [
        EvalResult(
            run_id=row.run_id,
            task_id=row.task_id,
            model_id=row.model_id,
            prompt_id=prompt_id,
            example_id=row.example_id,
            split=row.split,
            raw_output_path=row.raw_output_path,
            parsed_output=row.parsed_output,
            score=row.score,
            parse_success=row.parse_success,
            error_type=row.error_type,
        )
        for row in eval_results
    ]
    optimization_calls = max(history_after_compile - history_start, 0)
    total_calls = max(history_end - history_start, optimization_calls + sum(eval_calls_by_split.values()))
    total_dollars = _history_dollars(history_rows)
    if total_dollars == 0.0:
        total_dollars = _estimate_dollars(prompt_text, eval_results)
    selection = MethodSelection(
        method=normalized_method,
        task_id=task.task_id,
        fold_id=config.fold_id,
        target_model=config.target_model,
        source_models=config.source_models,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        selected_edit_ids=[],
        target_calls=optimization_calls,
        source_calls=0,
        total_dollars=total_dollars,
    )
    result = DspyOptimizerResult(
        method=normalized_method,
        status="completed",
        selection=selection,
        eval_results=eval_results,
        optimization_calls=optimization_calls,
        eval_calls_by_split=eval_calls_by_split,
        total_calls=total_calls,
        total_dollars=total_dollars,
        prompt_text=prompt_text,
        reason=f"compile_latency_ms={compile_latency_ms}",
    )
    write_jsonl(Path(artifact_dir) / "eval_results" / f"{task.task_id}_{normalized_method}_dspy_eval.jsonl", eval_results)
    write_jsonl(status_path, [result])
    _restore_lm(dspy, previous_lm)
    return result


def _build_lm(dspy: Any, config: DspyOptimizerConfig) -> Any:
    kwargs = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "cache": True,
        "num_retries": config.max_retries,
    }
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["api_base"] = base_url.rstrip("/")
    return dspy.LM(f"openai/{config.api_model}", **kwargs)


def _build_program(dspy: Any, task: TaskAdapter, config: DspyOptimizerConfig) -> Any:
    signature = dspy.Signature(
        "question -> answer",
        instructions=_task_instruction(task),
    )
    program_type = config.program
    if program_type == "auto":
        program_type = "cot" if task.task_id in {"gsm8k", "bbh"} else "predict"
    if program_type == "cot":
        return dspy.ChainOfThought(signature)
    return dspy.Predict(signature)


def _task_instruction(task: TaskAdapter) -> str:
    seeds = seed_prompt_texts(task.task_id)
    task_seed = seeds[-1] if seeds else "Solve the task carefully."
    output_desc = {
        "gsm8k": "Return the final numeric answer in the answer field.",
        "bbh": "Return only the final answer in the answer field.",
        "classification": "Return exactly one label: sports, business, science, or world.",
        "extraction_qa": "Return the shortest answer span supported by the context.",
        "ifbench": "Return only the final response that satisfies every visible instruction-following constraint.",
        "ifbench_hard": "Return only the final response after satisfying every hard formatting, count, JSON, CSV, suffix, and forbidden-token constraint.",
        "ifbench_official": "Return only the final response after satisfying every visible instruction-following constraint.",
    }.get(task.task_id, "Return only the final answer.")
    return f"{task_seed}\n\n{output_desc}"


def _compile_optimizer(
    *,
    dspy: Any,
    method: str,
    program: Any,
    metric: Callable[..., Any],
    lm: Any,
    trainset: list[Any],
    valset: list[Any],
    config: DspyOptimizerConfig,
    log_dir: Path,
) -> Any:
    log_dir.mkdir(parents=True, exist_ok=True)
    if method == "gepa":
        optimizer_kwargs: dict[str, Any] = {
            "metric": metric,
            "reflection_lm": lm,
            "num_threads": max(1, int(config.num_threads)),
            "log_dir": str(log_dir),
            "track_stats": True,
            "seed": config.seed,
        }
        if config.max_metric_calls is not None:
            optimizer_kwargs["auto"] = None
            optimizer_kwargs["max_metric_calls"] = config.max_metric_calls
        else:
            optimizer_kwargs["auto"] = config.auto
        optimizer = dspy.GEPA(**optimizer_kwargs)
        return optimizer.compile(program, trainset=trainset, valset=valset)

    optimizer = dspy.MIPROv2(
        metric=metric,
        task_model=lm,
        prompt_model=lm,
        auto=config.auto,
        num_threads=max(1, int(config.num_threads)),
        seed=config.seed,
        log_dir=str(log_dir),
    )
    return optimizer.compile(
        program,
        trainset=trainset,
        valset=valset,
        max_bootstrapped_demos=config.max_bootstrapped_demos,
        max_labeled_demos=config.max_labeled_demos,
    )


def _to_dspy_examples(dspy: Any, task: TaskAdapter, examples: list[Example]) -> list[Any]:
    rows = []
    for example in examples:
        rows.append(
            dspy.Example(
                question=task.format_input(example),
                answer=_gold_to_text(example.gold),
                ipeo_gold=example.gold,
                ipeo_example_id=example.example_id,
            ).with_inputs("question")
        )
    return rows


def _build_metric(dspy: Any, task: TaskAdapter, *, feedback_mode: bool) -> Callable[..., Any]:
    def metric(gold: Any, pred: Any, trace: Any = None, pred_name: Any = None, pred_trace: Any = None) -> Any:
        raw_answer = str(_example_value(pred, "answer", pred))
        gold_value = _example_value(gold, "ipeo_gold", _example_value(gold, "answer", ""))
        parsed = task.parse_output(raw_answer)
        score = float(task.score(parsed, gold_value))
        if feedback_mode:
            feedback = _feedback_for_score(score, raw_answer, gold_value)
            return dspy.Prediction(score=score, feedback=feedback)
        return score

    return metric


def _evaluate_compiled_program(
    *,
    compiled: Any,
    task: TaskAdapter,
    examples: list[Example],
    run_id: str,
    target_model: str,
    method: str,
    raw_root: Path,
    split_name: str,
) -> list[EvalResult]:
    rows: list[EvalResult] = []
    raw_root.mkdir(parents=True, exist_ok=True)
    for example in examples:
        raw_text = ""
        parse_success = True
        error_type = None
        try:
            pred = compiled(question=task.format_input(example))
            raw_text = str(_example_value(pred, "answer", pred))
            parsed = task.parse_output(raw_text)
            score = float(task.score(parsed, example.gold))
        except Exception as exc:  # pragma: no cover - optional runtime guard
            parsed = {"answer": "", "raw": raw_text}
            score = 0.0
            parse_success = False
            error_type = exc.__class__.__name__
        raw_path = raw_root / f"{stable_hash({'method': method, 'example_id': example.example_id}, prefix='raw-')}.txt"
        ensure_parent(raw_path).write_text(raw_text, encoding="utf-8")
        rows.append(
            EvalResult(
                run_id=run_id,
                task_id=task.task_id,
                model_id=target_model,
                prompt_id=f"{method}-pending",
                example_id=example.example_id,
                split=split_name,  # type: ignore[arg-type]
                raw_output_path=str(raw_path),
                parsed_output=parsed,
                score=score,
                parse_success=parse_success,
                error_type=error_type,
            )
        )
    return rows


def _extract_prompt_text(program: Any, method: str) -> str:
    parts: list[str] = [f"DSPy {method} optimized program"]
    try:
        for idx, predictor in enumerate(program.predictors()):
            signature = getattr(predictor, "signature", None)
            instructions = getattr(signature, "instructions", None)
            if instructions:
                parts.append(f"predictor_{idx}_instructions:\n{instructions}")
            demos = getattr(predictor, "demos", None) or []
            if demos:
                parts.append(f"predictor_{idx}_demos: {len(demos)}")
    except Exception:
        pass
    if len(parts) == 1:
        parts.append(repr(program))
    return "\n\n".join(parts)


def _gold_to_text(gold: Any) -> str:
    if isinstance(gold, str):
        return gold
    if isinstance(gold, dict):
        for key in ("answer", "value", "label"):
            if key in gold:
                return str(gold[key])
    return str(gold)


def _example_value(obj: Any, key: str, default: Any = None) -> Any:
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return obj[key]
    except Exception:
        return default


def _feedback_for_score(score: float, raw_answer: str, gold_value: Any) -> str:
    if score >= 1.0:
        return "Correct. Preserve the task format and answer constraints."
    if not raw_answer.strip():
        return "The answer was empty. Produce a concise answer that satisfies the requested format."
    return (
        "The answer did not satisfy the task metric. "
        f"Observed answer: {raw_answer[:500]!r}. "
        f"Gold/validator target: {str(gold_value)[:500]!r}. "
        "Improve exact final-answer precision and hard format/constraint compliance."
    )


def _history_len(lm: Any) -> int:
    history = getattr(lm, "history", None)
    if isinstance(history, list):
        return len(history)
    return 0


def _history_slice(lm: Any, start: int, end: int) -> list[Any]:
    history = getattr(lm, "history", None)
    if isinstance(history, list):
        return history[start:end]
    return []


def _history_dollars(history_rows: list[Any]) -> float:
    total = 0.0
    for row in history_rows:
        value = _example_value(row, "cost", None)
        if isinstance(value, (int, float)):
            total += float(value)
            continue
        hidden = getattr(_example_value(row, "response", None), "_hidden_params", None)
        response_cost = _example_value(hidden, "response_cost", None)
        if isinstance(response_cost, (int, float)):
            total += float(response_cost)
            continue
        usage = _example_value(row, "usage", None)
        input_tokens, output_tokens = _usage_tokens(usage)
        total += input_tokens / 1000 * GPT_41_MINI_INPUT_PER_1K
        total += output_tokens / 1000 * GPT_41_MINI_OUTPUT_PER_1K
    return total


def _usage_tokens(usage: Any) -> tuple[int, int]:
    if not isinstance(usage, dict):
        return (0, 0)
    input_tokens = 0
    output_tokens = 0
    for value in usage.values():
        if isinstance(value, dict):
            input_tokens += int(value.get("prompt_tokens") or value.get("input_tokens") or 0)
            output_tokens += int(value.get("completion_tokens") or value.get("output_tokens") or 0)
    input_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return input_tokens, output_tokens


def _estimate_dollars(prompt_text: str, rows: list[EvalResult]) -> float:
    input_tokens = count_tokens(prompt_text) * max(1, len(rows))
    output_tokens = sum(count_tokens(str(row.parsed_output.get("raw", ""))) for row in rows)
    return input_tokens / 1000 * GPT_41_MINI_INPUT_PER_1K + output_tokens / 1000 * GPT_41_MINI_OUTPUT_PER_1K


def _restore_lm(dspy: Any, previous_lm: Any) -> None:
    if previous_lm is not None:
        try:
            dspy.configure(lm=previous_lm)
        except Exception:
            pass

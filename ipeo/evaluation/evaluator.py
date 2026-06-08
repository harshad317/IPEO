"""Prompt pool evaluator."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from ipeo.core.io import write_jsonl
from ipeo.core.schemas import EvalResult, Example, GenerationConfig, PromptCandidate
from ipeo.evaluation.cache import ResponseCache, make_cache_key
from ipeo.evaluation.cost_ledger import CostLedger
from ipeo.models.base import ModelAdapter
from ipeo.tasks.base import TaskAdapter


def evaluate_pool(
    *,
    run_id: str,
    task: TaskAdapter,
    models: list[ModelAdapter],
    pool: list[PromptCandidate],
    examples: list[Example],
    cache: ResponseCache,
    cost_ledger: CostLedger,
    generation_config: GenerationConfig,
    method: str,
    fold_id: str,
    seed: int = 0,
    phase: str = "evaluation",
    artifact_path: str | Path | None = None,
    show_tqdm: bool = True,
    workers: int = 1,
) -> list[EvalResult]:
    rows: list[EvalResult] = []
    total = len(models) * len(pool) * len(examples)
    running_score = 0.0

    def evaluate_one(model: ModelAdapter, prompt: PromptCandidate, example: Example) -> EvalResult:
        key = make_cache_key(model, prompt, example, generation_config)
        cache_hit = cache.exists(key)
        if cache_hit:
            response = cache.load(key)
        else:
            response = model.generate(prompt.text, task.format_input(example), generation_config)
            cache.save(key, response)
        parsed = task.parse_output(response.raw_text)
        try:
            score = float(task.score(parsed, example.gold))
            parse_success = True
            error_type = None
        except Exception as exc:  # pragma: no cover - defensive parser guard
            score = 0.0
            parse_success = False
            error_type = exc.__class__.__name__
        cost_ledger.log(
            run_id=run_id,
            task_id=task.task_id,
            model=model,
            method=method,
            fold_id=fold_id,
            seed=seed,
            phase=phase,
            prompt=prompt,
            edit_id=None,
            example_id=example.example_id,
            response=response,
            config=generation_config,
            cache_hit=cache_hit,
        )
        row = EvalResult(
            run_id=run_id,
            task_id=task.task_id,
            model_id=model.model_id,
            prompt_id=prompt.prompt_id,
            example_id=example.example_id,
            split=example.split,  # type: ignore[arg-type]
            raw_output_path=cache.raw_output_path(key),
            parsed_output=parsed,
            score=score,
            parse_success=parse_success,
            error_type=error_type,
        )
        return row

    iterator: list[tuple[ModelAdapter, PromptCandidate, Example]] = [
        (model, prompt, example) for model in models for prompt in pool for example in examples
    ]
    if workers <= 1:
        progress: Iterable[tuple[ModelAdapter, PromptCandidate, Example]] = tqdm(
            iterator,
            total=total,
            disable=not show_tqdm,
            leave=False,
            desc=f"{task.task_id}:{method}",
        )
        for idx, (model, prompt, example) in enumerate(progress, start=1):
            row = evaluate_one(model, prompt, example)
            rows.append(row)
            running_score += row.score
            if show_tqdm:
                progress.set_postfix(acc=f"{running_score / idx:.3f}", calls=idx)  # type: ignore[attr-defined]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(evaluate_one, model, prompt, example) for model, prompt, example in iterator]
            progress = tqdm(
                as_completed(futures),
                total=total,
                disable=not show_tqdm,
                leave=False,
                desc=f"{task.task_id}:{method}",
            )
            for idx, future in enumerate(progress, start=1):
                row = future.result()
                rows.append(row)
                running_score += row.score
                if show_tqdm:
                    progress.set_postfix(acc=f"{running_score / idx:.3f}", calls=idx)
    if artifact_path is not None:
        write_jsonl(artifact_path, rows)
    return rows


def aggregate_scores(results: list[EvalResult], split: str = "val") -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in results:
        if row.split != split:
            continue
        grouped.setdefault((row.prompt_id, row.model_id), []).append(row.score)
    return {key: sum(values) / len(values) for key, values in grouped.items() if values}


def score_prompt(results: list[EvalResult], prompt_id: str, model_id: str, split: str = "test") -> float:
    scores = [r.score for r in results if r.prompt_id == prompt_id and r.model_id == model_id and r.split == split]
    return sum(scores) / len(scores) if scores else 0.0

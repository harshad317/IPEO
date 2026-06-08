"""Deterministic offline dry-run CLI for IPEO."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ipeo.baselines.official_optimizers import official_optimizer_records
from ipeo.baselines.optional_wrappers import optional_baseline_statuses
from ipeo.core.ids import stable_hash
from ipeo.core.io import write_csv, write_jsonl
from ipeo.core.schemas import GenerationConfig, MethodSelection, PromptCandidate
from ipeo.effects.invariant_scorer import InvariantScorerConfig, estimate_invariant_effects
from ipeo.evaluation.cache import ResponseCache
from ipeo.evaluation.cost_ledger import CostLedger
from ipeo.evaluation.evaluator import evaluate_pool
from ipeo.methods.fixed_pool import (
    asha_selection,
    best_source_transfer,
    original_prompt,
    pooled_source_selection,
    promptbridge_emulation,
    random_search_selection,
    robust_source_selection,
    source_average_selection,
    target_only_bo_selection,
)
from ipeo.methods.ipeo_zero import (
    select_composed_vs_existing_prompt,
    select_existing_prompt_by_invariant_score,
    select_zero_target_prompt,
)
from ipeo.models.mock import get_models
from ipeo.prompts.pool_builder import build_frozen_pool
from ipeo.runners.progress import ProgressSettings, RichRunReporter
from ipeo.stats.ipeo_compare import build_composed_vs_existing_row
from ipeo.stats.method_summary import aggregate_method_summary_rows, build_method_summary_rows
from ipeo.stats.regret import build_transfer_rows
from ipeo.tasks.adapters import get_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic IPEO MVP dry run")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k"])
    parser.add_argument("--models", nargs="+", default=["mock_openai_a", "mock_openai_b", "mock_openai_c", "mock_openai_d"])
    parser.add_argument("--num_prompts", type=int, default=20)
    parser.add_argument("--num_examples", type=int, default=24)
    parser.add_argument("--fold_target", default="mock_openai_d")
    parser.add_argument("--cache_dir", default="artifacts/cache")
    parser.add_argument("--cost_log", default="artifacts/costs/dry_run.jsonl")
    parser.add_argument("--artifact_dir", default="artifacts")
    parser.add_argument("--progress", choices=["off", "rich", "tqdm", "both"], default="both")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_color", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _selection_to_prompt(selection: MethodSelection, pool: list[PromptCandidate]) -> PromptCandidate:
    for prompt in pool:
        if prompt.prompt_id == selection.prompt_id:
            return prompt
    return PromptCandidate(
        prompt_id=selection.prompt_id,
        task_id=selection.task_id,
        text=selection.prompt_text,
        edit_ids=selection.selected_edit_ids,
        edit_vector=[],
        source_generator="promptbridge_emulation" if "promptbridge" in selection.method else "ipeo_composed",
        parent_prompt_ids=[],
        prompt_tokens_by_model={"mock": len(selection.prompt_text.split())},
        estimated_deployment_cost={"mock": len(selection.prompt_text.split()) * 0.0001 / 1000},
        coherence_repaired=False,
        frozen_pool_version="mvp-v1",
    )


def _dedupe_prompts(prompts: list[PromptCandidate]) -> list[PromptCandidate]:
    seen: set[str] = set()
    rows: list[PromptCandidate] = []
    for prompt in prompts:
        if prompt.prompt_id in seen:
            continue
        seen.add(prompt.prompt_id)
        rows.append(prompt)
    return rows


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    artifact_dir = Path(args.artifact_dir)
    settings = ProgressSettings(mode=args.progress, quiet=args.quiet, no_color=args.no_color)
    reporter = RichRunReporter(settings)
    run_id = stable_hash({"tasks": args.tasks, "models": args.models, "seed": args.seed}, prefix="run-")
    fold_id = f"target-{args.fold_target}"
    models = get_models(args.models)
    model_by_id = {model.model_id: model for model in models}
    if args.fold_target not in model_by_id:
        raise ValueError(f"fold target {args.fold_target!r} is not in --models")
    source_models = [model_id for model_id in args.models if model_id != args.fold_target]
    cache = ResponseCache(args.cache_dir)
    cost_ledger = CostLedger(args.cost_log)
    config = GenerationConfig(temperature=0.0, max_tokens=64)
    invariant_config = InvariantScorerConfig(n_bootstrap=20)
    all_transfer_rows: list[dict[str, object]] = []
    all_comparison_rows: list[dict[str, object]] = []
    all_method_summary_rows: list[dict[str, object]] = []

    write_jsonl(artifact_dir / "stats" / "optional_baselines.jsonl", optional_baseline_statuses())
    write_jsonl(artifact_dir / "stats" / "official_optimizer_records.jsonl", official_optimizer_records())

    for task in get_tasks(args.tasks):
        reporter.status(f"Task {task.task_id}: building frozen pool")
        pool, edits = build_frozen_pool(task.task_id, num_prompts=args.num_prompts, seed=args.seed, artifact_dir=artifact_dir)
        val_examples = task.load_split("val", args.num_examples)
        test_examples = task.load_split("test", args.num_examples)

        reporter.status(f"Task {task.task_id}: evaluating frozen pool on validation and test splits")
        pool_val_results = evaluate_pool(
            run_id=run_id,
            task=task,
            models=models,
            pool=pool,
            examples=val_examples,
            cache=cache,
            cost_ledger=cost_ledger,
            generation_config=config,
            method="fixed_pool",
            fold_id=fold_id,
            seed=args.seed,
            phase="evaluation",
            artifact_path=artifact_dir / "eval_results" / f"{task.task_id}_pool_val.jsonl",
            show_tqdm=settings.use_tqdm,
        )
        pool_test_results = evaluate_pool(
            run_id=run_id,
            task=task,
            models=models,
            pool=pool,
            examples=test_examples,
            cache=cache,
            cost_ledger=cost_ledger,
            generation_config=config,
            method="fixed_pool",
            fold_id=fold_id,
            seed=args.seed,
            phase="final_test",
            artifact_path=artifact_dir / "eval_results" / f"{task.task_id}_pool_test.jsonl",
            show_tqdm=settings.use_tqdm,
        )
        pool_results = pool_val_results + pool_test_results

        reporter.status(f"Task {task.task_id}: estimating invariant edit effects from sources {', '.join(source_models)}")
        invariant_table = estimate_invariant_effects(
            task_id=task.task_id,
            source_model_ids=source_models,
            pool=pool,
            edits=edits,
            eval_results=pool_results,
            config=invariant_config,
            split="val",
            seed=args.seed,
        )
        write_jsonl(artifact_dir / "stats" / f"{task.task_id}_invariant_edits.jsonl", invariant_table)

        ipeo_prompt, ipeo_selection = select_zero_target_prompt(
            task_id=task.task_id,
            seed_prompt=pool[0],
            edits=edits,
            invariant_table=invariant_table,
            fold_id=fold_id,
            target_model=args.fold_target,
            source_models=source_models,
            max_edits_per_prompt=invariant_config.max_edits_per_prompt,
            min_sign_agreement=invariant_config.min_sign_agreement,
            min_lcb=-0.05,
        )
        ipeo_existing_selection = select_existing_prompt_by_invariant_score(
            task_id=task.task_id,
            pool=pool,
            invariant_table=invariant_table,
            fold_id=fold_id,
            target_model=args.fold_target,
            source_models=source_models,
        )
        ipeo_comparison_selection = select_composed_vs_existing_prompt(
            task_id=task.task_id,
            composed_prompt=ipeo_prompt,
            existing_selection=ipeo_existing_selection,
            pool=pool,
            invariant_table=invariant_table,
            fold_id=fold_id,
            target_model=args.fold_target,
            source_models=source_models,
        )

        selections = [
            original_prompt(task_id=task.task_id, fold_id=fold_id, target_model=args.fold_target, source_models=source_models, pool=pool),
            source_average_selection(
                task_id=task.task_id,
                fold_id=fold_id,
                target_model=args.fold_target,
                source_models=source_models,
                pool=pool,
                eval_results=pool_results,
            ),
            pooled_source_selection(
                task_id=task.task_id,
                fold_id=fold_id,
                target_model=args.fold_target,
                source_models=source_models,
                pool=pool,
                eval_results=pool_results,
            ),
            robust_source_selection(
                task_id=task.task_id,
                fold_id=fold_id,
                target_model=args.fold_target,
                source_models=source_models,
                pool=pool,
                eval_results=pool_results,
            ),
            random_search_selection(
                task_id=task.task_id,
                fold_id=fold_id,
                target_model=args.fold_target,
                source_models=source_models,
                pool=pool,
                seed=args.seed,
            ),
            target_only_bo_selection(
                task_id=task.task_id,
                fold_id=fold_id,
                target_model=args.fold_target,
                source_models=source_models,
                pool=pool,
                eval_results=pool_results,
                budget=min(8, len(pool)),
            ),
            asha_selection(
                task_id=task.task_id,
                fold_id=fold_id,
                target_model=args.fold_target,
                source_models=source_models,
                pool=pool,
                eval_results=pool_results,
            ),
            promptbridge_emulation(
                task_id=task.task_id,
                fold_id=fold_id,
                target_model=args.fold_target,
                source_models=source_models,
                pool=pool,
                eval_results=pool_results,
            ),
            ipeo_selection,
            ipeo_existing_selection,
            ipeo_comparison_selection,
        ]
        selections.extend(
            best_source_transfer(
                task_id=task.task_id,
                fold_id=fold_id,
                target_model=args.fold_target,
                source_models=source_models,
                pool=pool,
                eval_results=pool_results,
            )
        )
        write_jsonl(artifact_dir / "stats" / f"{task.task_id}_method_selections.jsonl", selections)

        final_prompts = _dedupe_prompts([_selection_to_prompt(selection, pool + [ipeo_prompt]) for selection in selections])
        reporter.status(f"Task {task.task_id}: evaluating selected methods on held-out target test examples")
        final_results = evaluate_pool(
            run_id=run_id,
            task=task,
            models=[model_by_id[args.fold_target]],
            pool=final_prompts,
            examples=test_examples,
            cache=cache,
            cost_ledger=cost_ledger,
            generation_config=config,
            method="selected_methods",
            fold_id=fold_id,
            seed=args.seed,
            phase="final_test",
            artifact_path=artifact_dir / "eval_results" / f"{task.task_id}_selected_test.jsonl",
            show_tqdm=settings.use_tqdm,
        )

        source_avg = next(selection for selection in selections if selection.method == "source_average")
        source_eval_calls = len(source_models) * len(pool) * len(val_examples)
        source_call_methods = {
            "source_average",
            "pooled_source",
            "worst_source_robust",
            "asha_fixed_pool",
            "ipeo_zero",
            "ipeo_select_existing",
            "ipeo_composed_vs_existing",
        }
        source_calls_by_method = {method: source_eval_calls for method in source_call_methods}
        for model_id in source_models:
            source_calls_by_method[f"best_source_transfer:{model_id}"] = len(pool) * len(val_examples)
        rows = build_transfer_rows(
            task_id=task.task_id,
            fold_id=fold_id,
            target_model=args.fold_target,
            source_models=source_models,
            selections=selections,
            final_results=final_results + pool_test_results,
            pool_results=pool_test_results,
            pool=pool,
            source_average_prompt_id=source_avg.prompt_id,
            source_calls_by_method=source_calls_by_method,
            target_calls_by_method={"target_only_bo_fixed_pool": min(8, len(pool)) * len(val_examples), "ipeo_zero": 0},
            dollars_by_method={
                "ipeo_zero": cost_ledger.method_cost("fixed_pool", model_ids=set(source_models)),
                "ipeo_select_existing": cost_ledger.method_cost("fixed_pool", model_ids=set(source_models)),
                "ipeo_composed_vs_existing": cost_ledger.method_cost("fixed_pool", model_ids=set(source_models)),
            },
        )
        all_transfer_rows.extend(rows)
        method_summary_rows = build_method_summary_rows(
            run_id=run_id,
            task_id=task.task_id,
            target_model=args.fold_target,
            source_models=source_models,
            selections=selections,
            transfer_rows=rows,
            pool_results=pool_results,
            final_results=final_results,
            cost_log_path=args.cost_log,
        )
        all_method_summary_rows.extend(method_summary_rows)
        existing_prompt = next(prompt for prompt in pool if prompt.prompt_id == ipeo_existing_selection.prompt_id)
        comparison_rows = [
            build_composed_vs_existing_row(
                task_id=task.task_id,
                fold_id=fold_id,
                target_model=args.fold_target,
                composed_method="ipeo_zero",
                composed_prompt=ipeo_prompt,
                existing_selection=ipeo_existing_selection,
                existing_prompt=existing_prompt,
                comparison_selection=ipeo_comparison_selection,
                invariant_table=invariant_table,
                eval_results=final_results + pool_test_results,
            )
        ]
        write_jsonl(artifact_dir / "stats" / f"{task.task_id}_ipeo_composed_vs_existing.jsonl", comparison_rows)
        all_comparison_rows.extend(comparison_rows)
        reporter.summary_table(f"{task.task_id} transfer regret", rows)

    write_csv(artifact_dir / "stats" / "transfer_regret.csv", all_transfer_rows)
    write_csv(artifact_dir / "stats" / "ipeo_composed_vs_existing.csv", all_comparison_rows)
    write_csv(artifact_dir / "stats" / "method_summary.csv", all_method_summary_rows)
    reporter.method_summary_panels(aggregate_method_summary_rows(all_method_summary_rows))
    invariant_rows = []
    for path in sorted((artifact_dir / "stats").glob("*_invariant_edits.jsonl")):
        from ipeo.core.io import read_jsonl

        invariant_rows.extend(read_jsonl(path))
    write_csv(artifact_dir / "stats" / "invariant_edits.csv", invariant_rows)
    return all_transfer_rows


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

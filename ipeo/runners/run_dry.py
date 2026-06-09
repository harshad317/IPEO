"""Deterministic offline dry-run CLI for IPEO."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ipeo.baselines.official_optimizers import official_optimizer_records
from ipeo.baselines.optional_wrappers import optional_baseline_statuses
from ipeo.core.ids import stable_hash
from ipeo.core.io import write_csv, write_jsonl
from ipeo.core.schemas import EvalResult, GenerationConfig, MethodSelection, PromptCandidate
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
from ipeo.methods.budgeted_ipeo import (
    BudgetedPromptCandidate,
    build_budgeted_source_subset,
    select_budgeted_prompt,
    select_budgeted_prompt_by_source_validation,
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
from ipeo.stats.split_contract import access_row, access_rows_by_method
from ipeo.tasks.adapters import get_tasks

IPEO_BUDGET_METHODS = {
    "ipeo_budget_200": 200,
    "ipeo_budget_500": 500,
    "ipeo_budget_1000": 1000,
}
IPEO_BUDGET_SELECT_METHOD = "ipeo_budget_select"
IPEO_BUDGET_SELECT_SOURCE_VAL_METHOD = "ipeo_budget_select_source_val"


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
    all_access_rows: list[dict[str, object]] = []

    write_jsonl(artifact_dir / "stats" / "optional_baselines.jsonl", optional_baseline_statuses())
    write_jsonl(artifact_dir / "stats" / "official_optimizer_records.jsonl", official_optimizer_records())
    write_jsonl(
        artifact_dir / "stats" / "split_contract.jsonl",
        [
            {
                "train_split": "opt",
                "validation_split": "val",
                "test_split": "test",
                "rule": "Methods may optimize/select on their declared train/validation access only; target test is final evaluation only.",
            }
        ],
    )

    for task in get_tasks(args.tasks):
        reporter.status(f"Task {task.task_id}: building frozen pool")
        pool, edits = build_frozen_pool(task.task_id, num_prompts=args.num_prompts, seed=args.seed, artifact_dir=artifact_dir)
        train_examples = task.load_split("opt", args.num_examples)
        val_examples = task.load_split("val", args.num_examples)
        test_examples = task.load_split("test", args.num_examples)

        reporter.status(f"Task {task.task_id}: evaluating frozen pool on source train and source/target validation splits")
        pool_train_results = evaluate_pool(
            run_id=run_id,
            task=task,
            models=[model_by_id[model_id] for model_id in source_models],
            pool=pool,
            examples=train_examples,
            cache=cache,
            cost_ledger=cost_ledger,
            generation_config=config,
            method="fixed_pool",
            fold_id=fold_id,
            seed=args.seed,
            phase="baseline_optimization",
            artifact_path=artifact_dir / "eval_results" / f"{task.task_id}_pool_train.jsonl",
            show_tqdm=settings.use_tqdm,
        )
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
            phase="calibration",
            artifact_path=artifact_dir / "eval_results" / f"{task.task_id}_pool_val.jsonl",
            show_tqdm=settings.use_tqdm,
        )
        pool_results = pool_train_results + pool_val_results

        reporter.status(f"Task {task.task_id}: estimating invariant edit effects from sources {', '.join(source_models)}")
        invariant_table = estimate_invariant_effects(
            task_id=task.task_id,
            source_model_ids=source_models,
            pool=pool,
            edits=edits,
            eval_results=pool_results,
            config=invariant_config,
            split="opt",
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
        budgeted_ipeo_prompts: list[PromptCandidate] = []
        budgeted_ipeo_selections: list[MethodSelection] = []
        budgeted_source_calls_by_method: dict[str, int] = {}
        budgeted_source_train_calls_by_method: dict[str, int] = {}
        budgeted_source_validation_calls_by_method: dict[str, int] = {}
        budgeted_dollars_by_method: dict[str, float] = {}
        budgeted_prompt_candidates: list[BudgetedPromptCandidate] = []
        budgeted_eval_rows_by_method: dict[str, list[EvalResult]] = {}
        for method_name, budget in sorted(IPEO_BUDGET_METHODS.items(), key=lambda item: item[1]):
            subset = build_budgeted_source_subset(
                pool=pool,
                train_examples=train_examples,
                source_model_ids=source_models,
                pool_train_results=pool_train_results,
                budget=budget,
                seed=args.seed,
            )
            budget_table = estimate_invariant_effects(
                task_id=task.task_id,
                source_model_ids=subset.source_model_ids,
                pool=subset.pool,
                edits=edits,
                eval_results=subset.eval_results,
                config=invariant_config,
                split="opt",
                seed=args.seed + budget,
            )
            write_jsonl(artifact_dir / "stats" / f"{task.task_id}_{method_name}_invariant_edits.jsonl", budget_table)
            write_jsonl(
                artifact_dir / "stats" / f"{task.task_id}_{method_name}_budget.jsonl",
                [
                    {
                        "method": method_name,
                        "requested_budget": budget,
                        "actual_source_calls": subset.source_calls,
                        "num_prompts": len(subset.pool),
                        "num_examples": len(subset.example_ids),
                        "source_models": subset.source_model_ids,
                    }
                ],
            )
            budget_prompt, budget_selection = select_zero_target_prompt(
                task_id=task.task_id,
                seed_prompt=pool[0],
                edits=edits,
                invariant_table=budget_table,
                fold_id=fold_id,
                target_model=args.fold_target,
                source_models=subset.source_model_ids,
                max_edits_per_prompt=invariant_config.max_edits_per_prompt,
                min_sign_agreement=0.0,
                min_lcb=-0.10,
                method_name=method_name,
            )
            budgeted_ipeo_prompts.append(budget_prompt)
            budgeted_ipeo_selections.append(budget_selection)
            budgeted_source_calls_by_method[method_name] = subset.source_calls
            budgeted_source_train_calls_by_method[method_name] = subset.source_calls
            budgeted_dollars_by_method[method_name] = cost_ledger.method_estimated_cost(
                "fixed_pool",
                run_id=run_id,
                phase="baseline_optimization",
                model_ids=set(subset.source_model_ids),
                task_id=task.task_id,
                prompt_ids=set(subset.prompt_ids),
                example_ids=set(subset.example_ids),
            )
            budgeted_eval_rows_by_method[method_name] = list(subset.eval_results)
            budgeted_prompt_candidates.append(
                BudgetedPromptCandidate(
                    method=method_name,
                    requested_budget=budget,
                    source_calls=subset.source_calls,
                    prompt=budget_prompt,
                    selection=budget_selection,
                    invariant_table=budget_table,
                )
            )
        unique_train_eval_rows = {
            (row.model_id, row.prompt_id, row.example_id): row
            for candidate in budgeted_prompt_candidates
            for row in budgeted_eval_rows_by_method.get(candidate.method, [])
        }
        budget_choice = select_budgeted_prompt(
            candidates=budgeted_prompt_candidates,
            task_id=task.task_id,
            fold_id=fold_id,
            target_model=args.fold_target,
            method_name=IPEO_BUDGET_SELECT_METHOD,
        )
        budgeted_ipeo_selections.append(budget_choice.selection)
        budgeted_source_calls_by_method[IPEO_BUDGET_SELECT_METHOD] = len(unique_train_eval_rows)
        budgeted_source_train_calls_by_method[IPEO_BUDGET_SELECT_METHOD] = len(unique_train_eval_rows)
        budgeted_dollars_by_method[IPEO_BUDGET_SELECT_METHOD] = cost_ledger.method_estimated_cost_for_eval_results(
            "fixed_pool",
            list(unique_train_eval_rows.values()),
            run_id=run_id,
            phase="baseline_optimization",
            task_id=task.task_id,
        )
        write_jsonl(
            artifact_dir / "stats" / f"{task.task_id}_{IPEO_BUDGET_SELECT_METHOD}.jsonl",
            [
                {
                    "method": IPEO_BUDGET_SELECT_METHOD,
                    "chosen_method": budget_choice.chosen_method,
                    "requested_budget": budget_choice.requested_budget,
                    "source_calls": budget_choice.source_calls,
                    "source_score": budget_choice.source_score,
                    "prompt_id": budget_choice.prompt.prompt_id,
                    "selected_edit_ids": budget_choice.selection.selected_edit_ids,
                    "candidate_scores": budget_choice.score_rows,
                }
            ],
        )
        validation_prompts = _dedupe_prompts([candidate.prompt for candidate in budgeted_prompt_candidates])
        budgeted_source_validation_results = evaluate_pool(
            run_id=run_id,
            task=task,
            models=[model_by_id[model_id] for model_id in source_models],
            pool=validation_prompts,
            examples=val_examples,
            cache=cache,
            cost_ledger=cost_ledger,
            generation_config=config,
            method="fixed_pool",
            fold_id=fold_id,
            seed=args.seed,
            phase="calibration",
            artifact_path=artifact_dir / "eval_results" / f"{task.task_id}_{IPEO_BUDGET_SELECT_SOURCE_VAL_METHOD}_source_val.jsonl",
            show_tqdm=settings.use_tqdm,
        )
        pool_results.extend(budgeted_source_validation_results)
        budget_val_choice = select_budgeted_prompt_by_source_validation(
            candidates=budgeted_prompt_candidates,
            validation_results=budgeted_source_validation_results,
            task_id=task.task_id,
            fold_id=fold_id,
            target_model=args.fold_target,
            method_name=IPEO_BUDGET_SELECT_SOURCE_VAL_METHOD,
        )
        budgeted_ipeo_selections.append(budget_val_choice.selection)
        unique_val_eval_rows = {
            (row.model_id, row.prompt_id, row.example_id): row
            for row in budgeted_source_validation_results
        }
        train_calls = len(unique_train_eval_rows)
        val_calls = len(unique_val_eval_rows)
        budgeted_source_train_calls_by_method[IPEO_BUDGET_SELECT_SOURCE_VAL_METHOD] = train_calls
        budgeted_source_validation_calls_by_method[IPEO_BUDGET_SELECT_SOURCE_VAL_METHOD] = val_calls
        budgeted_source_calls_by_method[IPEO_BUDGET_SELECT_SOURCE_VAL_METHOD] = train_calls + val_calls
        train_cost = cost_ledger.method_estimated_cost_for_eval_results(
            "fixed_pool",
            list(unique_train_eval_rows.values()),
            run_id=run_id,
            phase="baseline_optimization",
            task_id=task.task_id,
        )
        val_cost = cost_ledger.method_estimated_cost_for_eval_results(
            "fixed_pool",
            list(unique_val_eval_rows.values()),
            run_id=run_id,
            phase="calibration",
            task_id=task.task_id,
        )
        budgeted_dollars_by_method[IPEO_BUDGET_SELECT_SOURCE_VAL_METHOD] = train_cost + val_cost
        write_jsonl(
            artifact_dir / "stats" / f"{task.task_id}_{IPEO_BUDGET_SELECT_SOURCE_VAL_METHOD}.jsonl",
            [
                {
                    "method": IPEO_BUDGET_SELECT_SOURCE_VAL_METHOD,
                    "chosen_method": budget_val_choice.chosen_method,
                    "requested_budget": budget_val_choice.requested_budget,
                    "source_train_calls": train_calls,
                    "source_validation_calls": val_calls,
                    "source_calls": train_calls + val_calls,
                    "source_score": budget_val_choice.source_score,
                    "prompt_id": budget_val_choice.prompt.prompt_id,
                    "selected_edit_ids": budget_val_choice.selection.selected_edit_ids,
                    "candidate_scores": budget_val_choice.score_rows,
                }
            ],
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
            *budgeted_ipeo_selections,
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

        final_prompts = _dedupe_prompts([_selection_to_prompt(selection, pool + [ipeo_prompt] + budgeted_ipeo_prompts) for selection in selections])
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
        reporter.status(f"Task {task.task_id}: evaluating locked target test oracle for regret reporting")
        pool_test_results = evaluate_pool(
            run_id=run_id,
            task=task,
            models=[model_by_id[args.fold_target]],
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

        source_avg = next(selection for selection in selections if selection.method == "source_average")
        source_train_calls = len(source_models) * len(pool) * len(train_examples)
        source_validation_calls = len(source_models) * len(pool) * len(val_examples)
        source_train_cost = cost_ledger.method_estimated_cost(
            "fixed_pool",
            run_id=run_id,
            phase="baseline_optimization",
            model_ids=set(source_models),
            task_id=task.task_id,
        )
        source_validation_cost = cost_ledger.method_estimated_cost(
            "fixed_pool",
            run_id=run_id,
            phase="calibration",
            model_ids=set(source_models),
            task_id=task.task_id,
            prompt_ids={prompt.prompt_id for prompt in pool},
        )
        target_bo_prompt_ids = {prompt.prompt_id for prompt in pool[: min(8, len(pool))]}
        target_pool_val_cost = cost_ledger.method_estimated_cost(
            "fixed_pool",
            run_id=run_id,
            phase="calibration",
            model_ids={args.fold_target},
            task_id=task.task_id,
            prompt_ids=target_bo_prompt_ids,
        )
        final_cost_by_prompt = {
            prompt.prompt_id: cost_ledger.method_estimated_cost(
                "selected_methods",
                run_id=run_id,
                phase="final_test",
                model_ids={args.fold_target},
                task_id=task.task_id,
                prompt_ids={prompt.prompt_id},
            )
            for prompt in final_prompts
        }
        source_call_methods = {
            "source_average",
            "pooled_source",
            "worst_source_robust",
            "asha_fixed_pool",
            "promptbridge_emulation",
        }
        ipeo_source_call_methods = {
            "ipeo_zero",
            "ipeo_select_existing",
            "ipeo_composed_vs_existing",
        }
        source_calls_by_method = {method: source_validation_calls for method in source_call_methods}
        source_calls_by_method.update({method: source_train_calls for method in ipeo_source_call_methods})
        source_calls_by_method.update(budgeted_source_calls_by_method)
        source_train_calls_by_method = {method: source_train_calls for method in ipeo_source_call_methods}
        source_train_calls_by_method.update(budgeted_source_train_calls_by_method)
        source_validation_calls_by_method = {method: source_validation_calls for method in source_call_methods}
        source_validation_calls_by_method.update(budgeted_source_validation_calls_by_method)
        dollars_by_method = {method: source_validation_cost for method in source_call_methods}
        dollars_by_method.update({method: source_train_cost for method in ipeo_source_call_methods})
        dollars_by_method.update(budgeted_dollars_by_method)
        for model_id in source_models:
            source_calls_by_method[f"best_source_transfer:{model_id}"] = len(pool) * len(val_examples)
            dollars_by_method[f"best_source_transfer:{model_id}"] = cost_ledger.method_estimated_cost(
                "fixed_pool",
                run_id=run_id,
                phase="calibration",
                model_ids={model_id},
                task_id=task.task_id,
            )
        target_calls_by_method = {"target_only_bo_fixed_pool": min(8, len(pool)) * len(val_examples), "ipeo_zero": 0}
        dollars_by_method["target_only_bo_fixed_pool"] = target_pool_val_cost
        for selection in selections:
            dollars_by_method[selection.method] = dollars_by_method.get(selection.method, 0.0) + final_cost_by_prompt.get(selection.prompt_id, 0.0)
        task_access_rows = [
            access_row(
                task_id=task.task_id,
                selection=selection,
                source_train_calls=source_train_calls_by_method.get(selection.method, source_calls_by_method.get(selection.method, source_train_calls)),
                source_validation_calls=source_validation_calls_by_method.get(selection.method, source_validation_calls),
                target_validation_calls=target_calls_by_method.get(selection.method, 0),
                target_optimization_calls=target_calls_by_method.get(selection.method, selection.target_calls),
                final_target_test_calls=len(test_examples),
            )
            for selection in selections
        ]
        write_jsonl(artifact_dir / "stats" / f"{task.task_id}_data_access.jsonl", task_access_rows)
        all_access_rows.extend(task_access_rows)
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
            target_calls_by_method=target_calls_by_method,
            dollars_by_method=dollars_by_method,
            method_access_by_method=access_rows_by_method(task_access_rows),
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
    write_csv(artifact_dir / "stats" / "data_access.csv", all_access_rows)
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

"""Live OpenAI benchmark runner for IPEO."""

from __future__ import annotations

import argparse
from pathlib import Path

from ipeo.baselines.dspy_optimizers import DspyOptimizerConfig, DspyOptimizerResult, run_dspy_optimizer
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
    build_expanded_prompt_candidates,
    build_budgeted_source_subset,
    plan_budgeted_source_subset,
    select_budgeted_prompt,
    select_budgeted_prompt_by_source_validation,
    select_expanded_prompt_by_source_validation,
)
from ipeo.methods.ipeo_zero import (
    select_composed_vs_existing_prompt,
    select_existing_prompt_by_invariant_score,
    select_zero_target_prompt,
)
from ipeo.models.openai_adapter import build_openai_environments, clamp_openai_max_output_tokens
from ipeo.prompts.pool_builder import build_frozen_pool
from ipeo.runners.progress import ProgressSettings, RichRunReporter
from ipeo.stats.ipeo_compare import build_composed_vs_existing_row
from ipeo.stats.method_summary import aggregate_method_summary_rows, build_method_summary_rows
from ipeo.stats.regret import build_transfer_rows
from ipeo.stats.split_contract import access_row, access_rows_by_method
from ipeo.tasks.adapters import get_tasks

IPEO_COMPOSED_METHODS = {
    "ipeo_zero",
    "ipeo_no_generic",
    "ipeo_no_cost",
    "ipeo_no_generic_no_cost",
}
IPEO_BUDGET_METHODS = {
    "ipeo_budget_200": 200,
    "ipeo_budget_500": 500,
    "ipeo_budget_1000": 1000,
}
IPEO_BUDGET_SELECT_METHOD = "ipeo_budget_select"
IPEO_BUDGET_SELECT_SOURCE_VAL_METHOD = "ipeo_budget_select_source_val"
IPEO_BUDGET_SELECTOR_METHODS = {IPEO_BUDGET_SELECT_METHOD, IPEO_BUDGET_SELECT_SOURCE_VAL_METHOD}
IPEO_EXPAND_500_SOURCE_VAL_METHOD = "ipeo_expand_500_source_val"
IPEO_EXPAND_METHODS = {IPEO_EXPAND_500_SOURCE_VAL_METHOD}
IPEO_METHODS = IPEO_COMPOSED_METHODS | set(IPEO_BUDGET_METHODS) | IPEO_BUDGET_SELECTOR_METHODS | IPEO_EXPAND_METHODS | {"ipeo_select_existing", "ipeo_composed_vs_existing"}
FIXED_POOL_METHODS = {
    "original",
    "source_average",
    "pooled_source",
    "worst_source_robust",
    "random_search",
    "target_only_bo_fixed_pool",
    "asha_fixed_pool",
    "promptbridge_emulation",
    "best_source_transfer",
    "ipeo_zero",
    "ipeo_no_generic",
    "ipeo_no_cost",
    "ipeo_no_generic_no_cost",
    "ipeo_budget_200",
    "ipeo_budget_500",
    "ipeo_budget_1000",
    "ipeo_budget_select",
    "ipeo_budget_select_source_val",
    "ipeo_expand_500_source_val",
    "ipeo_select_existing",
    "ipeo_composed_vs_existing",
}
OFFICIAL_METHOD_ALIASES = {
    "gepa": "gepa",
    "mipro": "miprov2",
    "miprov2": "miprov2",
    "capo": "capo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live OpenAI IPEO benchmark")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k"])
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--num_prompts", type=int, default=20)
    parser.add_argument("--num_examples", type=int, default=24)
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress", choices=["off", "rich", "tqdm", "both"], default="both")
    parser.add_argument("--artifact_dir", default="artifacts/gpt41mini_benchmark")
    parser.add_argument("--cache_dir", default="artifacts/gpt41mini_benchmark/cache")
    parser.add_argument("--cost_log", default="artifacts/gpt41mini_benchmark/costs/run.jsonl")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_color", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout_seconds", type=int, default=240)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--dspy_auto", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--dspy_program", choices=["auto", "predict", "cot"], default="auto")
    parser.add_argument("--dspy_train_examples", type=int, default=16)
    parser.add_argument("--dspy_val_examples", type=int, default=16)
    parser.add_argument("--dspy_max_metric_calls", type=int, default=None)
    parser.add_argument("--dspy_max_bootstrapped_demos", type=int, default=4)
    parser.add_argument("--dspy_max_labeled_demos", type=int, default=4)
    parser.add_argument("--dspy_max_tokens", type=int, default=None)
    return parser.parse_args()


def normalize_methods(values: list[str]) -> tuple[set[str], set[str]]:
    raw: list[str] = []
    for value in values:
        raw.extend(part.strip().lower() for part in value.split(",") if part.strip())
    if not raw or "all" in raw:
        return set(FIXED_POOL_METHODS), set(OFFICIAL_METHOD_ALIASES.values())
    fixed: set[str] = set()
    official: set[str] = set()
    unknown: list[str] = []
    for method in raw:
        if method in FIXED_POOL_METHODS:
            fixed.add(method)
        elif method in OFFICIAL_METHOD_ALIASES:
            official.add(OFFICIAL_METHOD_ALIASES[method])
        else:
            unknown.append(method)
    if unknown:
        supported = sorted(FIXED_POOL_METHODS | set(OFFICIAL_METHOD_ALIASES))
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}. Supported: {', '.join(supported)}")
    return fixed, official


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
        prompt_tokens_by_model={"openai": len(selection.prompt_text.split())},
        estimated_deployment_cost={"openai": len(selection.prompt_text.split()) * 0.0004 / 1000},
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
    fixed_methods, official_methods = normalize_methods(args.methods)
    artifact_dir = Path(args.artifact_dir)
    settings = ProgressSettings(mode=args.progress, quiet=args.quiet, no_color=args.no_color)
    reporter = RichRunReporter(settings)
    models = build_openai_environments(
        args.model,
        count=4,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    model_ids = [model.model_id for model in models]
    fold_target = model_ids[-1]
    source_models = model_ids[:-1]
    model_by_id = {model.model_id: model for model in models}
    fold_id = f"target-{fold_target}"
    run_id = stable_hash(
        {"tasks": args.tasks, "model": args.model, "methods": sorted(fixed_methods | official_methods), "seed": args.seed},
        prefix="run-openai-",
    )
    cache = ResponseCache(args.cache_dir)
    cost_ledger = CostLedger(args.cost_log)
    invariant_config = InvariantScorerConfig(n_bootstrap=20)
    all_transfer_rows: list[dict[str, object]] = []
    all_comparison_rows: list[dict[str, object]] = []
    all_method_summary_rows: list[dict[str, object]] = []
    all_access_rows: list[dict[str, object]] = []

    write_jsonl(artifact_dir / "stats" / "optional_baselines.jsonl", optional_baseline_statuses())
    official_records = [record for record in official_optimizer_records() if record.name in official_methods]
    write_jsonl(artifact_dir / "stats" / "requested_official_methods.jsonl", official_records)
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
        generation_config = GenerationConfig(
            temperature=args.temperature,
            max_tokens=clamp_openai_max_output_tokens(args.max_tokens if args.max_tokens is not None else task.max_tokens),
        )
        reporter.status(f"Task {task.task_id}: building frozen pool")
        pool, edits = build_frozen_pool(task.task_id, num_prompts=args.num_prompts, seed=args.seed, artifact_dir=artifact_dir)
        train_examples = task.load_split("opt", args.num_examples)
        val_examples = task.load_split("val", args.num_examples)
        test_examples = task.load_split("test", args.num_examples)

        budget_method_names = fixed_methods & set(IPEO_BUDGET_METHODS)
        candidate_budget_method_names = set(budget_method_names)
        if fixed_methods & IPEO_BUDGET_SELECTOR_METHODS:
            candidate_budget_method_names.update(IPEO_BUDGET_METHODS)
        if IPEO_EXPAND_500_SOURCE_VAL_METHOD in fixed_methods:
            candidate_budget_method_names.add("ipeo_budget_500")
        full_ipeo_method_names = fixed_methods & (IPEO_COMPOSED_METHODS | {"ipeo_select_existing", "ipeo_composed_vs_existing"})
        source_validation_method_names = fixed_methods & {
            "source_average",
            "pooled_source",
            "worst_source_robust",
            "asha_fixed_pool",
            "promptbridge_emulation",
            "best_source_transfer",
        }
        target_validation_method_names = fixed_methods & {"target_only_bo_fixed_pool"}
        budget_plans = {
            method_name: plan_budgeted_source_subset(
                pool=pool,
                train_examples=train_examples,
                source_model_ids=source_models,
                budget=budget,
                seed=args.seed,
            )
            for method_name, budget in IPEO_BUDGET_METHODS.items()
            if method_name in candidate_budget_method_names
        }
        source_train_pool = pool
        source_train_examples_for_eval = train_examples
        source_train_model_ids = source_models
        if budget_plans and not full_ipeo_method_names:
            prompt_ids = {prompt_id for plan in budget_plans.values() for prompt_id in plan.prompt_ids}
            example_ids = {example_id for plan in budget_plans.values() for example_id in plan.example_ids}
            model_ids_for_budget = {model_id for plan in budget_plans.values() for model_id in plan.source_model_ids}
            source_train_pool = [prompt for prompt in pool if prompt.prompt_id in prompt_ids]
            source_train_examples_for_eval = [example for example in train_examples if example.example_id in example_ids]
            source_train_model_ids = [model_id for model_id in source_models if model_id in model_ids_for_budget]

        pool_train_results = []
        if full_ipeo_method_names or budget_plans:
            reporter.status(f"Task {task.task_id}: evaluating source train split")
            pool_train_results = evaluate_pool(
                run_id=run_id,
                task=task,
                models=[model_by_id[model_id] for model_id in source_train_model_ids],
                pool=source_train_pool,
                examples=source_train_examples_for_eval,
                cache=cache,
                cost_ledger=cost_ledger,
                generation_config=generation_config,
                method="fixed_pool",
                fold_id=fold_id,
                seed=args.seed,
                phase="baseline_optimization",
                artifact_path=artifact_dir / "eval_results" / f"{task.task_id}_pool_train.jsonl",
                show_tqdm=settings.use_tqdm,
                workers=args.workers,
            )
        validation_model_ids: list[str] = []
        if source_validation_method_names:
            validation_model_ids.extend(source_models)
        if target_validation_method_names:
            validation_model_ids.append(fold_target)

        pool_val_results = []
        if validation_model_ids:
            reporter.status(f"Task {task.task_id}: evaluating validation split")
            pool_val_results = evaluate_pool(
                run_id=run_id,
                task=task,
                models=[model_by_id[model_id] for model_id in validation_model_ids],
                pool=pool,
                examples=val_examples,
                cache=cache,
                cost_ledger=cost_ledger,
                generation_config=generation_config,
                method="fixed_pool",
                fold_id=fold_id,
                seed=args.seed,
                phase="calibration",
                artifact_path=artifact_dir / "eval_results" / f"{task.task_id}_pool_val.jsonl",
                show_tqdm=settings.use_tqdm,
                workers=args.workers,
            )
        pool_results = pool_train_results + pool_val_results

        selections: list[MethodSelection] = []
        ipeo_prompts: list[PromptCandidate] = []
        invariant_table = []
        ipeo_composed_prompts: dict[str, PromptCandidate] = {}
        ipeo_existing_selection: MethodSelection | None = None
        ipeo_comparison_selection: MethodSelection | None = None
        budgeted_source_calls_by_method: dict[str, int] = {}
        budgeted_source_train_calls_by_method: dict[str, int] = {}
        budgeted_source_validation_calls_by_method: dict[str, int] = {}
        budgeted_dollars_by_method: dict[str, float] = {}
        budgeted_prompt_candidates: list[BudgetedPromptCandidate] = []
        budgeted_eval_rows_by_method: dict[str, list[EvalResult]] = {}
        ipeo_methods = fixed_methods & IPEO_METHODS
        if ipeo_methods:
            if full_ipeo_method_names:
                reporter.status(f"Task {task.task_id}: estimating invariant edit effects from {args.model} source environments")
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
                ipeo_variant_configs = {
                    "ipeo_zero": {"exclude_generic": False, "exclude_edit_types": set()},
                    "ipeo_no_generic": {"exclude_generic": True, "exclude_edit_types": set()},
                    "ipeo_no_cost": {"exclude_generic": False, "exclude_edit_types": {"cost_reduction"}},
                    "ipeo_no_generic_no_cost": {"exclude_generic": True, "exclude_edit_types": {"cost_reduction"}},
                }
                composed_methods_to_build = set(fixed_methods & IPEO_COMPOSED_METHODS)
                if "ipeo_composed_vs_existing" in fixed_methods:
                    composed_methods_to_build.add("ipeo_zero")
                for method_name in sorted(composed_methods_to_build):
                    variant = ipeo_variant_configs[method_name]
                    ipeo_prompt, ipeo_selection = select_zero_target_prompt(
                        task_id=task.task_id,
                        seed_prompt=pool[0],
                        edits=edits,
                        invariant_table=invariant_table,
                        fold_id=fold_id,
                        target_model=fold_target,
                        source_models=source_models,
                        max_edits_per_prompt=invariant_config.max_edits_per_prompt,
                        min_sign_agreement=invariant_config.min_sign_agreement,
                        min_lcb=-0.05,
                        exclude_generic=variant["exclude_generic"],
                        exclude_edit_types=variant["exclude_edit_types"],
                        method_name=method_name,
                    )
                    ipeo_prompts.append(ipeo_prompt)
                    ipeo_composed_prompts[method_name] = ipeo_prompt
                    if method_name in fixed_methods:
                        selections.append(ipeo_selection)
                if fixed_methods & {"ipeo_select_existing", "ipeo_composed_vs_existing"}:
                    ipeo_existing_selection = select_existing_prompt_by_invariant_score(
                        task_id=task.task_id,
                        pool=pool,
                        invariant_table=invariant_table,
                        fold_id=fold_id,
                        target_model=fold_target,
                        source_models=source_models,
                    )
                    if "ipeo_select_existing" in fixed_methods:
                        selections.append(ipeo_existing_selection)
                if "ipeo_composed_vs_existing" in fixed_methods and ipeo_existing_selection is not None:
                    ipeo_comparison_selection = select_composed_vs_existing_prompt(
                        task_id=task.task_id,
                        composed_prompt=ipeo_composed_prompts["ipeo_zero"],
                        existing_selection=ipeo_existing_selection,
                        pool=pool,
                        invariant_table=invariant_table,
                        fold_id=fold_id,
                        target_model=fold_target,
                        source_models=source_models,
                    )
                    selections.append(ipeo_comparison_selection)
            for method_name, budget in sorted(IPEO_BUDGET_METHODS.items(), key=lambda item: item[1]):
                if method_name not in candidate_budget_method_names:
                    continue
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
                ipeo_prompt, ipeo_selection = select_zero_target_prompt(
                    task_id=task.task_id,
                    seed_prompt=pool[0],
                    edits=edits,
                    invariant_table=budget_table,
                    fold_id=fold_id,
                    target_model=fold_target,
                    source_models=subset.source_model_ids,
                    max_edits_per_prompt=invariant_config.max_edits_per_prompt,
                    min_sign_agreement=0.0,
                    min_lcb=-0.10,
                    method_name=method_name,
                )
                ipeo_prompts.append(ipeo_prompt)
                if method_name in fixed_methods:
                    selections.append(ipeo_selection)
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
                        prompt=ipeo_prompt,
                        selection=ipeo_selection,
                        invariant_table=budget_table,
                    )
                )
            unique_train_eval_rows = {
                (row.model_id, row.prompt_id, row.example_id): row
                for candidate in budgeted_prompt_candidates
                for row in budgeted_eval_rows_by_method.get(candidate.method, [])
            }
            if IPEO_BUDGET_SELECT_METHOD in fixed_methods:
                budget_choice = select_budgeted_prompt(
                    candidates=budgeted_prompt_candidates,
                    task_id=task.task_id,
                    fold_id=fold_id,
                    target_model=fold_target,
                    method_name=IPEO_BUDGET_SELECT_METHOD,
                )
                selections.append(budget_choice.selection)
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
            if IPEO_BUDGET_SELECT_SOURCE_VAL_METHOD in fixed_methods:
                validation_prompts = _dedupe_prompts([candidate.prompt for candidate in budgeted_prompt_candidates])
                reporter.status(f"Task {task.task_id}: validating budgeted IPEO candidates on source validation split")
                budgeted_source_validation_results = evaluate_pool(
                    run_id=run_id,
                    task=task,
                    models=[model_by_id[model_id] for model_id in source_models],
                    pool=validation_prompts,
                    examples=val_examples,
                    cache=cache,
                    cost_ledger=cost_ledger,
                    generation_config=generation_config,
                    method="fixed_pool",
                    fold_id=fold_id,
                    seed=args.seed,
                    phase="calibration",
                    artifact_path=artifact_dir / "eval_results" / f"{task.task_id}_{IPEO_BUDGET_SELECT_SOURCE_VAL_METHOD}_source_val.jsonl",
                    show_tqdm=settings.use_tqdm,
                    workers=args.workers,
                )
                pool_results.extend(budgeted_source_validation_results)
                budget_val_choice = select_budgeted_prompt_by_source_validation(
                    candidates=budgeted_prompt_candidates,
                    validation_results=budgeted_source_validation_results,
                    task_id=task.task_id,
                    fold_id=fold_id,
                    target_model=fold_target,
                    method_name=IPEO_BUDGET_SELECT_SOURCE_VAL_METHOD,
                )
                selections.append(budget_val_choice.selection)
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
            if IPEO_EXPAND_500_SOURCE_VAL_METHOD in fixed_methods:
                base_candidate = next(
                    candidate
                    for candidate in budgeted_prompt_candidates
                    if candidate.method == "ipeo_budget_500"
                )
                expanded_candidates = build_expanded_prompt_candidates(
                    task_id=task.task_id,
                    seed_prompt=pool[0],
                    pool=pool,
                    edits=edits,
                    invariant_table=base_candidate.invariant_table,
                    method_name=IPEO_EXPAND_500_SOURCE_VAL_METHOD,
                    max_candidates=8,
                    max_edits_per_prompt=invariant_config.max_edits_per_prompt,
                    min_sign_agreement=0.0,
                    min_lcb=-0.10,
                )
                reporter.status(f"Task {task.task_id}: validating expanded IPEO candidates on source validation split")
                expanded_source_validation_results = evaluate_pool(
                    run_id=run_id,
                    task=task,
                    models=[model_by_id[model_id] for model_id in source_models],
                    pool=expanded_candidates,
                    examples=val_examples,
                    cache=cache,
                    cost_ledger=cost_ledger,
                    generation_config=generation_config,
                    method="fixed_pool",
                    fold_id=fold_id,
                    seed=args.seed,
                    phase="calibration",
                    artifact_path=artifact_dir / "eval_results" / f"{task.task_id}_{IPEO_EXPAND_500_SOURCE_VAL_METHOD}_source_val.jsonl",
                    show_tqdm=settings.use_tqdm,
                    workers=args.workers,
                )
                pool_results.extend(expanded_source_validation_results)
                expanded_choice = select_expanded_prompt_by_source_validation(
                    candidates=expanded_candidates,
                    validation_results=expanded_source_validation_results,
                    task_id=task.task_id,
                    fold_id=fold_id,
                    target_model=fold_target,
                    source_models=source_models,
                    method_name=IPEO_EXPAND_500_SOURCE_VAL_METHOD,
                )
                selections.append(expanded_choice.selection)
                ipeo_prompts.append(expanded_choice.prompt)
                train_rows = budgeted_eval_rows_by_method.get("ipeo_budget_500", [])
                unique_val_eval_rows = {
                    (row.model_id, row.prompt_id, row.example_id): row
                    for row in expanded_source_validation_results
                }
                train_calls = len(train_rows)
                val_calls = len(unique_val_eval_rows)
                budgeted_source_train_calls_by_method[IPEO_EXPAND_500_SOURCE_VAL_METHOD] = train_calls
                budgeted_source_validation_calls_by_method[IPEO_EXPAND_500_SOURCE_VAL_METHOD] = val_calls
                budgeted_source_calls_by_method[IPEO_EXPAND_500_SOURCE_VAL_METHOD] = train_calls + val_calls
                train_cost = cost_ledger.method_estimated_cost_for_eval_results(
                    "fixed_pool",
                    train_rows,
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
                budgeted_dollars_by_method[IPEO_EXPAND_500_SOURCE_VAL_METHOD] = train_cost + val_cost
                write_jsonl(
                    artifact_dir / "stats" / f"{task.task_id}_{IPEO_EXPAND_500_SOURCE_VAL_METHOD}.jsonl",
                    [
                        {
                            "method": IPEO_EXPAND_500_SOURCE_VAL_METHOD,
                            "base_method": "ipeo_budget_500",
                            "prompt_id": expanded_choice.prompt.prompt_id,
                            "validation_score": expanded_choice.validation_score,
                            "source_train_calls": train_calls,
                            "source_validation_calls": val_calls,
                            "source_calls": train_calls + val_calls,
                            "selected_edit_ids": expanded_choice.selection.selected_edit_ids,
                            "candidate_scores": expanded_choice.score_rows,
                        }
                    ],
                )

        if "original" in fixed_methods:
            selections.append(original_prompt(task_id=task.task_id, fold_id=fold_id, target_model=fold_target, source_models=source_models, pool=pool))
        if "source_average" in fixed_methods:
            selections.append(source_average_selection(task_id=task.task_id, fold_id=fold_id, target_model=fold_target, source_models=source_models, pool=pool, eval_results=pool_results))
        if "pooled_source" in fixed_methods:
            selections.append(pooled_source_selection(task_id=task.task_id, fold_id=fold_id, target_model=fold_target, source_models=source_models, pool=pool, eval_results=pool_results))
        if "worst_source_robust" in fixed_methods:
            selections.append(robust_source_selection(task_id=task.task_id, fold_id=fold_id, target_model=fold_target, source_models=source_models, pool=pool, eval_results=pool_results))
        if "random_search" in fixed_methods:
            selections.append(random_search_selection(task_id=task.task_id, fold_id=fold_id, target_model=fold_target, source_models=source_models, pool=pool, seed=args.seed))
        if "target_only_bo_fixed_pool" in fixed_methods:
            selections.append(target_only_bo_selection(task_id=task.task_id, fold_id=fold_id, target_model=fold_target, source_models=source_models, pool=pool, eval_results=pool_results, budget=min(8, len(pool))))
        if "asha_fixed_pool" in fixed_methods:
            selections.append(asha_selection(task_id=task.task_id, fold_id=fold_id, target_model=fold_target, source_models=source_models, pool=pool, eval_results=pool_results))
        if "promptbridge_emulation" in fixed_methods:
            selections.append(promptbridge_emulation(task_id=task.task_id, fold_id=fold_id, target_model=fold_target, source_models=source_models, pool=pool, eval_results=pool_results))
        if "best_source_transfer" in fixed_methods:
            selections.extend(best_source_transfer(task_id=task.task_id, fold_id=fold_id, target_model=fold_target, source_models=source_models, pool=pool, eval_results=pool_results))

        official_results: list[DspyOptimizerResult] = []
        if official_methods:
            dspy_config = DspyOptimizerConfig(
                api_model=args.model,
                fold_id=fold_id,
                target_model=fold_target,
                source_models=source_models,
                seed=args.seed,
                auto=args.dspy_auto,
                program=args.dspy_program,
                train_examples=args.dspy_train_examples,
                val_examples=args.dspy_val_examples,
                max_bootstrapped_demos=args.dspy_max_bootstrapped_demos,
                max_labeled_demos=args.dspy_max_labeled_demos,
                max_metric_calls=args.dspy_max_metric_calls,
                num_threads=args.workers,
                temperature=args.temperature,
                max_tokens=args.dspy_max_tokens if args.dspy_max_tokens is not None else max(128, generation_config.max_tokens),
                max_retries=args.max_retries,
            )
            for method_name in sorted(official_methods):
                if method_name == "capo":
                    official_results.append(
                        DspyOptimizerResult(
                            method="capo",
                            status="skipped",
                            reason="CAPO requires the optional promptolution package and a compatible runner API",
                        )
                    )
                    continue
                reporter.status(f"Task {task.task_id}: running official {method_name} optimizer through DSPy")
                result = run_dspy_optimizer(
                    method=method_name,
                    run_id=run_id,
                    task=task,
                    train_examples=train_examples,
                    val_examples=val_examples,
                    test_examples=test_examples,
                    artifact_dir=artifact_dir,
                    config=dspy_config,
                )
                official_results.append(result)
                if result.selection is not None:
                    selections.append(result.selection)
            write_jsonl(artifact_dir / "stats" / f"{task.task_id}_official_optimizer_results.jsonl", official_results)

        write_jsonl(artifact_dir / "stats" / f"{task.task_id}_method_selections.jsonl", selections)
        if not selections:
            reporter.status(f"Task {task.task_id}: no runnable methods completed; skipping transfer table")
            continue
        official_method_names = {result.method for result in official_results if result.selection is not None}
        official_eval_results = [row for result in official_results for row in result.eval_results]
        comparison_prompts = list(ipeo_composed_prompts.values()) if "ipeo_composed_vs_existing" in fixed_methods else []
        prompt_pool = pool + ipeo_prompts + comparison_prompts
        final_prompts = _dedupe_prompts(
            [_selection_to_prompt(selection, prompt_pool) for selection in selections if selection.method not in official_method_names]
        )
        final_prompts = _dedupe_prompts(final_prompts + comparison_prompts)
        reporter.status(f"Task {task.task_id}: evaluating selected methods on target {args.model} environment")
        final_results = evaluate_pool(
            run_id=run_id,
            task=task,
            models=[model_by_id[fold_target]],
            pool=final_prompts,
            examples=test_examples,
            cache=cache,
            cost_ledger=cost_ledger,
            generation_config=generation_config,
            method="selected_methods",
            fold_id=fold_id,
            seed=args.seed,
            phase="final_test",
            artifact_path=artifact_dir / "eval_results" / f"{task.task_id}_selected_test.jsonl",
            show_tqdm=settings.use_tqdm,
            workers=args.workers,
        )
        combined_final_results = final_results + official_eval_results
        reporter.status(f"Task {task.task_id}: evaluating locked target test oracle for regret reporting")
        pool_test_results = evaluate_pool(
            run_id=run_id,
            task=task,
            models=[model_by_id[fold_target]],
            pool=pool,
            examples=test_examples,
            cache=cache,
            cost_ledger=cost_ledger,
            generation_config=generation_config,
            method="fixed_pool",
            fold_id=fold_id,
            seed=args.seed,
            phase="final_test",
            artifact_path=artifact_dir / "eval_results" / f"{task.task_id}_pool_test.jsonl",
            show_tqdm=settings.use_tqdm,
            workers=args.workers,
        )
        source_average_prompt_id = next((selection.prompt_id for selection in selections if selection.method == "source_average"), selections[0].prompt_id)
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
        final_cost_by_prompt = {
            prompt.prompt_id: cost_ledger.method_estimated_cost(
                "selected_methods",
                run_id=run_id,
                phase="final_test",
                model_ids={fold_target},
                task_id=task.task_id,
                prompt_ids={prompt.prompt_id},
            )
            for prompt in final_prompts
        }
        target_bo_prompt_ids = {prompt.prompt_id for prompt in pool[: min(8, len(pool))]}
        target_pool_val_cost = cost_ledger.method_estimated_cost(
            "fixed_pool",
            run_id=run_id,
            phase="calibration",
            model_ids={fold_target},
            task_id=task.task_id,
            prompt_ids=target_bo_prompt_ids,
        )
        source_calls_by_method = {
            method: source_validation_calls
            for method in {
                "source_average",
                "pooled_source",
                "worst_source_robust",
                "asha_fixed_pool",
                "promptbridge_emulation",
            }
        }
        source_calls_by_method.update(
            {
                method: source_train_calls
                for method in {
                    "ipeo_zero",
                    "ipeo_no_generic",
                    "ipeo_no_cost",
                    "ipeo_no_generic_no_cost",
                    "ipeo_select_existing",
                    "ipeo_composed_vs_existing",
                }
            }
        )
        source_calls_by_method.update(budgeted_source_calls_by_method)
        source_train_calls_by_method = {
            method: source_train_calls
            for method in {
                "ipeo_zero",
                "ipeo_no_generic",
                "ipeo_no_cost",
                "ipeo_no_generic_no_cost",
                "ipeo_select_existing",
                "ipeo_composed_vs_existing",
            }
        }
        source_train_calls_by_method.update(budgeted_source_train_calls_by_method)
        source_validation_calls_by_method = {
            method: source_validation_calls
            for method in {
                "source_average",
                "pooled_source",
                "worst_source_robust",
                "asha_fixed_pool",
                "promptbridge_emulation",
            }
        }
        source_validation_calls_by_method.update(budgeted_source_validation_calls_by_method)
        dollars_by_method = {
            method: source_validation_cost
            for method in {
                "source_average",
                "pooled_source",
                "worst_source_robust",
                "asha_fixed_pool",
                "promptbridge_emulation",
            }
        }
        dollars_by_method.update(
            {
                method: source_train_cost
                for method in {
                    "ipeo_zero",
                    "ipeo_no_generic",
                    "ipeo_no_cost",
                    "ipeo_no_generic_no_cost",
                    "ipeo_select_existing",
                    "ipeo_composed_vs_existing",
                }
            }
        )
        dollars_by_method.update(budgeted_dollars_by_method)
        for model_id in source_models:
            source_calls_by_method[f"best_source_transfer:{model_id}"] = len(pool) * len(val_examples)
            dollars_by_method[f"best_source_transfer:{model_id}"] = (
                cost_ledger.method_estimated_cost(
                    "fixed_pool",
                    run_id=run_id,
                    phase="calibration",
                    model_ids={model_id},
                    task_id=task.task_id,
                )
            )
        target_calls_by_method = {"target_only_bo_fixed_pool": min(8, len(pool)) * len(val_examples)}
        dollars_by_method.update({"target_only_bo_fixed_pool": target_pool_val_cost})
        for result in official_results:
            if result.selection is None:
                continue
            target_calls_by_method[result.method] = result.optimization_calls
            dollars_by_method[result.method] = result.total_dollars
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
            target_model=fold_target,
            source_models=source_models,
            selections=selections,
            final_results=combined_final_results + pool_test_results,
            pool_results=pool_test_results,
            pool=pool,
            source_average_prompt_id=source_average_prompt_id,
            source_calls_by_method=source_calls_by_method,
            target_calls_by_method=target_calls_by_method,
            dollars_by_method=dollars_by_method,
            method_access_by_method=access_rows_by_method(task_access_rows),
        )
        all_transfer_rows.extend(rows)
        method_summary_rows = build_method_summary_rows(
            run_id=run_id,
            task_id=task.task_id,
            target_model=fold_target,
            source_models=source_models,
            selections=selections,
            transfer_rows=rows,
            pool_results=pool_results,
            final_results=combined_final_results,
            cost_log_path=args.cost_log,
        )
        all_method_summary_rows.extend(method_summary_rows)
        task_comparison_rows: list[dict[str, object]] = []
        if invariant_table and ipeo_existing_selection is not None and "ipeo_zero" in ipeo_composed_prompts:
            existing_prompt = next(prompt for prompt in pool if prompt.prompt_id == ipeo_existing_selection.prompt_id)
            task_comparison_rows.append(
                build_composed_vs_existing_row(
                    task_id=task.task_id,
                    fold_id=fold_id,
                    target_model=fold_target,
                    composed_method="ipeo_zero",
                    composed_prompt=ipeo_composed_prompts["ipeo_zero"],
                    existing_selection=ipeo_existing_selection,
                    existing_prompt=existing_prompt,
                    comparison_selection=ipeo_comparison_selection,
                    invariant_table=invariant_table,
                    eval_results=combined_final_results + pool_test_results,
                )
            )
            write_jsonl(artifact_dir / "stats" / f"{task.task_id}_ipeo_composed_vs_existing.jsonl", task_comparison_rows)
            all_comparison_rows.extend(task_comparison_rows)
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

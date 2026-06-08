"""Live OpenAI benchmark runner for IPEO."""

from __future__ import annotations

import argparse
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
from ipeo.models.openai_adapter import build_openai_environments, clamp_openai_max_output_tokens
from ipeo.prompts.pool_builder import build_frozen_pool
from ipeo.runners.progress import ProgressSettings, RichRunReporter
from ipeo.stats.ipeo_compare import build_composed_vs_existing_row
from ipeo.stats.method_summary import aggregate_method_summary_rows, build_method_summary_rows
from ipeo.stats.regret import build_transfer_rows
from ipeo.tasks.adapters import get_tasks

IPEO_COMPOSED_METHODS = {
    "ipeo_zero",
    "ipeo_no_generic",
    "ipeo_no_cost",
    "ipeo_no_generic_no_cost",
}
IPEO_METHODS = IPEO_COMPOSED_METHODS | {"ipeo_select_existing", "ipeo_composed_vs_existing"}
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

    write_jsonl(artifact_dir / "stats" / "optional_baselines.jsonl", optional_baseline_statuses())
    official_records = [record for record in official_optimizer_records() if record.name in official_methods]
    write_jsonl(artifact_dir / "stats" / "requested_official_methods.jsonl", official_records)

    for task in get_tasks(args.tasks):
        generation_config = GenerationConfig(
            temperature=args.temperature,
            max_tokens=clamp_openai_max_output_tokens(args.max_tokens if args.max_tokens is not None else task.max_tokens),
        )
        reporter.status(f"Task {task.task_id}: building frozen pool")
        pool, edits = build_frozen_pool(task.task_id, num_prompts=args.num_prompts, seed=args.seed, artifact_dir=artifact_dir)
        val_examples = task.load_split("val", args.num_examples)
        test_examples = task.load_split("test", args.num_examples)

        reporter.status(f"Task {task.task_id}: evaluating {args.model} environments on validation and test splits")
        pool_val_results = evaluate_pool(
            run_id=run_id,
            task=task,
            models=models,
            pool=pool,
            examples=val_examples,
            cache=cache,
            cost_ledger=cost_ledger,
            generation_config=generation_config,
            method="fixed_pool",
            fold_id=fold_id,
            seed=args.seed,
            phase="evaluation",
            artifact_path=artifact_dir / "eval_results" / f"{task.task_id}_pool_val.jsonl",
            show_tqdm=settings.use_tqdm,
            workers=args.workers,
        )
        pool_test_results = evaluate_pool(
            run_id=run_id,
            task=task,
            models=models,
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
        pool_results = pool_val_results + pool_test_results

        selections: list[MethodSelection] = []
        ipeo_prompts: list[PromptCandidate] = []
        invariant_table = []
        ipeo_composed_prompts: dict[str, PromptCandidate] = {}
        ipeo_existing_selection: MethodSelection | None = None
        ipeo_comparison_selection: MethodSelection | None = None
        ipeo_methods = fixed_methods & IPEO_METHODS
        if ipeo_methods:
            reporter.status(f"Task {task.task_id}: estimating invariant edit effects from {args.model} source environments")
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

        write_jsonl(artifact_dir / "stats" / f"{task.task_id}_method_selections.jsonl", selections)
        comparison_prompts = list(ipeo_composed_prompts.values()) if "ipeo_composed_vs_existing" in fixed_methods else []
        prompt_pool = pool + ipeo_prompts + comparison_prompts
        final_prompts = _dedupe_prompts([_selection_to_prompt(selection, prompt_pool) for selection in selections])
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
        source_average_prompt_id = next((selection.prompt_id for selection in selections if selection.method == "source_average"), selections[0].prompt_id)
        source_eval_calls = len(source_models) * len(pool) * len(val_examples)
        source_eval_cost = cost_ledger.method_cost(
            "fixed_pool",
            phase="evaluation",
            model_ids=set(source_models),
            task_id=task.task_id,
        )
        final_cost_by_prompt = {
            prompt.prompt_id: cost_ledger.method_cost(
                "selected_methods",
                phase="final_test",
                model_ids={fold_target},
                task_id=task.task_id,
                prompt_ids={prompt.prompt_id},
            )
            for prompt in final_prompts
        }
        target_bo_prompt_ids = {prompt.prompt_id for prompt in pool[: min(8, len(pool))]}
        target_pool_val_cost = cost_ledger.method_cost(
            "fixed_pool",
            phase="evaluation",
            model_ids={fold_target},
            task_id=task.task_id,
            prompt_ids=target_bo_prompt_ids,
        )
        source_calls_by_method = {
            method: source_eval_calls
            for method in {
                "source_average",
                "pooled_source",
                "worst_source_robust",
                "asha_fixed_pool",
                "ipeo_zero",
                "ipeo_no_generic",
                "ipeo_no_cost",
                "ipeo_no_generic_no_cost",
            }
        }
        source_calls_by_method.update({method: source_eval_calls for method in {"ipeo_select_existing", "ipeo_composed_vs_existing"}})
        dollars_by_method = {
            method: source_eval_cost
            for method in {
                "source_average",
                "pooled_source",
                "worst_source_robust",
                "asha_fixed_pool",
                "ipeo_zero",
                "ipeo_no_generic",
                "ipeo_no_cost",
                "ipeo_no_generic_no_cost",
                "ipeo_select_existing",
                "ipeo_composed_vs_existing",
            }
        }
        for model_id in source_models:
            source_calls_by_method[f"best_source_transfer:{model_id}"] = len(pool) * len(val_examples)
            dollars_by_method[f"best_source_transfer:{model_id}"] = (
                cost_ledger.method_cost(
                    "fixed_pool",
                    phase="evaluation",
                    model_ids={model_id},
                    task_id=task.task_id,
                )
            )
        dollars_by_method.update({"target_only_bo_fixed_pool": target_pool_val_cost})
        for selection in selections:
            dollars_by_method[selection.method] = dollars_by_method.get(selection.method, 0.0) + final_cost_by_prompt.get(selection.prompt_id, 0.0)
        rows = build_transfer_rows(
            task_id=task.task_id,
            fold_id=fold_id,
            target_model=fold_target,
            source_models=source_models,
            selections=selections,
            final_results=final_results + pool_test_results,
            pool_results=pool_test_results,
            pool=pool,
            source_average_prompt_id=source_average_prompt_id,
            source_calls_by_method=source_calls_by_method,
            target_calls_by_method={"target_only_bo_fixed_pool": min(8, len(pool)) * len(val_examples)},
            dollars_by_method=dollars_by_method,
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
            final_results=final_results,
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
                    eval_results=final_results + pool_test_results,
                )
            )
            write_jsonl(artifact_dir / "stats" / f"{task.task_id}_ipeo_composed_vs_existing.jsonl", task_comparison_rows)
            all_comparison_rows.extend(task_comparison_rows)
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

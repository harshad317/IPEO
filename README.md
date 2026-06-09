# IPEO

Runnable MVP for Invariant Prompt-Edit Optimization across black-box model
environments. The default dry run is deterministic and requires no API keys.

## Smoke Test

```bash
python -m pytest
python -m ipeo.runners.run_dry \
  --tasks gsm8k \
  --models mock_openai_a mock_openai_b mock_openai_c mock_openai_d \
  --num_prompts 8 \
  --num_examples 8 \
  --fold_target mock_openai_d
```

## Full Offline Dry Run

```bash
python -m ipeo.runners.run_dry \
  --tasks gsm8k bbh classification extraction_qa \
  --models mock_openai_a mock_openai_b mock_openai_c mock_openai_d \
  --num_prompts 20 \
  --num_examples 24 \
  --fold_target mock_openai_d \
  --cache_dir artifacts/cache \
  --cost_log artifacts/costs/dry_run.jsonl
```

Artifacts are written under `artifacts/` as JSONL and CSV files.

## Benchmark Split Contract

IPEO now treats the fixture splits as a strict benchmark contract:

- `opt` is the **train** split for optimization.
- `val` is the **validation** split for prompt/model selection.
- `test` is the locked **final target evaluation** split.

The runner writes `stats/split_contract.jsonl`, `stats/data_access.csv`, and
`stats/*_data_access.jsonl` so every result states which data each method was
allowed to use. `transfer_regret.csv` also includes columns such as
`benchmark_track`, `selection_access`, `uses_target_validation`,
`uses_target_test_for_selection`, `source_train_calls`,
`source_validation_calls`, `target_validation_calls`, and
`target_optimization_calls`.

The main comparison tracks are:

- `zero_target_transfer`: IPEO methods use source train data and no target
  train/validation examples.
- `source_transfer`: source-selection baselines use source validation data.
- `target_optimization`: GEPA/MIPROv2 and target-only fixed-pool search use
  target train/validation data.

No method is allowed to use target test for selection; target test is evaluated
only after method selection is complete.

`total_dollars` in benchmark tables is a **fair estimated uncached cost** based
on logged token counts and model prices. The cache can reduce what you actually
pay during repeated local runs, but cached source baselines are still charged
fairly in the benchmark reports.

## Analyze A Completed Run

After a benchmark finishes, summarize per-task winners, benchmark tracks,
cost/performance frontier rows, IPEO-vs-baseline deltas, and bootstrap
confidence intervals over task-level paired deltas:

```bash
python -m ipeo.runners.analyze_run \
  --artifact_dir artifacts/gpt41mini_fair_split_v1
```

Use `--bootstrap_samples`, `--bootstrap_seed`, and `--confidence_level` to
control the deterministic bootstrap summary. These intervals are over tasks in
the completed artifact directory; run multiple seeds when you need true
seed-level stability evidence.

Focus on one stress task:

```bash
python -m ipeo.runners.analyze_run \
  --artifact_dir artifacts/gpt41mini_fair_split_v1 \
  --focus_task ifbench_hard
```

This writes:

- `stats/analysis_per_task_winners.csv`
- `stats/analysis_track_summary.csv`
- `stats/analysis_method_task_summary.csv`
- `stats/analysis_ipeo_vs_baselines.csv`
- `stats/analysis_bootstrap_comparisons.csv`
- `stats/analysis_cost_frontier.csv`
- `stats/analysis_budget_select_decisions.csv`
- `stats/analysis_budget_select_summary.csv`

With `--focus_task`, the files get a task suffix such as
`stats/analysis_per_task_winners_ifbench_hard.csv`.

## Multi-Seed Analysis

For stability evidence, run the same benchmark across several seeds into
separate artifact directories:

```bash
for seed in 0 1 2 3 4; do
  python -m ipeo.runners.run_openai \
    --tasks ifbench_hard \
    --model gpt-4.1-mini \
    --num_prompts 30 \
    --num_examples 48 \
    --methods ipeo_budget_200 ipeo_budget_500 ipeo_budget_1000 ipeo_budget_select ipeo_budget_select_source_val miprov2 gepa source_average target_only_bo_fixed_pool best_source_transfer \
    --workers 8 \
    --timeout_seconds 300 \
    --max_retries 6 \
    --dspy_auto light \
    --dspy_program auto \
    --dspy_train_examples 16 \
    --dspy_val_examples 16 \
    --dspy_max_bootstrapped_demos 4 \
    --dspy_max_labeled_demos 4 \
    --dspy_max_tokens 128 \
    --seed "$seed" \
    --progress both \
    --artifact_dir "artifacts/gpt41mini_ifbench_hard_seed_${seed}" \
    --cache_dir "artifacts/gpt41mini_ifbench_hard_seed_${seed}/cache" \
    --cost_log "artifacts/gpt41mini_ifbench_hard_seed_${seed}/costs/run.jsonl"
done
```

Then aggregate the completed runs:

```bash
python -m ipeo.runners.analyze_many \
  --artifact_glob "artifacts/gpt41mini_ifbench_hard_seed_*" \
  --output_dir artifacts/gpt41mini_ifbench_hard_multiseed \
  --focus_task ifbench_hard \
  --bootstrap_samples 5000 \
  --confidence_level 0.95
```

This writes `stats/multi_run_method_summary*.csv`,
`stats/multi_run_ipeo_vs_baselines*.csv`,
`stats/multi_run_cost_frontier*.csv`, and
`stats/multi_run_combined_transfer_rows*.csv`. When a budget selector is
present, it also writes `stats/multi_run_budget_select_decisions*.csv` and
`stats/multi_run_budget_select_summary*.csv`; these show the selector, chosen
budget, best realized budget candidate, selector regret, and chosen/oracle
budget counts across seeds. These reports include mean score, bootstrap
confidence intervals, score win/tie/loss rate, mean calls, mean dollars, and
the aggregate cost/performance frontier.

## Live OpenAI Benchmark

Set `OPENAI_API_KEY` first, then run:

```bash
python -m ipeo.runners.run_openai \
  --tasks gsm8k bbh classification extraction_qa \
  --model gpt-4.1-mini \
  --num_prompts 20 \
  --num_examples 24 \
  --methods all \
  --workers 8 \
  --progress both \
  --artifact_dir artifacts/gpt41mini_benchmark \
  --cache_dir artifacts/gpt41mini_benchmark/cache \
  --cost_log artifacts/gpt41mini_benchmark/costs/run.jsonl
```

`--methods all` runs the implemented fixed-pool methods, IPEO ablations, and
the optional official optimizer wrappers. GEPA and MIPROv2 execute through
DSPy when `dspy`, `optuna`, and `OPENAI_API_KEY` are available. CAPO is still
reported as skipped until a compatible `promptolution` runner is wired.

Useful IPEO ablations:

```bash
--methods ipeo_zero ipeo_budget_200 ipeo_budget_500 ipeo_budget_1000 ipeo_budget_select ipeo_budget_select_source_val ipeo_select_existing ipeo_composed_vs_existing ipeo_no_generic ipeo_no_cost ipeo_no_generic_no_cost source_average pooled_source target_only_bo_fixed_pool
```

`ipeo_budget_200`, `ipeo_budget_500`, and `ipeo_budget_1000` estimate invariant
edits from deterministic source-train subsets capped by the requested source
call budget. They are zero-target transfer variants meant for direct
cost-matched comparison against GEPA/MIPROv2. Actual calls can land slightly
below the named budget because the sampler keeps complete
prompt/example/source-model grids; for example, 30 prompts over 3 source
environments gives 180 calls for `ipeo_budget_200`. When a live run requests
only budgeted IPEO methods, `run_openai` evaluates only the union of those
budget grids instead of the full source-train pool.

`ipeo_budget_select` builds the 200/500/1000 budget candidates and chooses one
using only source-side invariant evidence: selected edit scores, lower
confidence bounds, sign agreement, rank stability, prompt length, and source
call cost. It writes `stats/*_ipeo_budget_select.jsonl` so the budget choice is
auditable. `analyze_run` and `analyze_many` join that audit file with
`transfer_regret.csv` to report whether the selector chose the best realized
budget candidate.

`ipeo_budget_select_source_val` builds the same budget candidates, then
evaluates only those candidate prompts on the source validation split and
selects by held-out source score. It still uses zero target data, but reports
both source-train and source-validation calls in the data-access ledger.

`ipeo_select_existing` scores each frozen-pool prompt by the sum of invariant
scores for its edit vector and selects the best existing prompt.
`ipeo_composed_vs_existing` compares the zero-target composed prompt with that
existing-prompt selector without target leakage, then writes
`stats/*_ipeo_composed_vs_existing.jsonl` and
`stats/ipeo_composed_vs_existing.csv` to report which side actually won on the
held-out target test split.

To run only IPEO plus GEPA/MIPROv2:

```bash
python -m ipeo.runners.run_openai \
  --tasks gsm8k bbh classification extraction_qa ifbench_hard \
  --model gpt-4.1-mini \
  --num_prompts 30 \
  --num_examples 48 \
  --methods ipeo_zero ipeo_budget_200 ipeo_budget_500 ipeo_budget_1000 ipeo_budget_select ipeo_budget_select_source_val ipeo_select_existing ipeo_composed_vs_existing gepa mipro \
  --workers 8 \
  --timeout_seconds 300 \
  --max_retries 6 \
  --dspy_auto light \
  --dspy_program auto \
  --dspy_train_examples 16 \
  --dspy_val_examples 16 \
  --dspy_max_bootstrapped_demos 4 \
  --dspy_max_labeled_demos 4 \
  --dspy_max_tokens 128 \
  --progress both \
  --artifact_dir artifacts/gpt41mini_dspy_methods \
  --cache_dir artifacts/gpt41mini_dspy_methods/cache \
  --cost_log artifacts/gpt41mini_dspy_methods/costs/run.jsonl
```

Use `--dspy_auto medium` or `--dspy_auto heavy` only when you are ready to
spend more optimizer calls. `--dspy_max_metric_calls N` caps GEPA with an
explicit metric-call budget. `--dspy_program auto` uses Chain-of-Thought for
math/reasoning tasks and direct prediction for strict-format tasks.

## IFBench Stress Tests

Use the harder local fixture first. It has compositional constraints for
keyword counts, JSON/CSV exactness, line and paragraph counts, suffix tokens,
forbidden words, and punctuation.

```bash
python -m ipeo.runners.run_openai \
  --tasks ifbench_hard \
  --model gpt-4.1-mini \
  --num_prompts 20 \
  --num_examples 24 \
  --methods ipeo_no_generic_no_cost ipeo_select_existing ipeo_composed_vs_existing source_average pooled_source worst_source_robust asha_fixed_pool best_source_transfer \
  --workers 8 \
  --timeout_seconds 300 \
  --max_retries 6 \
  --max_tokens 160 \
  --progress both \
  --artifact_dir artifacts/gpt41mini_ifbench_hard \
  --cache_dir artifacts/gpt41mini_ifbench_hard/cache \
  --cost_log artifacts/gpt41mini_ifbench_hard/costs/run.jsonl
```

For the official AllenAI IFBench evaluator, clone the upstream repo and point
IPEO at it. The adapter uses prompt-level loose accuracy by default, matching
the upstream reporting note.

```bash
git clone https://github.com/allenai/IFBench.git external/IFBench
export IFBENCH_REPO="$PWD/external/IFBench"
python -m pip install -r "$IFBENCH_REPO/requirements.txt"

python -m ipeo.runners.run_openai \
  --tasks ifbench_official \
  --model gpt-4.1-mini \
  --num_prompts 20 \
  --num_examples 24 \
  --methods ipeo_no_generic_no_cost ipeo_select_existing ipeo_composed_vs_existing source_average pooled_source worst_source_robust asha_fixed_pool best_source_transfer \
  --workers 8 \
  --timeout_seconds 300 \
  --max_retries 6 \
  --max_tokens 512 \
  --progress both \
  --artifact_dir artifacts/gpt41mini_ifbench_official \
  --cache_dir artifacts/gpt41mini_ifbench_official/cache \
  --cost_log artifacts/gpt41mini_ifbench_official/costs/run.jsonl
```

Set `IFBENCH_DATA_PATH=/path/to/IFBench_test.jsonl` if the data file is outside
the cloned repo. Set `IFBENCH_EVAL_MODE=strict` to use strict official scoring.

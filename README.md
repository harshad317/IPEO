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
--methods ipeo_zero ipeo_select_existing ipeo_composed_vs_existing ipeo_no_generic ipeo_no_cost ipeo_no_generic_no_cost source_average pooled_source target_only_bo_fixed_pool
```

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
  --methods ipeo_zero ipeo_select_existing ipeo_composed_vs_existing gepa mipro \
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

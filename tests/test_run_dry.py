from __future__ import annotations

import argparse
from pathlib import Path

from ipeo.baselines.optional_wrappers import optional_baseline_statuses
from ipeo.core.io import read_jsonl
from ipeo.runners.run_dry import run


def test_optional_baseline_statuses_are_explicit() -> None:
    statuses = optional_baseline_statuses()
    assert {status.name for status in statuses} == {"gepa", "miprov2", "capo"}
    assert all(status.package for status in statuses)


def test_dry_run_writes_core_artifacts(tmp_path: Path) -> None:
    args = argparse.Namespace(
        tasks=["gsm8k"],
        models=["mock_openai_a", "mock_openai_b", "mock_openai_c", "mock_openai_d"],
        num_prompts=8,
        num_examples=4,
        fold_target="mock_openai_d",
        cache_dir=str(tmp_path / "cache"),
        cost_log=str(tmp_path / "artifacts" / "costs" / "dry_run.jsonl"),
        artifact_dir=str(tmp_path / "artifacts"),
        progress="off",
        quiet=True,
        no_color=True,
        seed=0,
    )
    rows = run(args)
    artifact_dir = tmp_path / "artifacts"
    assert rows
    assert (artifact_dir / "prompts" / "gsm8k_pool.jsonl").exists()
    assert (artifact_dir / "edits" / "gsm8k_edits.jsonl").exists()
    assert (artifact_dir / "eval_results" / "gsm8k_pool_val.jsonl").exists()
    assert (artifact_dir / "stats" / "transfer_regret.csv").exists()
    assert (artifact_dir / "stats" / "data_access.csv").exists()
    assert (artifact_dir / "stats" / "split_contract.jsonl").exists()
    assert (artifact_dir / "stats" / "gsm8k_data_access.jsonl").exists()
    assert (artifact_dir / "stats" / "ipeo_composed_vs_existing.csv").exists()
    assert (artifact_dir / "stats" / "method_summary.csv").exists()
    assert (artifact_dir / "stats" / "gsm8k_ipeo_composed_vs_existing.jsonl").exists()
    invariant_rows = read_jsonl(artifact_dir / "stats" / "gsm8k_invariant_edits.jsonl")
    assert invariant_rows
    assert any(row["method"] == "ipeo_zero" for row in rows)
    assert any(row["method"] == "ipeo_budget_200" for row in rows)
    assert any(row["method"] == "ipeo_budget_select" for row in rows)
    assert any(row["method"] == "ipeo_budget_select_source_val" for row in rows)
    assert any(row["method"] == "ipeo_expand_500_source_val" for row in rows)
    assert any(row["method"] == "ipeo_select_existing" for row in rows)
    assert any(row["method"] == "ipeo_composed_vs_existing" for row in rows)
    assert all(row["uses_target_test_for_selection"] is False for row in rows)
    assert {row["benchmark_track"] for row in rows} >= {"zero_target_transfer", "target_optimization"}
    access_rows = read_jsonl(artifact_dir / "stats" / "gsm8k_data_access.jsonl")
    ipeo_access = next(row for row in access_rows if row["method"] == "ipeo_zero")
    assert ipeo_access["train_access"] == "source_train"
    assert ipeo_access["uses_target_validation"] is False
    target_bo_access = next(row for row in access_rows if row["method"] == "target_only_bo_fixed_pool")
    assert target_bo_access["validation_access"] == "target_validation"
    budget_access = next(row for row in access_rows if row["method"] == "ipeo_budget_200")
    assert budget_access["source_train_calls"] <= 200
    budget_select_access = next(row for row in access_rows if row["method"] == "ipeo_budget_select")
    assert budget_select_access["uses_target_validation"] is False
    assert budget_select_access["source_train_calls"] >= budget_access["source_train_calls"]
    budget_source_val_access = next(row for row in access_rows if row["method"] == "ipeo_budget_select_source_val")
    assert budget_source_val_access["train_access"] == "source_train"
    assert budget_source_val_access["validation_access"] == "source_validation"
    assert budget_source_val_access["selection_access"] == "source_train,source_validation"
    assert budget_source_val_access["uses_target_validation"] is False
    assert budget_source_val_access["source_validation_calls"] > 0
    expanded_access = next(row for row in access_rows if row["method"] == "ipeo_expand_500_source_val")
    assert expanded_access["train_access"] == "source_train"
    assert expanded_access["validation_access"] == "source_validation"
    assert expanded_access["uses_target_validation"] is False
    assert expanded_access["source_validation_calls"] > 0
    assert read_jsonl(artifact_dir / "stats" / "gsm8k_ipeo_budget_select.jsonl")
    assert read_jsonl(artifact_dir / "stats" / "gsm8k_ipeo_budget_select_source_val.jsonl")
    assert read_jsonl(artifact_dir / "stats" / "gsm8k_ipeo_expand_500_source_val.jsonl")
    method_summary = (artifact_dir / "stats" / "method_summary.csv").read_text(encoding="utf-8")
    assert "optimization" in method_summary
    assert "test" in method_summary


def test_ifbench_dry_run(tmp_path: Path) -> None:
    args = argparse.Namespace(
        tasks=["ifbench"],
        models=["mock_openai_a", "mock_openai_b", "mock_openai_c", "mock_openai_d"],
        num_prompts=6,
        num_examples=3,
        fold_target="mock_openai_d",
        cache_dir=str(tmp_path / "cache"),
        cost_log=str(tmp_path / "artifacts" / "costs" / "dry_run.jsonl"),
        artifact_dir=str(tmp_path / "artifacts"),
        progress="off",
        quiet=True,
        no_color=True,
        seed=0,
    )
    rows = run(args)
    assert rows
    assert any(row["task_id"] == "ifbench" for row in rows)


def test_ifbench_hard_dry_run(tmp_path: Path) -> None:
    args = argparse.Namespace(
        tasks=["ifbench_hard"],
        models=["mock_openai_a", "mock_openai_b", "mock_openai_c", "mock_openai_d"],
        num_prompts=20,
        num_examples=8,
        fold_target="mock_openai_d",
        cache_dir=str(tmp_path / "cache"),
        cost_log=str(tmp_path / "artifacts" / "costs" / "dry_run.jsonl"),
        artifact_dir=str(tmp_path / "artifacts"),
        progress="off",
        quiet=True,
        no_color=True,
        seed=0,
    )
    rows = run(args)
    assert rows
    assert any(row["task_id"] == "ifbench_hard" for row in rows)
    scores = {row["method"]: row["target_score"] for row in rows}
    assert scores["ipeo_zero"] > scores["source_average"]

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
    invariant_rows = read_jsonl(artifact_dir / "stats" / "gsm8k_invariant_edits.jsonl")
    assert invariant_rows
    assert any(row["method"] == "ipeo_zero" for row in rows)


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

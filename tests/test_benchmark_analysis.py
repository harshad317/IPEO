from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from ipeo.core.io import read_csv, write_csv, write_jsonl
from ipeo.runners.analyze_run import run
from ipeo.stats.benchmark_analysis import analyze_artifact_dir


def test_analyze_artifact_dir_writes_per_task_outputs(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    rows = [
        _row("gsm8k", "ipeo_select_existing", "zero_target_transfer", 0.95, 100, 0, 0.10),
        _row("gsm8k", "miprov2", "target_optimization", 0.97, 0, 40, 0.05),
        _row("gsm8k", "source_average", "source_transfer", 0.90, 80, 0, 0.04),
        _row("ifbench_hard", "ipeo_select_existing", "zero_target_transfer", 0.80, 100, 0, 0.10),
        _row("ifbench_hard", "miprov2", "target_optimization", 0.75, 0, 40, 0.05),
        _row("ifbench_hard", "best_source_transfer:model_a", "source_transfer", 0.78, 30, 0, 0.03),
    ]
    write_csv(artifact_dir / "stats" / "transfer_regret.csv", rows)

    outputs = analyze_artifact_dir(artifact_dir)

    winners = {row["task_id"]: row for row in outputs["per_task_winners"]}
    assert winners["gsm8k"]["cheapest_best_method"] == "miprov2"
    assert winners["ifbench_hard"]["best_ipeo_method"] == "ipeo_select_existing"
    deltas = outputs["ipeo_vs_baselines"]
    assert any(row["task_id"] == "ifbench_hard" and row["baseline_method"] == "miprov2" and row["winner"] == "ipeo" for row in deltas)
    bootstrap_rows = outputs["bootstrap_comparisons"]
    assert any(row["ipeo_method"] == "ipeo_select_existing" and row["baseline_method"] == "miprov2" for row in bootstrap_rows)
    assert (artifact_dir / "stats" / "analysis_per_task_winners.csv").exists()
    assert (artifact_dir / "stats" / "analysis_bootstrap_comparisons.csv").exists()
    assert read_csv(artifact_dir / "stats" / "analysis_track_summary.csv")


def test_analyze_run_focus_task_filters_outputs(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    write_csv(
        artifact_dir / "stats" / "transfer_regret.csv",
        [
            _row("gsm8k", "ipeo_zero", "zero_target_transfer", 1.0, 100, 0, 0.10),
            _row("ifbench_hard", "ipeo_zero", "zero_target_transfer", 0.8, 100, 0, 0.10),
            _row("ifbench_hard", "gepa", "target_optimization", 0.7, 0, 20, 0.05),
        ],
    )

    outputs = run(
        Namespace(
            artifact_dir=str(artifact_dir),
            focus_task="ifbench_hard",
            ipeo_methods=["ipeo_zero"],
            baseline_methods=["all"],
            bootstrap_samples=100,
            bootstrap_seed=0,
            confidence_level=0.95,
            quiet=True,
            no_color=True,
        )
    )

    assert {row["task_id"] for row in outputs["method_task_summary"]} == {"ifbench_hard"}
    assert outputs["bootstrap_comparisons"][0]["num_tasks"] == 1
    assert (artifact_dir / "stats" / "analysis_per_task_winners_ifbench_hard.csv").exists()


def test_bootstrap_comparisons_detect_consistent_score_delta(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    rows = []
    for task_id in ["task_a", "task_b", "task_c"]:
        rows.append(_row(task_id, "ipeo_budget_200", "zero_target_transfer", 0.90, 100, 0, 0.02))
        rows.append(_row(task_id, "miprov2", "target_optimization", 0.80, 0, 150, 0.08))
    write_csv(artifact_dir / "stats" / "transfer_regret.csv", rows)

    outputs = analyze_artifact_dir(
        artifact_dir,
        ipeo_methods=["ipeo_budget_200"],
        baseline_methods=["miprov2"],
        bootstrap_samples=200,
        bootstrap_seed=7,
    )

    comparison = outputs["bootstrap_comparisons"][0]
    assert comparison["score_delta_ci_low"] > 0
    assert comparison["score_outcome"] == "ipeo"
    assert comparison["probability_ipeo_fewer_calls"] == 1.0


def test_analyze_artifact_dir_reports_budget_selector_regret(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    write_csv(
        artifact_dir / "stats" / "transfer_regret.csv",
        [
            _row("ifbench_hard", "ipeo_budget_200", "zero_target_transfer", 0.833, 180, 0, 0.02),
            _row("ifbench_hard", "ipeo_budget_500", "zero_target_transfer", 0.771, 450, 0, 0.04),
            _row("ifbench_hard", "ipeo_budget_1000", "zero_target_transfer", 0.771, 990, 0, 0.08),
            _row("ifbench_hard", "ipeo_budget_select", "zero_target_transfer", 0.771, 990, 0, 0.08),
        ],
    )
    write_jsonl(
        artifact_dir / "stats" / "ifbench_hard_ipeo_budget_select.jsonl",
        [
            {
                "method": "ipeo_budget_select",
                "chosen_method": "ipeo_budget_1000",
                "requested_budget": 1000,
                "source_calls": 990,
                "source_score": 2.0,
                "prompt_id": "prompt-1000",
                "candidate_scores": [
                    {"method": "ipeo_budget_200", "requested_budget": 200, "source_calls": 180, "source_score": 1.0},
                    {"method": "ipeo_budget_500", "requested_budget": 500, "source_calls": 450, "source_score": 1.5},
                    {"method": "ipeo_budget_1000", "requested_budget": 1000, "source_calls": 990, "source_score": 2.0},
                ],
            }
        ],
    )

    outputs = analyze_artifact_dir(
        artifact_dir,
        focus_task="ifbench_hard",
        ipeo_methods=["ipeo_budget_select"],
        baseline_methods=["all"],
        bootstrap_samples=100,
    )

    decision = outputs["budget_select_decisions"][0]
    assert decision["chosen_method"] == "ipeo_budget_1000"
    assert decision["oracle_budget_method"] == "ipeo_budget_200"
    assert decision["budget_selector_regret"] == pytest.approx(0.062)
    assert decision["budget_selection_outcome"] == "miss"
    summary = outputs["budget_select_summary"][0]
    assert summary["selection_accuracy"] == 0.0
    assert summary["chosen_method_counts"] == "ipeo_budget_1000:1"
    assert (artifact_dir / "stats" / "analysis_budget_select_decisions_ifbench_hard.csv").exists()
    assert read_csv(artifact_dir / "stats" / "analysis_budget_select_summary_ifbench_hard.csv")


def test_analyze_artifact_dir_accepts_stats_dir(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "run"
    write_csv(
        artifact_dir / "stats" / "transfer_regret.csv",
        [_row("gsm8k", "ipeo_zero", "zero_target_transfer", 1.0, 100, 0, 0.10)],
    )

    outputs = analyze_artifact_dir(artifact_dir / "stats", ipeo_methods=["ipeo_zero"], baseline_methods=["all"])

    assert outputs["per_task_winners"][0]["task_id"] == "gsm8k"
    assert (artifact_dir / "stats" / "analysis_per_task_winners.csv").exists()


def test_analyze_artifact_dir_missing_transfer_lists_nearby_runs(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    write_csv(
        artifact_root / "completed" / "stats" / "transfer_regret.csv",
        [_row("gsm8k", "ipeo_zero", "zero_target_transfer", 1.0, 100, 0, 0.10)],
    )

    with pytest.raises(ValueError, match="Nearby transfer_regret.csv files") as exc_info:
        analyze_artifact_dir(artifact_root / "missing")

    assert "completed" in str(exc_info.value)


def _row(
    task_id: str,
    method: str,
    track: str,
    score: float,
    source_calls: int,
    target_calls: int,
    dollars: float,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "fold_id": "target-model",
        "target_model": "model:target",
        "source_models": "model:a,model:b",
        "method": method,
        "prompt_id": f"prompt-{method}",
        "target_score": score,
        "fixed_pool_oracle_score": max(score, 0.97),
        "fixed_pool_regret": max(score, 0.97) - score,
        "source_calls": source_calls,
        "target_calls": target_calls,
        "total_dollars": dollars,
        "benchmark_track": track,
        "selection_access": track,
        "uses_target_validation": track == "target_optimization",
        "uses_target_test_for_selection": False,
    }

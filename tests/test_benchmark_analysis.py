from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from ipeo.core.io import read_csv, write_csv
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

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from ipeo.core.io import read_csv, write_csv, write_jsonl
from ipeo.runners.analyze_many import run
from ipeo.stats.multi_run_analysis import analyze_many_artifact_dirs


def test_analyze_many_artifact_dirs_writes_multi_run_outputs(tmp_path: Path) -> None:
    seed0 = _write_run(tmp_path, "seed0", ipeo_score=0.90, mipro_score=0.80)
    seed1 = _write_run(tmp_path, "seed1", ipeo_score=0.70, mipro_score=0.80)
    output_dir = tmp_path / "analysis"

    outputs = analyze_many_artifact_dirs(
        [seed0, seed1],
        output_dir=output_dir,
        focus_task="ifbench_hard",
        ipeo_methods=["ipeo_budget_200"],
        baseline_methods=["miprov2"],
        bootstrap_samples=200,
        bootstrap_seed=3,
    )

    summary = {
        (row["task_id"], row["method"]): row
        for row in outputs["method_summary"]
    }
    ipeo_summary = summary[("ifbench_hard", "ipeo_budget_200")]
    assert ipeo_summary["num_runs"] == 2
    assert ipeo_summary["mean_target_score"] == 0.8
    assert ipeo_summary["best_score_win_rate"] == 0.5
    comparison = outputs["ipeo_vs_baselines"][0]
    assert comparison["num_pairs"] == 2
    assert comparison["score_win_rate"] == 0.5
    assert comparison["score_loss_rate"] == 0.5
    assert (output_dir / "stats" / "multi_run_method_summary_ifbench_hard.csv").exists()
    assert read_csv(output_dir / "stats" / "multi_run_ipeo_vs_baselines_ifbench_hard.csv")


def test_analyze_many_runner_accepts_artifact_glob(tmp_path: Path) -> None:
    _write_run(tmp_path, "seed0", ipeo_score=0.90, mipro_score=0.80)
    _write_run(tmp_path, "seed1", ipeo_score=0.90, mipro_score=0.80)
    output_dir = tmp_path / "analysis"

    outputs = run(
        Namespace(
            artifact_dirs=[],
            artifact_glob=[str(tmp_path / "seed*")],
            output_dir=str(output_dir),
            focus_task="ifbench_hard",
            ipeo_methods=["ipeo_budget_200"],
            baseline_methods=["miprov2"],
            bootstrap_samples=100,
            bootstrap_seed=0,
            confidence_level=0.95,
            quiet=True,
            no_color=True,
        )
    )

    assert len(outputs["input_runs"]) == 2
    comparison = outputs["ipeo_vs_baselines"][0]
    assert comparison["score_outcome"] == "ipeo"
    assert comparison["probability_ipeo_fewer_calls"] == 1.0


def test_analyze_many_runner_reports_unmatched_glob_with_nearby_runs(tmp_path: Path) -> None:
    completed = _write_run(tmp_path, "sourceval_seed_0", ipeo_score=0.90, mipro_score=0.80)

    with pytest.raises(ValueError) as exc_info:
        run(
            Namespace(
                artifact_dirs=[],
                artifact_glob=[str(tmp_path / "missing_seed_*")],
                output_dir=str(tmp_path / "analysis"),
                focus_task="ifbench_hard",
                ipeo_methods=["ipeo_budget_200"],
                baseline_methods=["miprov2"],
                bootstrap_samples=100,
                bootstrap_seed=0,
                confidence_level=0.95,
                quiet=True,
                no_color=True,
            )
        )

    message = str(exc_info.value)
    assert "matched zero directories" in message
    assert "Nearby completed run directories" in message
    assert str(completed) in message


def test_analyze_many_artifact_dirs_reports_missing_transfer_rows(tmp_path: Path) -> None:
    missing_run = tmp_path / "missing_seed_0"

    with pytest.raises(ValueError) as exc_info:
        analyze_many_artifact_dirs(
            [missing_run],
            output_dir=tmp_path / "analysis",
            focus_task="ifbench_hard",
            ipeo_methods=["ipeo_budget_200"],
            baseline_methods=["miprov2"],
        )

    message = str(exc_info.value)
    assert "No transfer rows found" in message
    assert "status=missing" in message
    assert str(missing_run) in message


def test_analyze_many_artifact_dirs_reports_focus_task_mismatch(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "seed0"
    write_csv(
        artifact_dir / "stats" / "transfer_regret.csv",
        [_row("gsm8k", "ipeo_budget_200", "zero_target_transfer", 0.9, 180, 0, 0.02)],
    )

    with pytest.raises(ValueError) as exc_info:
        analyze_many_artifact_dirs(
            [artifact_dir],
            output_dir=tmp_path / "analysis",
            focus_task="ifbench_hard",
            ipeo_methods=["ipeo_budget_200"],
            baseline_methods=["miprov2"],
        )

    message = str(exc_info.value)
    assert "No rows matched --focus_task 'ifbench_hard'" in message
    assert "status=filtered_by_focus_task" in message
    assert "available_tasks=gsm8k" in message


def test_analyze_many_reports_budget_selector_regret(tmp_path: Path) -> None:
    seed0 = _write_budget_select_run(
        tmp_path,
        "seed0",
        chosen_method="ipeo_budget_200",
        select_score=0.90,
        budget_scores={"ipeo_budget_200": 0.90, "ipeo_budget_500": 0.80, "ipeo_budget_1000": 0.70},
    )
    seed1 = _write_budget_select_run(
        tmp_path,
        "seed1",
        chosen_method="ipeo_budget_1000",
        select_score=0.70,
        budget_scores={"ipeo_budget_200": 0.90, "ipeo_budget_500": 0.80, "ipeo_budget_1000": 0.70},
    )
    output_dir = tmp_path / "analysis"

    outputs = analyze_many_artifact_dirs(
        [seed0, seed1],
        output_dir=output_dir,
        focus_task="ifbench_hard",
        ipeo_methods=["ipeo_budget_select"],
        baseline_methods=["all"],
        bootstrap_samples=100,
        bootstrap_seed=0,
    )

    summary = outputs["budget_select_summary"][0]
    assert summary["num_runs"] == 2
    assert summary["selection_accuracy"] == 0.5
    assert summary["mean_budget_selector_regret"] == pytest.approx(0.10)
    assert summary["chosen_method_counts"] == "ipeo_budget_1000:1,ipeo_budget_200:1"
    decisions = {row["run_label"]: row for row in outputs["budget_select_decisions"]}
    assert decisions["seed1"]["oracle_budget_method"] == "ipeo_budget_200"
    assert decisions["seed1"]["budget_selection_outcome"] == "miss"
    assert (output_dir / "stats" / "multi_run_budget_select_summary_ifbench_hard.csv").exists()
    assert read_csv(output_dir / "stats" / "multi_run_budget_select_decisions_ifbench_hard.csv")


def _write_run(tmp_path: Path, name: str, *, ipeo_score: float, mipro_score: float) -> Path:
    artifact_dir = tmp_path / name
    write_csv(
        artifact_dir / "stats" / "transfer_regret.csv",
        [
            _row("ifbench_hard", "ipeo_budget_200", "zero_target_transfer", ipeo_score, 180, 0, 0.02),
            _row("ifbench_hard", "miprov2", "target_optimization", mipro_score, 0, 201, 0.09),
            _row("ifbench_hard", "gepa", "target_optimization", 0.75, 0, 500, 0.08),
        ],
    )
    return artifact_dir


def _write_budget_select_run(
    tmp_path: Path,
    name: str,
    *,
    chosen_method: str,
    select_score: float,
    budget_scores: dict[str, float],
) -> Path:
    artifact_dir = tmp_path / name
    rows = [
        _row("ifbench_hard", method, "zero_target_transfer", score, _budget_calls(method), 0, 0.02)
        for method, score in budget_scores.items()
    ]
    rows.append(_row("ifbench_hard", "ipeo_budget_select", "zero_target_transfer", select_score, _budget_calls(chosen_method), 0, 0.02))
    write_csv(artifact_dir / "stats" / "transfer_regret.csv", rows)
    write_jsonl(
        artifact_dir / "stats" / "ifbench_hard_ipeo_budget_select.jsonl",
        [
            {
                "method": "ipeo_budget_select",
                "chosen_method": chosen_method,
                "requested_budget": int(chosen_method.rsplit("_", 1)[-1]),
                "source_calls": _budget_calls(chosen_method),
                "source_score": 3.0 if chosen_method == "ipeo_budget_1000" else 1.0,
                "prompt_id": f"prompt-{chosen_method}",
                "candidate_scores": [
                    {
                        "method": method,
                        "requested_budget": int(method.rsplit("_", 1)[-1]),
                        "source_calls": _budget_calls(method),
                        "source_score": 3.0 if method == "ipeo_budget_1000" else 1.0,
                    }
                    for method in sorted(budget_scores)
                ],
            }
        ],
    )
    return artifact_dir


def _budget_calls(method: str) -> int:
    return {"ipeo_budget_200": 180, "ipeo_budget_500": 450, "ipeo_budget_1000": 990}[method]


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
        "fixed_pool_oracle_score": max(score, 0.90),
        "fixed_pool_regret": max(score, 0.90) - score,
        "source_calls": source_calls,
        "target_calls": target_calls,
        "total_dollars": dollars,
        "benchmark_track": track,
        "selection_access": track,
        "uses_target_validation": track == "target_optimization",
        "uses_target_test_for_selection": False,
    }

"""Benchmark identity, pairing, and durable CLI behavior."""

from __future__ import annotations

import json

import pytest

from llmoose.benchmark.runner import _changed_fields, aggregate, benchmark_command


def _manifest():
    return {"suite_id": "paired-test", "seeds": [1, 2], "source_sha256": "source"}


def _job(seed, team, winner, *, complete=True):
    return {
        "job_id": f"seed-{seed}-candidate-team-{team}",
        "seed": seed,
        "candidate_team": team,
        "complete": complete,
        "candidate_won": winner == team if complete else None,
        "candidate_stone_margin": (3 if winner == team else -3) if complete else None,
    }


def test_mirrors_attribute_wins_to_candidate_and_cluster_by_seed():
    jobs = [
        _job(1, 0, 0), _job(1, 1, 0),
        _job(2, 0, 1), _job(2, 1, 1),
    ]
    report = aggregate(_manifest(), jobs)
    assert report["candidate_win_rate"] == 0.5
    assert [pair["candidate_win_fraction"] for pair in report["pairs"]] == [0.5, 0.5]
    assert report["candidate_win_rate_95pct_bootstrap"] == [0.5, 0.5]
    assert report["coverage"]["complete_pairs"] == 2


def test_incomplete_mirror_is_coverage_gap_not_candidate_loss():
    jobs = [
        _job(1, 0, 0), _job(1, 1, 1),
        _job(2, 0, 0), _job(2, 1, 1, complete=False),
    ]
    report = aggregate(_manifest(), jobs)
    assert report["candidate_win_rate"] == 1.0
    assert report["coverage"] == {
        "planned_games": 4, "completed_games": 3,
        "complete_pairs": 1, "planned_pairs": 2,
    }
    assert report["pairs"][1]["candidate_win_fraction"] is None
    assert report["jobs"][3]["complete"] is False


def test_cli_runs_and_reuses_validated_pair(tmp_path):
    output = tmp_path / "bench"
    args = [
        "--suite-id", "baseline-smoke", "--seeds", "12",
        "--candidate", "heuristic", "--opponent", "conservative",
        "--target-score", "8", "--output", str(output),
    ]
    assert benchmark_command(args) == 0
    first = json.loads((output / "summary.json").read_text())
    assert first["coverage"]["complete_pairs"] == 1
    assert first["candidate_win_rate"] in (0.0, 0.5, 1.0)
    assert all(row["validation_status"] == "finalized" for row in first["jobs"])
    assert all((output / row["run_dir"] / "records.jsonl").exists() for row in first["jobs"])
    journal = output / first["jobs"][0]["run_dir"] / "records.jsonl"
    before = journal.read_bytes()
    assert benchmark_command(args) == 0
    assert journal.read_bytes() == before


def test_identical_deterministic_policies_get_exactly_half(tmp_path):
    output = tmp_path / "symmetric"
    assert benchmark_command([
        "--suite-id", "symmetry", "--seeds", "21",
        "--candidate", "conservative", "--opponent", "conservative",
        "--target-score", "8", "--output", str(output),
    ]) == 0
    report = json.loads((output / "summary.json").read_text())
    assert report["candidate_win_rate"] == 0.5
    assert report["candidate_win_rate_95pct_bootstrap"] == [0.5, 0.5]


def test_existing_suite_error_names_changed_fields(tmp_path, capsys):
    output = tmp_path / "bench"
    args = [
        "--suite-id", "smoke", "--seeds", "12",
        "--candidate", "heuristic", "--opponent", "conservative",
        "--target-score", "8", "--output", str(output),
    ]
    assert benchmark_command(args) == 0
    with pytest.raises(SystemExit):
        benchmark_command(["9" if part == "8" else part for part in args])
    assert "differs in target_score" in capsys.readouterr().err
    assert _changed_fields({"optional": None}, {}) == ["optional"]

"""Benchmark identity, pairing, and durable CLI behavior."""

from __future__ import annotations

import json
from io import BytesIO

import pytest

from llmoose.benchmark.runner import _changed_fields, aggregate, benchmark_command
from llmoose.simulation.artifacts import canonical_bytes, digest, write_checksums
from llmoose.simulation.validation import validate_run


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


@pytest.mark.parametrize(
    ("candidate_model", "opponent_model", "expected_option"),
    [
        ("CANDIDATE_MODEL", "provider/real-model", "--model"),
        ("provider/real-model", "OPPONENT_MODEL", "--opponent-model"),
    ],
)
def test_example_model_ids_fail_before_creating_a_suite(
    tmp_path, capsys, candidate_model, opponent_model, expected_option
):
    output = tmp_path / "unused"
    with pytest.raises(SystemExit) as error:
        benchmark_command([
            "--suite-id", "example", "--seeds", "12",
            "--candidate", "llm", "--model", candidate_model,
            "--opponent", "llm", "--opponent-model", opponent_model,
            "--output", str(output),
        ])
    assert error.value.code == 2
    assert f"{expected_option}=" in capsys.readouterr().err
    assert not output.exists()


def test_two_models_swap_teams_with_independent_endpoints(tmp_path, monkeypatch):
    requests = []

    class FakeOpener:
        def open(self, request, timeout):
            payload = json.loads(request.data)
            requests.append((request.full_url, payload["model"]))
            choices = (
                payload["messages"][1]["content"]
                .split("Legal actions (ID: meaning):\n")[1]
                .splitlines()
            )
            choices = dict(line.strip().split(": ", 1) for line in choices)
            chosen = next(
                (
                    key for kind in ("no mus", "órdago", "call", "bet", "pass", "fold")
                    for key, meaning in choices.items() if meaning.startswith(kind)
                ),
                next(iter(choices)),
            )
            return BytesIO(json.dumps({
                "model": payload["model"],
                "choices": [{"message": {"content": json.dumps({"action_id": int(chosen)})}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            }).encode())

    monkeypatch.setattr("llmoose.agents.providers.build_opener", lambda *_: FakeOpener())
    output = tmp_path / "models"
    assert benchmark_command([
        "--suite-id", "two-model-smoke", "--seeds", "12",
        "--candidate", "llm", "--model", "candidate/a",
        "--base-url", "http://localhost:8001/v1",
        "--opponent", "llm", "--opponent-model", "opponent/b",
        "--opponent-base-url", "http://localhost:8002/v1",
        "--target-score", "4", "--max-steps", "100", "--output", str(output),
    ]) == 0
    assert requests
    assert {model for _, model in requests} == {"candidate/a", "opponent/b"}
    assert all(
        ("8001" in url) == (model == "candidate/a")
        for url, model in requests
    )
    for team in (0, 1):
        path = output / "jobs" / f"seed-12-candidate-team-{team}"
        manifest = json.loads((path / "manifest.json").read_text())
        policies = manifest["policies"]
        assert policies["NORTH"]["requested_model"] == (
            "candidate/a" if team == 0 else "opponent/b"
        )
        assert policies["EAST"]["requested_model"] == (
            "candidate/a" if team == 1 else "opponent/b"
        )
        assert manifest["intent"]["requested_models"] == [
            policies["NORTH"]["requested_model"],
            policies["EAST"]["requested_model"],
        ]
    old_run = output / "jobs" / "seed-12-candidate-team-0"
    journal = old_run / "records.jsonl"
    rows = [json.loads(line) for line in journal.read_bytes().splitlines()]
    attempt = next(row for row in rows if row["record_type"] == "attempt_started")
    body = attempt["payload"]["request_body"]
    legacy_order = ("model", "messages", "stream", "max_tokens", "temperature", "reasoning")
    legacy_body = {key: body[key] for key in legacy_order if key in body}
    attempt["payload"]["request_body_sha256"] = digest(json.dumps(legacy_body).encode())
    journal.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))
    write_checksums(old_run)
    assert validate_run(old_run)["status"] == "finalized"

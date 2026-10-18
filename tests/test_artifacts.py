import json
import subprocess
import sys

import pytest

from llmoose.agents.providers import Completion, ProviderError
from llmoose.cli import play_command
from llmoose.simulation.artifacts import read_journal
from llmoose.simulation.evidence import read_evidence
from llmoose.simulation.traces import load_trace
from llmoose.simulation.validation import (
    load_resume_prefix,
    recover_run,
    resume_lock,
    validate_command,
    validate_run,
)


def test_new_run_has_linked_durable_evidence_and_validates(tmp_path):
    directory = tmp_path / "run"
    assert play_command(["--seed", "12", "--target-score", "8", "--run-dir", str(directory)]) == 0
    manifest = json.loads((directory / "manifest.json").read_text())
    records, complete = read_journal(directory / "records.jsonl")
    assert complete
    assert manifest["schema_version"] == 2
    assert manifest["provenance"]["source_sha256"]
    assert records[0]["record_type"] == "run_started"
    assert records[1]["record_type"] == "decision_opened"
    assert records[2]["record_type"] == "action_selected"
    assert records[3]["record_type"] == "action_applied"
    assert records[1]["decision_id"] == records[3]["decision_id"]
    assert "seed" not in records[1]["payload"]["observation"]
    assert validate_run(directory)["status"] == "finalized"


def test_model_attempts_are_linked_even_on_provider_failure(tmp_path, monkeypatch):
    calls = 0
    directory = tmp_path / "run"

    def provider(_messages):
        nonlocal calls
        calls += 1
        assert (directory / "manifest.json").exists()
        records, complete = read_journal(directory / "records.jsonl")
        assert complete
        assert records[-1]["record_type"] == "attempt_started"
        if calls == 1:
            return Completion("bad")
        raise ProviderError("offline test failure")

    monkeypatch.setattr("llmoose.cli._build_provider", lambda _: provider)
    with pytest.raises(SystemExit):
        play_command(["--policy", "llm", "--model", "test/model", "--run-dir", str(directory)])
    records, _ = read_journal(directory / "records.jsonl")
    started = [row for row in records if row["record_type"] == "attempt_started"]
    finished = [row for row in records if row["record_type"] == "attempt_finished"]
    assert len(started) == len(finished) == 2
    assert [row["payload"]["status"] for row in finished] == ["invalid", "provider_error"]
    assert not any(row["record_type"] == "action_applied" for row in records)
    assert validate_run(directory)["status"] == "finalized"


def test_validator_detects_a_changed_reward_and_legacy_files_stay_readable(tmp_path):
    directory = tmp_path / "run"
    assert play_command(["--seed", "12", "--target-score", "8", "--run-dir", str(directory)]) == 0
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for name in ("match.json", "moves.jsonl", "report.json"):
        (legacy / name).write_bytes((directory / name).read_bytes())
    assert validate_run(legacy)["format"] == "v1"
    assert read_evidence(legacy).actions[0].source is None
    path = directory / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    applied = next(row for row in rows if row["record_type"] == "action_applied")
    applied["payload"]["reward"]["team_stone_delta"] = [99, 99]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        validate_run(directory)


def test_recovery_marks_a_torn_run_partial_without_reissuing_requests(tmp_path):
    directory = tmp_path / "run"
    assert play_command(["--seed", "12", "--target-score", "8", "--run-dir", str(directory)]) == 0
    rows = (directory / "records.jsonl").read_bytes().splitlines(keepends=True)
    (directory / "records.jsonl").write_bytes(b"".join(rows[:-2]) + b'{"schema":')
    (directory / "checksums.json").unlink()
    (directory / "result.json").unlink()
    with pytest.raises(ValueError, match="Torn final journal line"):
        validate_run(directory)
    first = recover_run(directory)
    second = recover_run(directory)
    assert first == second
    assert first["status"] == "partial"
    assert first["provider_requests_reissued"] is False
    assert json.loads((directory / "recovery.json").read_text()) == first


def test_recovery_finishes_atomic_result_after_a_crash_window(tmp_path):
    directory = tmp_path / "run"
    assert play_command(["--seed", "12", "--target-score", "8", "--run-dir", str(directory)]) == 0
    (directory / "checksums.json").unlink()
    (directory / "result.json").unlink()
    with pytest.raises(ValueError, match="Missing result.json"):
        validate_run(directory)
    assert recover_run(directory)["status"] == "finalized"
    assert validate_run(directory)["status"] == "finalized"


def test_validator_rejects_hidden_data_added_to_a_seat_observation(tmp_path):
    directory = tmp_path / "run"
    assert play_command(["--seed", "12", "--target-score", "8", "--run-dir", str(directory)]) == 0
    (directory / "checksums.json").unlink()
    path = directory / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    opened = next(row for row in rows if row["record_type"] == "decision_opened")
    opened["payload"]["observation"]["opponent_hand"] = ["hidden card"]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="Seat observation mismatch or hidden data"):
        validate_run(directory)


def test_validation_cli_saves_hashed_sidecar_for_legacy_run(tmp_path, capsys):
    directory = tmp_path / "run"
    assert play_command(["--seed", "12", "--target-score", "8", "--run-dir", str(directory)]) == 0
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for name in ("match.json", "moves.jsonl", "report.json"):
        (legacy / name).write_bytes((directory / name).read_bytes())

    assert validate_command([str(legacy), "--save"]) == 0
    capsys.readouterr()
    sidecar = json.loads((legacy / "validation.json").read_text())
    assert sidecar["result"]["format"] == "v1"
    assert sidecar["result"]["unavailable_provenance"]["request_start_times"]
    assert set(sidecar["source_sha256"]) == {"match.json", "moves.jsonl", "report.json"}


def _cut_after(directory, predicate, *, torn=False):
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    index = next(index for index, row in enumerate(rows) if predicate(row))
    data = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows[: index + 1]).encode()
    (directory / "records.jsonl").write_bytes(data + (b'{"record_type":' if torn else b""))
    for name in ("report.json", "result.json", "checksums.json", "match.json"):
        (directory / name).unlink(missing_ok=True)


def test_resumed_random_match_preserves_committed_actions_and_policy_sequence(tmp_path):
    directory = tmp_path / "run"
    assert play_command(["--seed", "12", "--policy", "random", "--target-score", "2", "--run-dir", str(directory)]) == 0
    original = load_trace(directory / "match.json")
    cutoff = max(1, len(original.actions) // 2)
    _cut_after(
        directory,
        lambda row: row["record_type"] == "action_applied" and row["payload"]["action_index"] == cutoff - 1,
        torn=True,
    )
    assert play_command(["--resume-run", str(directory)]) == 0
    resumed = load_trace(directory / "match.json")
    assert resumed.actions == original.actions
    assert validate_run(directory)["status"] == "finalized"
    assert len((directory / "moves.jsonl").read_text().splitlines()) == len(original.actions)
    assert list(directory.glob("records.torn.*.bin"))
    assert list(directory.glob("moves.pre-resume.*.bin"))


@pytest.mark.parametrize("cut_type,expected", [
    ("attempt_finished", "action_selected"),
    ("attempt_started", "decision_resolved"),
])
def test_resume_uses_saved_choice_or_explicit_fallback_without_reissuing_request(
    tmp_path, monkeypatch, cut_type, expected
):
    directory = tmp_path / "run"

    def provider(messages):
        choices = dict(
            line.strip().split(": ", 1)
            for line in messages[1]["content"].split("Legal actions (ID: meaning):\n")[1].splitlines()
        )
        return Completion(f"<answer>{next(iter(choices))}</answer>")

    monkeypatch.setattr("llmoose.cli._build_provider", lambda _: provider)
    assert play_command([
        "--team-0", "llm", "--team-1", "conservative", "--model", "test/model",
        "--target-score", "2", "--run-dir", str(directory),
    ]) == 0
    _cut_after(
        directory,
        lambda row: row["record_type"] == cut_type
        and (cut_type != "attempt_finished" or row["payload"]["status"] == "chosen"),
    )
    if cut_type == "attempt_started":
        (directory / "decisions.jsonl").write_bytes(b'{"torn":')
    assert play_command(["--resume-run", str(directory)]) == 0
    records, _ = read_journal(directory / "records.jsonl")
    resumed = next(index for index, row in enumerate(records) if row["record_type"] == "run_resumed")
    assert records[resumed + 1]["record_type"] == expected
    if cut_type == "attempt_started":
        assert records[resumed + 1]["payload"]["status"] == "recovery_fallback"
        assert records[resumed + 2]["record_type"] == "action_selected"
        assert records[resumed + 3]["record_type"] == "action_applied"
        assert list(directory.glob("decisions.torn.*.bin"))
        for line in (directory / "decisions.jsonl").read_text().splitlines():
            json.loads(line)
    else:
        assert records[resumed + 2]["record_type"] == "action_applied"
    assert validate_run(directory)["status"] == "finalized"


def test_common_reader_preserves_v2_extensions_without_current_engine(tmp_path):
    directory = tmp_path / "run"
    assert play_command(["--seed", "12", "--target-score", "8", "--run-dir", str(directory)]) == 0
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    rows[1]["future_optional_field"] = {"kept": True}
    (directory / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["prompt"]["system_sha256"] = "older-prompt-hash"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    evidence = read_evidence(directory)
    assert evidence.records[1]["future_optional_field"] == {"kept": True}
    assert evidence.actions[0].source is not None
    with pytest.raises(ValueError):
        validate_run(directory)


def test_resume_rejects_changed_source_before_modifying_evidence(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    assert play_command(["--seed", "12", "--target-score", "8", "--run-dir", str(directory)]) == 0
    _cut_after(directory, lambda row: row["record_type"] == "action_applied")
    original = (directory / "records.jsonl").read_bytes()
    monkeypatch.setattr("llmoose.simulation.validation.current_source_digest", lambda: "changed")
    with pytest.raises(ValueError, match="Current source differs"):
        load_resume_prefix(directory)
    assert (directory / "records.jsonl").read_bytes() == original


def test_hard_kill_can_resume_from_fsynced_action_prefix(tmp_path):
    directory = tmp_path / "killed"
    script = """
import os
import sys
from llmoose.cli import play_command
from llmoose.simulation.runs import RunRecorder

original = RunRecorder.on_transition
def stop_after_seven(self, before, action, after):
    original(self, before, action, after)
    if self.action_count == 7:
        os._exit(71)
RunRecorder.on_transition = stop_after_seven
play_command(['--seed', '12', '--target-score', '8', '--run-dir', sys.argv[1]])
"""
    killed = subprocess.run([sys.executable, "-c", script, str(directory)], check=False)
    assert killed.returncode == 71
    assert not (directory / "result.json").exists()
    assert play_command(["--resume-run", str(directory)]) == 0
    assert validate_run(directory)["status"] == "finalized"
    assert len(load_trace(directory / "match.json").actions) > 7


def test_hard_kill_after_random_selection_reuses_that_exact_action(tmp_path):
    reference = tmp_path / "reference"
    assert play_command([
        "--seed", "12", "--policy", "random", "--target-score", "2",
        "--run-dir", str(reference),
    ]) == 0
    expected = load_trace(reference / "match.json").actions
    directory = tmp_path / "killed"
    script = """
import os
import sys
from llmoose.cli import play_command
from llmoose.simulation.runs import RunRecorder

original = RunRecorder.on_action_selected
def stop_after_selection(self, state, action):
    original(self, state, action)
    if self.action_count == 5:
        os._exit(72)
RunRecorder.on_action_selected = stop_after_selection
play_command(['--seed', '12', '--policy', 'random', '--target-score', '2', '--run-dir', sys.argv[1]])
"""
    killed = subprocess.run([sys.executable, "-c", script, str(directory)], check=False)
    assert killed.returncode == 72
    records, _ = read_journal(directory / "records.jsonl")
    assert records[-1]["record_type"] == "action_selected"
    assert play_command(["--resume-run", str(directory)]) == 0
    assert load_trace(directory / "match.json").actions == expected
    assert validate_run(directory)["status"] == "finalized"


def test_second_resume_cannot_extend_a_locked_run(tmp_path):
    directory = tmp_path / "run"
    assert play_command(["--seed", "12", "--target-score", "8", "--run-dir", str(directory)]) == 0
    _cut_after(directory, lambda row: row["record_type"] == "action_applied")
    original = (directory / "records.jsonl").read_bytes()
    with resume_lock(directory):
        with pytest.raises(SystemExit):
            play_command(["--resume-run", str(directory)])
    with pytest.raises(SystemExit):
        play_command(["--resume-run", str(directory), "--seed", "99"])
    assert (directory / "records.jsonl").read_bytes() == original


def test_provider_error_before_hard_kill_resumes_with_explicit_fallback(tmp_path, monkeypatch):
    directory = tmp_path / "run"

    def offline(_messages):
        raise ProviderError("offline")

    monkeypatch.setattr("llmoose.cli._build_provider", lambda _: offline)
    with pytest.raises(SystemExit):
        play_command([
            "--team-0", "llm", "--team-1", "conservative", "--model", "test/model",
            "--target-score", "2", "--run-dir", str(directory),
        ])
    _cut_after(
        directory,
        lambda row: row["record_type"] == "attempt_finished"
        and row["payload"]["status"] == "provider_error",
    )
    def legal(messages):
        first = messages[1]["content"].split("Legal actions (ID: meaning):\n")[1].splitlines()[0]
        return Completion(f"<answer>{first.strip().split(': ', 1)[0]}</answer>")

    monkeypatch.setattr("llmoose.cli._build_provider", lambda _: legal)
    assert play_command(["--resume-run", str(directory)]) == 0
    records, _ = read_journal(directory / "records.jsonl")
    assert any(
        row["record_type"] == "decision_resolved"
        and row["payload"]["status"] == "recovery_fallback"
        for row in records
    )
    assert validate_run(directory)["status"] == "finalized"


def test_resume_finalizes_terminal_action_prefix_without_new_move(tmp_path):
    directory = tmp_path / "run"
    assert play_command(["--seed", "12", "--target-score", "8", "--run-dir", str(directory)]) == 0
    trace = (directory / "match.json").read_bytes()
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    last_action = max(
        index for index, row in enumerate(rows) if row["record_type"] == "action_applied"
    )
    (directory / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows[: last_action + 1])
    )
    for name in ("report.json", "result.json", "checksums.json"):
        (directory / name).unlink()
    assert play_command(["--resume-run", str(directory)]) == 0
    assert (directory / "match.json").read_bytes() == trace
    assert validate_run(directory)["status"] == "finalized"

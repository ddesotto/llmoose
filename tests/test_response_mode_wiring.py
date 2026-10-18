import json
from io import BytesIO

import pytest

from llmoose.benchmark.runner import benchmark_command
from llmoose.cli import play_command


def install_transport(monkeypatch):
    requests = []

    class FakeOpener:
        def open(self, request, timeout):
            payload = json.loads(request.data)
            requests.append(payload)
            menu = payload["messages"][1]["content"].split(
                "Legal actions (ID: meaning):\n"
            )[1]
            choices = dict(line.strip().split(": ", 1) for line in menu.splitlines())
            chosen = next(
                (
                    key
                    for kind in ("no mus", "órdago", "call", "bet", "pass", "fold")
                    for key, meaning in choices.items()
                    if meaning.startswith(kind)
                ),
                next(iter(choices)),
            )
            answer = json.dumps({"action_id": int(chosen)})
            return BytesIO(json.dumps({
                "choices": [{"message": {"content": "My final choice is:\n" + answer}}],
            }).encode())

    monkeypatch.setattr("llmoose.agents.providers.build_opener", lambda *_: FakeOpener())
    return requests


def test_compatible_mode_is_recorded_and_restored_on_resume(tmp_path, monkeypatch):
    requests = install_transport(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _: "q")
    directory = tmp_path / "run"
    assert play_command([
        "--team-0", "llm", "--team-1", "conservative", "--model", "test/model",
        "--base-url", "http://localhost:8000/v1", "--target-score", "2",
        "--run-dir", str(directory), "--step", "--llm-strict",
        "--llm-response-mode", "compatible",
    ]) == 130
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["intent"]["llm_response_mode"] == "compatible"
    assert manifest["request_policy"]["response_mode"] == "compatible"
    assert not requests
    # Model a process interruption before finalization, retaining its durable
    # journal prefix. A clean interactive quit is finalized and cannot resume.
    journal = directory / "records.jsonl"
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    journal.write_text("".join(
        json.dumps(row) + "\n" for row in rows if row["record_type"] != "run_finished"
    ))
    for name in ("report.json", "result.json", "checksums.json", "match.json"):
        (directory / name).unlink(missing_ok=True)
    # Explicit overrides are refused; the default strict mode is replaced by
    # the persisted compatible mode when resuming without overrides.
    with pytest.raises(SystemExit) as error:
        play_command([
            "--resume-run", str(directory), "--llm-response-mode", "strict",
        ])
    assert error.value.code == 2
    assert play_command(["--resume-run", str(directory)]) == 0
    assert requests
    assert all("response_format" not in payload for payload in requests)


def test_benchmark_forwards_mode_and_rejects_changed_mode(tmp_path, monkeypatch, capsys):
    requests = install_transport(monkeypatch)
    output = tmp_path / "suite"
    args = [
        "--suite-id", "compatible-test", "--seeds", "12", "--candidate", "llm",
        "--model", "test/model", "--base-url", "http://localhost:8000/v1",
        "--opponent", "conservative", "--target-score", "2", "--output", str(output),
        "--llm-response-mode", "compatible",
    ]
    assert benchmark_command(args) == 0
    manifest = json.loads((output / "benchmark.json").read_text())
    assert manifest["model_settings"]["response_mode"] == "compatible"
    assert requests
    assert all("response_format" not in payload for payload in requests)
    with pytest.raises(SystemExit) as error:
        benchmark_command([*args[:-1], "strict"])
    assert error.value.code == 2
    assert "model_settings.response_mode" in capsys.readouterr().err


def test_compatible_mode_requires_llm_player():
    with pytest.raises(SystemExit) as error:
        play_command(["--llm-response-mode", "compatible"])
    assert error.value.code == 2

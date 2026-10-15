import json

import pytest

from llmoose.cli import play_command
from llmoose.simulation.inspect import inspect_command, read_jsonl
from llmoose.simulation.reporting import LLMMetrics


def test_inspect_and_replay_baseline(tmp_path, monkeypatch, capsys):
    directory = tmp_path / "run"
    play_command(["--target-score", "8", "--run-dir", str(directory)])
    before = {p.name: p.read_bytes() for p in directory.iterdir()}

    def forbidden(*args, **kwargs):
        pytest.fail("Inspection must not contact a provider")

    monkeypatch.setattr("llmoose.agents.providers.build_opener", forbidden)
    assert inspect_command([str(directory), "--replay"]) == 0
    output = capsys.readouterr().out
    assert "Status: completed" in output
    assert "Hand 1" in output
    assert "No API calls" in output
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize(
    "reasoning, details, expected",
    [
        (None, [], "No readable reasoning saved"),
        (
            "Fold because the stake is too high.",
            [],
            "Fold because the stake is too high.",
        ),
        (
            None,
            [{"type": "reasoning.text", "text": "The hand is weak."}],
            "The hand is weak.",
        ),
        (
            None,
            [{"type": "reasoning.text", "signature": "opaque-signature"}],
            "No readable reasoning saved",
        ),
    ],
)
def test_inspect_attempts_in_an_unfinished_run(
    tmp_path, capsys, reasoning, details, expected
):
    record = {
        "action_index": 2,
        "hand_number": 0,
        "seat": 0,
        "phase": "grande",
        "status": "chosen",
        "attempt": 1,
        "response": "<answer>4</answer>",
        "messages": [{"role": "user", "content": "private observation"}],
        "reasoning": reasoning,
        "reasoning_details": details,
        "usage": {"completion_tokens_details": {"reasoning_tokens": 12}},
    }
    (tmp_path / "decisions.jsonl").write_text(json.dumps(record) + "\n")
    assert inspect_command([str(tmp_path), "--decisions", "--action", "2"]) == 0
    output = capsys.readouterr().out
    assert "No final report yet" in output
    assert "reasoning_tokens: 12" in output
    assert "private observation" in output
    assert "<answer>4</answer>" in output
    assert expected in output
    assert "opaque-signature" not in output


def test_live_log_reader_ignores_only_an_unfinished_last_record(tmp_path):
    path = tmp_path / "moves.jsonl"
    path.write_text('{"action_index": 0}\n{"action_')
    assert read_jsonl(path) == [{"action_index": 0}]
    path.write_text('{"action_index": 0}\nnot-json\n')
    with pytest.raises(ValueError):
        read_jsonl(path)


def test_reasoning_tokens_are_counted_with_coverage():
    metrics = LLMMetrics()
    metrics.add(
        {
            "status": "chosen",
            "attempt": 1,
            "usage": {"completion_tokens_details": {"reasoning_tokens": 15}},
        }
    )
    metrics.add({"status": "chosen", "attempt": 1})
    assert metrics.to_dict()["usage"]["reasoning_tokens"] == {
        "reported_total": 15,
        "requests_reporting": 1,
    }

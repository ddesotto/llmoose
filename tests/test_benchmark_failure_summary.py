"""Match failures and artifact integrity are independent benchmark outcomes."""

import json
from io import BytesIO

import pytest

from llmoose.benchmark import runner
from llmoose.simulation.validation import validate_run


def test_strict_failure_keeps_valid_recording_and_readable_cause(tmp_path, monkeypatch):
    class FakeOpener:
        def open(self, request, timeout):
            return BytesIO(json.dumps({
                "choices": [{"message": {"content": "prose <answer>0</answer>"}}],
            }).encode())

    monkeypatch.setattr("llmoose.agents.providers.build_opener", lambda *_: FakeOpener())
    output = tmp_path / "failed-benchmark"
    runner.benchmark_command([
        "--suite-id", "strict-failure", "--seeds", "12",
        "--candidate", "llm", "--model", "test/model",
        "--base-url", "http://localhost:8001/v1",
        "--opponent", "conservative", "--target-score", "8",
        "--output", str(output),
    ])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["coverage"]["completed_games"] == 0
    assert summary["candidate_win_rate"] is None
    assert len(summary["jobs"]) == 2
    for job in summary["jobs"]:
        assert job["status"] == "failed"
        assert job["validation_status"] == "finalized"
        assert job["complete"] is False
        assert "Model did not produce a legal action" in job["error"]
        assert validate_run(output / job["run_dir"])["status"] == "finalized"


@pytest.mark.parametrize("raises", [False, True])
def test_interrupt_exit_preserves_interrupt_semantics(monkeypatch, raises):
    def interrupt(command):
        if raises:
            raise SystemExit(130)
        return 130

    monkeypatch.setattr(runner, "play_command", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner._play([])

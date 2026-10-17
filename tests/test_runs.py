import json
from dataclasses import replace

import pytest

from llmoose.agents.prompt import load_system_prompt, render_messages
from llmoose.agents.providers import Completion, ProviderError
from llmoose.cli import play_command
from llmoose.game.actions import ActionCodec
from llmoose.game.engine import apply_action
from llmoose.game.state import LanceResult, Phase, new_game
from llmoose.observations.core import observe
from llmoose.rules.hands import Lance
from llmoose.rules.ruleset import Ruleset
from llmoose.simulation.reporting import LLMMetrics
from llmoose.simulation.runs import score_game
from llmoose.simulation.traces import load_trace, replay_trace


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def legal_completion(messages):
    choices = dict(
        line.strip().split(": ", 1)
        for line in messages[1]["content"]
        .split("Legal actions (ID: meaning):\n")[1]
        .splitlines()
    )
    chosen = next(
        (
            key
            for kind in ("no mus", "pass", "fold")
            for key, value in choices.items()
            if value == kind or value.startswith(kind + " —")
        ),
        next(iter(choices)),
    )
    return Completion(
        f"<answer>{chosen}</answer>",
        model="resolved/test",
        usage={"prompt_tokens": 100, "completion_tokens": 8, "cost": 0.001},
    )


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_run_links_model_attempts_to_replay_and_scores(tmp_path, monkeypatch, version):
    requests = []

    def provider(messages):
        requests.append([dict(m) for m in messages])
        # Exercise a retry, including its additional usage.
        if len(requests) == 1:
            return Completion(
                "bad", usage={"prompt_tokens": 10, "completion_tokens": 1}
            )
        return legal_completion(messages)

    monkeypatch.setattr("llmoose.cli._build_provider", lambda _: provider)
    directory = tmp_path / "run"
    assert (
        play_command(
            [
                "--policy",
                "llm",
                "--model",
                "test/model",
                "--seed",
                "12",
                "--target-score",
                "8",
                "--prompt-version",
                str(version),
                "--run-dir",
                str(directory),
            ]
        )
        == 0
    )
    report = json.loads((directory / "report.json").read_text())
    decisions = read_lines(directory / "decisions.jsonl")
    moves = read_lines(directory / "moves.jsonl")
    trace = load_trace(directory / "match.json")
    final = replay_trace(trace)
    assert report["status"] == "completed"
    assert report["prompt_version"] == version
    assert report["renderer_version"] == (2 if version <= 3 else 3)
    assert trace.metadata["renderer_version"] == str(2 if version <= 3 else 3)
    assert report["game"] == score_game(final, len(moves))
    assert report["game"]["actions"] == len(trace.actions)
    assert report["run_id"] == trace.metadata["run_id"]
    assert report["llm"]["resolved_models"] == ["resolved/test"]
    assert len({d["action_index"] for d in decisions}) == len(moves)
    state = new_game(trace.seed, trace.ruleset, trace.dealer)
    codec = ActionCodec(trace.ruleset.target_score)
    for index, (move, action) in enumerate(zip(moves, trace.actions)):
        assert move["action_index"] == index
        assert move["action_id"] == codec.encode(action)
        assert move["scores_before"] == list(state.scores)
        state = apply_action(state, action)
        assert move["scores_after"] == list(state.scores)
    for record in decisions:
        move = moves[record["action_index"]]
        assert record["seat"] == move["seat"]
        assert record["run_id"] == move["run_id"] == report["run_id"]
        assert record["prompt_version"] == version
        assert record["renderer_version"] == (2 if version <= 3 else 3)
        assert record["ruleset_id"] == trace.ruleset.id
        assert record["messages"][0]["content"].startswith(load_system_prompt(version))
        if record["status"] in ("chosen", "forced"):
            assert record["chosen_id"] == move["action_id"]
        if record["status"] != "forced":
            assert record["latency_seconds"] >= 0
    assert decisions[0]["decision_index"] == decisions[1]["decision_index"]
    assert decisions[0]["action_index"] == decisions[1]["action_index"]
    metrics = report["llm"]["total"]
    assert metrics["requests"] == len(requests)
    assert metrics["decisions"] == len(moves)
    assert metrics["invalid_outputs"] == metrics["retries"] == 1
    assert metrics["fallbacks"] == 0
    assert metrics["first_try_legal_rate"] == pytest.approx(
        (metrics["model_decisions"] - 1) / metrics["model_decisions"]
    )
    assert metrics["usage"]["prompt_tokens"]["reported_total"] == 10 + 100 * (
        len(requests) - 1
    )
    assert metrics["usage"]["cost"]["requests_reporting"] == len(requests) - 1
    for field in ("requests", "decisions", "fallbacks", "invalid_outputs"):
        assert metrics[field] == sum(
            m[field] for m in report["llm"]["by_team"].values()
        )


def test_game_score_counts_points_and_folds_without_inventing_a_winner():
    state = replace(
        new_game(12, Ruleset(target_score=8)),
        scores=(3, 2),
        lance_results=(
            LanceResult(Lance.GRANDE, 0, 1, "fold"),
            LanceResult(Lance.PARES, 0, 2, "uncontested"),
            LanceResult(Lance.CHICA, 1, 2, "called"),
        ),
    )
    score = score_game(state, 10)
    assert score["winner"] is None
    for team in score["teams"]:
        assert team["outcome"] == "incomplete"
        assert team["win_points"] is None
        assert (
            sum(lance["points"] for lance in team["by_lance"].values())
            == team["stones"]
        )
    assert score["teams"][0]["stone_difference"] == 1
    assert score["teams"][1]["stone_difference"] == -1
    assert score["teams"][0]["by_lance"]["grande"]["won_by_reason"] == {"fold": 1}
    completed = score_game(replace(state, phase=Phase.COMPLETE, scores=(8, 2)), 11)
    assert completed["winner"] == 0
    assert [t["win_points"] for t in completed["teams"]] == [1, 0]


@pytest.mark.parametrize("failure", ["invalid", "provider", "interrupt"])
def test_failed_llm_run_keeps_attempt_and_partial_report(
    tmp_path, monkeypatch, failure
):
    def provider(_):
        if failure == "provider":
            raise ProviderError("test connection failure")
        if failure == "interrupt":
            raise KeyboardInterrupt
        return Completion("bad")

    monkeypatch.setattr("llmoose.cli._build_provider", lambda _: provider)
    directory = tmp_path / "run"
    args = [
        "--policy",
        "llm",
        "--model",
        "test/model",
        "--llm-strict",
        "--run-dir",
        str(directory),
    ]
    if failure == "interrupt":
        assert play_command(args) == 130
    else:
        with pytest.raises(SystemExit) as error:
            play_command(args)
        assert error.value.code == 1
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == ("interrupted" if failure == "interrupt" else "failed")
    assert report["game"]["winner"] is None
    assert report["game"]["actions"] == 0
    assert report["llm"]["total"]["requests"] == 1
    assert report["llm"]["total"]["fallbacks"] == 0
    assert len(read_lines(directory / "decisions.jsonl")) == 1
    assert not (directory / "match.json").exists()


def test_action_limit_preserves_applied_moves(tmp_path):
    directory = tmp_path / "run"
    with pytest.raises(SystemExit):
        play_command(["--run-dir", str(directory), "--max-steps", "3"])
    report = json.loads((directory / "report.json").read_text())
    moves = read_lines(directory / "moves.jsonl")
    assert report["status"] == "failed"
    assert report["game"]["actions"] == len(moves) == 3
    assert report["game"]["winner"] is None
    assert report["llm"]["total"]["first_try_legal_rate"] is None


def test_existing_run_directory_cannot_be_overwritten(tmp_path):
    sentinel = tmp_path / "report.json"
    sentinel.write_text("existing report")
    with pytest.raises(SystemExit) as error:
        play_command(["--run-dir", str(tmp_path)])
    assert error.value.code == 2
    assert sentinel.read_text() == "existing report"


@pytest.mark.parametrize("flag", ["--trace", "--llm-log"])
def test_run_directory_rejects_conflicting_paths(tmp_path, flag):
    directory = tmp_path / "run"
    with pytest.raises(SystemExit) as error:
        play_command(["--run-dir", str(directory), flag, "other.json"])
    assert error.value.code == 2
    assert not directory.exists()


def test_metrics_distinguish_retries_fallbacks_forced_actions_and_missing_usage():
    metrics = LLMMetrics()
    metrics.add({"status": "forced", "attempt": 0})
    for attempt in (1, 2, 3):
        metrics.add({"status": "invalid", "attempt": attempt, "latency_seconds": 0.5})
    metrics.add({"status": "fallback", "attempt": 3})
    metrics.add(
        {
            "status": "chosen",
            "attempt": 1,
            "latency_seconds": 1.0,
            "usage": {"prompt_tokens": 12, "cost": 0},
        }
    )
    result = metrics.to_dict()
    assert result["decisions"] == 3
    assert result["model_decisions"] == 2
    assert result["requests"] == 4
    assert result["retries"] == 2
    assert result["first_try_legal_rate"] == result["fallback_rate"] == 0.5
    assert result["invalid_response_rate"] == 0.75
    assert result["mean_request_seconds"] == 2.5 / 4
    assert result["usage"]["completion_tokens"]["reported_total"] is None
    assert result["usage"]["cost"] == {"reported_total": 0, "requests_reporting": 1}


def test_prompt_selection_changes_only_shared_instructions():
    state = new_game(12)
    observation = observe(state, state.current_seat, ActionCodec())
    old = render_messages(observation, prompt_version=1)
    new = render_messages(observation, prompt_version=2)
    assert old[0] != new[0]
    assert old[1] == new[1]
    assert old[0]["content"].splitlines()[-1] == new[0]["content"].splitlines()[-1]
    with pytest.raises(ValueError, match="Unknown prompt version"):
        load_system_prompt(999)

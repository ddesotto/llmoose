import json
from io import BytesIO

import pytest

from llmoose.cli import play_command
from llmoose.simulation.traces import load_trace, replay_trace


@pytest.mark.parametrize("model_override", [None, "shell/model", "cli/model"])
def test_dotenv_loads_before_model_and_credentials_are_resolved(
    tmp_path, monkeypatch, model_override
):
    # Register restoration before load_dotenv mutates the process environment.
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL"):
        monkeypatch.setenv(name, "temporary")
        monkeypatch.delenv(name)
    (tmp_path / ".env").write_text(
        "# Local game settings\n"
        'OPENROUTER_API_KEY="file-secret"\n'
        "OPENROUTER_MODEL='file/model'\n"
    )
    if model_override is not None:
        monkeypatch.setenv("OPENROUTER_MODEL", "shell/model")
        monkeypatch.setenv("OPENROUTER_API_KEY", "shell-secret")
    args = ["--policy", "llm", "--step"]
    if model_override == "cli/model":
        args += ["--model", model_override]
    providers = []
    from llmoose.agents.providers import ChatCompletionProvider

    def capture_provider(**settings):
        provider = ChatCompletionProvider(**settings)
        providers.append(provider)
        return provider

    monkeypatch.setattr("llmoose.cli.ChatCompletionProvider", capture_provider)
    monkeypatch.setattr("builtins.input", lambda _: "q")
    assert play_command(args) == 130  # Validate configuration without an API call.
    assert providers[0].model == (model_override or "file/model")
    assert providers[0].api_key == ("shell-secret" if model_override else "file-secret")


def test_watching_preserves_match_and_trace(tmp_path, capsys):
    quiet_path = tmp_path / "quiet.json"
    watch_path = tmp_path / "watch.json"
    args = [
        "--seed",
        "12",
        "--target-score",
        "8",
        "--team-0",
        "heuristic",
        "--team-1",
        "conservative",
    ]
    assert play_command([*args, "--trace", str(quiet_path)]) == 0
    quiet_output = capsys.readouterr().out
    assert quiet_output.count("\n") == 1
    assert (
        play_command([*args, "--watch", "--delay", "0", "--trace", str(watch_path)])
        == 0
    )
    output = capsys.readouterr().out
    assert "North + South (heuristic)" in output
    assert "East + West (conservative)" in output
    assert "Hand 1 | mano: West" in output
    assert "Hand 2 | mano: North" in output
    assert "North: bets 2 stones" in output
    assert "East: folds" in output
    assert "Juego: team 0 +4 (uncontested)" in output
    assert output.endswith(quiet_output)
    quiet_trace = load_trace(quiet_path)
    watch_trace = load_trace(watch_path)
    assert quiet_trace == watch_trace
    assert replay_trace(watch_trace).scores == watch_trace.scores


def test_step_advances_once_per_enter_and_can_quit(tmp_path, monkeypatch, capsys):
    responses = iter(["", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    path = tmp_path / "unfinished.json"
    assert play_command(["--step", "--trace", str(path)]) == 130
    output = capsys.readouterr().out
    assert "1. [mus vote]" in output
    assert "2. [grande]" not in output
    assert "Stopped before match completion." in output
    assert "winner=" not in output
    assert not path.exists()


def test_step_handles_closed_input(monkeypatch, capsys):
    def closed_input(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", closed_input)
    assert play_command(["--step"]) == 130
    assert "Stopped before match completion." in capsys.readouterr().out


def test_watch_shows_replacement_hands(capsys):
    assert (
        play_command(
            ["--policy", "random", "--seed", "1", "--target-score", "8", "--delay", "0"]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "discards" in output
    assert "Now holds:" in output


@pytest.mark.parametrize("delay", ["-1", "nan", "inf"])
def test_invalid_pacing_is_rejected(delay, capsys):
    with pytest.raises(SystemExit) as error:
        play_command(["--delay", delay])
    assert error.value.code == 2
    assert "finite, non-negative" in capsys.readouterr().err


def test_action_limit_reports_an_incomplete_match(tmp_path, capsys):
    path = tmp_path / "unfinished.json"
    with pytest.raises(SystemExit) as error:
        play_command(["--max-steps", "1", "--trace", str(path)])
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "Match stopped:" in output.err
    assert "winner=" not in output.out
    assert not path.exists()


def test_llm_match_uses_openrouter_wire_format_and_saves_inspectable_decisions(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    requests = []

    class FakeOpener:
        def open(self, request, timeout):
            payload = json.loads(request.data)
            requests.append(payload)
            choices = (
                payload["messages"][1]["content"]
                .split("Legal actions (ID: meaning):\n")[1]
                .splitlines()
            )
            choices = dict(line.strip().split(": ", 1) for line in choices)
            chosen = next(
                (
                    key
                    for kind in ("no mus", "pass", "fold")
                    for key, value in choices.items()
                    if value == kind
                ),
                next(iter(choices)),
            )
            return BytesIO(
                json.dumps(
                    {
                        "model": "test/resolved",
                        "choices": [
                            {
                                "message": {"content": f"<answer>{chosen}</answer>"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 8},
                    }
                ).encode()
            )

    monkeypatch.setattr(
        "llmoose.agents.providers.build_opener", lambda *_: FakeOpener()
    )
    log = tmp_path / "runs" / "decisions.jsonl"
    trace = tmp_path / "match.json"
    assert (
        play_command(
            [
                "--seed",
                "12",
                "--target-score",
                "8",
                "--team-0",
                "llm",
                "--team-1",
                "heuristic",
                "--model",
                "test/model",
                "--watch",
                "--delay",
                "0",
                "--llm-log",
                str(log),
                "--trace",
                str(trace),
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert "winner=" in output.out
    assert "LLM model: test/model" in output.out
    assert "invalid_outputs=0" in output.err
    assert requests and all(request["model"] == "test/model" for request in requests)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    chosen = [record for record in records if record["status"] == "chosen"]
    assert len(chosen) == len(requests)
    assert all(record["seat"] in (0, 2) for record in records)
    assert all(record["chosen_id"] in record["legal_ids"] for record in records)
    assert [record["messages"] for record in chosen] == [
        request["messages"] for request in requests
    ]
    assert all(record["model"] == "test/resolved" for record in chosen)
    assert "test-secret" not in log.read_text()
    saved = load_trace(trace)
    assert saved.metadata["model"] == "test/model"
    assert replay_trace(saved).scores == saved.scores


def test_quitting_before_first_llm_turn_makes_no_api_call(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    monkeypatch.setattr("builtins.input", lambda _: "q")

    def unexpected(*_):
        pytest.fail("Quit must happen before requesting inference")

    monkeypatch.setattr("llmoose.agents.providers.build_opener", unexpected)
    assert play_command(["--policy", "llm", "--model", "test/model", "--step"]) == 130
    assert "requests=0" in capsys.readouterr().err


def test_missing_openrouter_key_fails_before_starting_a_match(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit) as error:
        play_command(["--policy", "llm", "--model", "test/model"])
    assert error.value.code == 2
    assert "Set OPENROUTER_API_KEY" in capsys.readouterr().err


def test_strict_cli_stops_and_preserves_invalid_response_log(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")

    class InvalidOpener:
        def open(self, request, timeout):
            return BytesIO(b'{"choices":[{"message":{"content":"bad"}}]}')

    monkeypatch.setattr(
        "llmoose.agents.providers.build_opener", lambda *_: InvalidOpener()
    )
    log = tmp_path / "decisions.jsonl"
    trace = tmp_path / "match.json"
    with pytest.raises(SystemExit) as error:
        play_command(
            [
                "--policy",
                "llm",
                "--model",
                "test/model",
                "--llm-strict",
                "--llm-log",
                str(log),
                "--trace",
                str(trace),
            ]
        )
    assert error.value.code == 1
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["status"] == "invalid"
    assert records[0]["response"] == "bad"
    assert not trace.exists()
    assert "requests=1 invalid_outputs=1 fallbacks=0" in capsys.readouterr().err

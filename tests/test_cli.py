import pytest

from llmoose.cli import play_command
from llmoose.simulation.traces import load_trace, replay_trace


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

import pytest

from llmoose.agents.baselines import ConservativePolicy, HeuristicPolicy
from llmoose.benchmark import DealSuite, baseline_factory, run_benchmark
from llmoose.game import EnvEpisode, MusEnv
from llmoose.game.actions import ActionCodec
from llmoose.game.state import SEATS
from llmoose.rules.ruleset import Ruleset
from llmoose.simulation import (
    load_trace,
    replay_trace,
    run_match,
    save_trace,
    trace_from_match,
)


def _conservative_match(target_score: int = 4):
    ruleset = Ruleset(target_score=target_score)
    codec = ActionCodec(target_score)
    return run_match({seat: ConservativePolicy(codec) for seat in SEATS}, 12, ruleset)


def test_environment_accepts_action_ids_and_reports_score_delta() -> None:
    env = MusEnv(Ruleset(target_score=2), seed=9)
    observation = env.reset()
    first_action = next(
        action_id
        for action_id, allowed in enumerate(observation.legal_action_mask)
        if allowed
    )
    result = env.step(first_action)

    assert result.info["actor"] == observation.seat
    assert result.reward == (0.0, 0.0)
    assert result.observation is not None
    assert result.observation.seat == env.state.current_seat


@pytest.mark.parametrize("action_id", (-1, 10_000, True))
def test_environment_rejects_out_of_range_or_non_integer_action_ids(action_id) -> None:
    env = MusEnv(seed=1)
    env.reset()

    with pytest.raises(ValueError, match="action ID"):
        env.step(action_id)


def test_environment_rejects_an_unknown_episode_granularity() -> None:
    assert MusEnv(seed=1).episode is EnvEpisode.MATCH
    assert MusEnv(seed=1, episode="hand").episode is EnvEpisode.HAND

    with pytest.raises(ValueError):
        MusEnv(seed=1, episode="hnd")


def test_trace_round_trip_and_replay(tmp_path) -> None:
    trace = trace_from_match(_conservative_match(), {"policy": "conservative"})
    path = tmp_path / "match.json"
    save_trace(trace, path)
    loaded = load_trace(path)

    assert loaded == trace
    assert replay_trace(loaded).scores == trace.scores


def test_lance_results_are_preserved_over_multiple_hands() -> None:
    result = _conservative_match(target_score=40)
    resolved_events = [
        event for event in result.final_state.events if event.kind == "lance_resolved"
    ]

    assert result.final_state.hand_number > 0
    assert len(result.final_state.lance_results) == len(resolved_events)


def test_heuristic_policy_finishes_a_match() -> None:
    ruleset = Ruleset(target_score=8)
    codec = ActionCodec(ruleset.target_score)
    result = run_match({seat: HeuristicPolicy(codec) for seat in SEATS}, 9, ruleset)

    assert result.scores[result.winner] >= ruleset.target_score


def test_fixed_deal_benchmark_aggregates_results() -> None:
    ruleset = Ruleset(target_score=4)
    report = run_benchmark(
        DealSuite("smoke-v1", (1, 2, 3)),
        baseline_factory(ConservativePolicy),
        ruleset,
    )

    assert report.games == 3
    assert sum(report.team_wins) == report.games
    assert report.lance_counts

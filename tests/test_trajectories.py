import json

import pytest

from llmoose.agents.baselines import HeuristicPolicy, RandomPolicy
from llmoose.game import EnvEpisode, MusEnv
from llmoose.game.actions import ActionCodec
from llmoose.game.state import SEATS, Seat, team_for
from llmoose.rules.ruleset import Ruleset
from llmoose.training import collect_episode, read_episodes, write_episodes

RULESET = Ruleset(target_score=8)
CODEC = ActionCodec(RULESET.target_score)

SCORING_HAND = {"seed": 0, "hand_number": 0}
SHORT_HAND = {"seed": 0, "hand_number": 1}


def _policies():
    return {seat: HeuristicPolicy(CODEC) for seat in SEATS}


def _random_policies():
    return {seat: RandomPolicy(CODEC, int(seat)) for seat in SEATS}


def _hand_env():
    return MusEnv(RULESET, episode=EnvEpisode.HAND)


def _collect(policies=None, render=None, **hand):
    return collect_episode(
        _hand_env(),
        policies or _policies(),
        render=render,
        **(hand or SCORING_HAND),
    )


def _played_action_count(**hand) -> int:
    """Play the same hand outside the collector and count its actions."""

    env = _hand_env()
    policies = _policies()
    observation = env.reset(**(hand or SCORING_HAND))
    actions = 0
    while True:
        result = env.step(policies[observation.seat].act(observation))
        actions += 1
        if result.terminated:
            return actions
        observation = result.observation


def test_every_seat_gets_an_episode_of_the_same_hand() -> None:
    episodes = _collect()

    assert tuple(episode.seat for episode in episodes) == SEATS
    assert {(episode.seed, episode.hand_number) for episode in episodes} == {
        (SCORING_HAND["seed"], SCORING_HAND["hand_number"])
    }
    assert all(episode.target_score == RULESET.target_score for episode in episodes)


def test_turn_counts_per_seat_sum_to_the_hands_action_count() -> None:
    episodes = _collect()

    assert sum(len(episode.turns) for episode in episodes) == _played_action_count()
    assert all(episode.turns for episode in episodes)


def test_returns_are_equal_and_opposite_between_the_partnerships() -> None:
    episodes = _collect()
    returns = {episode.team: episode.team_return for episode in episodes}

    assert len(returns) == 2
    assert returns[0] != 0, "a drawn hand would not prove the sign convention"
    assert returns[0] == -returns[1]
    assert all(episode.team_return == returns[episode.team] for episode in episodes), (
        "partners share one return"
    )


def test_mano_offset_is_zero_for_mano_and_counts_up_the_table() -> None:
    env = _hand_env()
    env.reset(**SCORING_HAND)
    mano = env.state.mano
    episodes = _collect()

    by_seat = {episode.seat: episode for episode in episodes}
    assert by_seat[mano].mano_offset == 0
    for seat, episode in by_seat.items():
        assert episode.mano_offset == (int(seat) - int(mano)) % 4
        assert all(turn.mano_offset == episode.mano_offset for turn in episode.turns)
        assert all(turn.seat is seat for turn in episode.turns)


def test_every_recorded_choice_was_legal_at_that_turn() -> None:
    for episode in _collect():
        for turn in episode.turns:
            assert turn.chosen_id in turn.legal_ids
            assert CODEC.decode(turn.chosen_id) is not None


def test_a_renderer_supplies_the_prompt_text_and_is_optional() -> None:
    rendered = _collect(render=lambda observation: f"seat={int(observation.seat)}")

    assert all(
        turn.prompt_text == f"seat={int(turn.seat)}"
        for episode in rendered
        for turn in episode.turns
    )
    assert all(
        turn.prompt_text == "" for episode in _collect() for turn in episode.turns
    )


def test_collecting_rejects_a_match_env_or_an_incomplete_policy_mapping() -> None:
    with pytest.raises(ValueError, match="episode='hand'"):
        collect_episode(MusEnv(RULESET), _policies())

    policies = _policies()
    del policies[Seat.WEST]
    with pytest.raises(ValueError, match="every seat"):
        collect_episode(_hand_env(), policies)


def test_episodes_round_trip_through_jsonl(tmp_path) -> None:
    episodes = _collect(render=lambda observation: observation.phase.value)
    path = tmp_path / "episodes.jsonl"

    assert write_episodes(episodes, path) == 4
    assert read_episodes(path) == episodes


def test_jsonl_appends_so_datasets_grow_cheaply(tmp_path) -> None:
    path = tmp_path / "episodes.jsonl"
    write_episodes(_collect(**SCORING_HAND), path)
    written = write_episodes(_collect(**SHORT_HAND), path, append=True)
    loaded = read_episodes(path)

    assert written == 4
    assert len(loaded) == 8
    assert {episode.hand_number for episode in loaded} == {0, 1}


def test_a_seat_that_never_acted_keeps_its_return_and_can_be_dropped(
    tmp_path,
) -> None:
    episodes = _collect(_random_policies(), **SHORT_HAND)
    silent = [episode for episode in episodes if not episode.turns]

    assert silent, "this hand is meant to end before every seat has acted"
    for episode in silent:
        assert episode.team == team_for(episode.seat)
        assert episode.team_return == next(
            other.team_return for other in episodes if other.team == episode.team
        )

    path = tmp_path / "episodes.jsonl"
    assert write_episodes(episodes, path, drop_empty=True) == 4 - len(silent)
    assert all(episode.turns for episode in read_episodes(path))


def test_every_jsonl_line_shares_one_key_set(tmp_path) -> None:
    """A dataset loader infers one schema for the file, so lines cannot differ."""

    path = tmp_path / "episodes.jsonl"
    write_episodes(_collect(**SCORING_HAND) + _collect(**SHORT_HAND), path)
    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert len(records) == 8
    assert len({tuple(sorted(record)) for record in records}) == 1
    assert {tuple(sorted(turn)) for record in records for turn in record["turns"]} == {
        ("chosen_id", "legal_ids", "mano_offset", "phase", "prompt_text", "seat")
    }

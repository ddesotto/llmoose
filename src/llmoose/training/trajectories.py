"""Per-seat records of what a policy saw, chose, and earned in one hand."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Union

from llmoose.agents.protocol import Policy
from llmoose.game.environment import EnvEpisode, MusEnv
from llmoose.game.state import SEATS, Phase, Seat, team_for
from llmoose.observations.core import Observation

EPISODE_SCHEMA_VERSION = 1

PromptRenderer = Callable[[Observation], str]


@dataclass(frozen=True)
class TurnRecord:
    """A decision point in a hand."""

    seat: Seat
    mano_offset: int
    phase: Phase
    prompt_text: str
    legal_ids: Tuple[int, ...]
    chosen_id: int


@dataclass(frozen=True)
class Episode:
    """One seat's turns in a hand, with the hand's team differential as the return.

    ``team_return`` is the undiscounted return for a team in ``turns``. 
    !! Not normalized, leave that to trainign loop
    """

    turns: Tuple[TurnRecord, ...]
    team_return: float
    seed: int
    hand_number: int
    seat: Seat
    team: int
    mano_offset: int
    target_score: int


def collect_episode(
    env: MusEnv,
    policies: Mapping[Seat, Policy],
    seed: Optional[int] = None,
    hand_number: int = 0,
    render: Optional[PromptRenderer] = None,
    max_steps: int = 1_000,
) -> Tuple[Episode, ...]:
    """Play one hand, record all decisions"""

    if env.episode is not EnvEpisode.HAND:
        raise ValueError("collect_episode needs an env with episode='hand'")
    if set(policies) != set(SEATS):
        raise ValueError("A hand needs exactly one policy for every seat")

    observation = env.reset(seed=seed, hand_number=hand_number)
    mano = env.state.mano
    played_seed = env.seed
    played_hand = env.state.hand_number
    turns: list[TurnRecord] = []
    while True:
        if len(turns) >= max_steps:
            raise RuntimeError("Hand exceeded max_steps without terminating")
        seat = observation.seat
        action = policies[seat].act(observation)
        turns.append(
            TurnRecord(
                seat=seat,
                mano_offset=_mano_offset(seat, mano),
                phase=observation.phase,
                prompt_text="" if render is None else render(observation),
                legal_ids=tuple(
                    action_id
                    for action_id, allowed in enumerate(observation.legal_action_mask)
                    if allowed
                ),
                chosen_id=env.codec.encode(action),
            )
        )
        result = env.step(action)
        if result.terminated:
            break
        observation = result.observation

    hand_reward = result.info.get("hand_reward")
    if hand_reward is None:
        raise RuntimeError("A terminated hand episode must report a hand_reward")
    return tuple(
        Episode(
            turns=tuple(turn for turn in turns if turn.seat is seat),
            team_return=float(
                hand_reward[team_for(seat)] - hand_reward[1 - team_for(seat)]
            ),
            seed=played_seed,
            hand_number=played_hand,
            seat=seat,
            team=team_for(seat),
            mano_offset=_mano_offset(seat, mano),
            target_score=env.ruleset.target_score,
        )
        for seat in SEATS
    )


def write_episodes(
    episodes: Iterable[Episode],
    path: Union[str, Path],
    append: bool = False,
    drop_empty: bool = False,
) -> int:
    """Write one JSON episode per line and return how many lines were written."""

    lines = [
        json.dumps(episode_to_dict(episode), sort_keys=True)
        for episode in episodes
        if episode.turns or not drop_empty
    ]
    with Path(path).open("a" if append else "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")
    return len(lines)


def read_episodes(path: Union[str, Path]) -> Tuple[Episode, ...]:
    """Read a JSONL episode file written by :func:`write_episodes`."""

    with Path(path).open(encoding="utf-8") as handle:
        return tuple(
            episode_from_dict(json.loads(line)) for line in handle if line.strip()
        )


def episode_to_dict(episode: Episode) -> dict[str, Any]:
    """Render one episode as a flat JSON object, self-describing per line."""

    return {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "seed": episode.seed,
        "hand_number": episode.hand_number,
        "seat": int(episode.seat),
        "team": episode.team,
        "mano_offset": episode.mano_offset,
        "target_score": episode.target_score,
        "team_return": episode.team_return,
        "turns": [_turn_to_dict(turn) for turn in episode.turns],
    }


def episode_from_dict(value: Mapping[str, Any]) -> Episode:

    if value.get("schema_version") != EPISODE_SCHEMA_VERSION:
        raise ValueError("Unsupported episode schema version")
    return Episode(
        turns=tuple(_turn_from_dict(turn) for turn in value["turns"]),
        team_return=float(value["team_return"]),
        seed=value["seed"],
        hand_number=value["hand_number"],
        seat=Seat(value["seat"]),
        team=value["team"],
        mano_offset=value["mano_offset"],
        target_score=value["target_score"],
    )


def _mano_offset(seat: Seat, mano: Seat) -> int:
    """Return a seat's position relative to mano, who is always offset zero."""
    return (int(seat) - int(mano)) % len(SEATS)


def _turn_to_dict(turn: TurnRecord) -> dict[str, Any]:
    return {
        "seat": int(turn.seat),
        "mano_offset": turn.mano_offset,
        "phase": turn.phase.value,
        "prompt_text": turn.prompt_text,
        "legal_ids": list(turn.legal_ids),
        "chosen_id": turn.chosen_id,
    }


def _turn_from_dict(value: Mapping[str, Any]) -> TurnRecord:
    return TurnRecord(
        seat=Seat(value["seat"]),
        mano_offset=value["mano_offset"],
        phase=Phase(value["phase"]),
        prompt_text=value["prompt_text"],
        legal_ids=tuple(value["legal_ids"]),
        chosen_id=value["chosen_id"],
    )

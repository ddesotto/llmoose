"""Run and replay complete matches between four policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from llmoose.agents.protocol import Policy
from llmoose.game.actions import Action, ActionCodec
from llmoose.game.engine import apply_action
from llmoose.game.state import SEATS, GameState, Phase, Seat, new_game
from llmoose.observations.core import observe
from llmoose.rules.ruleset import Ruleset


@dataclass(frozen=True)
class MatchResult:
    """The terminal result and action trace of one seeded match."""

    seed: int
    scores: Tuple[int, int]
    winner: int
    actions: Tuple[Action, ...]
    final_state: GameState
    initial_dealer: Seat


def run_match(
    policies: Mapping[Seat, Policy],
    seed: int,
    ruleset: Optional[Ruleset] = None,
    max_steps: int = 10_000,
    dealer: Seat = Seat.NORTH,
) -> MatchResult:
    """Run four policies until one partnership reaches the target score."""

    if set(policies) != set(SEATS):
        raise ValueError("A match needs exactly one policy for every seat")
    state = new_game(seed=seed, ruleset=ruleset, dealer=dealer)
    codec = ActionCodec(state.ruleset.target_score)
    actions = []
    while state.phase is not Phase.COMPLETE:
        if len(actions) >= max_steps:
            raise RuntimeError("Match exceeded max_steps without reaching a winner")
        actor = state.current_seat
        if actor is None:
            raise RuntimeError("A non-terminal state must name a current seat")
        action = policies[actor].act(observe(state, actor, codec))
        state = apply_action(state, action)
        actions.append(action)
    winner = 0 if state.scores[0] >= state.ruleset.target_score else 1
    return MatchResult(seed, state.scores, winner, tuple(actions), state, dealer)


def replay_match(
    actions: Sequence[Action],
    seed: int,
    ruleset: Optional[Ruleset] = None,
    dealer: Seat = Seat.NORTH,
) -> GameState:
    """Replay an action trace and return its terminal game state."""

    state = new_game(seed=seed, ruleset=ruleset, dealer=dealer)
    for action in actions:
        if state.phase is Phase.COMPLETE:
            raise ValueError("Trace contains actions after the match completed")
        state = apply_action(state, action)
    if state.phase is not Phase.COMPLETE:
        raise ValueError("Trace ended before the match completed")
    return state

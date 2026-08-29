"""
Observations on game state, for a single seat at a single decision point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from llmoose.game.actions import ActionCodec
from llmoose.game.engine import legal_actions
from llmoose.game.state import Event, GameState, Phase, Seat, team_for
from llmoose.rules.cards import Card
from llmoose.rules.hands import Lance


@dataclass(frozen=True)
class Observation:
    """The complete policy input for one seat at one decision point."""

    seat: Seat
    team: int
    private_hand: Tuple[Card, ...]
    phase: Phase
    current_seat: Seat
    scores: Tuple[int, int]
    current_lance: Optional[Lance]
    discard_counts: Tuple[int, int, int, int]
    public_events: Tuple[Event, ...]
    legal_action_mask: Tuple[bool, ...]


def observe(
    state: GameState,
    seat: Seat,
    codec: ActionCodec,
) -> Observation:
    return Observation(
        seat=seat,
        team=team_for(seat),
        private_hand=state.hands[seat],
        phase=state.phase,
        current_seat=state.current_seat,
        scores=state.scores,
        current_lance=state.current_lance,
        discard_counts=state.discard_counts,
        public_events=state.events,
        legal_action_mask=codec.mask(legal_actions(state, seat)),
    )

"""
Observations on game state, for a single seat at a single decision point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from llmoose.game.actions import ActionCodec
from llmoose.game.engine import legal_actions
from llmoose.game.state import Event, GameState, PendingLance, Phase, Seat, team_for
from llmoose.rules.cards import Card
from llmoose.rules.hands import Lance
from llmoose.rules.ruleset import LEGACY_RULESET_ID


@dataclass(frozen=True)
class Observation:
    """The complete policy input for one seat at one decision point."""

    seat: Seat
    team: int
    private_hand: Tuple[Card, ...]
    phase: Phase
    current_seat: Optional[Seat]
    scores: Tuple[int, int]
    current_lance: Optional[Lance]
    discard_counts: Tuple[int, int, int, int]
    own_discard_history: Tuple[Tuple[Card, ...], ...]
    public_events: Tuple[Event, ...]
    legal_action_mask: Tuple[bool, ...]
    mano: Seat
    hand_number: int
    target_score: int
    stake: int
    bidding_team: Optional[int]
    refusal_points: int
    lance_seats: Tuple[Seat, ...]
    ruleset_id: str = LEGACY_RULESET_ID
    is_ordago: bool = False
    pending_lances: Tuple[PendingLance, ...] = ()


def observe(
    state: GameState,
    seat: Seat,
    codec: ActionCodec,
) -> Observation:
    if codec.size != ActionCodec(state.ruleset.target_score).size:
        raise ValueError("Observation codec does not match ruleset target")
    return Observation(
        seat=seat,
        team=team_for(seat),
        private_hand=state.hands[seat],
        phase=state.phase,
        current_seat=state.current_seat,
        scores=state.scores,
        current_lance=state.current_lance,
        discard_counts=state.discard_counts,
        own_discard_history=state.own_discard_history[int(seat)],
        public_events=state.events,
        legal_action_mask=codec.mask(legal_actions(state, seat)),
        mano=state.mano,
        hand_number=state.hand_number,
        target_score=state.ruleset.target_score,
        stake=state.stake,
        bidding_team=state.bidding_team,
        refusal_points=state.refusal_points,
        lance_seats=state.lance_seats,
        ruleset_id=state.ruleset.id,
        is_ordago=state.is_ordago,
        pending_lances=state.pending_lances,
    )

"""Authoritative legal-action and transition functions for Mus v1.

The engine owns all rule validation. Policies receive an observation and can only
return a proposed action; they never mutate or interpret ``GameState`` directly.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from random import Random
from typing import Iterable, Optional, Sequence, Tuple

from llmoose.game.actions import Action, ActionKind
from llmoose.game.state import (
    SEATS,
    Event,
    GameState,
    LanceResult,
    Phase,
    Seat,
    deal_hand,
    freeze_hands,
    next_seat,
    team_for,
)
from llmoose.rules.hands import Lance, evaluate_hand, winner_for_lance
from llmoose.rules.ruleset import BidResponseProtocol


class IllegalAction(ValueError):
    """Raised when an action is not legal for the current state and actor."""


def legal_actions(state: GameState, seat: Seat) -> Tuple[Action, ...]:
    """Return every action ``seat`` may take in ``state``.

    A non-acting seat and a completed match always have an empty action set.
    """

    if state.phase is Phase.COMPLETE or seat != state.current_seat:
        return ()
    if state.phase is Phase.MUS_VOTE:
        return (Action(ActionKind.MUS), Action(ActionKind.NO_MUS))
    if state.phase is Phase.DISCARD:
        return tuple(
            Action(ActionKind.DISCARD, card_indices=indices)
            for size in range(1, 5)
            for indices in combinations(range(4), size)
        )
    if state.phase is Phase.DECLARE_PARES:
        return (
            (Action(ActionKind.DECLARE_PARES),)
            if evaluate_hand(state.hand_for(seat)).pares.points
            else (Action(ActionKind.DECLARE_NO_PARES),)
        )
    if state.phase is Phase.DECLARE_JUEGO:
        return (
            (Action(ActionKind.DECLARE_JUEGO),)
            if evaluate_hand(state.hand_for(seat)).has_juego
            else (Action(ActionKind.DECLARE_NO_JUEGO),)
        )
    if state.current_lance is None:
        raise RuntimeError("A lance phase must name its active lance")
    if state.stake == 0:
        return (
            Action(ActionKind.PASS),
            *_stake_actions(ActionKind.BET, 2, state.ruleset.target_score),
            Action(ActionKind.ORDAGO),
        )
    if state.bidding_team == team_for(seat):
        raise RuntimeError("The bidding team cannot be asked to answer its own bet")
    responses = (
        Action(ActionKind.CALL),
        Action(ActionKind.FOLD),
        *_stake_actions(ActionKind.RAISE, state.stake + 1, state.ruleset.target_score),
    )
    if state.stake == state.ruleset.target_score:
        return responses
    return responses + (Action(ActionKind.ORDAGO),)


def apply_action(state: GameState, action: Action) -> GameState:
    """Validate and apply one action, returning the next immutable state."""

    actor = state.current_seat
    if actor is None or action not in legal_actions(state, actor):
        raise IllegalAction("{} is not legal in the current state".format(action))
    if state.phase is Phase.MUS_VOTE:
        return _apply_mus_vote(state, actor, action)
    if state.phase is Phase.DISCARD:
        return _apply_discard(state, actor, action)
    if state.phase in {Phase.DECLARE_PARES, Phase.DECLARE_JUEGO}:
        return _apply_declaration(state, actor, action)
    return _apply_lance_action(state, actor, action)


def _apply_mus_vote(state: GameState, actor: Seat, action: Action) -> GameState:
    state = _event(state, "mus_vote", actor, action=action.kind.value)
    if action.kind is ActionKind.NO_MUS:
        return _start_lance(state, Lance.GRANDE, Phase.GRANDE, SEATS)
    voters = state.mus_voters + (actor,)
    if len(voters) == len(SEATS):
        return replace(
            state,
            phase=Phase.DISCARD,
            current_seat=state.mano,
            mus_voters=(),
        )
    return replace(state, current_seat=_next_priority(state, actor), mus_voters=voters)


def _apply_discard(state: GameState, actor: Seat, action: Action) -> GameState:
    hand = state.hand_for(actor)
    removed = tuple(hand[index] for index in action.card_indices)
    kept = tuple(
        card for index, card in enumerate(hand) if index not in action.card_indices
    )
    replacements, stock, discards = _draw_replacements(
        state, len(removed), state.discards + removed
    )
    hands = dict(state.hands)
    hands[actor] = kept + replacements
    counts = list(state.discard_counts)
    counts[int(actor)] += len(removed)
    state = _event(state, "discard", actor, count=str(len(removed)))
    state = replace(
        state,
        hands=freeze_hands(hands),
        stock=stock,
        discards=discards,
        discard_counts=tuple(counts),
    )
    if actor == state.priority[-1]:
        return replace(
            state,
            phase=Phase.MUS_VOTE,
            current_seat=state.mano,
            mus_voters=(),
        )
    return replace(state, current_seat=_next_priority(state, actor))


def _apply_declaration(state: GameState, actor: Seat, action: Action) -> GameState:
    is_pares = state.phase is Phase.DECLARE_PARES
    declared = action.kind in {ActionKind.DECLARE_PARES, ActionKind.DECLARE_JUEGO}
    state = _event(state, "declaration", actor, value=action.kind.value)
    declaration_seats = state.declaration_seats + (actor,)
    if is_pares:
        declared_pares = state.declared_pares + ((actor,) if declared else ())
        state = replace(
            state,
            declared_pares=declared_pares,
            declaration_seats=declaration_seats,
        )
    else:
        declared_juego = state.declared_juego + ((actor,) if declared else ())
        state = replace(
            state,
            declared_juego=declared_juego,
            declaration_seats=declaration_seats,
        )
    if len(declaration_seats) < len(SEATS):
        return replace(state, current_seat=_next_priority(state, actor))
    return _after_declarations(state, is_pares)


def _after_declarations(state: GameState, is_pares: bool) -> GameState:
    declared = state.declared_pares if is_pares else state.declared_juego
    if is_pares:
        if declared:
            state = _start_or_resolve_lance(state, Lance.PARES, Phase.PARES, declared)
            if state.current_lance is not None:
                return state
        return _begin_juego_declaration(state)
    if declared:
        state = _start_or_resolve_lance(
            state, Lance.JUEGO, Phase.JUEGO_OR_PUNTO, declared
        )
        if state.current_lance is not None:
            return state
        return _start_lance(state, Lance.PUNTO, Phase.JUEGO_OR_PUNTO, SEATS)
    return _start_lance(state, Lance.PUNTO, Phase.JUEGO_OR_PUNTO, SEATS)


def _begin_juego_declaration(state: GameState) -> GameState:
    return replace(
        state,
        phase=Phase.DECLARE_JUEGO,
        current_seat=state.mano,
        declaration_seats=(),
        current_lance=None,
        lance_seats=(),
        passed_seats=(),
        bidding_team=None,
        stake=0,
        refusal_points=0,
    )


def _start_or_resolve_lance(
    state: GameState, lance: Lance, phase: Phase, eligible: Sequence[Seat]
) -> GameState:
    eligible = tuple(seat for seat in state.priority if seat in eligible)
    if len({team_for(seat) for seat in eligible}) == 1:
        return _resolve_natural_lance(state, lance, eligible, "uncontested")
    return _start_lance(state, lance, phase, eligible)


def _start_lance(
    state: GameState, lance: Lance, phase: Phase, eligible: Iterable[Seat]
) -> GameState:
    lance_seats = tuple(seat for seat in state.priority if seat in eligible)
    if not lance_seats:
        raise ValueError("A lance needs at least one eligible seat")
    state = _event(state, "lance_started", lance=lance.value)
    return replace(
        state,
        phase=phase,
        current_seat=lance_seats[0],
        current_lance=lance,
        lance_seats=lance_seats,
        passed_seats=(),
        bidding_team=None,
        stake=0,
        refusal_points=0,
    )


def _apply_lance_action(state: GameState, actor: Seat, action: Action) -> GameState:
    lance = state.current_lance
    assert lance is not None
    if action.kind is ActionKind.PASS:
        passed = state.passed_seats + (actor,)
        state = _event(state, "pass", actor, lance=lance.value)
        if len(passed) == len(state.lance_seats):
            return _resolve_natural_lance(
                replace(state, passed_seats=passed), lance, state.lance_seats, "passed"
            )
        return replace(
            state,
            passed_seats=passed,
            current_seat=_next_from(state, actor, state.lance_seats),
        )
    if action.kind in {ActionKind.BET, ActionKind.RAISE, ActionKind.ORDAGO}:
        refusal_points = state.stake or 1
        stake = (
            state.ruleset.target_score
            if action.kind is ActionKind.ORDAGO
            else action.amount
        )
        state = _event(state, "bid", actor, lance=lance.value, stake=str(stake))
        responders = _responders(state, actor)
        return replace(
            state,
            current_seat=_next_from(state, actor, responders),
            bidding_team=team_for(actor),
            stake=stake,
            refusal_points=refusal_points,
            passed_seats=(),
        )
    if action.kind is ActionKind.FOLD:
        assert state.bidding_team is not None
        assert state.refusal_points > 0
        state = _event(state, "fold", actor, lance=lance.value)
        return _resolve_lance(
            state, lance, state.bidding_team, state.refusal_points, "fold"
        )
    if action.kind is ActionKind.CALL:
        state = _event(state, "call", actor, lance=lance.value, stake=str(state.stake))
        winner = _winner(state, lance, state.lance_seats)
        points = (
            state.ruleset.target_score
            if state.stake == state.ruleset.target_score
            else state.stake
        )
        return _resolve_lance(state, lance, team_for(winner), points, "called")
    raise AssertionError("Unexpected lance action: {}".format(action.kind))


def _resolve_natural_lance(
    state: GameState, lance: Lance, eligible: Sequence[Seat], reason: str
) -> GameState:
    winner = _winner(state, lance, eligible)
    winning_team = team_for(winner)
    points = _natural_points(state, lance, winning_team, eligible)
    return _resolve_lance(state, lance, winning_team, points, reason)


def _winner(state: GameState, lance: Lance, eligible: Sequence[Seat]) -> Seat:
    hands = {seat: state.hand_for(seat) for seat in eligible}
    priority = tuple(seat for seat in state.priority if seat in hands)
    return winner_for_lance(hands, priority, lance)


def _natural_points(
    state: GameState,
    lance: Lance,
    winning_team: int,
    eligible: Sequence[Seat],
) -> int:
    if lance is Lance.PARES:
        return sum(
            evaluate_hand(state.hand_for(seat)).pares.points
            for seat in eligible
            if team_for(seat) == winning_team
        )
    if lance is Lance.JUEGO:
        return sum(
            evaluate_hand(state.hand_for(seat)).juego_points
            for seat in eligible
            if team_for(seat) == winning_team
        )
    return 1


def _resolve_lance(
    state: GameState, lance: Lance, winning_team: int, points: int, reason: str
) -> GameState:
    scores = list(state.scores)
    scores[winning_team] += points
    result = LanceResult(lance, winning_team, points, reason)
    state = _event(
        state,
        "lance_resolved",
        lance=lance.value,
        team=str(winning_team),
        points=str(points),
        reason=reason,
    )
    state = replace(
        state,
        scores=tuple(scores),
        current_lance=None,
        lance_seats=(),
        passed_seats=(),
        bidding_team=None,
        stake=0,
        refusal_points=0,
        lance_results=state.lance_results + (result,),
    )
    if scores[winning_team] >= state.ruleset.target_score:
        return replace(state, phase=Phase.COMPLETE, current_seat=None)
    if lance is Lance.GRANDE:
        return _start_lance(state, Lance.CHICA, Phase.CHICA, SEATS)
    if lance is Lance.CHICA:
        return replace(
            state,
            phase=Phase.DECLARE_PARES,
            current_seat=state.mano,
            declaration_seats=(),
        )
    if lance is Lance.PARES:
        return _begin_juego_declaration(state)
    return deal_hand(
        seed=state.seed,
        ruleset=state.ruleset,
        dealer=next_seat(state.dealer),
        scores=tuple(scores),
        hand_number=state.hand_number + 1,
        lance_results=state.lance_results,
        events=state.events,
    )


def _draw_replacements(
    state: GameState, count: int, discards: Tuple
) -> Tuple[Tuple, Tuple, Tuple]:
    stock = list(state.stock)
    recycle = list(discards)
    if len(stock) < count:
        recycle_seed = "{}:{}:{}".format(
            state.seed, state.hand_number, len(state.events)
        )
        Random(recycle_seed).shuffle(recycle)
        stock.extend(recycle)
        recycle = []
    if len(stock) < count:
        raise RuntimeError("Not enough cards to refill a Mus discard")
    return tuple(stock.pop() for _ in range(count)), tuple(stock), tuple(recycle)


def _next_priority(state: GameState, actor: Seat) -> Seat:
    return _next_from(state, actor, state.priority)


def _responders(state: GameState, actor: Seat) -> Tuple[Seat, ...]:
    if state.ruleset.bid_response_protocol is BidResponseProtocol.NEXT_OPPOSING_SEAT:
        return tuple(
            seat for seat in state.lance_seats if team_for(seat) != team_for(actor)
        )
    raise ValueError(
        "Unsupported bid response protocol: {}".format(
            state.ruleset.bid_response_protocol
        )
    )


def _next_from(state: GameState, actor: Seat, seats: Sequence[Seat]) -> Seat:
    actor_index = state.priority.index(actor)
    candidates = state.priority[actor_index + 1 :] + state.priority[: actor_index + 1]
    for candidate in candidates:
        if candidate in seats:
            return candidate
    raise ValueError("No next seat available")


def _stake_actions(kind: ActionKind, lower: int, upper: int) -> Tuple[Action, ...]:
    return tuple(Action(kind, amount=amount) for amount in range(lower, upper + 1))


def _event(
    state: GameState, kind: str, actor: Optional[Seat] = None, **detail: str
) -> GameState:
    event = Event(
        index=len(state.events),
        kind=kind,
        actor=actor,
        detail=tuple(sorted(detail.items())),
    )
    return replace(state, events=state.events + (event,))

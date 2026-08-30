"""Immutable state records for a deterministic, no-señas Mus match."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from random import Random
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

from llmoose.rules.cards import Card, spanish_deck
from llmoose.rules.hands import Lance
from llmoose.rules.ruleset import Ruleset


class Seat(IntEnum):
    """Seats in counter-clockwise table order."""

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


SEATS: Tuple[Seat, ...] = tuple(Seat)


def next_seat(seat: Seat) -> Seat:
    """Return the next seat in counter-clockwise table order."""

    return Seat((int(seat) + 1) % len(SEATS))


def team_for(seat: Seat) -> int:
    """Return a seat's partnership (North/South or East/West)."""

    return int(seat) % 2


def mano_for(dealer: Seat) -> Seat:
    """Return the player to the dealer's right, who acts first."""

    return Seat((int(dealer) - 1) % len(SEATS))


def dealer_for(initial_dealer: Seat, hand_number: int) -> Seat:
    """Return the dealer for a given hand number, counting from the first one."""

    return Seat((int(initial_dealer) + hand_number) % len(SEATS))


def priority_from_mano(mano: Seat) -> Tuple[Seat, ...]:
    """Return table priority order, with mano first for ties."""

    order = [mano]
    while len(order) < len(SEATS):
        order.append(next_seat(order[-1]))
    return tuple(order)


class Phase(str, Enum):
    MUS_VOTE = "mus_vote"
    DISCARD = "discard"
    GRANDE = "grande"
    CHICA = "chica"
    DECLARE_PARES = "declare_pares"
    PARES = "pares"
    DECLARE_JUEGO = "declare_juego"
    JUEGO_OR_PUNTO = "juego_or_punto"
    COMPLETE = "complete"


@dataclass(frozen=True)
class Event:
    """A public, replayable event. Card identities never appear here."""

    index: int
    kind: str
    actor: Optional[Seat] = None
    detail: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class LanceResult:
    lance: Lance
    winning_team: int
    points: int
    reason: str


def freeze_hands(
    hands: Mapping[Seat, Sequence[Card]],
) -> Mapping[Seat, Tuple[Card, ...]]:
    """Return a non-mutable, complete seat-to-hand mapping."""

    if set(hands) != set(SEATS):
        raise ValueError("A game state must contain one hand for every seat")
    return MappingProxyType({seat: tuple(hands[seat]) for seat in SEATS})


@dataclass(frozen=True)
class GameState:
    """Complete engine state for one match, including its current hand."""

    ruleset: Ruleset
    seed: int
    dealer: Seat
    mano: Seat
    phase: Phase
    current_seat: Optional[Seat]
    scores: Tuple[int, int]
    hands: Mapping[Seat, Tuple[Card, ...]]
    stock: Tuple[Card, ...]
    hand_number: int = 0
    discards: Tuple[Card, ...] = ()
    discard_counts: Tuple[int, int, int, int] = (0, 0, 0, 0)
    mus_voters: Tuple[Seat, ...] = ()
    declared_pares: Tuple[Seat, ...] = ()
    declared_juego: Tuple[Seat, ...] = ()
    declaration_seats: Tuple[Seat, ...] = ()
    current_lance: Optional[Lance] = None
    lance_seats: Tuple[Seat, ...] = ()
    passed_seats: Tuple[Seat, ...] = ()
    bidding_team: Optional[int] = None
    stake: int = 0
    refusal_points: int = 0
    lance_results: Tuple[LanceResult, ...] = ()
    events: Tuple[Event, ...] = ()

    @property
    def priority(self) -> Tuple[Seat, ...]:
        return priority_from_mano(self.mano)

    def hand_for(self, seat: Seat) -> Tuple[Card, ...]:
        return self.hands[seat]


def new_game(
    seed: int,
    ruleset: Optional[Ruleset] = None,
    dealer: Seat = Seat.NORTH,
) -> GameState:
    """Create a fully dealt deterministic match at the first Mus vote."""

    return deal_hand(seed=seed, ruleset=ruleset or Ruleset(), dealer=dealer)


def deal_hand(
    *,
    seed: int,
    ruleset: Ruleset,
    dealer: Seat,
    scores: Tuple[int, int] = (0, 0),
    hand_number: int = 0,
    lance_results: Tuple[LanceResult, ...] = (),
    events: Tuple[Event, ...] = (),
) -> GameState:
    """Deal one deterministic hand while retaining match scores and history."""

    stock = list(spanish_deck())
    Random("{}:{}".format(seed, hand_number)).shuffle(stock)
    mano = mano_for(dealer)
    hands = {seat: [] for seat in SEATS}
    for _ in range(4):
        for recipient in priority_from_mano(mano):
            hands[recipient].append(stock.pop())
    event = Event(
        index=len(events),
        kind="dealt",
        detail=(("cards_per_seat", "4"), ("hand_number", str(hand_number))),
    )
    return GameState(
        ruleset=ruleset,
        seed=seed,
        dealer=dealer,
        mano=mano,
        phase=Phase.MUS_VOTE,
        current_seat=mano,
        scores=scores,
        hands=freeze_hands(hands),
        stock=tuple(stock),
        hand_number=hand_number,
        lance_results=lance_results,
        events=events + (event,),
    )

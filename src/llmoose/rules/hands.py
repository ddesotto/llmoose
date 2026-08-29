"""Pure Mus hand evaluation with no game-state or RNG dependency.

Source: https://www.pagat.com/vying/mus.html ("Grande", "Chica", "Pares", and "Juego", 2026-08-28).

Grande: biggest rank wins, with 2=1 and 3=King.  Ties are broken by the next-highest card, then the next, then the next.
Chica: smallest rank wins, with 2=1 and 3=King.  Ties are broken by the next-lowest card, then the next, then the next.
Pares: 0, 1, or 2 pairs of equal rank.  A pair of 2s is worth the same as a pair of Kings.  Ties are broken by the highest pair, then the next pair, then the next pair, then the kicker.
Juego: 31 is the best, then 32, then 40, then 37, 36, 35, 34, 33.  Ties are broken by the next-highest card, then the next, then the next.  If a hand has no Juego (total < 31), it is not eligible to win this lance.
Punto (only played if no one has Juego): 30 is the best, then 29, then 28, then 27, then 26, then 25, then 24, then 23, then 22, then 21.  Ties are broken by the next-highest card, then the next, then the next.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Mapping, Optional, Sequence, Tuple, TypeVar

from llmoose.rules.cards import Card


class Lance(str, Enum):
    GRANDE = "grande"
    CHICA = "chica"
    PARES = "pares"
    JUEGO = "juego"
    PUNTO = "punto"


class PairKind(IntEnum):
    NONE = 0
    SIMPLE = 1
    MEDIAS = 2
    DUPLES = 3


@dataclass(frozen=True)
class PairValue:
    kind: PairKind
    ranks: Tuple[int, ...] = ()

    @property
    def points(self) -> int:
        """Unbet pares score for this player's hand."""

        return int(self.kind)


@dataclass(frozen=True)
class HandValue:
    """A hand is composed of four cards, which can be evaluated for each lance."""

    grande: Tuple[int, int, int, int]
    chica: Tuple[int, int, int, int]
    pares: PairValue
    total: int
    juego_order: Optional[int]

    @property
    def has_juego(self) -> bool:
        return self.juego_order is not None

    @property
    def juego_points(self) -> int:

        if not self.has_juego:
            return 0

        return 3 if self.total == 31 else 2


# A lower index is a stronger Juego
# !! 38 and 39 are impossible
JUEGO_ORDER = {31: 0, 32: 1, 40: 2, 37: 3, 36: 4, 35: 5, 34: 6, 33: 7}


def evaluate_hand(cards: Sequence[Card]) -> HandValue:
    """Evaluate a hand without hand tie-breakers"""

    if len(cards) != 4:
        raise ValueError("A Mus hand must contain exactly four cards")
    if len(set(cards)) != 4:
        raise ValueError("A hand cannot contain the same physical card twice")

    ranks = tuple(card.mus_rank for card in cards)
    total = sum(card.juego_points for card in cards)
    return HandValue(
        grande=tuple(sorted(ranks, reverse=True)),
        chica=tuple(sorted(ranks)),
        pares=_evaluate_pares(ranks),
        total=total,
        juego_order=JUEGO_ORDER.get(total),
    )


def _evaluate_pares(ranks: Sequence[int]) -> PairValue:
    counts = Counter(ranks)
    groups = sorted(counts.items(), key=lambda item: item[0], reverse=True)
    multiplicities = sorted(counts.values(), reverse=True)

    if multiplicities == [4]:
        rank = groups[0][0]
        return PairValue(PairKind.DUPLES, (rank, rank))
    if multiplicities == [2, 2]:
        return PairValue(PairKind.DUPLES, tuple(rank for rank, _ in groups))
    if multiplicities == [3, 1]:
        return PairValue(PairKind.MEDIAS, (groups[0][0],))
    if multiplicities == [2, 1, 1]:
        pair_rank = next(rank for rank, count in groups if count == 2)
        return PairValue(PairKind.SIMPLE, (pair_rank,))
    return PairValue(PairKind.NONE)


def compare_lance(left: HandValue, right: HandValue, lance: Lance) -> int:
    """Return 1 when left wins, -1 when right wins, 0 on an exact tie."""

    if lance is Lance.GRANDE:
        return _compare_high(left.grande, right.grande)
    if lance is Lance.CHICA:
        return _compare_high(right.chica, left.chica)
    if lance is Lance.PARES:
        return _compare_pares(left.pares, right.pares)
    if lance is Lance.JUEGO:
        return _compare_juego(left, right)
    if lance is Lance.PUNTO:
        return _compare_high((left.total,), (right.total,))
    raise ValueError("Unsupported lance: {0}".format(lance))


def _compare_high(left: Tuple[int, ...], right: Tuple[int, ...]) -> int:
    return (left > right) - (left < right)


def _compare_pares(left: PairValue, right: PairValue) -> int:
    return _compare_high(
        (int(left.kind),) + left.ranks, (int(right.kind),) + right.ranks
    )


def _compare_juego(left: HandValue, right: HandValue) -> int:
    if not left.has_juego or not right.has_juego:
        raise ValueError("Juego may only compare hands that have Juego")
    assert left.juego_order is not None
    assert right.juego_order is not None
    return (right.juego_order > left.juego_order) - (
        right.juego_order < left.juego_order
    )


SeatT = TypeVar("SeatT")


def winner_for_lance(
    hands: Mapping[SeatT, Sequence[Card]],
    priority: Sequence[SeatT],
    lance: Lance,
) -> SeatT:
    """Return the winner using ``priority`` to break equal hands.

    ``priority`` => table seating order starting at "mano"
    """

    if not priority:
        raise ValueError("At least one eligible seat is required")
    if set(priority) - set(hands):
        raise ValueError("Priority names a seat without a hand")

    winner = priority[0]
    winner_value = evaluate_hand(hands[winner])
    for seat in priority[1:]:
        candidate = evaluate_hand(hands[seat])
        if compare_lance(candidate, winner_value, lance) > 0:
            winner = seat
            winner_value = candidate
    return winner

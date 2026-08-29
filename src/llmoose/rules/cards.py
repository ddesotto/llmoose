"""Card identity and the normalisation used by the baseline Mus ruleset.

Source: https://www.pagat.com/vying/mus.html ("Players and Cards", 2026-08-28).

- Spanish deck
- That means 4 suits with 10 cards each.
- 1, 2, 3, 4, 5, 6, 7, 10 (Jack), 11 (Knight), 12 (King)
- 1 = 2 and 3 = King for Mus purposes, but we preserve distinction on card level.
#
- For the first three phases (grande, chica, pares), the rank of a card is its Mus rank.
- For the "point" phase, the point value of a card is its rank, except that 10, 11, and 12 are worth 10 points each.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Tuple


class Suit(str, Enum):
    SWORDS = "swords"
    BATONS = "batons"
    CUPS = "cups"
    COINS = "coins"


class Rank(IntEnum):
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    SOTA = 10
    CABALLO = 11
    REY = 12


@dataclass(frozen=True, order=True)
class Card:
    """A distinct physical card from a Spanish 40-card deck."""

    rank: Rank
    suit: Suit

    @property
    def is_figure(self) -> bool:
        """True if the card is a face card (Sota, Caballo, Rey)."""

        return self.rank in {Rank.SOTA, Rank.CABALLO, Rank.REY}

    @property
    def mus_rank(self) -> int:
        """Get in-game rank for Mus purposes, where 2=1 and 3=King."""

        # 2 = 1
        if self.rank is Rank.TWO:
            return int(Rank.ACE)

        # 3 = King
        if self.rank is Rank.THREE:
            return int(Rank.REY)

        return int(self.rank)

    @property
    def juego_points(self) -> int:
        """Point value for the Juego/Punto phase."""

        # Mus treats a three as a king, so use the normalized rank here too.
        # All figures (and normalized kings) count as ten points.
        return 10 if self.mus_rank >= int(Rank.SOTA) else self.mus_rank


ALL_RANKS: Tuple[Rank, ...] = (
    Rank.ACE,
    Rank.TWO,
    Rank.THREE,
    Rank.FOUR,
    Rank.FIVE,
    Rank.SIX,
    Rank.SEVEN,
    Rank.SOTA,
    Rank.CABALLO,
    Rank.REY,
)


def spanish_deck() -> Tuple[Card, ...]:
    """Returns unshuffled Spanish deck of 40 cards, in suit-major order."""
    return tuple(Card(rank, suit) for suit in Suit for rank in ALL_RANKS)

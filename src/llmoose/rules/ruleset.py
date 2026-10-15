"""Versioned benchmark adaptations of eight-kings/eight-aces Mus.

See README.md's rules contract for the reference and explicit adaptations.
The v1 identifier retains the original immediate-scoring transitions for replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

LEGACY_RULESET_ID = "mus_8r8a_no_senas_v1"
FEDERATION_RULESET_ID = "mus_8r8a_federation_no_senas_v2"
RULESET_IDS = (LEGACY_RULESET_ID, FEDERATION_RULESET_ID)


class MusProtocol(str, Enum):
    # all players must vote to continue playing
    INDIVIDUAL_UNANIMOUS = "individual_unanimous"


class BidResponseProtocol(str, Enum):
    """How a partnership chooses its single deterministic bid response."""

    NEXT_OPPOSING_SEAT = "next_opposing_seat"


class CommunicationMode(str, Enum):
    # We disable señas, leave it encoded
    DISABLED = "disabled"


@dataclass(frozen=True)
class Ruleset:
    # the 8 kings, 8 aces (1=2, 3=king), no señas variant
    id: str = FEDERATION_RULESET_ID
    target_score: int = 40

    # how to decide when to "mus" (vote to continue playing) and when to "no mus" (stop playing)
    mus_protocol: MusProtocol = MusProtocol.INDIVIDUAL_UNANIMOUS

    # Each bid is answered by the first eligible opposing seat in table order.
    # This deliberately replaces live partnership speech with a reproducible rule.
    bid_response_protocol: BidResponseProtocol = BidResponseProtocol.NEXT_OPPOSING_SEAT

    # if señas are allowed
    communication: CommunicationMode = CommunicationMode.DISABLED

    def __post_init__(self) -> None:
        if self.id not in RULESET_IDS:
            raise ValueError(f"Unsupported ruleset: {self.id}")
        if self.target_score < 2:
            raise ValueError("target_score must be at least 2")

    @property
    def deferred_scoring(self) -> bool:
        return self.id == FEDERATION_RULESET_ID

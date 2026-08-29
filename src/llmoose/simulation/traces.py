"""Portable, versioned records for deterministic match replay."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

from llmoose.game.actions import Action, ActionKind
from llmoose.game.state import Seat
from llmoose.rules.ruleset import (
    BidResponseProtocol,
    CommunicationMode,
    MusProtocol,
    Ruleset,
)
from llmoose.simulation.match import MatchResult, replay_match

TRACE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MatchTrace:
    """The minimum self-contained record needed to replay a completed match."""

    schema_version: int
    seed: int
    dealer: Seat
    ruleset: Ruleset
    actions: Tuple[Action, ...]
    scores: Tuple[int, int]
    winner: int
    metadata: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "dealer": int(self.dealer),
            "ruleset": _ruleset_to_dict(self.ruleset),
            "actions": [_action_to_dict(action) for action in self.actions],
            "scores": list(self.scores),
            "winner": self.winner,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MatchTrace":
        if value.get("schema_version") != TRACE_SCHEMA_VERSION:
            raise ValueError("Unsupported match-trace schema version")
        actions = tuple(_action_from_dict(action) for action in value["actions"])
        scores = tuple(value["scores"])
        if len(scores) != 2 or not all(isinstance(score, int) for score in scores):
            raise ValueError("A trace must contain exactly two integer scores")
        winner = value["winner"]
        if winner not in (0, 1):
            raise ValueError("Trace winner must be team 0 or team 1")
        return cls(
            schema_version=TRACE_SCHEMA_VERSION,
            seed=value["seed"],
            dealer=Seat(value["dealer"]),
            ruleset=_ruleset_from_dict(value["ruleset"]),
            actions=actions,
            scores=scores,
            winner=winner,
            metadata={
                str(key): str(item) for key, item in value.get("metadata", {}).items()
            },
        )


def trace_from_match(
    result: MatchResult, metadata: Optional[Mapping[str, str]] = None
) -> MatchTrace:
    """Create a portable trace from a completed match result."""

    return MatchTrace(
        schema_version=TRACE_SCHEMA_VERSION,
        seed=result.seed,
        dealer=result.initial_dealer,
        ruleset=result.final_state.ruleset,
        actions=result.actions,
        scores=result.scores,
        winner=result.winner,
        metadata=dict(metadata or {}),
    )


def save_trace(trace: MatchTrace, path: Union[str, Path]) -> None:
    """Write a human-readable JSON trace without exposing intermediate hands."""

    Path(path).write_text(
        json.dumps(trace.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_trace(path: Union[str, Path]) -> MatchTrace:
    """Load and validate a JSON trace."""

    return MatchTrace.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def replay_trace(trace: MatchTrace):
    """Replay a trace and reject a record whose recorded outcome is inconsistent."""

    state = replay_match(trace.actions, trace.seed, trace.ruleset, trace.dealer)
    winner = 0 if state.scores[0] >= trace.ruleset.target_score else 1
    if state.scores != trace.scores or winner != trace.winner:
        raise ValueError("Trace outcome does not match deterministic replay")
    return state


def _action_to_dict(action: Action) -> dict[str, Any]:
    value = asdict(action)
    value["kind"] = action.kind.value
    value["card_indices"] = list(action.card_indices)
    return value


def _action_from_dict(value: Mapping[str, Any]) -> Action:
    return Action(
        kind=ActionKind(value["kind"]),
        amount=value.get("amount", 0),
        card_indices=tuple(value.get("card_indices", ())),
    )


def _ruleset_to_dict(ruleset: Ruleset) -> dict[str, Any]:
    return {
        "id": ruleset.id,
        "target_score": ruleset.target_score,
        "mus_protocol": ruleset.mus_protocol.value,
        "bid_response_protocol": ruleset.bid_response_protocol.value,
        "communication": ruleset.communication.value,
    }


def _ruleset_from_dict(value: Mapping[str, Any]) -> Ruleset:
    return Ruleset(
        id=value["id"],
        target_score=value["target_score"],
        mus_protocol=MusProtocol(value["mus_protocol"]),
        bid_response_protocol=BidResponseProtocol(value["bid_response_protocol"]),
        communication=CommunicationMode(value["communication"]),
    )

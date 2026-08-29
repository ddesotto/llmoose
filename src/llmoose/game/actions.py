"""
I looked at https://github.com/datamllab/rlcard as a reference for encoding card game.

- It lets small policies emit an integer action while masking on ``legal_actions``;
- Rules engine remains the only authority that validates the choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Dict, Tuple


class ActionKind(str, Enum):
    MUS = "mus"
    NO_MUS = "no_mus"
    DISCARD = "discard"
    PASS = "pass"
    BET = "bet"
    CALL = "call"
    FOLD = "fold"
    RAISE = "raise"
    ORDAGO = "ordago"
    DECLARE_PARES = "declare_pares"
    DECLARE_NO_PARES = "declare_no_pares"
    DECLARE_JUEGO = "declare_juego"
    DECLARE_NO_JUEGO = "declare_no_juego"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    amount: int = 0
    card_indices: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.kind in {ActionKind.BET, ActionKind.RAISE} and self.amount < 2:
            raise ValueError(
                "Bets and raises must name an absolute stake of at least 2"
            )
        if self.kind not in {ActionKind.BET, ActionKind.RAISE} and self.amount:
            raise ValueError("Only bets and raises may carry an amount")
        if self.kind is ActionKind.DISCARD:
            if not 1 <= len(self.card_indices) <= 4:
                raise ValueError("A discard must contain between one and four cards")
            if len(set(self.card_indices)) != len(self.card_indices):
                raise ValueError("Discard card indices must be distinct")
            if any(index < 0 or index > 3 for index in self.card_indices):
                raise ValueError("Discard card indices must be in [0, 3]")
        elif self.card_indices:
            raise ValueError("Only discards may name card indices")


class ActionCodec:
    def __init__(self, target_score: int = 40) -> None:
        if target_score < 2:
            raise ValueError("target_score must be at least 2")

        base = (
            Action(ActionKind.MUS),
            Action(ActionKind.NO_MUS),
            Action(ActionKind.PASS),
            Action(ActionKind.CALL),
            Action(ActionKind.FOLD),
            Action(ActionKind.ORDAGO),
            Action(ActionKind.DECLARE_PARES),
            Action(ActionKind.DECLARE_NO_PARES),
            Action(ActionKind.DECLARE_JUEGO),
            Action(ActionKind.DECLARE_NO_JUEGO),
        )

        discards = tuple(
            Action(ActionKind.DISCARD, card_indices=indices)
            for size in range(1, 5)
            for indices in combinations(range(4), size)
        )

        stakes = tuple(
            action
            for amount in range(2, target_score + 1)
            for action in (
                Action(ActionKind.BET, amount=amount),
                Action(ActionKind.RAISE, amount=amount),
            )
        )

        self._actions = base + discards + stakes
        self._ids: Dict[Action, int] = {
            action: index for index, action in enumerate(self._actions)
        }

    @property
    def actions(self) -> Tuple[Action, ...]:
        return self._actions

    @property
    def size(self) -> int:
        return len(self._actions)

    def encode(self, action: Action) -> int:
        return self._ids[action]

    def decode(self, action_id: int) -> Action:
        if type(action_id) is not int or not 0 <= action_id < self.size:
            raise ValueError("action ID must be an integer in the codec range")
        return self._actions[action_id]

    def mask(self, legal: Tuple[Action, ...]) -> Tuple[bool, ...]:
        legal_set = set(legal)
        return tuple(action in legal_set for action in self._actions)

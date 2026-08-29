"""Small deterministic and stochastic policies used to exercise the engine."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from random import Random
from typing import Optional, Tuple

from llmoose.game.actions import Action, ActionCodec, ActionKind
from llmoose.game.state import Phase
from llmoose.observations.core import Observation
from llmoose.rules.hands import Lance, evaluate_hand


def _legal_actions(observation: Observation, codec: ActionCodec) -> Tuple[Action, ...]:
    return tuple(
        action
        for action, allowed in zip(codec.actions, observation.legal_action_mask)
        if allowed
    )


@dataclass
class RandomPolicy:
    """Choose uniformly among the engine-supplied legal actions."""

    codec: ActionCodec
    seed: int
    _random: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._random = Random(self.seed)

    def act(self, observation: Observation) -> Action:
        actions = _legal_actions(observation, self.codec)
        if not actions:
            raise ValueError("RandomPolicy was asked to act without a legal action")
        return self._random.choice(actions)


@dataclass(frozen=True)
class ConservativePolicy:
    """A deterministic baseline that ends Mus and plays every lance cheaply."""

    codec: ActionCodec

    def act(self, observation: Observation) -> Action:
        actions = _legal_actions(observation, self.codec)
        priorities = (
            ActionKind.NO_MUS,
            ActionKind.PASS,
            ActionKind.FOLD,
            ActionKind.CALL,
            ActionKind.DECLARE_PARES,
            ActionKind.DECLARE_NO_PARES,
            ActionKind.DECLARE_JUEGO,
            ActionKind.DECLARE_NO_JUEGO,
            ActionKind.DISCARD,
        )
        for kind in priorities:
            for action in actions:
                if action.kind is kind:
                    return action
        if not actions:
            raise ValueError(
                "ConservativePolicy was asked to act without a legal action"
            )
        return actions[0]


@dataclass(frozen=True)
class HeuristicPolicy:
    """A transparent, modest hand-aware reference policy.

    It is deliberately not presented as strong Mus strategy: it keeps made
    combinations, stops Mus with a promising hand, and only bids/calls with a
    hand that is naturally strong for the active lance.
    """

    codec: ActionCodec

    def act(self, observation: Observation) -> Action:
        actions = _legal_actions(observation, self.codec)
        if not actions:
            raise ValueError("HeuristicPolicy was asked to act without a legal action")
        value = evaluate_hand(observation.private_hand)
        if observation.phase is Phase.MUS_VOTE:
            return (
                _find(actions, ActionKind.NO_MUS)
                if _promising(value)
                else _find(actions, ActionKind.MUS)
            )
        if observation.phase is Phase.DISCARD:
            return _discard_action(observation, actions)
        if observation.phase in {Phase.DECLARE_PARES, Phase.DECLARE_JUEGO}:
            return actions[0]
        strength = _lance_strength(value, observation.current_lance)
        if _contains(actions, ActionKind.CALL):
            return (
                _find(actions, ActionKind.CALL)
                if strength >= 2
                else _find(actions, ActionKind.FOLD)
            )
        if strength >= 3 and _contains(actions, ActionKind.BET):
            return min(
                (action for action in actions if action.kind is ActionKind.BET),
                key=lambda action: action.amount,
            )
        return _find(actions, ActionKind.PASS)


def _contains(actions: Tuple[Action, ...], kind: ActionKind) -> bool:
    return any(action.kind is kind for action in actions)


def _find(actions: Tuple[Action, ...], kind: ActionKind) -> Action:
    return next(action for action in actions if action.kind is kind)


def _promising(value) -> bool:
    return value.has_juego or value.pares.points > 0 or value.grande[0] >= 10


def _lance_strength(value, lance: Optional[Lance]) -> int:
    if lance is Lance.PARES:
        return value.pares.points
    if lance is Lance.JUEGO:
        return 3 if value.total == 31 else 2 if value.has_juego else 0
    if lance is Lance.PUNTO:
        return 3 if value.total >= 29 else 2 if value.total >= 27 else 1
    if lance is Lance.GRANDE:
        return 3 if value.grande[0] == 12 else 2 if value.grande[0] >= 10 else 1
    if lance is Lance.CHICA:
        return 3 if value.chica[0] == 1 else 2 if value.chica[0] <= 4 else 1
    return 0


def _discard_action(observation: Observation, actions: Tuple[Action, ...]) -> Action:
    """Discard low singleton cards while preserving pairs and figures."""

    ranks = [card.mus_rank for card in observation.private_hand]
    counts = Counter(ranks)
    candidates = [
        index
        for index, card in enumerate(observation.private_hand)
        if counts[card.mus_rank] == 1 and card.mus_rank < 10
    ]
    if not candidates:
        candidates = [min(range(4), key=lambda index: ranks[index])]
    indices = tuple(candidates[:2])
    desired = Action(ActionKind.DISCARD, card_indices=indices)
    return desired if desired in actions else actions[0]

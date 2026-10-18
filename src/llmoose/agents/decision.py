"""Adapters for classifiers, value models, and other non-text decision policies."""

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Sequence

from llmoose.game.actions import Action, ActionCodec
from llmoose.observations.core import Observation
from llmoose.observations.encoding import legal_ids, policy_input


@dataclass
class ActionScorePolicy:
    """Choose the highest-scoring legal action, breaking ties by lowest ID.

    The scorer receives the same seat information as the LLM and returns one
    finite score per codec ID. A classifier must map its classes_ to codec IDs;
    missing classes must not silently shift the action mapping. Feature extraction
    belongs inside the scorer and has no access to the environment or run journal.
    """

    codec: ActionCodec
    scorer: Callable[[dict], Sequence[float]]

    def act(self, observation: Observation) -> Action:
        ids = legal_ids(observation, self.codec)
        if not ids:
            raise ValueError("Policy was asked to act without legal actions")
        if len(ids) == 1:
            return self.codec.decode(ids[0])
        scores = tuple(self.scorer(policy_input(observation)))
        if len(scores) != self.codec.size or not all(isfinite(s) for s in scores):
            raise ValueError("Scorer must return one finite score per codec ID")
        return self.codec.decode(max(ids, key=lambda i: scores[i]))

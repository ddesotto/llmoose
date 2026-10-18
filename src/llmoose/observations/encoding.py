"""Versioned, JSON-compatible policy input, independent of inference libraries."""

import json
from dataclasses import asdict

from llmoose.game.actions import ActionCodec
from llmoose.observations.core import Observation

INPUT_SCHEMA_VERSION = 1


def legal_ids(observation: Observation, codec: ActionCodec) -> tuple[int, ...]:
    """Validate the action-space contract before any policy consumes its mask."""
    if (codec.size != len(observation.legal_action_mask)
            or codec.size != ActionCodec(observation.target_score).size):
        raise ValueError("Policy codec does not match observation")
    if any(type(value) is not bool for value in observation.legal_action_mask):
        raise ValueError("Action mask must contain booleans")
    return tuple(i for i, allowed in enumerate(observation.legal_action_mask) if allowed)


def policy_input(observation: Observation) -> dict:
    """Return only seat-visible facts and legal actions; never accept GameState.

    Consumers may engineer features from this packet. Any lossy feature/history
    transform must be versioned separately by the consuming policy. No seed,
    evaluator labels, model output, or other seat's private input belongs here.
    """
    if not isinstance(observation, Observation):
        raise TypeError("policy_input requires a seat Observation")
    codec = ActionCodec(observation.target_score)
    ids = legal_ids(observation, codec)
    return json.loads(json.dumps({
        "schema_version": INPUT_SCHEMA_VERSION,
        "observation": asdict(observation),
        "legal_actions": [
            {"id": i, **asdict(codec.decode(i))} for i in ids
        ],
    }))

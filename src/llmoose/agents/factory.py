"""Construct seat policies; game execution lives in simulation.match."""

from __future__ import annotations

from typing import Mapping

from llmoose.agents.baselines import ConservativePolicy, HeuristicPolicy, RandomPolicy
from llmoose.agents.llm import DecisionLogger, LLMProvider, StructuredLLMPolicy
from llmoose.agents.prompt import PROMPT_VERSION
from llmoose.agents.protocol import Policy
from llmoose.game.actions import ActionCodec
from llmoose.game.state import SEATS, Seat

POLICY_NAMES = ("conservative", "heuristic", "random", "llm")


def build_policies(
    names: Mapping[Seat, str],
    codec: ActionCodec,
    *,
    seed: int,
    provider: LLMProvider | Mapping[Seat, LLMProvider] | None = None,
    strict: bool = False,
    on_record: DecisionLogger | None = None,
    prompt_version: int = PROMPT_VERSION,
) -> dict[Seat, Policy]:
    """Create one independent policy per seat.

    A provider mapping gives each model seat its own transport.
    """
    if set(names) != set(SEATS):
        raise ValueError("Provide one policy name for every seat")
    if any(name not in POLICY_NAMES for name in names.values()):
        raise ValueError(f"Policy names must be one of {POLICY_NAMES}")
    if "llm" in names.values() and provider is None:
        raise ValueError("LLM policies require a provider")
    if isinstance(provider, Mapping) and any(
        seat not in provider for seat in SEATS if names[seat] == "llm"
    ):
        raise ValueError("Every LLM seat requires a provider")

    policies: dict[Seat, Policy] = {}
    for seat in SEATS:
        name = names[seat]
        if name == "llm":
            seat_provider = provider[seat] if isinstance(provider, Mapping) else provider
            policies[seat] = StructuredLLMPolicy(
                codec,
                seat_provider,
                max_retries=0 if strict else 2,
                fallback_on_invalid=not strict,
                on_record=on_record,
                prompt_version=prompt_version,
            )
        elif name == "random":
            policies[seat] = RandomPolicy(codec, seed + int(seat))
        elif name == "heuristic":
            policies[seat] = HeuristicPolicy(codec)
        else:
            policies[seat] = ConservativePolicy(codec)
    return policies

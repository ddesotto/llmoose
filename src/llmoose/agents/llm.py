"""Observation-only LLM policy with explicit evaluation recovery and telemetry."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Mapping, Sequence

from llmoose.agents.baselines import ConservativePolicy
from llmoose.agents.prompt import (
    PROMPT_VERSION,
    load_system_prompt,
    render_messages,
    renderer_version_for_prompt,
)
from llmoose.agents.providers import ChatCompletionProvider, Completion, ProviderError
from llmoose.game.actions import Action, ActionCodec
from llmoose.observations.core import Observation

LLMProvider = Callable[[Sequence[Mapping[str, str]]], Completion]
DecisionLogger = Callable[[dict[str, object]], None]
AttemptStartLogger = Callable[
    [Observation, int, Sequence[Mapping[str, str]], Mapping[str, object] | None], None
]


class InvalidModelAction(RuntimeError):
    """The model exhausted its allowed attempts without a legal answer."""


@dataclass
class PolicyTelemetry:
    decisions: int = 0
    forced_actions: int = 0
    requests: int = 0
    invalid_outputs: int = 0
    fallbacks: int = 0
    provider_errors: int = 0


def parse_action(text: str, legal_ids: tuple[int, ...]) -> int:
    match = re.fullmatch(r"\s*<answer>([0-9]+)</answer>\s*", text)
    if match is None:
        raise ValueError("Expected exactly <answer>ID</answer>")
    # Avoid converting an arbitrarily long malformed integer.
    if len(match[1]) > 8 or int(match[1]) not in legal_ids:
        raise ValueError("Answer is not a legal action ID")
    return int(match[1])


@dataclass
class StructuredLLMPolicy:
    codec: ActionCodec
    provider: LLMProvider
    max_retries: int = 2
    fallback_on_invalid: bool = True
    on_record: DecisionLogger | None = None
    on_attempt_start: AttemptStartLogger | None = None
    prompt_version: int = PROMPT_VERSION
    telemetry: PolicyTelemetry = field(default_factory=PolicyTelemetry, init=False)

    def __post_init__(self) -> None:
        load_system_prompt(self.prompt_version)
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

    def act(self, observation: Observation) -> Action:
        if len(observation.legal_action_mask) != self.codec.size:
            raise ValueError("Policy codec does not match observation")
        legal_ids = tuple(
            i for i, allowed in enumerate(observation.legal_action_mask) if allowed
        )
        if not legal_ids:
            raise ValueError("LLM policy was asked to act without legal actions")
        self.telemetry.decisions += 1
        messages = render_messages(observation, prompt_version=self.prompt_version)
        context = {
            "schema_version": 1,
            "decision_index": self.telemetry.decisions - 1,
            "prompt_version": self.prompt_version,
            "renderer_version": renderer_version_for_prompt(self.prompt_version),
            "ruleset_id": observation.ruleset_id,
            "seat": int(observation.seat),
            "hand_number": observation.hand_number,
            "phase": observation.phase.value,
            "mano_offset": (observation.seat - observation.mano) % 4,
            "legal_ids": list(legal_ids),
        }

        def record(**values: object) -> None:
            if self.on_record is not None:
                self.on_record(
                    {**context, "messages": [dict(m) for m in messages], **values}
                )

        if len(legal_ids) == 1:
            self.telemetry.forced_actions += 1
            record(status="forced", chosen_id=legal_ids[0], attempt=0)
            return self.codec.decode(legal_ids[0])
        for attempt in range(self.max_retries + 1):
            self.telemetry.requests += 1
            if self.on_attempt_start is not None:
                request_body = (
                    self.provider.request_payload(messages)
                    if isinstance(self.provider, ChatCompletionProvider)
                    else None
                )
                self.on_attempt_start(
                    observation, attempt + 1, messages, request_body
                )
            started = perf_counter()
            try:
                completion = self.provider(messages)
            except KeyboardInterrupt:
                record(
                    status="interrupted",
                    attempt=attempt + 1,
                    latency_seconds=perf_counter() - started,
                )
                raise
            except ProviderError as exc:
                self.telemetry.provider_errors += 1
                record(
                    status="provider_error",
                    error=str(exc),
                    attempt=attempt + 1,
                    latency_seconds=perf_counter() - started,
                )
                raise
            details = {
                "latency_seconds": perf_counter() - started,
                "response": completion.text,
                "reasoning": completion.reasoning,
                "reasoning_details": list(completion.reasoning_details),
                "model": completion.model,
                "provider_request_id": completion.request_id,
                "finish_reason": completion.finish_reason,
                "usage": dict(completion.usage),
                "attempt": attempt + 1,
            }
            try:
                chosen_id = parse_action(completion.text, legal_ids)
            except ValueError as exc:
                self.telemetry.invalid_outputs += 1
                record(status="invalid", error=str(exc), **details)
                if attempt < self.max_retries:
                    messages.extend(
                        [
                            {"role": "assistant", "content": completion.text},
                            {
                                "role": "user",
                                "content": f"Invalid action. Reply only <answer>ID</answer>. Legal IDs: {list(legal_ids)}",
                            },
                        ]
                    )
            else:
                record(status="chosen", chosen_id=chosen_id, **details)
                return self.codec.decode(chosen_id)
        if not self.fallback_on_invalid:
            raise InvalidModelAction(
                "Model did not produce a legal action; strict mode stopped the match"
            )
        action = ConservativePolicy(self.codec).act(observation)
        self.telemetry.fallbacks += 1
        record(
            status="fallback",
            chosen_id=self.codec.encode(action),
            attempt=self.max_retries + 1,
        )
        return action

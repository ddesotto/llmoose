"""Observation-only LLM policy with explicit evaluation recovery and telemetry."""

from __future__ import annotations

import json
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
from llmoose.observations.encoding import legal_ids as observation_legal_ids

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


def action_response_format(legal_ids: tuple[int, ...]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "mus_action",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"action_id": {"type": "integer", "enum": list(legal_ids)}},
                "required": ["action_id"],
                "additionalProperties": False,
            },
        },
    }


def parse_json_action(text: str, legal_ids: tuple[int, ...]) -> int:
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate response key")
            obj[key] = value
        return obj

    try:
        value = json.loads(text, object_pairs_hook=unique_object)
    except (ValueError, RecursionError):
        raise ValueError('Expected exactly {"action_id": ID}') from None
    if (not isinstance(value, dict) or set(value) != {"action_id"}
            or type(value["action_id"]) is not int
            or value["action_id"] not in legal_ids):
        raise ValueError("Expected exactly one legal integer action_id")
    return value["action_id"]


def parse_response(
    text: str, legal_ids: tuple[int, ...], *, prompt_version: int,
    response_mode: str = "strict",
) -> tuple[int, bool]:
    """Parse the declared protocol, optionally normalizing a final answer envelope.

    Compatibility never interprets numbers in prose or validates its claims. It
    requires one unambiguous final envelope and reports normalization separately.
    """
    if response_mode not in {"strict", "compatible"}:
        raise ValueError("Unknown LLM response mode")
    parser = parse_json_action if prompt_version >= 5 else parse_action
    try:
        return parser(text, legal_ids), False
    except ValueError:
        if response_mode == "strict":
            raise
    if prompt_version < 5:
        matches = list(re.finditer(r"<answer>[0-9]+</answer>", text))
        if (len(matches) != 1 or text.count("<answer>") != 1
                or text.count("</answer>") != 1 or '"action_id"' in text):
            raise ValueError("Expected one unambiguous final answer envelope")
        match = matches[0]
        if text[match.end():].strip():
            raise ValueError("Answer envelope must be final")
        return parser(match[0], legal_ids), True
    # Ignore numeric prose; recognize only the named action envelope. Counting
    # markers also rejects duplicated keys, conflicting answers, and examples.
    if (text.count('"action_id"') != 1 or "<answer>" in text
            or "</answer>" in text):
        raise ValueError("Expected one unambiguous final action object")
    stripped = text.strip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].rstrip()
    start = stripped.rfind("{")
    if start < 0:
        raise ValueError("Missing final action object")
    prefix = stripped[:start]
    if "{" in prefix or "}" in prefix:
        raise ValueError("Ambiguous action object")
    return parser(stripped[start:], legal_ids), True


@dataclass
class StructuredLLMPolicy:
    codec: ActionCodec
    provider: LLMProvider
    max_retries: int = 2
    fallback_on_invalid: bool = True
    on_record: DecisionLogger | None = None
    on_attempt_start: AttemptStartLogger | None = None
    prompt_version: int = PROMPT_VERSION
    response_mode: str = "strict"
    telemetry: PolicyTelemetry = field(default_factory=PolicyTelemetry, init=False)

    def __post_init__(self) -> None:
        load_system_prompt(self.prompt_version)
        if self.response_mode not in {"strict", "compatible"}:
            raise ValueError("Unknown LLM response mode")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

    def act(self, observation: Observation) -> Action:
        legal_ids = observation_legal_ids(observation, self.codec)
        if not legal_ids:
            raise ValueError("LLM policy was asked to act without legal actions")
        self.telemetry.decisions += 1
        messages = render_messages(observation, prompt_version=self.prompt_version)
        context = {
            "schema_version": 1,
            "decision_index": self.telemetry.decisions - 1,
            "prompt_version": self.prompt_version,
            "response_mode": self.response_mode,
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
        response_format = (
            action_response_format(legal_ids)
            if self.prompt_version >= 5 and self.response_mode == "strict" else None
        )
        transport_options = {"response_format": response_format} if response_format else {}
        for attempt in range(self.max_retries + 1):
            self.telemetry.requests += 1
            if self.on_attempt_start is not None:
                request_body = (
                    self.provider.request_payload(messages, **transport_options)
                    if isinstance(self.provider, ChatCompletionProvider)
                    else None
                )
                self.on_attempt_start(
                    observation, attempt + 1, messages, request_body
                )
            started = perf_counter()
            try:
                completion = (
                    self.provider(messages, **transport_options)
                    if isinstance(self.provider, ChatCompletionProvider)
                    else self.provider(messages)
                )
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
                chosen_id, normalized = parse_response(
                    completion.text, legal_ids, prompt_version=self.prompt_version,
                    response_mode=self.response_mode,
                )
            except ValueError as exc:
                self.telemetry.invalid_outputs += 1
                record(status="invalid", error=str(exc), **details)
                if attempt < self.max_retries and self.prompt_version >= 5:
                    messages = render_messages(observation, prompt_version=self.prompt_version)
                    messages[-1]["content"] += (
                        '\nYour previous response was rejected. Return only {"action_id": ID} '
                        'with one legal integer ID from this unchanged snapshot.'
                    )
                elif attempt < self.max_retries:
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
                record(status="chosen", chosen_id=chosen_id, response_normalized=normalized, **details)
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

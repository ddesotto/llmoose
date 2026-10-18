"""Output for an experiment: decision logs and result summaries."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, TextIO

from llmoose.agents.llm import StructuredLLMPolicy
from llmoose.agents.protocol import Policy
from llmoose.simulation.match import MatchResult


@dataclass
class LLMMetrics:
    """Count attempts separately from decisions; missing usage is not zero."""

    decisions: int = 0
    forced_actions: int = 0
    requests: int = 0
    retries: int = 0
    valid_responses: int = 0
    normalized_responses: int = 0
    invalid_outputs: int = 0
    provider_errors: int = 0
    interrupted_requests: int = 0
    unknown_outcomes: int = 0
    fallbacks: int = 0
    first_try_legal: int = 0
    request_seconds: float = 0.0
    usage_totals: dict[str, float] = field(default_factory=dict)
    usage_counts: dict[str, int] = field(default_factory=dict)

    def add(self, record: Mapping[str, object]) -> None:
        status = record["status"]
        attempt = record["attempt"]
        if status == "forced" or (attempt == 1 and status != "fallback"):
            self.decisions += 1
        if status == "forced":
            self.forced_actions += 1
        elif status in {"fallback", "recovery_fallback"}:
            self.fallbacks += 1
        else:
            self.requests += 1
            self.retries += int(attempt > 1)
            self.valid_responses += int(status == "chosen")
            self.normalized_responses += int(
                status == "chosen" and record.get("response_normalized") is True
            )
            self.invalid_outputs += int(status == "invalid")
            self.provider_errors += int(status == "provider_error")
            self.interrupted_requests += int(status == "interrupted")
            self.unknown_outcomes += int(status == "outcome_unknown")
            self.first_try_legal += int(status == "chosen" and attempt == 1)
            self.request_seconds += record.get("latency_seconds", 0.0)
            usage = record.get("usage", {})
            values = {
                name: usage.get(name)
                for name in ("prompt_tokens", "completion_tokens", "cost")
            }
            details = usage.get("completion_tokens_details")
            values["reasoning_tokens"] = (
                details.get("reasoning_tokens") if isinstance(details, dict) else None
            )
            for name, value in values.items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value >= 0
                ):
                    self.usage_totals[name] = self.usage_totals.get(name, 0) + value
                    self.usage_counts[name] = self.usage_counts.get(name, 0) + 1

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.pop("usage_totals")
        result.pop("usage_counts")
        model_decisions = self.decisions - self.forced_actions
        responses = self.valid_responses + self.invalid_outputs
        result.update(
            model_decisions=model_decisions,
            first_try_legal_rate=(
                self.first_try_legal / model_decisions if model_decisions else None
            ),
            invalid_response_rate=(
                self.invalid_outputs / responses if responses else None
            ),
            fallback_rate=self.fallbacks / model_decisions if model_decisions else None,
            mean_request_seconds=self.request_seconds / self.requests
            if self.requests
            else None,
            usage={
                name: {
                    "reported_total": self.usage_totals.get(name),
                    "requests_reporting": self.usage_counts.get(name, 0),
                }
                for name in (
                    "prompt_tokens",
                    "completion_tokens",
                    "reasoning_tokens",
                    "cost",
                )
            },
        )
        return result


class DecisionReporter:
    """A policy callback whose optional JSONL file is owned by a with block."""

    def __init__(
        self, path: str | None, metadata: Mapping[str, object], *, append: bool = False
    ) -> None:
        self.path = Path(path) if path else None
        self.metadata = dict(metadata)
        self._handle: TextIO | None = None
        self.action_index = 0
        self.total = LLMMetrics()
        self.by_seat: dict[int, LLMMetrics] = {}
        self.by_team: dict[int, LLMMetrics] = {}
        self.models: set[str] = set()
        self.on_journal = None
        self.append = append

    def __enter__(self) -> DecisionReporter:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a" if self.append else "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __call__(self, record: dict[str, object]) -> None:
        if self.on_journal is not None:
            self.on_journal(record)
        self.restore(record)
        if self._handle is not None:
            self._handle.write(
                json.dumps(
                    {
                        **self.metadata,
                        **record,
                        "action_index": self.action_index,
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            self._handle.flush()
        if record["status"] in {"invalid", "fallback", "recovery_fallback"}:
            print(
                f"LLM seat {record['seat']}: {record['status']} (attempt {record['attempt']})",
                file=sys.stderr,
                flush=True,
            )

    def restore(self, record: Mapping[str, object]) -> None:
        """Rebuild metrics from committed records without writing or re-emitting them."""
        seat = record["seat"]
        self.total.add(record)
        self.by_seat.setdefault(seat, LLMMetrics()).add(record)
        self.by_team.setdefault(seat % 2, LLMMetrics()).add(record)
        if record.get("model"):
            self.models.add(record["model"])

    def metrics(self) -> dict[str, object]:
        return {
            "total": self.total.to_dict(),
            "by_seat": {str(k): v.to_dict() for k, v in self.by_seat.items()},
            "by_team": {str(k): v.to_dict() for k, v in self.by_team.items()},
            "resolved_models": sorted(self.models),
        }


def match_summary(result: MatchResult) -> str:
    return (
        f"winner=team-{result.winner} scores={result.scores} "
        f"actions={len(result.actions)} events={len(result.final_state.events)}"
    )


def llm_summary(policies: Iterable[Policy]) -> str | None:
    """Aggregate model telemetry, including interrupted or failed matches."""
    counts = [
        asdict(policy.telemetry)
        for policy in policies
        if isinstance(policy, StructuredLLMPolicy)
    ]
    if not counts:
        return None
    totals = {key: sum(count[key] for count in counts) for key in counts[0]}
    return "LLM " + " ".join(f"{key}={value}" for key, value in totals.items())

"""Output for an experiment: decision logs and result summaries."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, TextIO

from llmoose.agents.llm import StructuredLLMPolicy
from llmoose.agents.protocol import Policy
from llmoose.simulation.match import MatchResult


class DecisionReporter:
    """A policy callback whose optional JSONL file is owned by a with block."""

    def __init__(self, path: str | None, metadata: Mapping[str, object]) -> None:
        self.path = Path(path) if path else None
        self.metadata = dict(metadata)
        self._handle: TextIO | None = None

    def __enter__(self) -> DecisionReporter:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __call__(self, record: dict[str, object]) -> None:
        if self._handle is not None:
            self._handle.write(
                json.dumps({**self.metadata, **record}, sort_keys=True) + "\n"
            )
            self._handle.flush()
        if record["status"] in {"invalid", "fallback"}:
            print(
                f"LLM seat {record['seat']}: {record['status']} (attempt {record['attempt']})",
                file=sys.stderr,
                flush=True,
            )


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

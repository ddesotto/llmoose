"""Engine-independent reader for preserved run evidence.

Readers keep the original records intact. Missing v1 facts stay unknown; callers
can project this common shape without pretending old runs had v2 provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from llmoose.simulation.artifacts import (
    MANIFEST_SCHEMA,
    SCHEMA,
    SCHEMA_VERSION,
    read_journal,
)


@dataclass(frozen=True)
class StoredAction:
    action_index: int
    action: Mapping[str, Any]
    action_id: int
    source: str | None
    state_sha256_after: str | None
    original: Mapping[str, Any]


@dataclass(frozen=True)
class RunEvidence:
    format_version: int
    run_id: str | None
    manifest: Mapping[str, Any] | None
    records: tuple[Mapping[str, Any], ...]
    actions: tuple[StoredAction, ...]
    complete_line: bool
    provenance: Mapping[str, str]


def _lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"Torn line in {path.name}")
    return [json.loads(line) for line in data.splitlines() if line]


def read_evidence(directory: Path) -> RunEvidence:
    """Read v1/v2 records without invoking today's engine or prompt renderer."""
    if not directory.is_dir():
        raise ValueError("Run directory does not exist")
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported run manifest schema")
        records, complete = read_journal(directory / "records.jsonl")
        if any(
            record.get("schema") != SCHEMA or record.get("schema_version") != SCHEMA_VERSION
            for record in records
        ):
            raise ValueError("Unsupported journal record schema")
        actions = tuple(
            StoredAction(
                action_index=record["payload"]["action_index"],
                action=record["payload"]["action"],
                action_id=record["payload"]["action_id"],
                source=record["payload"]["source"],
                state_sha256_after=record["payload"]["state_sha256_after"],
                original=record,
            )
            for record in records
            if record["record_type"] == "action_applied"
        )
        return RunEvidence(
            2,
            manifest["run_id"],
            manifest,
            tuple(records),
            actions,
            complete,
            {"engine_revision": "recorded", "request_start_times": "recorded"},
        )
    moves = _lines(directory / "moves.jsonl")
    decisions = _lines(directory / "decisions.jsonl")
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    if report is None and not moves:
        raise ValueError("No recognized run records")
    actions = tuple(
        StoredAction(
            action_index=move["action_index"],
            action=move["action"],
            action_id=move["action_id"],
            source=None,
            state_sha256_after=None,
            original=move,
        )
        for move in moves
    )
    return RunEvidence(
        1,
        report.get("run_id") if report else None,
        None,
        tuple(moves + decisions),
        actions,
        True,
        {
            "engine_revision": "unknown unless independently archived",
            "request_start_times": "not recorded in v1",
            "provider_revision": "not recorded in v1",
            "token_probabilities": "not recorded in v1",
        },
    )

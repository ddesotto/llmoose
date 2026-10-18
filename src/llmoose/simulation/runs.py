"""Durable game records and objective scores for a single experiment."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Mapping, TextIO

from llmoose.game.actions import Action, ActionCodec
from llmoose.game.state import GameState, Phase
from llmoose.observations.core import observe
from llmoose.rules.hands import Lance
from llmoose.simulation.artifacts import (
    SCHEMA_VERSION,
    Journal,
    atomic_json,
    build_manifest,
    canonical_bytes,
    digest,
    file_digest,
    state_digest,
    write_checksums,
)
from llmoose.simulation.reporting import DecisionReporter


def score_game(state: GameState, action_count: int) -> dict[str, object]:
    """Score observed outcomes, without claiming to grade strategic decisions.

    Win points are 1/0 only for completed matches. Lance wins include folds
    and uncontested awards; they are not estimates of showdown strength.
    """
    complete = state.phase is Phase.COMPLETE
    winner = (
        (0 if state.scores[0] >= state.ruleset.target_score else 1)
        if complete
        else None
    )
    teams = []
    for team in (0, 1):
        by_lance = {}
        for lance in Lance:
            resolved = [r for r in state.lance_results if r.lance is lance]
            won = [r for r in resolved if r.winning_team == team]
            by_lance[lance.value] = {
                "resolved": len(resolved),
                "won": len(won),
                "points": sum(r.points for r in won),
                "won_by_reason": {
                    reason: sum(r.reason == reason for r in won)
                    for reason in sorted({r.reason for r in resolved})
                },
            }
        teams.append(
            {
                "team": team,
                "outcome": ("win" if winner == team else "loss")
                if complete
                else "incomplete",
                "win_points": int(winner == team) if complete else None,
                "stones": state.scores[team],
                "stone_difference": state.scores[team] - state.scores[1 - team],
                "by_lance": by_lance,
            }
        )
    return {
        "complete": complete,
        "winner": winner,
        "hands_dealt": state.hand_number + 1,
        "actions": action_count,
        "teams": teams,
    }


class RunRecorder:
    """Flush every applied move and preserve a report on completion or failure."""

    def __init__(
        self,
        directory: Path,
        initial_state: GameState,
        metadata: Mapping[str, object],
        decisions: DecisionReporter,
        *,
        resume_records: list[dict[str, object]] | None = None,
    ) -> None:
        self.directory = directory
        self.state = initial_state
        self.metadata = dict(metadata)
        self.decisions = decisions
        self.action_count = 0
        self._moves: TextIO | None = None
        self.journal: Journal | None = None
        self._current_decision_id: str | None = None
        self._current_policy_record: dict[str, object] | None = None
        self._current_policy_name: str | None = None
        self._resume_records = resume_records

    def __enter__(self) -> RunRecorder:
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started = perf_counter()
        if self._resume_records is None:
            manifest = build_manifest(self.directory, self.state, self.metadata)
            atomic_json(self.directory / "manifest.json", manifest)
            self.journal = Journal(
                self.directory / "records.jsonl",
                str(self.metadata["run_id"]),
                str(manifest["match_id"]),
            )
            self.journal.append(
                "run_started",
                {
                    "manifest_sha256": file_digest(self.directory / "manifest.json"),
                    "episode_mode": "match",
                },
            )
        else:
            manifest = json.loads((self.directory / "manifest.json").read_text())
            self.journal = Journal(
                self.directory / "records.jsonl",
                str(self.metadata["run_id"]),
                str(manifest["match_id"]),
                sequence=len(self._resume_records),
            )
            self.journal.append(
                "run_resumed",
                {
                    "previous_record_id": self._resume_records[-1]["record_id"],
                    "action_index": self.action_count,
                    "state_sha256": state_digest(self.state),
                },
            )
            pending_id = f"{self.journal.match_id}:decision:{self.action_count}"
            pending = [
                record for record in self._resume_records
                if record.get("decision_id") == pending_id
            ]
            if pending and pending[0]["record_type"] == "decision_opened":
                self._current_decision_id = pending_id
                self._current_policy_name = pending[0]["payload"]["policy"]
                results = [
                    record["payload"] for record in pending
                    if record["record_type"] in {"attempt_finished", "decision_resolved"}
                    and record["payload"]["status"] in {"chosen", "forced", "fallback", "recovery_fallback"}
                ]
                if results:
                    self._current_policy_record = results[-1]
        self._moves = (self.directory / "moves.jsonl").open(
            "a" if self._resume_records is not None else "w", encoding="utf-8"
        )
        return self

    def on_turn(self, state: GameState) -> None:
        assert self.journal is not None
        if self._current_decision_id is not None:
            assert state_digest(state) == state_digest(self.state)
            return
        actor = state.current_seat
        assert actor is not None
        codec = ActionCodec(state.ruleset.target_score)
        observation = observe(state, actor, codec)
        decision_id = f"{self.journal.match_id}:decision:{self.action_count}"
        self._current_decision_id = decision_id
        self._current_policy_record = None
        self._current_policy_name = self.metadata["teams"][int(actor) % 2]
        self.journal.append(
            "decision_opened",
            {
                "action_index": self.action_count,
                "hand_id": f"{self.journal.match_id}:hand:{state.hand_number}",
                "hand_number": state.hand_number,
                "seat": int(actor),
                "team": int(actor) % 2,
                "policy": self._current_policy_name,
                "observation_schema_version": 1,
                "observation": asdict(observation),
                "legal_ids": [
                    index
                    for index, allowed in enumerate(observation.legal_action_mask)
                    if allowed
                ],
                "public_event_cursor": len(state.events) - 1,
                "scores": list(state.scores),
                "state_sha256": state_digest(state),
            },
            decision_id=decision_id,
            visibility="seat_private",
        )

    def on_attempt_start(self, observation, attempt: int, messages, request_body) -> None:
        assert self.journal is not None and self._current_decision_id is not None
        self.journal.append(
            "attempt_started",
            {
                "attempt_id": f"{self._current_decision_id}:attempt:{attempt}",
                "attempt_number": attempt,
                "seat": int(observation.seat),
                "messages": [dict(message) for message in messages],
                "messages_sha256": digest(canonical_bytes(messages)),
                "request_body": request_body,
                "request_body_sha256": (
                    digest(json.dumps(request_body).encode("utf-8"))
                    if request_body is not None
                    else None
                ),
                "request_body_unavailable_reason": (
                    None if request_body is not None else "custom provider interface"
                ),
                "request_settings": {
                    key: self.metadata[key]
                    for key in ("requested_model", "temperature", "max_tokens", "timeout", "include_reasoning")
                },
            },
            decision_id=self._current_decision_id,
            visibility="seat_private",
        )

    def on_policy_record(self, record: dict[str, object]) -> None:
        assert self.journal is not None and self._current_decision_id is not None
        status = str(record["status"])
        self._current_policy_record = record
        payload = {key: value for key, value in record.items() if key != "messages"}
        payload["action_index"] = self.action_count
        if status in {"chosen", "invalid", "provider_error", "interrupted"}:
            attempt = int(record["attempt"])
            payload["attempt_id"] = f"{self._current_decision_id}:attempt:{attempt}"
            self.journal.append(
                "attempt_finished", payload, decision_id=self._current_decision_id
            )
        else:
            self.journal.append(
                "decision_resolved", payload, decision_id=self._current_decision_id
            )

    def _selection(self) -> tuple[str, str | None]:
        policy_record = self._current_policy_record or {}
        status = policy_record.get("status")
        source = (
            {"chosen": "model", "forced": "forced", "fallback": "fallback", "recovery_fallback": "fallback"}.get(
                str(status), "baseline"
            )
            if self._current_policy_name == "llm"
            else "sampled" if self._current_policy_name == "random" else "baseline"
        )
        attempt_id = (
            f"{self._current_decision_id}:attempt:{policy_record['attempt']}"
            if status == "chosen"
            else None
        )
        return source, attempt_id

    def on_action_selected(self, state: GameState, action: Action) -> None:
        """Commit a valid choice before the engine mutates its in-memory state."""
        assert self.journal is not None and self._current_decision_id is not None
        source, attempt_id = self._selection()
        self.journal.append(
            "action_selected",
            {
                "action_index": self.action_count,
                "action_id": ActionCodec(state.ruleset.target_score).encode(action),
                "action": asdict(action),
                "source": source,
                "selected_attempt_id": attempt_id,
                "state_sha256_before": state_digest(state),
            },
            decision_id=self._current_decision_id,
        )

    def on_transition(
        self, before: GameState, action: Action, after: GameState
    ) -> None:
        index = self.action_count
        record = {
            "schema_version": 1,
            "run_id": self.metadata["run_id"],
            "action_index": index,
            "hand_number": before.hand_number,
            "seat": int(before.current_seat),
            "phase": before.phase.value,
            "action_id": ActionCodec(before.ruleset.target_score).encode(action),
            "action": asdict(action),
            "scores_before": before.scores,
            "scores_after": after.scores,
            "events": [asdict(e) for e in after.events[len(before.events) :]],
        }
        assert self.journal is not None and self._current_decision_id is not None
        source, selected_attempt_id = self._selection()
        self.journal.append(
            "action_applied",
            {
                "action_index": index,
                "hand_id": f"{self.journal.match_id}:hand:{before.hand_number}",
                "hand_number": before.hand_number,
                "seat": int(before.current_seat),
                "team": int(before.current_seat) % 2,
                "source": source,
                "selected_attempt_id": selected_attempt_id,
                "action_id": record["action_id"],
                "action": record["action"],
                "scores_before": list(before.scores),
                "scores_after": list(after.scores),
                "state_sha256_before": state_digest(before),
                "state_sha256_after": state_digest(after),
                "reward": {
                    "definition": "team_stone_delta_v1",
                    "team_stone_delta": [
                        new - old for old, new in zip(before.scores, after.scores)
                    ],
                },
                "events": [
                    {"event_id": f"{self.journal.match_id}:event:{event.index}", **asdict(event)}
                    for event in after.events[len(before.events) :]
                ],
                "terminated": after.phase is Phase.COMPLETE,
                "truncated": False,
                "next_actor": int(after.current_seat) if after.current_seat is not None else None,
            },
            decision_id=self._current_decision_id,
        )
        self.state = after
        self.action_count += 1
        self._moves.write(json.dumps(record, sort_keys=True) + "\n")
        self._moves.flush()
        self._current_decision_id = None
        self._current_policy_record = None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._moves is not None:
            self._moves.close()
        status = (
            "completed"
            if exc_type is None
            else (
                "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed"
            )
        )
        report = {
            "schema_version": 1,
            **self.metadata,
            "status": status,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": perf_counter() - self.started,
            "error": str(exc_value) if exc_value is not None else None,
            "game": score_game(self.state, self.action_count),
            "llm": self.decisions.metrics(),
        }
        atomic_json(self.directory / "report.json", report)
        assert self.journal is not None
        terminated = self.state.phase is Phase.COMPLETE
        self.journal.append(
            "episode_finished",
            {
                "status": status,
                "terminated": terminated,
                "truncated": not terminated,
                "reason": str(exc_value) if exc_value is not None else None,
                "winner": score_game(self.state, self.action_count)["winner"],
                "scores": list(self.state.scores),
                "actions": self.action_count,
            },
        )
        self.journal.append(
            "run_finished",
            {
                "status": status,
                "actions": self.action_count,
                "report_sha256": file_digest(self.directory / "report.json"),
            },
        )
        self.journal.close()
        atomic_json(
            self.directory / "result.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.metadata["run_id"],
                "status": status,
                "terminated": terminated,
                "truncated": not terminated,
                "actions": self.action_count,
                "scores": list(self.state.scores),
                "winner": score_game(self.state, self.action_count)["winner"],
                "error": str(exc_value) if exc_value is not None else None,
            },
        )
        write_checksums(self.directory)

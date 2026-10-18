"""Validate recorded run evidence against engine transitions and schema links."""

from __future__ import annotations

import argparse
import fcntl
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from llmoose.agents.prompt import load_system_prompt, render_messages
from llmoose.game.actions import Action, ActionCodec, ActionKind
from llmoose.game.engine import apply_action, legal_actions
from llmoose.game.state import GameState, Phase, Seat, new_game
from llmoose.observations.core import observe
from llmoose.rules.ruleset import (
    BidResponseProtocol,
    CommunicationMode,
    MusProtocol,
    Ruleset,
)
from llmoose.simulation.artifacts import (
    MANIFEST_SCHEMA,
    SCHEMA,
    SCHEMA_VERSION,
    atomic_bytes,
    atomic_json,
    canonical_bytes,
    current_source_digest,
    digest,
    file_digest,
    state_digest,
    write_checksums,
)
from llmoose.simulation.evidence import read_evidence
from llmoose.simulation.traces import load_trace, replay_trace


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _action(value: dict[str, Any]) -> Action:
    return Action(
        ActionKind(value["kind"]),
        amount=value.get("amount", 0),
        card_indices=tuple(value.get("card_indices", ())),
    )


def _ruleset(value: dict[str, Any]) -> Ruleset:
    return Ruleset(
        id=value["id"],
        target_score=value["target_score"],
        mus_protocol=MusProtocol(value["mus_protocol"]),
        bid_response_protocol=BidResponseProtocol(value["bid_response_protocol"]),
        communication=CommunicationMode(value["communication"]),
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    _require(not data or data.endswith(b"\n"), f"Torn line in {path.name}")
    return [json.loads(line) for line in data.splitlines() if line]


def _verify_hashes(directory: Path) -> None:
    path = directory / "checksums.json"
    if not path.exists():
        return
    checksums = json.loads(path.read_text())
    _require(checksums.get("schema_version") == 1, "Unknown checksum schema")
    for name, expected in checksums["sha256"].items():
        file = directory / name
        _require(file.is_file(), f"Missing checksummed file: {name}")
        _require(file_digest(file) == expected, f"Checksum mismatch: {name}")


def _verify_source_link(
    payload: dict[str, Any],
    decision_id: str,
    opened: dict[str, dict[str, Any]],
    attempts: dict[str, dict[str, Any]],
    attempt_status: dict[str, str],
    resolved: dict[str, str],
    records: Sequence[dict[str, Any]],
) -> None:
    source = payload["source"]
    selected = payload["selected_attempt_id"]
    if source == "model":
        _require(selected in attempts and attempt_status.get(selected) == "chosen", "Model action lacks chosen attempt")
        _require(opened[decision_id]["policy"] == "llm", "Model source on non-model policy")
        selected_id = next(
            row["chosen_id"]
            for row in (record["payload"] for record in records)
            if row.get("attempt_id") == selected and row.get("status") == "chosen"
        )
        _require(selected_id == payload["action_id"], "Chosen/selected action mismatch")
    elif source in {"forced", "fallback"}:
        _require(
            (resolved.get(decision_id) == source or source == "fallback" and resolved.get(decision_id) == "recovery_fallback")
            and selected is None,
            "Selector link mismatch",
        )
        _require(opened[decision_id]["policy"] == "llm", "Selector source on non-model policy")
    else:
        _require(source in {"baseline", "sampled"} and selected is None, "Unknown action source")
        expected_source = "sampled" if opened[decision_id]["policy"] == "random" else "baseline"
        _require(source == expected_source and opened[decision_id]["policy"] != "llm", "Policy source mismatch")


def validate_run(directory: Path, *, allow_partial: bool = False) -> dict[str, object]:
    """Check a v2 journal or a preserved v1 run without rewriting either."""
    _require(directory.is_dir(), "Run directory does not exist")
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return _validate_legacy(directory)
    _verify_hashes(directory)
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "Unknown manifest family")
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "Unsupported manifest version")
    run_id = manifest["run_id"]
    match_id = manifest["match_id"]
    snapshot = directory / manifest["provenance"]["source_snapshot"]
    _require(snapshot.is_file(), "Missing source snapshot")
    _require(
        file_digest(snapshot) == manifest["provenance"]["source_sha256"],
        "Source snapshot hash mismatch",
    )
    evidence = read_evidence(directory)
    records, complete_line = evidence.records, evidence.complete_line
    _require(complete_line or allow_partial, "Torn final journal line; run needs recovery")
    _require(records, "Empty journal")
    _require(records[0]["record_type"] == "run_started", "Missing run_started")
    _require(
        records[0]["payload"]["manifest_sha256"] == file_digest(manifest_path),
        "Journal does not match manifest",
    )
    state = new_game(
        manifest["rng"]["deal_seed"],
        _ruleset(manifest["ruleset"]),
        Seat(manifest["initial_dealer"]),
    )
    codec = ActionCodec(state.ruleset.target_score)
    _require(
        manifest["codec"]["table_sha256"]
        == digest(canonical_bytes([asdict(action) for action in codec.actions])),
        "Current action codec differs from archived run",
    )
    _require(
        manifest["prompt"]["system_sha256"]
        == digest(load_system_prompt(manifest["prompt"]["version"]).encode()),
        "Current system prompt differs from archived run",
    )
    opened: dict[str, dict[str, Any]] = {}
    attempts: dict[str, dict[str, Any]] = {}
    attempt_status: dict[str, str] = {}
    resolved: dict[str, str] = {}
    selected_actions: dict[str, dict[str, Any]] = {}
    applied = 0
    finished = False
    episode_finished = False
    execution_status = None
    episode_reason = None
    for sequence, record in enumerate(records):
        _require(record.get("schema") == SCHEMA, "Unknown journal family")
        _require(record.get("schema_version") == SCHEMA_VERSION, "Unsupported journal version")
        _require(record.get("sequence") == sequence, "Journal sequence gap")
        _require(record.get("record_id") == f"{run_id}:record:{sequence}", "Bad record ID")
        _require(record.get("run_id") == run_id, "Run ID mismatch")
        _require(record.get("match_id") == match_id, "Match ID mismatch")
        kind = record["record_type"]
        payload = record["payload"]
        decision_id = record.get("decision_id")
        if kind == "run_started":
            _require(sequence == 0, "Duplicate run_started")
        elif kind == "run_resumed":
            _require(not finished, "Resume after run completion")
            _require(payload["previous_record_id"] == records[sequence - 1]["record_id"], "Resume link mismatch")
            _require(payload["action_index"] == applied, "Resume action index mismatch")
            _require(payload["state_sha256"] == state_digest(state), "Resume state hash mismatch")
        elif kind == "decision_opened":
            _require(state.phase is not Phase.COMPLETE, "Decision after match completion")
            expected_id = f"{match_id}:decision:{applied}"
            _require(decision_id == expected_id, "Decision ID/action index mismatch")
            _require(decision_id not in opened, "Duplicate decision")
            actor = state.current_seat
            _require(actor is not None, "Decision without actor")
            expected = json.loads(canonical_bytes(asdict(observe(state, actor, codec))))
            _require(payload["observation"] == expected, "Seat observation mismatch or hidden data")
            legal_ids = [index for index, allowed in enumerate(observe(state, actor, codec).legal_action_mask) if allowed]
            _require(payload["legal_ids"] == legal_ids, "Legal IDs mismatch")
            _require(payload["seat"] == int(actor), "Decision seat mismatch")
            _require(payload["action_index"] == applied, "Decision action index mismatch")
            _require(payload["public_event_cursor"] == len(state.events) - 1, "Public cursor mismatch")
            _require(payload["state_sha256"] == state_digest(state), "Decision state hash mismatch")
            opened[decision_id] = payload
        elif kind == "attempt_started":
            _require(decision_id in opened, "Attempt without decision")
            attempt_id = payload["attempt_id"]
            _require(attempt_id not in attempts, "Duplicate attempt ID")
            _require(attempt_id == f"{decision_id}:attempt:{payload['attempt_number']}", "Bad attempt ID")
            previous = [item for item in attempts if item.startswith(decision_id + ":attempt:")]
            _require(payload["attempt_number"] == len(previous) + 1, "Attempt number gap")
            _require(payload["messages_sha256"] == digest(canonical_bytes(payload["messages"])), "Message hash mismatch")
            if payload["request_body"] is not None:
                body = payload["request_body"]
                expected_model = manifest["policies"][Seat(payload["seat"]).name]["requested_model"]
                _require(body["model"] == expected_model, "Request model differs from assigned seat model")
                legacy_order = (
                    "model", "messages", "stream", "max_tokens", "temperature", "reasoning"
                )
                legacy_body = {key: body[key] for key in legacy_order if key in body}
                _require(
                    payload["request_body_sha256"]
                    in {
                        digest(canonical_bytes(body)),
                        digest(json.dumps(legacy_body).encode("utf-8")),
                    },
                    "Request body hash mismatch",
                )
                _require(payload["request_body"]["messages"] == payload["messages"], "Request messages mismatch")
            if payload["attempt_number"] == 1:
                expected_messages = render_messages(
                    observe(state, state.current_seat, codec),
                    prompt_version=manifest["prompt"]["version"],
                )
                _require(payload["messages"] == expected_messages, "First request messages mismatch")
            attempts[attempt_id] = payload
        elif kind == "attempt_finished":
            attempt_id = payload["attempt_id"]
            _require(attempt_id in attempts, "Attempt result without start")
            _require(attempt_id not in attempt_status, "Duplicate attempt result")
            _require(payload["status"] in {"chosen", "invalid", "provider_error", "interrupted"}, "Unknown attempt status")
            if payload["status"] == "chosen":
                _require(payload["chosen_id"] in opened[decision_id]["legal_ids"], "Chosen ID was not legal")
            attempt_status[attempt_id] = payload["status"]
        elif kind == "decision_resolved":
            _require(decision_id in opened, "Resolution without decision")
            _require(payload["status"] in {"forced", "fallback", "recovery_fallback"}, "Unknown selector status")
            _require(decision_id not in resolved, "Duplicate selector")
            if payload["status"] == "forced":
                _require(len(opened[decision_id]["legal_ids"]) == 1, "Forced action had multiple choices")
            elif payload["status"] == "fallback":
                _require(any(
                    status == "invalid"
                    for attempt_id, status in attempt_status.items()
                    if attempt_id.startswith(decision_id + ":attempt:")
                ), "Fallback without invalid attempt")
            else:
                _require(any(
                    attempt_id.startswith(decision_id + ":attempt:")
                    and (attempt_id not in attempt_status or attempt_status[attempt_id] in {"invalid", "provider_error", "interrupted"})
                    for attempt_id in attempts
                ), "Recovery fallback without pending or invalid attempt")
            resolved[decision_id] = payload["status"]
        elif kind == "action_selected":
            _require(decision_id in opened, "Selection without decision")
            _require(decision_id not in selected_actions, "Duplicate action selection")
            _require(payload["action_index"] == applied, "Selection action index mismatch")
            action = _action(payload["action"])
            _require(codec.encode(action) == payload["action_id"], "Selected action ID mismatch")
            _require(payload["action_id"] in opened[decision_id]["legal_ids"], "Illegal selected action")
            _require(payload["state_sha256_before"] == state_digest(state), "Selection state hash mismatch")
            _verify_source_link(payload, decision_id, opened, attempts, attempt_status, resolved, records)
            selected_actions[decision_id] = payload
        elif kind == "action_applied":
            _require(decision_id in opened, "Applied action without decision")
            _require(payload["action_index"] == applied, "Action index gap")
            action = _action(payload["action"])
            _require(codec.encode(action) == payload["action_id"], "Action ID mismatch")
            _require(payload["action_id"] in opened[decision_id]["legal_ids"], "Illegal applied action")
            if decision_id in selected_actions:
                selected = selected_actions[decision_id]
                _require(selected["action_id"] == payload["action_id"], "Selected/applied action mismatch")
                _require(selected["source"] == payload["source"], "Selected/applied source mismatch")
                _require(selected["selected_attempt_id"] == payload["selected_attempt_id"], "Selected/applied attempt mismatch")
            _verify_source_link(payload, decision_id, opened, attempts, attempt_status, resolved, records)
            before = state
            state = apply_action(state, action)
            _require(payload["state_sha256_before"] == state_digest(before), "Pre-action state hash mismatch")
            _require(payload["state_sha256_after"] == state_digest(state), "Post-action state hash mismatch")
            _require(payload["scores_before"] == list(before.scores), "Pre-action score mismatch")
            _require(payload["scores_after"] == list(state.scores), "Post-action score mismatch")
            _require(payload["reward"]["team_stone_delta"] == [new - old for old, new in zip(before.scores, state.scores)], "Reward mismatch")
            events = payload["events"]
            expected_events = [
                {"event_id": f"{match_id}:event:{event.index}", **json.loads(canonical_bytes(asdict(event)))}
                for event in state.events[len(before.events) :]
            ]
            _require(events == expected_events, "Public event mismatch")
            _require(payload["terminated"] == (state.phase is Phase.COMPLETE), "Termination mismatch")
            _require(payload["next_actor"] == (int(state.current_seat) if state.current_seat is not None else None), "Next actor mismatch")
            applied += 1
        elif kind == "episode_finished":
            _require(not episode_finished, "Duplicate episode_finished")
            episode_finished = True
            episode_reason = payload["reason"]
            _require(payload["actions"] == applied, "Episode action count mismatch")
            _require(payload["scores"] == list(state.scores), "Episode score mismatch")
            _require(payload["terminated"] == (state.phase is Phase.COMPLETE), "Episode termination mismatch")
        elif kind == "run_finished":
            _require(episode_finished and not finished, "Duplicate or premature run_finished")
            _require(payload["actions"] == applied, "Run action count mismatch")
            _require(payload["report_sha256"] == file_digest(directory / "report.json"), "Report hash mismatch")
            execution_status = payload["status"]
            finished = True
        else:
            raise ValueError(f"Unknown journal record type: {kind}")
    _require(records[-1]["record_type"] == "run_finished" if finished else True, "Records after run_finished")
    _require(finished or allow_partial, "Missing run_finished; run needs recovery")
    for attempt_id in attempts:
        decision = attempt_id.rsplit(":attempt:", 1)[0]
        _require(
            attempt_id in attempt_status
            or not finished
            or resolved.get(decision) == "recovery_fallback",
            "Missing attempt result",
        )
    result_path = directory / "result.json"
    if finished and result_path.exists():
        result = json.loads(result_path.read_text())
        _require(result["actions"] == applied, "Result action count mismatch")
        _require(result["scores"] == list(state.scores), "Result score mismatch")
        _require(result["terminated"] == (state.phase is Phase.COMPLETE), "Result termination mismatch")
    elif finished:
        _require(allow_partial, "Missing result.json; run needs recovery")
    trace_path = directory / "match.json"
    if trace_path.exists():
        trace = load_trace(trace_path)
        _require(len(trace.actions) == applied, "Trace action count mismatch")
        _require(replay_trace(trace).scores == state.scores, "Trace replay mismatch")
    return {
        "format": "v2",
        "status": "finalized" if finished and result_path.exists() else "needs_finalization" if finished else "partial",
        "execution_status": execution_status,
        "reason": episode_reason,
        "run_id": run_id,
        "actions": applied,
        "scores": list(state.scores),
        "attempts": len(attempts),
        "pending_decision_id": (
            f"{match_id}:decision:{applied}" if f"{match_id}:decision:{applied}" in opened else None
        ),
        "unfinished_attempt_ids": sorted(set(attempts) - set(attempt_status)),
        "winner": 0 if state.scores[0] >= state.ruleset.target_score else 1 if state.scores[1] >= state.ruleset.target_score else None,
    }


@dataclass(frozen=True)
class ResumePrefix:
    manifest: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    actions: tuple[Action, ...]
    state: GameState
    complete_line: bool
    pending_decision_id: str | None
    unfinished_attempt_ids: tuple[str, ...]


@contextmanager
def resume_lock(directory: Path):
    """Allow one process at a time to validate and extend a partial run."""
    with (directory / ".resume.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process is already resuming this run") from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_resume_prefix(directory: Path) -> ResumePrefix:
    """Replay and validate the committed prefix before another action is chosen."""
    summary = validate_run(directory, allow_partial=True)
    _require(summary["format"] == "v2", "Only v2 runs can be resumed")
    _require(summary["status"] == "partial", "Run is already finished; use --recover to finalize")
    _require(not (directory / "checksums.json").exists(), "Partial run has finalized checksums")
    evidence = read_evidence(directory)
    manifest = dict(evidence.manifest)
    _require(
        manifest["provenance"]["source_sha256"] == current_source_digest(),
        "Current source differs from the run snapshot; inspect the old run, but resume with its original code",
    )
    state = new_game(
        manifest["rng"]["deal_seed"],
        _ruleset(manifest["ruleset"]),
        Seat(manifest["initial_dealer"]),
    )
    actions = tuple(_action(dict(item.action)) for item in evidence.actions)
    for action in actions:
        state = apply_action(state, action)
    return ResumePrefix(
        manifest=manifest,
        records=tuple(dict(record) for record in evidence.records),
        actions=actions,
        state=state,
        complete_line=evidence.complete_line,
        pending_decision_id=summary["pending_decision_id"],
        unfinished_attempt_ids=tuple(summary["unfinished_attempt_ids"]),
    )


def trim_torn_journal(directory: Path, prefix: ResumePrefix) -> str | None:
    """Keep the original torn bytes, then atomically publish the complete prefix."""
    if prefix.complete_line:
        return None
    path = directory / "records.jsonl"
    original = path.read_bytes()
    end = original.rfind(b"\n") + 1
    _require(end > 0, "No complete journal record to resume")
    backup = directory / f"records.torn.{uuid4().hex}.bin"
    atomic_bytes(backup, original)
    atomic_bytes(path, original[:end])
    return backup.name


def reconcile_move_log(directory: Path, prefix: ResumePrefix) -> str | None:
    """Regenerate lagging v1 moves from committed v2 actions, keeping old bytes."""
    state = new_game(
        prefix.manifest["rng"]["deal_seed"],
        _ruleset(prefix.manifest["ruleset"]),
        Seat(prefix.manifest["initial_dealer"]),
    )
    rows = []
    codec = ActionCodec(state.ruleset.target_score)
    for index, action in enumerate(prefix.actions):
        before = state
        state = apply_action(state, action)
        rows.append({
            "schema_version": 1,
            "run_id": prefix.manifest["run_id"],
            "action_index": index,
            "hand_number": before.hand_number,
            "seat": int(before.current_seat),
            "phase": before.phase.value,
            "action_id": codec.encode(action),
            "action": asdict(action),
            "scores_before": before.scores,
            "scores_after": state.scores,
            "events": [asdict(event) for event in state.events[len(before.events):]],
        })
    expected = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    path = directory / "moves.jsonl"
    original = path.read_bytes() if path.exists() else b""
    if original == expected:
        return None
    backup = directory / f"moves.pre-resume.{uuid4().hex}.bin"
    atomic_bytes(backup, original)
    atomic_bytes(path, expected)
    return backup.name


def trim_torn_decision_log(directory: Path) -> str | None:
    """Keep a torn compatibility-log tail from becoming an interior JSONL line."""
    path = directory / "decisions.jsonl"
    if not path.exists():
        return None
    original = path.read_bytes()
    if not original or original.endswith(b"\n"):
        return None
    end = original.rfind(b"\n") + 1
    backup = directory / f"decisions.torn.{uuid4().hex}.bin"
    atomic_bytes(backup, original)
    atomic_bytes(path, original[:end])
    return backup.name


def recover_run(directory: Path) -> dict[str, object]:
    """Mark a hard-stopped run partial without reissuing provider requests."""
    summary = validate_run(directory, allow_partial=True)
    _require(summary["format"] == "v2", "Recovery requires a v2 journal")
    if summary["status"] == "needs_finalization":
        atomic_json(
            directory / "result.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": summary["run_id"],
                "status": summary["execution_status"],
                "terminated": summary["winner"] is not None,
                "truncated": summary["winner"] is None,
                "actions": summary["actions"],
                "scores": summary["scores"],
                "winner": summary["winner"],
                "error": summary["reason"],
            },
        )
        write_checksums(directory)
        return validate_run(directory)
    if summary["status"] == "finalized":
        if not (directory / "checksums.json").exists():
            write_checksums(directory)
        return summary
    recovery = {
        "schema_version": SCHEMA_VERSION,
        "status": "partial",
        "reason": "journal ended before run_finished",
        "run_id": summary["run_id"],
        "actions": summary["actions"],
        "scores": summary["scores"],
        "pending_decision_id": summary["pending_decision_id"],
        "unfinished_attempt_ids": summary["unfinished_attempt_ids"],
        "provider_requests_reissued": False,
    }
    atomic_json(directory / "recovery.json", recovery)
    return recovery


def _validate_legacy(directory: Path) -> dict[str, object]:
    moves = _jsonl(directory / "moves.jsonl")
    decisions = _jsonl(directory / "decisions.jsonl")
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    _require(report is not None or moves, "No recognized run records")
    if report:
        state = new_game(report["seed"], _ruleset(report["ruleset"]))
        codec = ActionCodec(state.ruleset.target_score)
        for index, move in enumerate(moves):
            _require(move["action_index"] == index, "Legacy action index gap")
            action = _action(move["action"])
            _require(action in legal_actions(state, state.current_seat), "Legacy illegal action")
            _require(codec.encode(action) == move["action_id"], "Legacy action ID mismatch")
            before = state
            state = apply_action(state, action)
            _require(move["scores_before"] == list(before.scores), "Legacy pre-score mismatch")
            _require(move["scores_after"] == list(state.scores), "Legacy post-score mismatch")
        _require(report["game"]["actions"] == len(moves), "Legacy report action count mismatch")
        _require(report["game"]["teams"][0]["stones"] == state.scores[0], "Legacy score mismatch")
        _require(report["game"]["teams"][1]["stones"] == state.scores[1], "Legacy score mismatch")
    for record in decisions:
        _require(record["action_index"] <= len(moves), "Legacy decision index out of range")
        if record.get("chosen_id") is not None:
            _require(record["chosen_id"] in record["legal_ids"], "Legacy illegal choice")
    trace_path = directory / "match.json"
    if trace_path.exists():
        replay_trace(load_trace(trace_path))
    return {
        "format": "v1",
        "status": report["status"] if report else "partial",
        "run_id": report.get("run_id") if report else None,
        "actions": len(moves),
        "attempts": len(decisions),
        "unavailable_provenance": {
            "engine_revision": "unknown unless independently archived",
            "request_start_times": "not recorded in v1",
            "provider_revision": "not recorded in v1",
            "token_probabilities": "not recorded in v1",
        },
    }


def validate_command(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Mark a hard-stopped v2 journal partial or finish an atomic result",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write a validation.json sidecar with source file hashes",
    )
    args = parser.parse_args(argv)
    try:
        result = recover_run(args.run_dir) if args.recover else validate_run(args.run_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Validation failed: {exc}\n")
    if args.save:
        try:
            names = (
                "manifest.json",
                "source.zip",
                "records.jsonl",
                "result.json",
                "moves.jsonl",
                "decisions.jsonl",
                "match.json",
                "report.json",
            )
            atomic_json(
                args.run_dir / "validation.json",
                {
                    "schema": "llmoose.validation",
                    "schema_version": 1,
                    "validated_at": datetime.now(timezone.utc).isoformat(),
                    "result": result,
                    "source_sha256": {
                        name: file_digest(args.run_dir / name)
                        for name in names
                        if (args.run_dir / name).exists()
                    },
                },
            )
        except OSError as exc:
            parser.exit(1, f"Could not save validation: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(validate_command())

"""Versioned, durable run evidence kept alongside the original v1 files."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from llmoose.agents.prompt import load_system_prompt
from llmoose.game.actions import ActionCodec
from llmoose.game.state import GameState, Seat

SCHEMA = "llmoose.run-event"
SCHEMA_VERSION = 2
MANIFEST_SCHEMA = "llmoose.run-manifest"
CODEC_VERSION = 1
ROOT = Path(__file__).resolve().parents[3]


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def state_digest(state: GameState) -> str:
    """Evaluator-only digest of every replay-relevant state field."""
    return digest(canonical_bytes(_jsonable(state)))


def atomic_json(path: Path, value: object) -> None:
    """Publish one complete JSON value, syncing file and parent directory."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_bytes(path: Path, value: bytes) -> None:
    """Publish exact bytes with the same durability boundary as atomic_json."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def source_archive_bytes() -> bytes:
    """Build the reproducible local source archive without touching a run."""
    files = sorted((ROOT / "src" / "llmoose").rglob("*.py"))
    files += sorted((ROOT / "src" / "llmoose").rglob("*.txt"))
    files += [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in files:
            if not source.exists():
                continue
            info = zipfile.ZipInfo(source.relative_to(ROOT).as_posix())
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, source.read_bytes())
    return output.getvalue()


def current_source_digest() -> str:
    return digest(source_archive_bytes())


def source_snapshot(directory: Path) -> str:
    """Keep the exact local source used, including uncommitted changes."""
    path = directory / "source.zip"
    with path.open("xb") as handle:
        handle.write(source_archive_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    return file_digest(path)


def git_provenance() -> dict[str, object]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "status": status,
    }


def build_manifest(
    directory: Path,
    state: GameState,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    run_id = str(metadata["run_id"])
    codec = ActionCodec(state.ruleset.target_score)
    actions = [asdict(action) for action in codec.actions]
    prompt_version = int(metadata["prompt_version"])
    source_sha = source_snapshot(directory)
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "match_id": f"{run_id}:match:0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episode_mode": "match",
        "intent": dict(metadata),
        "initial_dealer": int(state.dealer),
        "ruleset": asdict(state.ruleset),
        "codec": {
            "version": CODEC_VERSION,
            "size": codec.size,
            "table_sha256": digest(canonical_bytes(actions)),
        },
        "prompt": {
            "version": prompt_version,
            "renderer_version": metadata["renderer_version"],
            "system_sha256": digest(load_system_prompt(prompt_version).encode()),
        },
        "policies": {
            seat.name: {
                "seat": int(seat),
                "team": int(seat) % 2,
                "kind": metadata["teams"][int(seat) % 2],
                "requested_model": (
                    metadata["requested_model"]
                    if metadata["teams"][int(seat) % 2] == "llm"
                    else None
                ),
                "checkpoint_revision": None,
                "tokenizer_revision": None,
                "revision_reason": "not exposed by the current provider or baseline interface",
            }
            for seat in Seat
        },
        "request_policy": {
            "max_retries": 0 if metadata["strict"] else 2,
            "fallback_on_invalid": not metadata["strict"],
            "stream": False,
            "top_p": None,
            "top_k": None,
            "unsupported_sampling_fields_reason": "not configurable in the current transport",
            "max_steps": metadata["max_steps"],
        },
        "rng": {
            "deal_seed": state.seed,
            "scripted_policy_seeds": {
                seat.name: state.seed + int(seat)
                for seat in Seat
                if metadata["teams"][int(seat) % 2] == "random"
            },
            "model_sampling_seed": None,
            "model_sampling_seed_reason": "not controlled by the current provider interface",
        },
        "reward": {
            "definition": "team_stone_delta_v1",
            "horizon": "match",
            "discount_time_unit": "engine_action",
        },
        "provenance": {
            "git": git_provenance(),
            "source_snapshot": "source.zip",
            "source_sha256": source_sha,
            "lock_sha256": file_digest(ROOT / "uv.lock"),
            "pyproject_sha256": file_digest(ROOT / "pyproject.toml"),
            "python": sys.version,
        },
        "experiment_id": None,
        "data_split": None,
        "data_split_reason": "ad hoc CLI run; no benchmark split declared",
    }


class Journal:
    """Append one fsynced JSON record per line in monotonic writer order."""

    def __init__(self, path: Path, run_id: str, match_id: str, *, sequence: int = 0):
        self.path = path
        self.run_id = run_id
        self.match_id = match_id
        self.sequence = sequence
        self._handle = path.open("r+b" if sequence else "xb")
        if sequence:
            self._handle.seek(0, os.SEEK_END)
        else:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

    def append(
        self,
        record_type: str,
        payload: Mapping[str, object],
        *,
        decision_id: str | None = None,
        visibility: str = "restricted",
    ) -> dict[str, object]:
        record = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "record_type": record_type,
            "run_id": self.run_id,
            "match_id": self.match_id,
            "record_id": f"{self.run_id}:record:{self.sequence}",
            "sequence": self.sequence,
            "decision_id": decision_id,
            "visibility": visibility,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "payload": dict(payload),
        }
        line = canonical_bytes(record) + b"\n"
        written = self._handle.write(line)
        if written != len(line):
            raise OSError("Incomplete journal write")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.sequence += 1
        return record

    def close(self) -> None:
        self._handle.close()


def read_journal(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Read complete lines and report, rather than invent, a torn final line."""
    data = path.read_bytes()
    complete = data.endswith(b"\n") or not data
    rows = data.split(b"\n")
    if not complete:
        rows.pop()
    records = [json.loads(row) for row in rows if row]
    return records, complete


def write_checksums(directory: Path) -> dict[str, str]:
    names = (
        "manifest.json",
        "source.zip",
        "records.jsonl",
        "result.json",
        "moves.jsonl",
        "decisions.jsonl",
        "match.json",
        "report.json",
    ) + tuple(
        path.name for path in sorted(directory.glob("*.bin"))
        if path.name.startswith(("records.torn.", "moves.pre-resume.", "decisions.torn."))
    )
    hashes = {name: file_digest(directory / name) for name in names if (directory / name).exists()}
    atomic_json(directory / "checksums.json", {"schema_version": 1, "sha256": hashes})
    return hashes

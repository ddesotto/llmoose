import json

import pytest

from llmoose.agents.providers import ChatCompletionProvider, Completion
from llmoose.cli import play_command
from llmoose.simulation.artifacts import canonical_bytes, digest, write_checksums
from llmoose.simulation.validation import validate_run


class LocalProvider(ChatCompletionProvider):
    def __call__(self, messages, *, response_format=None):
        if response_format is not None:
            ids = response_format["json_schema"]["schema"]["properties"]["action_id"]["enum"]
        else:
            menu = messages[1]["content"].split("Legal actions (ID: meaning):\n")[1]
            ids = [int(line.strip().split(":", 1)[0]) for line in menu.splitlines()]
        chosen = next((i for i in (1, 5, 3) if i in ids), ids[0])
        if response_format is not None:
            response = json.dumps({"action_id": chosen})
        elif '"action_id"' in messages[0]["content"]:
            response = 'Selected legal action. ' + json.dumps({"action_id": chosen})
        else:
            response = f"<answer>{chosen}</answer>"
        return Completion(response, model=self.model)


def recorded_run(tmp_path, monkeypatch, version=5, response_mode="strict"):
    provider = LocalProvider("test/model", base_url="http://localhost:9999/v1")
    monkeypatch.setattr("llmoose.cli._build_provider", lambda *_, **__: provider)
    directory = tmp_path / "run"
    assert play_command([
        "--policy", "llm", "--model", "test/model", "--seed", "12",
        "--target-score", "8", "--prompt-version", str(version),
        "--llm-response-mode", response_mode,
        "--run-dir", str(directory),
    ]) == 0
    assert validate_run(directory)["status"] == "finalized"
    return directory


def rewrite_records(directory, records):
    (directory / "records.jsonl").write_bytes(
        b"".join(canonical_bytes(row) + b"\n" for row in records)
    )
    write_checksums(directory)


@pytest.mark.parametrize("tamper", ["missing_schema", "wrong_enum", "boolean_enum", "response_mismatch", "response_prose"])
def test_v5_rejects_semantic_tampering_after_rehash(tmp_path, monkeypatch, tamper):
    directory = recorded_run(tmp_path, monkeypatch)
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    if tamper in {"missing_schema", "wrong_enum", "boolean_enum"}:
        payload = next(row["payload"] for row in records if row["record_type"] == "attempt_started")
        body = payload["request_body"]
        if tamper == "missing_schema":
            del body["response_format"]
        else:
            body["response_format"]["json_schema"]["schema"]["properties"]["action_id"]["enum"] = [False, True] if tamper == "boolean_enum" else [99]
        payload["request_body_sha256"] = digest(canonical_bytes(body))
        error = "Request action schema"
    else:
        payload = next(row["payload"] for row in records if row["record_type"] == "attempt_finished")
        assert payload["chosen_id"] == 1
        payload["response"] = '{"action_id": 0}' if tamper == "response_mismatch" else 'Invented future state {"action_id": 1}'
        error = "Response/chosen action mismatch" if tamper == "response_mismatch" else "Expected exactly"
    rewrite_records(directory, records)
    with pytest.raises(ValueError, match=error):
        validate_run(directory)


def test_v4_keeps_legacy_response_validation(tmp_path, monkeypatch):
    directory = recorded_run(tmp_path, monkeypatch, version=4)
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    payload = next(row["payload"] for row in records if row["record_type"] == "attempt_finished")
    payload["response"] = "historical response representation"
    rewrite_records(directory, records)
    assert validate_run(directory)["status"] == "finalized"


@pytest.mark.parametrize("tamper", [None, "mode", "normalization", "schema"])
@pytest.mark.parametrize("version", [4, 5])
def test_compatible_journal_uses_manifest_mode(tmp_path, monkeypatch, tamper, version):
    # The local provider returns prose without a schema, which compatible JSON
    # policies normalize while preserving the exact response in the journal.
    directory = recorded_run(tmp_path, monkeypatch, version=version, response_mode="compatible")
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    chosen = next(row["payload"] for row in records if row["record_type"] == "attempt_finished")
    assert chosen["response_normalized"] is (version == 5)
    if tamper is None:
        return
    if tamper == "mode":
        chosen["response_mode"] = "strict"
        error = "Recorded response mode differs"
    elif tamper == "normalization":
        chosen["response_normalized"] = not chosen["response_normalized"]
        error = "Response normalization flag mismatch"
    else:
        started = next(row["payload"] for row in records if row["record_type"] == "attempt_started")
        started["request_body"]["response_format"] = {"type": "json_object"}
        started["request_body_sha256"] = digest(canonical_bytes(started["request_body"]))
        error = "Compatible request unexpectedly constrained"
    rewrite_records(directory, records)
    with pytest.raises(ValueError, match=error):
        validate_run(directory)

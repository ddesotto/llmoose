"""Compatibility parsing remains visible without rewriting model evidence."""

import json
from copy import deepcopy

from llmoose.simulation.reporting import DecisionReporter, LLMMetrics


def test_only_explicitly_normalized_chosen_responses_are_counted():
    metrics = LLMMetrics()
    rows = [
        {"status": "chosen", "attempt": 1, "response_normalized": True},
        {"status": "chosen", "attempt": 1, "response_normalized": False},
        {"status": "chosen", "attempt": 1},  # Historical records remain valid.
        {"status": "chosen", "attempt": 1, "response_normalized": "true"},
        {"status": "invalid", "attempt": 1, "response_normalized": True},
        {"status": "forced", "attempt": 0, "response_normalized": True},
        {"status": "fallback", "attempt": 2, "response_normalized": True},
    ]
    for row in rows:
        metrics.add(row)
    result = metrics.to_dict()
    assert result["normalized_responses"] == 1
    assert result["valid_responses"] == 4
    assert result["invalid_outputs"] == 1
    assert result["forced_actions"] == 1
    assert result["fallbacks"] == 1


def test_reporter_preserves_raw_response_and_restores_normalization_counts(tmp_path):
    rows = [
        {
            "seat": 0, "status": "chosen", "attempt": 1, "chosen_id": 1,
            "response": 'My choice is {"action_id": 1}.',
            "response_mode": "compatible", "response_normalized": True,
        },
        {
            "seat": 1, "status": "chosen", "attempt": 1, "chosen_id": 0,
            "response": '{"action_id": 0}',
            "response_mode": "strict", "response_normalized": False,
        },
    ]
    original = deepcopy(rows)
    path = tmp_path / "decisions.jsonl"
    with DecisionReporter(str(path), {}) as reporter:
        for row in rows:
            reporter(row)
        metrics = reporter.metrics()
    assert rows == original
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    for source, record in zip(original, saved, strict=True):
        assert all(record[key] == value for key, value in source.items())
    restored = DecisionReporter(None, {})
    for record in saved:
        restored.restore(record)
    assert restored.metrics() == metrics
    assert metrics["total"]["normalized_responses"] == 1
    assert metrics["by_seat"]["0"]["normalized_responses"] == 1
    assert metrics["by_team"]["0"]["normalized_responses"] == 1
    assert metrics["by_team"]["1"]["normalized_responses"] == 0

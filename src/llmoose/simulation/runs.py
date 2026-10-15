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
from llmoose.rules.hands import Lance
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
    ) -> None:
        self.directory = directory
        self.state = initial_state
        self.metadata = dict(metadata)
        self.decisions = decisions
        self.action_count = 0
        self._moves: TextIO | None = None

    def __enter__(self) -> RunRecorder:
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started = perf_counter()
        self._moves = (self.directory / "moves.jsonl").open("w", encoding="utf-8")
        return self

    def on_transition(
        self, before: GameState, action: Action, after: GameState
    ) -> None:
        index = self.action_count
        self.state = after
        self.action_count += 1
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
        self._moves.write(json.dumps(record, sort_keys=True) + "\n")
        self._moves.flush()

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
        (self.directory / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

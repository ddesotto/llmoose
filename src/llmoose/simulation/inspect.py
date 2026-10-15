"""Read saved runs and replay completed games without contacting a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from llmoose.game.engine import apply_action
from llmoose.game.state import Seat, new_game
from llmoose.simulation.reporting import DecisionReporter
from llmoose.simulation.terminal import TerminalSpectator
from llmoose.simulation.traces import load_trace, replay_trace


def read_jsonl(path: Path) -> list[dict]:
    """Read flushed records, tolerating an unfinished last line during live play."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not text.endswith("\n"):
                break
            raise
    return records


def _summary(
    directory: Path, report: dict | None, moves: list, decisions: list
) -> None:
    print(f"Run: {directory}")
    if report is not None:
        game = report["game"]
        print(
            f"Status: {report['status']} | model: {report.get('requested_model') or 'none'}"
        )
        print(
            f"Target: {report['target_score']} | hands: {game['hands_dealt']} | actions: {game['actions']}"
        )
        for team in game["teams"]:
            name = report["teams"][team["team"]]
            print(
                f"Team {team['team']} ({name}): {team['outcome']}, {team['stones']} stones, differential {team['stone_difference']:+d}"
            )
            print(
                "  Lance points: "
                + ", ".join(f"{k}={v['points']}" for k, v in team["by_lance"].items())
            )
        if report.get("error"):
            print(f"Error: {report['error']}")
    else:
        print(f"No final report yet; {len(moves)} applied moves saved.")
        if moves:
            print(
                f"Latest score: {moves[-1]['scores_after']} | hand {moves[-1]['hand_number'] + 1}"
            )
    if decisions:
        # Recalculate so old runs also show newly supported usage fields.
        metrics = DecisionReporter(None, {})
        for record in decisions:
            metrics.total.add(record)
        counts = metrics.total.to_dict()
        rate = counts["first_try_legal_rate"]
        valid = f"{rate:.1%}" if rate is not None else "n/a"
        print(
            f"LLM: {counts['requests']} requests | first-try legal {valid} | {counts['invalid_outputs']} invalid | {counts['fallbacks']} fallbacks"
        )
        for name, usage in counts["usage"].items():
            total = usage["reported_total"]
            print(
                f"  {name}: {total if total is not None else 'not reported'} ({usage['requests_reporting']}/{counts['requests']} requests reporting)"
            )


def _show_action(index: int, moves: list, decisions: list) -> None:
    matching_moves = [m for m in moves if m["action_index"] == index]
    attempts = [d for d in decisions if d["action_index"] == index]
    if not matching_moves and not attempts:
        raise ValueError(f"No saved action or attempt at index {index}")
    for move in matching_moves:
        print(
            f"\nAction {index}: {Seat(move['seat']).name} {move['phase']} {move['action']}"
        )
        print(f"Scores: {move['scores_before']} -> {move['scores_after']}")
    if not attempts:
        print("No LLM record for this action (baseline seat).")
    for record in attempts:
        print(f"\nAttempt {record['attempt']}: {record['status']}")
        for message in record["messages"]:
            print(f"[{message['role']}]\n{message['content']}")
        print(f"Response: {record.get('response', '(forced action; no request)')}")
        print(f"Usage: {record.get('usage', {})}")
        if record.get("reasoning"):
            print(f"Provider reasoning:\n{record['reasoning']}")
        else:
            readable = [
                d.get("text") or d.get("summary")
                for d in record.get("reasoning_details", [])
                if d.get("type") in {"reasoning.text", "reasoning.summary"}
            ]
            if any(readable):
                print(
                    "Provider reasoning/summary:\n"
                    + "\n".join(t for t in readable if t)
                )
            else:
                print("No readable reasoning saved for this attempt.")


def inspect_command(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--decisions", action="store_true", help="List LLM decisions and attempts"
    )
    parser.add_argument(
        "--action",
        type=int,
        help="Show a zero-based action index, prompts, and reasoning",
    )
    parser.add_argument(
        "--replay", action="store_true", help="Replay a completed match in the terminal"
    )
    parser.add_argument("--step", action="store_true", help="Advance replay with Enter")
    args = parser.parse_args(argv)
    if args.step and not args.replay:
        parser.error("--step requires --replay")
    if args.action is not None and args.action < 0:
        parser.error("--action must be non-negative")
    try:
        if not args.run_dir.is_dir():
            raise ValueError("Run directory does not exist")
        report_path = args.run_dir / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        moves = read_jsonl(args.run_dir / "moves.jsonl")
        decisions = read_jsonl(args.run_dir / "decisions.jsonl")
        _summary(args.run_dir, report, moves, decisions)
        if args.decisions:
            for d in decisions:
                print(
                    f"action={d['action_index']:3} hand={d['hand_number'] + 1} {Seat(d['seat']).name:5} {d['phase']:15} attempt={d['attempt']} {d['status']:12} {d.get('response', '')}"
                )
        if args.action is not None:
            _show_action(args.action, moves, decisions)
        if args.replay:
            trace = load_trace(args.run_dir / "match.json")
            replay_trace(trace)
            state = new_game(trace.seed, trace.ruleset, trace.dealer)
            spectator = TerminalSpectator(step=args.step)
            print("\nReplay spectator view: all four hands are visible. No API calls.")
            for action in trace.actions:
                spectator.before_action(state)
                after = apply_action(state, action)
                spectator(state, action, after)
                state = after
    except KeyboardInterrupt:
        print("\nReplay stopped.")
        return 130
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Cannot inspect run: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(inspect_command())

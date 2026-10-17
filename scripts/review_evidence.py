"""Reproduce the review's engine check and summarize existing local run evidence.

Run from an installed checkout: python scripts/review_evidence.py
This makes no model requests and changes no game or run files. Its JSON output
can be redirected to a new evidence snapshot. Missing optional runs are reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

from llmoose.agents.baselines import ConservativePolicy, HeuristicPolicy
from llmoose.game.actions import ActionCodec
from llmoose.game.state import SEATS, team_for
from llmoose.rules.cards import Card, Rank, Suit
from llmoose.rules.hands import Lance, compare_lance, evaluate_hand
from llmoose.rules.ruleset import Ruleset
from llmoose.simulation.match import run_match
from llmoose.simulation.traces import load_trace, replay_trace

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranking_check() -> dict:
    def evaluate(ranks):
        return evaluate_hand(
            tuple(Card(Rank(rank), suit) for rank, suit in zip(ranks, Suit))
        )

    left = evaluate((4, 4, 4, 12))
    right = evaluate((7, 7, 7, 1))
    actual = compare_lance(left, right, Lance.PARES)
    return {
        "left_physical_ranks": [4, 4, 4, 12],
        "right_physical_ranks": [7, 7, 7, 1],
        "left_evaluated_triple_rank": list(left.pares.ranks),
        "right_evaluated_triple_rank": list(right.pares.ranks),
        "expected_comparison": -1,
        "actual_comparison": actual,
        "comparison_convention": "1: left wins; -1: right wins; 0: tie",
        "matches_expected": actual == -1,
    }


def mirrored_baseline() -> dict:
    """Reproduce the small offline exercise in docs/benchmark.md."""
    rules = Ruleset(target_score=8)
    rows = []
    for seed in range(16):
        for candidate_team in (0, 1):
            codec = ActionCodec(rules.target_score)
            policies = {
                seat: (
                    HeuristicPolicy
                    if team_for(seat) == candidate_team
                    else ConservativePolicy
                )(codec)
                for seat in SEATS
            }
            result = run_match(policies, seed=seed, ruleset=rules)
            rows.append(
                {
                    "seed": seed,
                    "candidate_team": candidate_team,
                    "candidate_won": result.winner == candidate_team,
                    "scores": list(result.scores),
                    "candidate_margin": result.scores[candidate_team]
                    - result.scores[1 - candidate_team],
                }
            )
    return {
        "candidate": "heuristic",
        "opponent": "conservative",
        "target_score": 8,
        "initial_dealer": "NORTH",
        "seed_clusters": 16,
        "games": len(rows),
        "candidate_wins": sum(row["candidate_won"] for row in rows),
        "candidate_mean_margin": sum(row["candidate_margin"] for row in rows)
        / len(rows),
        "results": rows,
        "scope": "Tutorial reproduction on current engine, not a frozen benchmark release.",
    }


def summarize_run(path: Path) -> dict:
    needed = ("report.json", "decisions.jsonl", "moves.jsonl", "match.json")
    missing = [name for name in needed if not (path / name).is_file()]
    if missing:
        return {"run": path.name, "available": False, "missing": missing}
    report = json.loads((path / "report.json").read_text())
    decisions = [
        json.loads(line)
        for line in (path / "decisions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    moves = [
        json.loads(line)
        for line in (path / "moves.jsonl").read_text().splitlines()
        if line.strip()
    ]
    chosen = [record for record in decisions if record["status"] == "chosen"]
    game = report["game"]
    trace = load_trace(path / "match.json")
    replayed = replay_trace(trace)
    result = {
        "run": path.name,
        "available": True,
        "sha256": {name: sha256(path / name) for name in needed},
        "seed": report["seed"],
        "target_score": report["target_score"],
        "prompt_version": report["prompt_version"],
        "requested_model": report["requested_model"],
        "status": report["status"],
        "teams": report["teams"],
        "scores": [team["stones"] for team in game["teams"]],
        "winner": game["winner"],
        "hands_dealt": game["hands_dealt"],
        "actions": game["actions"],
        "decision_status_counts": dict(Counter(r["status"] for r in decisions)),
        "chosen_action_counts": dict(Counter(str(r["chosen_id"]) for r in chosen)),
        "discard_actions": sum(m["action"]["kind"] == "discard" for m in moves),
        "replayed_scores": list(replayed.scores),
        "replay_matches_report": list(replayed.scores)
        == [team["stones"] for team in game["teams"]],
        "input_characters_first_last_inference": [
            sum(len(m["content"]) for m in record["messages"])
            for record in ([chosen[0], chosen[-1]] if chosen else [])
        ],
        "usage": report["llm"]["total"]["usage"],
        "lance_results": [team["by_lance"] for team in game["teams"]],
        "last_two_moves": [
            {
                key: move[key]
                for key in (
                    "action_index",
                    "seat",
                    "action",
                    "scores_before",
                    "scores_after",
                )
            }
            for move in moves[-2:]
        ],
    }
    # Store a pointer, not a copied reasoning transcript. The original remains
    # in the local decision artifact and may be absent from a shared checkout.
    result["review_decision_pointers"] = [
        {
            "action_index": r["action_index"],
            "chosen_id": r["chosen_id"],
            "phase": r["phase"],
            "reasoning_present": bool(r.get("reasoning") or r.get("reasoning_details")),
        }
        for r in chosen
        if path.name == "llm-long-40" and r["action_index"] in (195, 207)
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    sources = (
        "src/llmoose/rules/hands.py",
        "src/llmoose/game/engine.py",
        "src/llmoose/game/state.py",
        "src/llmoose/observations/core.py",
        "src/llmoose/agents/prompt.py",
        "src/llmoose/agents/llm.py",
        "src/llmoose/agents/baselines.py",
        "src/llmoose/simulation/match.py",
        "src/llmoose/agents/system_prompts/v2.txt",
        "scripts/review_evidence.py",
    )
    print(
        json.dumps(
            {
                "schema": "llmoose.review-evidence.v1",
                "python_version": platform.python_version(),
                "source_sha256": {name: sha256(ROOT / name) for name in sources},
                "medias_ranking_check": ranking_check(),
                "mirrored_baseline": mirrored_baseline(),
                "runs": [
                    summarize_run(args.runs_dir / name)
                    for name in ("llm-first", "llm-long-40")
                ],
                "scope": "Local artifacts, deterministic checks, and offline baselines; no model requests.",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Command-line entry points for running a local Mus match."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from llmoose.agents.baselines import ConservativePolicy, HeuristicPolicy, RandomPolicy
from llmoose.game.actions import ActionCodec
from llmoose.game.state import SEATS
from llmoose.rules.ruleset import Ruleset
from llmoose.simulation.match import run_match
from llmoose.simulation.traces import save_trace, trace_from_match


def play_command(argv: Optional[Sequence[str]] = None) -> int:
    """Run four baseline agents and print a compact terminal summary."""

    parser = argparse.ArgumentParser(description="Run a seeded no-señas Mus v1 match")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-score", type=int, default=40)
    parser.add_argument(
        "--policy",
        choices=("conservative", "heuristic", "random"),
        default="conservative",
    )
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument(
        "--trace", type=str, help="Write a replayable JSON trace to this path"
    )
    args = parser.parse_args(argv)
    ruleset = Ruleset(target_score=args.target_score)
    codec = ActionCodec(ruleset.target_score)
    if args.policy == "random":
        policies = {seat: RandomPolicy(codec, args.seed + int(seat)) for seat in SEATS}
    elif args.policy == "heuristic":
        policies = {seat: HeuristicPolicy(codec) for seat in SEATS}
    else:
        policies = {seat: ConservativePolicy(codec) for seat in SEATS}
    result = run_match(policies, args.seed, ruleset, args.max_steps)
    print(
        "winner=team-{winner} scores={scores} actions={actions} events={events}".format(
            winner=result.winner,
            scores=result.scores,
            actions=len(result.actions),
            events=len(result.final_state.events),
        )
    )
    if args.trace:
        save_trace(trace_from_match(result, {"policy": args.policy}), args.trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(play_command())

"""Command-line entry points for running a local Mus match."""

from __future__ import annotations

import argparse
import math
from typing import Optional, Sequence

from llmoose.agents.baselines import ConservativePolicy, HeuristicPolicy, RandomPolicy
from llmoose.game.actions import ActionCodec
from llmoose.game.state import SEATS, team_for
from llmoose.rules.ruleset import Ruleset
from llmoose.simulation.match import run_match
from llmoose.simulation.terminal import TerminalSpectator
from llmoose.simulation.traces import save_trace, trace_from_match


def play_command(argv: Optional[Sequence[str]] = None) -> int:
    """Run baseline agents, optionally watching or stepping through the match."""

    parser = argparse.ArgumentParser(description="Run a seeded no-señas Mus v1 match")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-score", type=int, default=40)
    parser.add_argument(
        "--policy",
        choices=("conservative", "heuristic", "random"),
        default="conservative",
    )
    parser.add_argument("--max-steps", type=int, default=10_000)
    for team in (0, 1):
        parser.add_argument(
            f"--team-{team}",
            choices=("conservative", "heuristic", "random"),
            help=f"Override --policy for team {team}",
        )
    parser.add_argument(
        "--watch", action="store_true", help="Show all hands, actions, and scoring"
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Press Enter before each action (implies --watch)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Seconds between actions (implies --watch; default: 0.4)",
    )
    parser.add_argument(
        "--trace", type=str, help="Write a replayable JSON trace to this path"
    )
    args = parser.parse_args(argv)
    if args.target_score < 2:
        parser.error("--target-score must be at least 2")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.delay is not None and (not math.isfinite(args.delay) or args.delay < 0):
        parser.error("--delay must be a finite, non-negative number")
    ruleset = Ruleset(target_score=args.target_score)
    codec = ActionCodec(ruleset.target_score)
    team_policies = (args.team_0 or args.policy, args.team_1 or args.policy)
    policies = {}
    for seat in SEATS:
        name = team_policies[team_for(seat)]
        if name == "random":
            policies[seat] = RandomPolicy(codec, args.seed + int(seat))
        else:
            policy_type = HeuristicPolicy if name == "heuristic" else ConservativePolicy
            policies[seat] = policy_type(codec)
    watch = args.watch or args.step or args.delay is not None
    spectator = None
    if watch:
        print(
            f"Mus | seed {args.seed} | first to {args.target_score} stones", flush=True
        )
        print(
            f"Team 0: North + South ({team_policies[0]}) | "
            f"Team 1: East + West ({team_policies[1]})",
            flush=True,
        )
        print("Spectator view: all four hands are visible.", flush=True)
        spectator = TerminalSpectator(
            delay=0.4 if args.delay is None else args.delay, step=args.step
        )
    try:
        result = run_match(
            policies, args.seed, ruleset, args.max_steps, on_transition=spectator
        )
    except KeyboardInterrupt:
        print("\nStopped before match completion.")
        return 130
    except RuntimeError as exc:
        parser.exit(1, f"Match stopped: {exc}\n")
    print(
        "winner=team-{winner} scores={scores} actions={actions} events={events}".format(
            winner=result.winner,
            scores=result.scores,
            actions=len(result.actions),
            events=len(result.final_state.events),
        )
    )
    if args.trace:
        save_trace(
            trace_from_match(
                result,
                {
                    "policy": args.policy,
                    "team_0": team_policies[0],
                    "team_1": team_policies[1],
                },
            ),
            args.trace,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(play_command())

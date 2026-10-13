"""Command-line entry points for running a local Mus match."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from llmoose.agents.baselines import ConservativePolicy, HeuristicPolicy, RandomPolicy
from llmoose.agents.llm import StructuredLLMPolicy
from llmoose.agents.providers import OPENROUTER_BASE_URL, ChatCompletionProvider
from llmoose.game.actions import ActionCodec
from llmoose.game.state import SEATS, team_for
from llmoose.rules.ruleset import Ruleset
from llmoose.simulation.match import run_match
from llmoose.simulation.terminal import TerminalSpectator
from llmoose.simulation.traces import save_trace, trace_from_match


def play_command(argv: Optional[Sequence[str]] = None) -> int:
    """Run baseline or LLM agents, optionally watching each decision."""

    parser = argparse.ArgumentParser(description="Run a seeded no-señas Mus v1 match")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-score", type=int, default=40)
    parser.add_argument(
        "--policy",
        choices=("conservative", "heuristic", "random", "llm"),
        default="conservative",
    )
    parser.add_argument("--max-steps", type=int, default=10_000)
    for team in (0, 1):
        parser.add_argument(
            f"--team-{team}",
            choices=("conservative", "heuristic", "random", "llm"),
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
    llm_options = parser.add_argument_group("LLM player (OpenRouter by default)")
    llm_options.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL"),
        help="Model ID (or OPENROUTER_MODEL)",
    )
    llm_options.add_argument(
        "--base-url", default=OPENROUTER_BASE_URL, help="OpenAI-compatible API base URL"
    )
    llm_options.add_argument(
        "--api-key-env",
        help="Environment variable containing the API key; defaults to OPENROUTER_API_KEY for OpenRouter",
    )
    llm_options.add_argument(
        "--max-tokens", type=int, default=1024, help="Maximum output tokens per request"
    )
    llm_options.add_argument("--temperature", type=float, default=0.0)
    llm_options.add_argument(
        "--timeout", type=float, default=60.0, help="Request timeout in seconds"
    )
    llm_options.add_argument(
        "--llm-strict",
        action="store_true",
        help="Stop at the first invalid output; no retries or fallback",
    )
    llm_options.add_argument(
        "--llm-log", help="Write prompts, responses, usage, and action choices to JSONL"
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
    provider = None
    if "llm" in team_policies:
        if not args.model:
            parser.error("--model or OPENROUTER_MODEL is required for an LLM player")
        key_env = args.api_key_env
        if key_env is None and args.base_url.rstrip("/") == OPENROUTER_BASE_URL:
            key_env = "OPENROUTER_API_KEY"
        api_key = os.environ.get(key_env, "") if key_env else ""
        if key_env and not api_key:
            parser.error(f"Set {key_env} before running an LLM player")
        try:
            provider = ChatCompletionProvider(
                model=args.model,
                api_key=api_key,
                base_url=args.base_url,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except ValueError as exc:
            parser.error(str(exc))
    elif args.llm_log or args.llm_strict:
        parser.error("--llm-log and --llm-strict require an llm policy")
    if (
        args.llm_log
        and args.trace
        and Path(args.llm_log).resolve() == Path(args.trace).resolve()
    ):
        parser.error("--llm-log and --trace must use different files")
    log_handle = None

    def log_decision(record):
        if log_handle is not None:
            log_handle.write(
                json.dumps(
                    {
                        "seed": args.seed,
                        "target_score": args.target_score,
                        "requested_model": args.model,
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                        "strict": args.llm_strict,
                        **record,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            log_handle.flush()
        if record["status"] in {"invalid", "fallback"}:
            print(
                f"LLM seat {record['seat']}: {record['status']} (attempt {record['attempt']})",
                file=sys.stderr,
                flush=True,
            )

    policies = {}
    for seat in SEATS:
        name = team_policies[team_for(seat)]
        if name == "llm":
            policies[seat] = StructuredLLMPolicy(
                codec,
                provider,
                max_retries=0 if args.llm_strict else 2,
                fallback_on_invalid=not args.llm_strict,
                on_record=log_decision,
            )
        elif name == "random":
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
        if provider is not None:
            print(f"LLM model: {args.model}", flush=True)
        spectator = TerminalSpectator(
            delay=0.4 if args.delay is None else args.delay, step=args.step
        )
    try:
        if args.llm_log:
            path = Path(args.llm_log)
            path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = path.open("w", encoding="utf-8")
        result = run_match(
            policies,
            args.seed,
            ruleset,
            args.max_steps,
            on_transition=spectator,
            on_turn=spectator.before_action if spectator else None,
        )
    except KeyboardInterrupt:
        print("\nStopped before match completion.")
        return 130
    except (RuntimeError, OSError) as exc:
        parser.exit(1, f"Match stopped: {exc}\n")
    finally:
        if log_handle is not None:
            log_handle.close()
        llm_policies = [
            p for p in policies.values() if isinstance(p, StructuredLLMPolicy)
        ]
        if llm_policies:
            totals = {
                key: sum(asdict(p.telemetry)[key] for p in llm_policies)
                for key in asdict(llm_policies[0].telemetry)
            }
            print(
                "LLM " + " ".join(f"{key}={value}" for key, value in totals.items()),
                file=sys.stderr,
                flush=True,
            )
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
                    **(
                        {"model": args.model, "llm_strict": str(args.llm_strict)}
                        if provider
                        else {}
                    ),
                },
            ),
            args.trace,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(play_command())

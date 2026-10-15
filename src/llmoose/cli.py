"""CLI wiring: parse options, build players, run a match, report the result."""

from __future__ import annotations

import argparse
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence
from uuid import uuid4

from dotenv import load_dotenv

from llmoose.agents.factory import POLICY_NAMES, build_policies
from llmoose.agents.prompt import PROMPT_VERSION, PROMPT_VERSIONS, RENDERER_VERSION
from llmoose.agents.providers import OPENROUTER_BASE_URL, ChatCompletionProvider
from llmoose.game.actions import ActionCodec
from llmoose.game.state import SEATS, new_game, team_for
from llmoose.rules.ruleset import FEDERATION_RULESET_ID, RULESET_IDS, Ruleset
from llmoose.simulation.match import run_match
from llmoose.simulation.reporting import DecisionReporter, llm_summary, match_summary
from llmoose.simulation.runs import RunRecorder
from llmoose.simulation.terminal import TerminalSpectator
from llmoose.simulation.traces import save_trace, trace_from_match


def play_command(argv: Optional[Sequence[str]] = None) -> int:
    """Wire the experiment together; policies choose moves and the engine runs Mus."""
    load_dotenv(Path.cwd() / ".env", override=False)
    parser = _build_parser()
    args = parser.parse_args(argv)
    teams = (args.team_0 or args.policy, args.team_1 or args.policy)
    try:
        _validate_options(args, teams)
        provider = _build_provider(args) if "llm" in teams else None
    except ValueError as exc:
        parser.error(str(exc))

    ruleset = Ruleset(id=args.ruleset, target_score=args.target_score)
    directory = Path(args.run_dir) if args.run_dir else None
    if directory is not None:
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            parser.error(f"Cannot create a new run directory: {exc}")
        args.trace = str(directory / "match.json")
        if provider is not None:
            args.llm_log = str(directory / "decisions.jsonl")
    metadata = {
        "seed": args.seed,
        "target_score": args.target_score,
        "ruleset": asdict(ruleset),
        "teams": list(teams),
        "requested_model": args.model,
        "base_url": args.base_url if provider is not None else None,
        "prompt_version": args.prompt_version,
        "renderer_version": RENDERER_VERSION,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "max_steps": args.max_steps,
        "strict": args.llm_strict,
        "include_reasoning": args.reasoning,
    }
    if directory is not None:
        metadata["run_id"] = uuid4().hex
    reporter = DecisionReporter(args.llm_log, metadata)
    policies = build_policies(
        {seat: teams[team_for(seat)] for seat in SEATS},
        ActionCodec(ruleset.target_score),
        seed=args.seed,
        provider=provider,
        strict=args.llm_strict,
        on_record=reporter,
        prompt_version=args.prompt_version,
    )
    spectator = _build_spectator(args, teams, provider is not None)
    recorder = (
        RunRecorder(directory, new_game(args.seed, ruleset), metadata, reporter)
        if directory is not None
        else None
    )
    action_count = 0

    def before_action(state):
        reporter.action_index = action_count
        if spectator is not None:
            spectator.before_action(state)

    def after_action(before, action, after):
        nonlocal action_count
        action_count += 1
        if recorder is not None:
            recorder.on_transition(before, action, after)
        if spectator is not None:
            spectator(before, action, after)

    try:
        with reporter, recorder if recorder is not None else nullcontext():
            result = run_match(
                policies,
                args.seed,
                ruleset,
                args.max_steps,
                on_transition=after_action,
                on_turn=before_action,
            )
            if args.trace:
                trace_metadata = {
                    "policy": args.policy,
                    "team_0": teams[0],
                    "team_1": teams[1],
                }
                if provider is not None:
                    trace_metadata.update(
                        model=args.model,
                        llm_strict=str(args.llm_strict),
                        prompt_version=str(args.prompt_version),
                    )
                if directory is not None:
                    trace_metadata["run_id"] = metadata["run_id"]
                trace_path = Path(args.trace)
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                save_trace(trace_from_match(result, trace_metadata), trace_path)
    except KeyboardInterrupt:
        print("\nStopped before match completion.")
        return 130
    except (RuntimeError, OSError) as exc:
        parser.exit(1, f"Match stopped: {exc}\n")
    finally:
        summary = llm_summary(policies.values())
        if summary is not None:
            print(summary, file=sys.stderr, flush=True)
        if directory is not None and (directory / "report.json").exists():
            print(f"Run report: {directory / 'report.json'}", file=sys.stderr)

    print(match_summary(result))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a seeded, versioned no-señas Mus match"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-score", type=int, default=40)
    parser.add_argument("--ruleset", choices=RULESET_IDS, default=FEDERATION_RULESET_ID)
    parser.add_argument(
        "--policy",
        choices=POLICY_NAMES,
        default="conservative",
    )
    parser.add_argument("--max-steps", type=int, default=10_000)
    for team in (0, 1):
        parser.add_argument(
            f"--team-{team}",
            choices=POLICY_NAMES,
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
    parser.add_argument(
        "--run-dir",
        help="Create a new directory with moves, LLM logs, replay, and scores",
    )
    llm_options = parser.add_argument_group("LLM player (OpenRouter by default)")
    llm_options.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL"),
        help="Model ID (or OPENROUTER_MODEL)",
    )
    llm_options.add_argument(
        "--prompt-version",
        type=int,
        choices=PROMPT_VERSIONS,
        default=PROMPT_VERSION,
        help=f"Shared system prompt version (default: {PROMPT_VERSION})",
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
    llm_options.add_argument(
        "--reasoning",
        action="store_true",
        help="Request provider reasoning for inspection (if supported)",
    )
    return parser


def _validate_options(args: argparse.Namespace, teams: tuple[str, str]) -> None:
    if args.run_dir and (args.trace or args.llm_log):
        raise ValueError("--run-dir supplies --trace and --llm-log; use it on its own")
    if args.target_score < 2:
        raise ValueError("--target-score must be at least 2")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.delay is not None and (not math.isfinite(args.delay) or args.delay < 0):
        raise ValueError("--delay must be a finite, non-negative number")
    if "llm" in teams:
        if not args.model:
            raise ValueError(
                "--model or OPENROUTER_MODEL is required for an LLM player"
            )
    elif args.llm_log or args.llm_strict or args.reasoning:
        raise ValueError(
            "--llm-log, --llm-strict, and --reasoning require an llm policy"
        )
    if (
        args.llm_log
        and args.trace
        and Path(args.llm_log).resolve() == Path(args.trace).resolve()
    ):
        raise ValueError("--llm-log and --trace must use different files")


def _build_provider(args: argparse.Namespace) -> ChatCompletionProvider:
    """Resolve CLI configuration and credentials at the application boundary."""
    key_env = args.api_key_env
    if key_env is None and args.base_url.rstrip("/") == OPENROUTER_BASE_URL:
        key_env = "OPENROUTER_API_KEY"
    api_key = os.environ.get(key_env, "") if key_env else ""
    if key_env and not api_key:
        raise ValueError(f"Set {key_env} before running an LLM player")
    return ChatCompletionProvider(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        include_reasoning=args.reasoning,
    )


def _build_spectator(
    args: argparse.Namespace,
    teams: tuple[str, str],
    has_model: bool,
) -> TerminalSpectator | None:
    if not (args.watch or args.step or args.delay is not None):
        return None
    print(f"Mus | seed {args.seed} | first to {args.target_score} stones", flush=True)
    print(
        f"Team 0: North + South ({teams[0]}) | Team 1: East + West ({teams[1]})",
        flush=True,
    )
    print("Spectator view: all four hands are visible.", flush=True)
    if has_model:
        print(f"LLM model: {args.model}", flush=True)
    return TerminalSpectator(
        delay=0.4 if args.delay is None else args.delay,
        step=args.step,
    )


if __name__ == "__main__":
    raise SystemExit(play_command())

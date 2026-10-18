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

from llmoose.agents.baselines import ConservativePolicy, RandomPolicy
from llmoose.agents.factory import POLICY_NAMES, build_policies
from llmoose.agents.llm import StructuredLLMPolicy
from llmoose.agents.prompt import (
    PROMPT_VERSION,
    PROMPT_VERSIONS,
    renderer_version_for_prompt,
)
from llmoose.agents.providers import OPENROUTER_BASE_URL, ChatCompletionProvider
from llmoose.game.actions import ActionCodec
from llmoose.game.engine import apply_action
from llmoose.game.state import SEATS, Seat, new_game, team_for
from llmoose.observations.core import observe
from llmoose.rules.ruleset import FEDERATION_RULESET_ID, RULESET_IDS, Ruleset
from llmoose.simulation.match import run_match
from llmoose.simulation.reporting import DecisionReporter, llm_summary, match_summary
from llmoose.simulation.runs import RunRecorder
from llmoose.simulation.terminal import TerminalSpectator
from llmoose.simulation.traces import save_trace, trace_from_match
from llmoose.simulation.validation import (
    load_resume_prefix,
    reconcile_move_log,
    resume_lock,
    trim_torn_decision_log,
    trim_torn_journal,
)


def play_command(argv: Optional[Sequence[str]] = None) -> int:
    """Wire the experiment together; policies choose moves and the engine runs Mus."""
    load_dotenv(Path.cwd() / ".env", override=False)
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.resume_run:
        supplied = list(argv) if argv is not None else sys.argv[1:]
        allowed = {
            "--resume-run", "--watch", "--step", "--delay", "--api-key-env",
            "--team-0-api-key-env", "--team-1-api-key-env",
        }
        unexpected = [
            token.split("=", 1)[0]
            for token in supplied
            if token.startswith("--") and token.split("=", 1)[0] not in allowed
        ]
        if unexpected:
            parser.error(f"--resume-run uses the original run settings; omit {unexpected[0]}")
        try:
            with resume_lock(Path(args.resume_run)):
                return _play_with_options(parser, args)
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot resume run: {exc}")
    return _play_with_options(parser, args)


def _play_with_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    prefix = None
    if args.resume_run:
        if args.run_dir or args.trace or args.llm_log:
            parser.error("--resume-run owns its existing files; omit --run-dir, --trace, and --llm-log")
        try:
            prefix = load_resume_prefix(Path(args.resume_run))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"Cannot resume run: {exc}")
        intent = prefix.manifest["intent"]
        for key in (
            "seed", "target_score", "max_steps", "prompt_version", "temperature",
            "max_tokens", "timeout",
        ):
            setattr(args, key, intent[key])
        args.ruleset = prefix.manifest["ruleset"]["id"]
        args.team_0, args.team_1 = intent["teams"]
        args.model = intent["requested_model"]
        args.base_url = intent["base_url"] or OPENROUTER_BASE_URL
        models = intent.get(
            "requested_models",
            [args.model if kind == "llm" else None for kind in intent["teams"]],
        )
        urls = intent.get(
            "base_urls",
            [args.base_url if kind == "llm" else None for kind in intent["teams"]],
        )
        args.team_0_model, args.team_1_model = models
        args.team_0_base_url, args.team_1_base_url = urls
        args.llm_strict = intent["strict"]
        args.llm_response_mode = intent.get("llm_response_mode", "strict")
        args.reasoning = intent["include_reasoning"]
    teams = (args.team_0 or args.policy, args.team_1 or args.policy)
    try:
        _validate_options(args, teams)
        models = tuple(
            (getattr(args, f"team_{team}_model") or args.model) if teams[team] == "llm" else None
            for team in (0, 1)
        )
        urls = tuple(
            (getattr(args, f"team_{team}_base_url") or args.base_url) if teams[team] == "llm" else None
            for team in (0, 1)
        )
        providers = {
            seat: _build_provider(
                args,
                model=models[team_for(seat)],
                base_url=urls[team_for(seat)],
                key_env=getattr(args, f"team_{team_for(seat)}_api_key_env") or args.api_key_env,
            )
            for seat in SEATS if teams[team_for(seat)] == "llm"
        }
    except ValueError as exc:
        parser.error(str(exc))

    ruleset = (
        prefix.state.ruleset if prefix is not None
        else Ruleset(id=args.ruleset, target_score=args.target_score)
    )
    directory = Path(args.resume_run or args.run_dir) if args.resume_run or args.run_dir else None
    if directory is not None and prefix is None:
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            parser.error(f"Cannot create a new run directory: {exc}")
        args.trace = str(directory / "match.json")
        if providers:
            args.llm_log = str(directory / "decisions.jsonl")
    elif prefix is not None:
        args.trace = str(directory / "match.json")
        if providers:
            args.llm_log = str(directory / "decisions.jsonl")
        trim_torn_journal(directory, prefix)
        reconcile_move_log(directory, prefix)
        trim_torn_decision_log(directory)
    metadata = dict(prefix.manifest["intent"]) if prefix is not None else {
        "seed": args.seed,
        "target_score": args.target_score,
        "ruleset": asdict(ruleset),
        "teams": list(teams),
        "requested_model": next(iter(set(model for model in models if model)), None)
        if len({model for model in models if model}) <= 1 else None,
        "requested_models": list(models),
        "base_url": next(iter(set(url for url in urls if url)), None)
        if len({url for url in urls if url}) <= 1 else None,
        "base_urls": list(urls),
        "prompt_version": args.prompt_version,
        "renderer_version": renderer_version_for_prompt(args.prompt_version),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "max_steps": args.max_steps,
        "strict": args.llm_strict,
        "llm_response_mode": args.llm_response_mode,
        "include_reasoning": args.reasoning,
    }
    if directory is not None and prefix is None:
        metadata["run_id"] = uuid4().hex
    reporter = DecisionReporter(args.llm_log, metadata, append=prefix is not None)
    if prefix is not None:
        finished_attempts = {
            row["payload"]["attempt_id"]
            for row in prefix.records if row["record_type"] == "attempt_finished"
        }
        for row in prefix.records:
            if row["record_type"] in {"attempt_finished", "decision_resolved"}:
                reporter.restore(row["payload"])
            elif row["record_type"] == "attempt_started":
                payload = row["payload"]
                if payload["attempt_id"] not in finished_attempts:
                    reporter.restore({
                        "seat": payload["seat"], "attempt": payload["attempt_number"],
                        "status": "outcome_unknown",
                    })
    policies = build_policies(
        {seat: teams[team_for(seat)] for seat in SEATS},
        ActionCodec(ruleset.target_score),
        seed=args.seed,
        provider=providers or None,
        strict=args.llm_strict,
        on_record=reporter,
        prompt_version=args.prompt_version,
        response_mode=args.llm_response_mode,
    )
    if prefix is not None:
        pending_without_attempt = (
            prefix.pending_decision_id is not None
            and not any(
                row.get("decision_id") == prefix.pending_decision_id
                and row["record_type"] in {"attempt_started", "decision_resolved"}
                for row in prefix.records
            )
        )
        for row in prefix.records:
            kind, payload = row["record_type"], row["payload"]
            seat = payload.get("seat")
            policy = policies.get(Seat(seat)) if seat is not None else None
            if not isinstance(policy, StructuredLLMPolicy):
                continue
            counts = policy.telemetry
            if kind == "decision_opened":
                counts.decisions += 1
            elif kind == "attempt_started":
                counts.requests += 1
            elif kind == "attempt_finished":
                counts.invalid_outputs += int(payload["status"] == "invalid")
                counts.provider_errors += int(payload["status"] == "provider_error")
            elif kind == "decision_resolved":
                counts.forced_actions += int(payload["status"] == "forced")
                counts.fallbacks += int(payload["status"] in {"fallback", "recovery_fallback"})
        if pending_without_attempt:
            actor = prefix.state.current_seat
            policy = policies[actor]
            if isinstance(policy, StructuredLLMPolicy):
                policy.telemetry.decisions -= 1
        state = new_game(args.seed, ruleset)
        codec = ActionCodec(ruleset.target_score)
        for recorded in prefix.records:
            if recorded["record_type"] != "action_applied":
                continue
            payload = recorded["payload"]
            action = codec.decode(payload["action_id"])
            actor = state.current_seat
            if payload["source"] == "sampled":
                policy = policies[actor]
                assert isinstance(policy, RandomPolicy)
                if policy.act(observe(state, actor, codec)) != action:
                    parser.error("Random policy sequence differs from recorded run")
            state = apply_action(state, action)
    spectator = _build_spectator(args, teams, models)
    recorder = (
        RunRecorder(
            directory,
            prefix.state if prefix is not None else new_game(args.seed, ruleset),
            metadata,
            reporter,
            resume_records=list(prefix.records) if prefix is not None else None,
        )
        if directory is not None
        else None
    )
    if prefix is not None:
        recorder.action_count = len(prefix.actions)
    if recorder is not None:
        reporter.on_journal = recorder.on_policy_record
        for policy in policies.values():
            if isinstance(policy, StructuredLLMPolicy):
                policy.on_attempt_start = recorder.on_attempt_start
    action_count = len(prefix.actions) if prefix is not None else 0
    reporter.action_index = action_count

    def before_action(state):
        reporter.action_index = action_count
        if recorder is not None:
            recorder.on_turn(state)
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
            initial_state = prefix.state if prefix is not None else None
            initial_actions = prefix.actions if prefix is not None else ()
            if prefix is not None and prefix.pending_decision_id is not None:
                pending = [
                    row for row in prefix.records
                    if row.get("decision_id") == prefix.pending_decision_id
                ]
                selected = [
                    row["payload"] for row in pending
                    if row["record_type"] == "action_selected"
                ]
                choices = [
                    row["payload"] for row in pending
                    if row["record_type"] in {"attempt_finished", "decision_resolved"}
                    and row["payload"]["status"] in {"chosen", "forced", "fallback", "recovery_fallback"}
                ]
                if selected:
                    action = ActionCodec(ruleset.target_score).decode(selected[-1]["action_id"])
                    if selected[-1]["source"] == "sampled":
                        actor = initial_state.current_seat
                        policy = policies[actor]
                        assert isinstance(policy, RandomPolicy)
                        if policy.act(observe(initial_state, actor, ActionCodec(ruleset.target_score))) != action:
                            raise RuntimeError("Pending random choice differs from recorded selection")
                elif choices:
                    action = ActionCodec(ruleset.target_score).decode(choices[-1]["chosen_id"])
                elif any(row["record_type"] == "attempt_started" for row in pending):
                    actor = initial_state.current_seat
                    action = ConservativePolicy(ActionCodec(ruleset.target_score)).act(
                        observe(initial_state, actor, ActionCodec(ruleset.target_score))
                    )
                    reporter({
                        "seat": int(actor),
                        "status": "recovery_fallback",
                        "attempt": max(
                            row["payload"]["attempt_number"]
                            for row in pending if row["record_type"] == "attempt_started"
                        ),
                        "chosen_id": ActionCodec(ruleset.target_score).encode(action),
                        "recovery_reason": "pending model decision; no request was reissued",
                    })
                else:
                    action = None
                if action is not None:
                    if not selected:
                        recorder.on_action_selected(initial_state, action)
                    after = apply_action(initial_state, action)
                    after_action(initial_state, action, after)
                    initial_state = after
                    initial_actions = (*initial_actions, action)
            result = run_match(
                policies,
                args.seed,
                ruleset,
                args.max_steps,
                on_transition=after_action,
                on_turn=before_action,
                on_action_selected=recorder.on_action_selected if recorder is not None else None,
                initial_state=initial_state,
                initial_actions=initial_actions,
            )
            if args.trace:
                trace_metadata = {
                    "policy": args.policy,
                    "team_0": teams[0],
                    "team_1": teams[1],
                }
                if providers:
                    if metadata["requested_model"] is not None:
                        trace_metadata["model"] = metadata["requested_model"]
                    for team, model in enumerate(models):
                        if model is not None:
                            trace_metadata[f"team_{team}_model"] = model
                    trace_metadata.update(
                        llm_strict=str(args.llm_strict),
                        llm_response_mode=args.llm_response_mode,
                        prompt_version=str(args.prompt_version),
                        renderer_version=str(
                            renderer_version_for_prompt(args.prompt_version)
                        ),
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
    parser.add_argument(
        "--resume-run",
        help="Continue a validated partial v2 run from its committed action prefix",
    )
    llm_options = parser.add_argument_group("LLM player (OpenRouter by default)")
    llm_options.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL"),
        help="Model ID (or OPENROUTER_MODEL)",
    )
    for team in (0, 1):
        llm_options.add_argument(
            f"--team-{team}-model",
            help=f"Model ID for LLM seats on team {team}; overrides --model",
        )
        llm_options.add_argument(
            f"--team-{team}-base-url",
            help=f"API base URL for LLM seats on team {team}; overrides --base-url",
        )
        llm_options.add_argument(
            f"--team-{team}-api-key-env",
            help=f"API key environment variable for team {team}",
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
        "--llm-response-mode",
        choices=("strict", "compatible"),
        default="strict",
        help="Response parsing: strict envelope (default) or compatible final envelope with extra wording",
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
    for team, kind in enumerate(teams):
        if kind != "llm" and any(
            getattr(args, f"team_{team}_{setting}") is not None
            for setting in ("model", "base_url", "api_key_env")
        ):
            raise ValueError(f"Team {team} model settings require --team-{team} llm")
    if "llm" in teams:
        for team, kind in enumerate(teams):
            if kind == "llm" and not (getattr(args, f"team_{team}_model") or args.model):
                raise ValueError(
                    f"--team-{team}-model, --model, or OPENROUTER_MODEL is required for an LLM player on team {team}"
                )
    elif args.llm_log or args.llm_strict or args.reasoning or args.llm_response_mode != "strict":
        raise ValueError(
            "--llm-log, --llm-strict, --reasoning, and --llm-response-mode compatible require an llm policy"
        )
    if (
        args.llm_log
        and args.trace
        and Path(args.llm_log).resolve() == Path(args.trace).resolve()
    ):
        raise ValueError("--llm-log and --trace must use different files")


def _build_provider(
    args: argparse.Namespace,
    *,
    model: str,
    base_url: str,
    key_env: str | None,
) -> ChatCompletionProvider:
    """Resolve CLI configuration and credentials at the application boundary."""
    if key_env is None and base_url.rstrip("/") == OPENROUTER_BASE_URL:
        key_env = "OPENROUTER_API_KEY"
    api_key = os.environ.get(key_env, "") if key_env else ""
    if key_env and not api_key:
        raise ValueError(f"Set {key_env} before running an LLM player")
    return ChatCompletionProvider(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        include_reasoning=args.reasoning,
    )


def _build_spectator(
    args: argparse.Namespace,
    teams: tuple[str, str],
    models: tuple[str | None, str | None],
) -> TerminalSpectator | None:
    if not (args.watch or args.step or args.delay is not None):
        return None
    print(f"Mus | seed {args.seed} | first to {args.target_score} stones", flush=True)
    print(
        f"Team 0: North + South ({teams[0]}) | Team 1: East + West ({teams[1]})",
        flush=True,
    )
    print("Spectator view: all four hands are visible.", flush=True)
    distinct_models = {model for model in models if model}
    if len(distinct_models) == 1:
        print(f"LLM model: {next(iter(distinct_models))}", flush=True)
    elif distinct_models:
        for team, model in enumerate(models):
            if model:
                print(f"Team {team} model: {model}", flush=True)
    return TerminalSpectator(
        delay=0.4 if args.delay is None else args.delay,
        step=args.step,
    )


if __name__ == "__main__":
    raise SystemExit(play_command())

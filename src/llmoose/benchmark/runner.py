"""Mirrored, fixed-opponent benchmark with durable per-game evidence."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

from llmoose.agents.factory import POLICY_NAMES
from llmoose.agents.prompt import PROMPT_VERSION, PROMPT_VERSIONS
from llmoose.agents.providers import OPENROUTER_BASE_URL, ChatCompletionProvider
from llmoose.cli import play_command
from llmoose.rules.ruleset import FEDERATION_RULESET_ID, RULESET_IDS
from llmoose.simulation.artifacts import atomic_json, current_source_digest
from llmoose.simulation.validation import recover_run, validate_run

SCHEMA = "llmoose.benchmark"
SCHEMA_VERSION = 1
_MISSING = object()
_EXAMPLE_MODEL_IDS = {"MODEL", "OTHER_MODEL", "CANDIDATE_MODEL", "OPPONENT_MODEL"}


@contextmanager
def _suite_lock(output: Path):
    with (output / ".benchmark.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process is running this benchmark") from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers") from None
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds must be nonempty and unique")
    return seeds


def _changed_fields(saved: Any, requested: Any, prefix: str = "") -> list[str]:
    """Name manifest differences without dumping large or private values."""
    if saved is _MISSING or requested is _MISSING:
        return [prefix]
    if isinstance(saved, dict) and isinstance(requested, dict):
        return [
            field
            for key in sorted(saved.keys() | requested.keys())
            for field in _changed_fields(
                saved.get(key, _MISSING), requested.get(key, _MISSING),
                f"{prefix}.{key}" if prefix else key
            )
        ]
    return [prefix] if saved != requested else []


def _manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "suite_id": args.suite_id,
        "seeds": args.seeds,
        "ruleset": args.ruleset,
        "target_score": args.target_score,
        "max_steps": args.max_steps,
        "candidate": {"kind": args.candidate, "model": args.model if args.candidate == "llm" else None},
        "opponent": {
            "kind": args.opponent,
            "model": args.opponent_model if args.opponent == "llm" else None,
        },
        "model_settings": {
            "base_url": args.base_url if args.candidate == "llm" else None,
            "prompt_version": args.prompt_version if "llm" in (args.candidate, args.opponent) else None,
            "temperature": args.temperature if "llm" in (args.candidate, args.opponent) else None,
            "max_tokens": args.max_tokens if "llm" in (args.candidate, args.opponent) else None,
            "timeout": args.timeout if "llm" in (args.candidate, args.opponent) else None,
            "strict": not args.assisted if "llm" in (args.candidate, args.opponent) else None,
            "response_mode": args.llm_response_mode if "llm" in (args.candidate, args.opponent) else None,
        },
        "opponent_base_url": args.opponent_base_url if args.opponent == "llm" else None,
        "pairing": "candidate plays team 0 and team 1 for every seed",
        "initial_dealer": "NORTH",
        "primary_metric": "candidate win fraction over complete seed pairs",
        "incomplete_pair_policy": "exclude from primary metric; retain in coverage and job rows",
        "source_sha256": current_source_digest(),
    }


def _job_path(output: Path, seed: int, candidate_team: int) -> Path:
    return output / "jobs" / f"seed-{seed}-candidate-team-{candidate_team}"


def _play_args(args: argparse.Namespace, seed: int, team: int, path: Path) -> list[str]:
    names = [args.opponent, args.opponent]
    names[team] = args.candidate
    command = [
        "--seed", str(seed), "--ruleset", args.ruleset,
        "--target-score", str(args.target_score), "--max-steps", str(args.max_steps),
        "--team-0", names[0], "--team-1", names[1],
        "--run-dir", str(path),
    ]
    if "llm" in (args.candidate, args.opponent):
        command += [
            "--prompt-version", str(args.prompt_version),
            "--llm-response-mode", args.llm_response_mode,
            "--temperature", str(args.temperature),
            "--max-tokens", str(args.max_tokens), "--timeout", str(args.timeout),
        ]
        if not args.assisted:
            command.append("--llm-strict")
        for assigned_team, kind in enumerate(names):
            if kind != "llm":
                continue
            candidate_side = assigned_team == team
            command += [
                f"--team-{assigned_team}-model",
                args.model if candidate_side else args.opponent_model,
                f"--team-{assigned_team}-base-url",
                args.base_url if candidate_side else args.opponent_base_url,
            ]
            key_env = args.api_key_env if candidate_side else args.opponent_api_key_env
            if key_env:
                command += [f"--team-{assigned_team}-api-key-env", key_env]
    return command


def _play(command: list[str]) -> str | None:
    # The CLI finalizes failed matches before exiting. Its exit status says
    # nothing about artifact integrity: let the caller validate those artifacts.
    try:
        code = play_command(command)
    except SystemExit as exc:
        code = exc.code
    if code == 130:
        raise KeyboardInterrupt
    if code not in (None, 0):
        return f"Match command exited with status {code}"
    return None


def _job_result(args: argparse.Namespace, seed: int, team: int, output: Path) -> dict[str, Any]:
    path = _job_path(output, seed, team)
    error = None
    try:
        if path.exists():
            verdict = validate_run(path, allow_partial=True)
            if verdict["status"] == "needs_finalization":
                recover_run(path)
            elif verdict["status"] == "partial":
                resume = ["--resume-run", str(path)]
                for assigned_team in (0, 1):
                    candidate_side = assigned_team == team
                    key_env = args.api_key_env if candidate_side else args.opponent_api_key_env
                    if key_env and (args.candidate if candidate_side else args.opponent) == "llm":
                        resume += [f"--team-{assigned_team}-api-key-env", key_env]
                error = _play(resume)
        else:
            error = _play(_play_args(args, seed, team, path))
        verdict = validate_run(path, allow_partial=True)
        validation_status = verdict["status"]
    except (OSError, ValueError, KeyError, TypeError, SystemExit) as exc:
        validation_status = "invalid"
        error = str(exc)
    result_path = path / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    report_path = path / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    complete = validation_status == "finalized" and result.get("status") == "completed" and result.get("winner") in (0, 1)
    scores = result.get("scores")
    game_teams = report.get("game", {}).get("teams", [])
    llm_teams = report.get("llm", {}).get("by_team", {})
    return {
        "job_id": f"seed-{seed}-candidate-team-{team}",
        "seed": seed,
        "candidate_team": team,
        "run_dir": str(path.relative_to(output)),
        "run_id": result.get("run_id"),
        "validation_status": validation_status,
        "status": result.get("status", "partial"),
        "error": (error or result.get("error")) if validation_status == "invalid"
        else (result.get("error") or error),
        "complete": complete,
        "candidate_won": result["winner"] == team if complete else None,
        "candidate_stone_margin": scores[team] - scores[1 - team] if complete else None,
        "actions": result.get("actions"),
        "candidate_game": game_teams[team] if len(game_teams) == 2 else None,
        "opponent_game": game_teams[1 - team] if len(game_teams) == 2 else None,
        "candidate_llm_metrics": llm_teams.get(str(team)),
        "opponent_llm_metrics": llm_teams.get(str(1 - team)),
        "resolved_models": report.get("llm", {}).get("resolved_models", []),
    }


def _interval(values: list[float], *, seed: int = 2026, repetitions: int = 2000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(repetitions)
    )
    return [estimates[int(0.025 * (repetitions - 1))], estimates[int(0.975 * (repetitions - 1))]]


def aggregate(manifest: dict[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate by seed, never silently treating a failed half-pair as a loss."""
    by_seed = {seed: [] for seed in manifest["seeds"]}
    for job in jobs:
        by_seed[job["seed"]].append(job)
    pairs = []
    for seed, rows in by_seed.items():
        rows = sorted(rows, key=lambda row: row["candidate_team"])
        complete = len(rows) == 2 and {row["candidate_team"] for row in rows} == {0, 1} and all(row["complete"] for row in rows)
        pairs.append({
            "seed": seed,
            "complete": complete,
            "candidate_win_fraction": sum(row["candidate_won"] for row in rows) / 2 if complete else None,
            "candidate_mean_stone_margin": sum(row["candidate_stone_margin"] for row in rows) / 2 if complete else None,
            "jobs": [row["job_id"] for row in rows],
        })
    wins = [pair["candidate_win_fraction"] for pair in pairs if pair["complete"]]
    margins = [pair["candidate_mean_stone_margin"] for pair in pairs if pair["complete"]]
    return {
        "schema": "llmoose.benchmark-report",
        "schema_version": SCHEMA_VERSION,
        "suite_id": manifest["suite_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": manifest["source_sha256"],
        "coverage": {
            "planned_games": len(manifest["seeds"]) * 2,
            "completed_games": sum(job["complete"] for job in jobs),
            "complete_pairs": len(wins),
            "planned_pairs": len(manifest["seeds"]),
        },
        "candidate_win_rate": sum(wins) / len(wins) if wins else None,
        "candidate_win_rate_95pct_bootstrap": _interval(wins),
        "confidence_method": {
            "type": "percentile_bootstrap", "cluster": "seed_pair",
            "repetitions": 2000, "seed": 2026, "confidence_level": 0.95,
        },
        "candidate_mean_stone_margin": sum(margins) / len(margins) if margins else None,
        "pairs": pairs,
        "jobs": jobs,
    }


def benchmark_command(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a mirrored fixed-opponent Mus benchmark")
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--seeds", required=True, type=_seeds)
    parser.add_argument("--candidate", required=True, choices=POLICY_NAMES)
    parser.add_argument("--opponent", required=True, choices=POLICY_NAMES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ruleset", choices=RULESET_IDS, default=FEDERATION_RULESET_ID)
    parser.add_argument("--target-score", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--model")
    parser.add_argument("--base-url", default=OPENROUTER_BASE_URL)
    parser.add_argument("--api-key-env")
    parser.add_argument("--opponent-model")
    parser.add_argument("--opponent-base-url", default=OPENROUTER_BASE_URL)
    parser.add_argument("--opponent-api-key-env")
    parser.add_argument("--prompt-version", type=int, choices=PROMPT_VERSIONS, default=PROMPT_VERSION)
    parser.add_argument(
        "--llm-response-mode", choices=("strict", "compatible"), default="strict",
        help="Response parsing: strict envelope (default) or compatible final envelope with extra wording",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--assisted", action="store_true", help="Enable retries and conservative fallback for LLM actions")
    args = parser.parse_args(argv)
    if not args.suite_id.strip() or args.target_score < 2 or args.max_steps < 1:
        parser.error("Provide a nonempty suite ID, target score >= 2, and positive max steps")
    if args.candidate == "llm" and not args.model:
        parser.error("--model is required for an LLM candidate")
    if args.opponent == "llm" and not args.opponent_model:
        parser.error("--opponent-model is required for an LLM opponent")
    for option, kind, model in (
        ("--model", args.candidate, args.model),
        ("--opponent-model", args.opponent, args.opponent_model),
    ):
        if kind == "llm" and model in _EXAMPLE_MODEL_IDS:
            parser.error(f"{option}={model} is an example placeholder; use a real provider model ID")
    if args.assisted and "llm" not in (args.candidate, args.opponent):
        parser.error("--assisted requires an LLM policy")
    if args.llm_response_mode != "strict" and "llm" not in (args.candidate, args.opponent):
        parser.error("--llm-response-mode compatible requires an LLM policy")
    if "llm" in (args.candidate, args.opponent):
        load_dotenv(Path.cwd() / ".env", override=False)
        try:
            for kind, model, base_url, requested_key_env in (
                (args.candidate, args.model, args.base_url, args.api_key_env),
                (args.opponent, args.opponent_model, args.opponent_base_url, args.opponent_api_key_env),
            ):
                if kind != "llm":
                    continue
                key_env = requested_key_env
                if key_env is None and base_url.rstrip("/") == OPENROUTER_BASE_URL:
                    key_env = "OPENROUTER_API_KEY"
                if key_env and not os.environ.get(key_env):
                    raise ValueError(f"Set {key_env} before running an LLM policy")
                ChatCompletionProvider(
                    model=model,
                    api_key=os.environ.get(key_env, "") if key_env else "",
                    base_url=base_url,
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
        except ValueError as exc:
            parser.error(str(exc))
    manifest = _manifest(args)
    path = args.output / "benchmark.json"
    if path.exists():
        differences = _changed_fields(json.loads(path.read_text()), manifest)
        if differences:
            parser.exit(
                2,
                f"{parser.prog}: Existing benchmark manifest differs in "
                + ", ".join(differences)
                + f". Inspect {path} or use a new output directory.\n",
            )
    else:
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Output directory is not empty and has no benchmark manifest")
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(path, manifest)
    jobs = []
    try:
        with _suite_lock(args.output):
            for seed in args.seeds:
                for team in (0, 1):
                    jobs.append(_job_result(args, seed, team, args.output))
                    atomic_json(args.output / "summary.json", aggregate(manifest, jobs))
    except ValueError as exc:
        parser.error(str(exc))
    report = aggregate(manifest, jobs)
    print(json.dumps({"coverage": report["coverage"], "candidate_win_rate": report["candidate_win_rate"], "confidence_interval": report["candidate_win_rate_95pct_bootstrap"]}, indent=2))
    return 0 if report["coverage"]["complete_pairs"] == report["coverage"]["planned_pairs"] else 1

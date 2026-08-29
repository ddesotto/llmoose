"""Fixed-deal evaluation and aggregate match metrics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Callable, Mapping, Optional, Tuple

from llmoose.agents.protocol import Policy
from llmoose.game.state import SEATS, Seat
from llmoose.rules.ruleset import Ruleset
from llmoose.simulation.match import MatchResult, run_match

PolicyFactory = Callable[[Ruleset, int], Mapping[Seat, Policy]]


@dataclass(frozen=True)
class DealSuite:
    """A named, ordered set of public seeds used for comparable evaluations."""

    id: str
    seeds: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("A deal suite needs an ID")
        if not self.seeds:
            raise ValueError("A deal suite needs at least one seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Deal-suite seeds must be unique")


@dataclass(frozen=True)
class BenchmarkReport:
    suite_id: str
    ruleset_id: str
    matches: Tuple[MatchResult, ...]
    team_wins: Tuple[int, int]
    mean_score_difference: float
    mean_actions: float
    lance_counts: Mapping[str, int]

    @property
    def games(self) -> int:
        return len(self.matches)

    @property
    def win_rates(self) -> Tuple[float, float]:
        return tuple(wins / self.games for wins in self.team_wins)


def run_benchmark(
    suite: DealSuite,
    policy_factory: PolicyFactory,
    ruleset: Optional[Ruleset] = None,
    max_steps: int = 10_000,
) -> BenchmarkReport:
    """Evaluate fresh policies against identical deals and aggregate outcomes."""

    active_ruleset = ruleset or Ruleset()
    matches = []
    for seed in suite.seeds:
        policies = policy_factory(active_ruleset, seed)
        matches.append(run_match(policies, seed, active_ruleset, max_steps=max_steps))
    matches = tuple(matches)
    wins = tuple(sum(match.winner == team for match in matches) for team in range(2))
    lance_counts: dict[str, int] = {}
    for match in matches:
        for event in match.final_state.events:
            if event.kind == "lance_resolved":
                lance = dict(event.detail)["lance"]
                lance_counts[lance] = lance_counts.get(lance, 0) + 1
    return BenchmarkReport(
        suite_id=suite.id,
        ruleset_id=active_ruleset.id,
        matches=matches,
        team_wins=wins,
        mean_score_difference=mean(
            match.scores[0] - match.scores[1] for match in matches
        ),
        mean_actions=mean(len(match.actions) for match in matches),
        lance_counts=lance_counts,
    )


def baseline_factory(policy_type: type[Policy]) -> PolicyFactory:
    """Build a convenience factory for stateless codec-based baseline policies."""

    def factory(ruleset: Ruleset, seed: int) -> Mapping[Seat, Policy]:
        from llmoose.game.actions import ActionCodec

        codec = ActionCodec(ruleset.target_score)
        return {seat: policy_type(codec) for seat in SEATS}

    return factory

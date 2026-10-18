import pytest

from llmoose.agents.baselines import RandomPolicy
from llmoose.agents.factory import build_policies
from llmoose.agents.providers import Completion
from llmoose.game.actions import ActionCodec, ActionKind
from llmoose.game.environment import MusEnv
from llmoose.game.state import SEATS, Seat
from llmoose.rules.ruleset import Ruleset
from llmoose.simulation.match import run_match


def test_factory_preserves_per_seat_random_streams():
    ruleset = Ruleset(target_score=8)
    codec = ActionCodec(8)
    expected = {seat: RandomPolicy(codec, 12 + int(seat)) for seat in SEATS}
    built = build_policies({seat: "random" for seat in SEATS}, codec, seed=12)
    assert run_match(built, 12, ruleset) == run_match(expected, 12, ruleset)


def test_individual_seat_can_use_a_model_without_cli_or_environment_variables():
    env = MusEnv(Ruleset(target_score=8))
    observation = env.reset(seed=12)
    records = []
    calls = []

    def provider(messages):
        calls.append(messages)
        return Completion('{"action_id": 1}' )

    names = {seat: "heuristic" for seat in SEATS}
    names[Seat.WEST] = "llm"
    policies = build_policies(
        names,
        env.codec,
        seed=12,
        provider=provider,
        strict=True,
        on_record=records.append,
    )
    assert len({id(policy) for policy in policies.values()}) == 4
    assert policies[Seat.WEST].act(observation).kind is ActionKind.NO_MUS
    assert len(calls) == 1
    assert records[0]["status"] == "chosen"
    assert policies[Seat.WEST].max_retries == 0
    assert not policies[Seat.WEST].fallback_on_invalid


def test_model_policy_requires_an_explicit_provider():
    with pytest.raises(ValueError, match="require a provider"):
        build_policies({seat: "llm" for seat in SEATS}, ActionCodec(), seed=0)


def test_model_seats_can_have_distinct_providers():
    providers = {
        seat: (lambda messages, seat=seat: Completion('{"action_id": 1}' , model=seat.name))
        for seat in SEATS
    }
    policies = build_policies(
        {seat: "llm" for seat in SEATS}, ActionCodec(),
        seed=0, provider=providers,
    )
    assert all(policies[seat].provider is providers[seat] for seat in SEATS)
    with pytest.raises(ValueError, match="Every LLM seat"):
        build_policies(
            {seat: "llm" for seat in SEATS}, ActionCodec(),
            seed=0, provider={Seat.NORTH: providers[Seat.NORTH]},
        )

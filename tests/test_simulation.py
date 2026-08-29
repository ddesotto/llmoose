from llmoose.agents.baselines import ConservativePolicy, RandomPolicy
from llmoose.game.actions import ActionCodec
from llmoose.game.state import SEATS
from llmoose.rules.ruleset import Ruleset
from llmoose.simulation.match import replay_match, run_match


def test_conservative_agents_finish_and_replay_a_match() -> None:
    ruleset = Ruleset(target_score=4)
    codec = ActionCodec(ruleset.target_score)
    policies = {seat: ConservativePolicy(codec) for seat in SEATS}

    result = run_match(policies, seed=12, ruleset=ruleset)

    assert result.scores[result.winner] >= ruleset.target_score
    assert replay_match(result.actions, seed=12, ruleset=ruleset) == result.final_state


def test_seeded_random_agents_produce_the_same_trace() -> None:
    ruleset = Ruleset(target_score=4)
    codec = ActionCodec(ruleset.target_score)

    def policies():
        return {seat: RandomPolicy(codec, seed=20 + int(seat)) for seat in SEATS}

    first = run_match(policies(), seed=8, ruleset=ruleset)
    second = run_match(policies(), seed=8, ruleset=ruleset)
    assert first.actions == second.actions
    assert first.scores == second.scores

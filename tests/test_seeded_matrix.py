"""Regression matrix for complete deterministic matches and replays."""

import pytest

from llmoose.agents.baselines import ConservativePolicy, HeuristicPolicy, RandomPolicy
from llmoose.game.actions import ActionCodec
from llmoose.game.state import SEATS
from llmoose.rules.ruleset import RULESET_IDS, Ruleset
from llmoose.simulation.match import replay_match, run_match


@pytest.mark.parametrize("policy_name", ("conservative", "heuristic", "random"))
@pytest.mark.parametrize("ruleset_id", RULESET_IDS)
def test_seeded_match_matrix_replays_exactly(policy_name: str, ruleset_id: str) -> None:
    """Exercise deal, Mus, bidding, scoring, redeal, and terminal paths repeatedly."""

    ruleset = Ruleset(id=ruleset_id, target_score=8)
    codec = ActionCodec(ruleset.target_score)
    for seed in range(100):
        if policy_name == "random":
            policies = {
                seat: RandomPolicy(codec, seed * 13 + int(seat)) for seat in SEATS
            }
        else:
            policy_type = (
                ConservativePolicy if policy_name == "conservative" else HeuristicPolicy
            )
            policies = {seat: policy_type(codec) for seat in SEATS}
        result = run_match(policies, seed, ruleset)

        assert result.scores[result.winner] >= ruleset.target_score
        assert replay_match(result.actions, seed, ruleset) == result.final_state

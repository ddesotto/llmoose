from dataclasses import replace

import pytest

from llmoose.agents.baselines import ConservativePolicy
from llmoose.agents.decision import ActionScorePolicy
from llmoose.agents.llm import InvalidModelAction, StructuredLLMPolicy
from llmoose.agents.prompt import render_messages, renderer_version_for_prompt
from llmoose.agents.providers import Completion
from llmoose.game.actions import Action, ActionCodec, ActionKind
from llmoose.game.engine import apply_action
from llmoose.game.state import Phase, Seat, freeze_hands, new_game
from llmoose.observations.core import observe
from llmoose.observations.encoding import policy_input
from llmoose.rules.ruleset import Ruleset


def test_hidden_state_is_irrelevant_through_declarations_bids_and_showdown():
    codec = ActionCodec(40)
    state = new_game(12, Ruleset(target_score=40))
    policy = ConservativePolicy(codec)
    phases = set()
    for _ in range(100):
        if state.phase is Phase.COMPLETE:
            break
        actor = state.current_seat
        phases.add(state.phase)
        others = [s for s in Seat if s != actor]
        hands = dict(state.hands)
        for i, seat in enumerate(others):
            hands[seat] = state.hands[others[(i + 1) % 3]]
        recall = list(state.own_discard_history)
        recall[int(others[0])] = (state.hands[others[0]],)
        alternate = replace(
            state, hands=freeze_hands(hands), seed=900,
            stock=tuple(reversed(state.stock)), discards=tuple(reversed(state.discards)),
            own_discard_history=tuple(recall),
        )
        original = observe(state, actor, codec)
        changed = observe(alternate, actor, codec)
        assert original == changed
        assert policy_input(original) == policy_input(changed)
        assert render_messages(original) == render_messages(changed)
        state = apply_action(state, policy.act(original))
    assert {Phase.DECLARE_PARES, Phase.DECLARE_JUEGO, Phase.GRANDE} <= phases
    assert state.hand_number > 0


def test_public_reveals_remain_available_to_llms_and_numeric_policies():
    codec = ActionCodec(40)
    state = new_game(12, Ruleset(target_score=40))
    policy = ConservativePolicy(codec)
    while state.hand_number == 0:
        state = apply_action(state, policy.act(observe(state, state.current_seat, codec)))
    observation = observe(state, state.current_seat, codec)
    packet = policy_input(observation)
    assert packet['schema_version'] == 1
    assert any(e['kind'] == 'cards_revealed' for e in packet['observation']['public_events'])
    assert 'cards_revealed' in render_messages(observation)[1]['content']
    assert 'cards_revealed' not in render_messages(observation, prompt_version=4)[1]['content']
    assert [renderer_version_for_prompt(i) for i in range(1, 6)] == [2, 2, 2, 3, 4]
    with pytest.raises(TypeError):
        policy_input(state)


def test_scorer_masks_illegal_ids_and_receives_only_observation():
    codec = ActionCodec(8)
    observation = observe(new_game(12, Ruleset(target_score=8)), Seat.WEST, codec)
    packets = []

    def scorer(packet):
        packets.append(packet)
        return list(range(codec.size))  # Highest overall ID is illegal here.

    policy = ActionScorePolicy(codec, scorer)
    assert policy.act(observation) == Action(ActionKind.NO_MUS)
    assert packets == [policy_input(observation)]
    for scores in ([0], [float('nan')] * codec.size):
        with pytest.raises(ValueError, match='finite score'):
            ActionScorePolicy(codec, lambda _: scores).act(observation)
    with pytest.raises(ValueError, match='codec'):
        ConservativePolicy(ActionCodec(40)).act(observation)


@pytest.mark.parametrize('kwargs', [
    {'kind': 'mus'}, {'kind': ActionKind.BET, 'amount': 2.0},
    {'kind': ActionKind.DISCARD, 'card_indices': (True,)},
    {'kind': ActionKind.DISCARD, 'card_indices': (1.0,)},
])
def test_action_objects_reject_python_equality_type_aliases(kwargs):
    with pytest.raises(ValueError):
        Action(**kwargs)


def test_invented_game_continuation_is_not_accepted_even_with_a_legal_answer():
    codec = ActionCodec(8)
    state = new_game(13, Ruleset(target_score=8))
    observation = observe(state, Seat.WEST, codec)
    response = ('EAST voted mus. SOUTH voted mus. Phase: exchange. '
                'Legal actions: 0 discard_count=1. <answer>0</answer>')
    policy = StructuredLLMPolicy(codec, lambda _: Completion(response),
                                 max_retries=0, fallback_on_invalid=False)
    with pytest.raises(InvalidModelAction):
        policy.act(observation)
    assert state.phase is Phase.MUS_VOTE
    assert len(state.events) == 1
    text = render_messages(observation)[1]['content']
    assert 'not a transcript to continue' in text
    assert 'No outstanding bid' not in text

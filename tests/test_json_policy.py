from dataclasses import replace

import pytest

from llmoose.agents.llm import (
    InvalidModelAction,
    StructuredLLMPolicy,
    action_response_format,
    parse_json_action,
)
from llmoose.agents.prompt import render_messages
from llmoose.agents.providers import Completion
from llmoose.game.environment import MusEnv
from llmoose.rules.ruleset import Ruleset


@pytest.mark.parametrize('text', [
    '{"action_id": true}', '{"action_id": 1.0}', '{"action_id": "1"}',
    '{"action_id": 99}', '{"action_id": 1, "action_id": 0}',
    '{"action_id": 1, "reason": "invented event"}', '[1]', 'null',
    'Here is the action: {"action_id": 1}', '```json\n{"action_id": 1}\n```',
    '{"action_id": 1}{"action_id": 0}', '<answer>1</answer>',
])
def test_json_action_rejects_noncontract_responses(text):
    with pytest.raises(ValueError):
        parse_json_action(text, (0, 1))


def test_json_action_accepts_only_legal_id_and_schema_matches():
    assert parse_json_action(' {"action_id": 13}\n', (10, 13)) == 13
    schema = action_response_format((10, 13))['json_schema']['schema']
    assert schema['properties']['action_id']['enum'] == [10, 13]
    assert schema['required'] == ['action_id']
    assert schema['additionalProperties'] is False


def test_json_retry_discards_invented_assistant_state():
    env = MusEnv(Ruleset(target_score=8))
    observation = env.reset(12)
    calls = []

    def provider(messages):
        calls.append([dict(m) for m in messages])
        return Completion('Invented future hand <answer>1</answer>' if len(calls) == 1
                          else '{"action_id": 1}')

    policy = StructuredLLMPolicy(env.codec, provider, prompt_version=5)
    assert env.codec.encode(policy.act(observation)) == 1
    assert len(calls) == 2
    assert len(calls[1]) == 2
    assert 'Invented future hand' not in str(calls[1])
    assert 'previous response was rejected' in calls[1][-1]['content']
    assert calls[0] == render_messages(observation, prompt_version=5)


def test_json_strict_policy_has_no_fallback_and_seats_have_fresh_context():
    env = MusEnv(Ruleset(target_score=8))
    observation = env.reset(12)
    policy = StructuredLLMPolicy(env.codec, lambda _: Completion('prose'),
                                 prompt_version=5, max_retries=0,
                                 fallback_on_invalid=False)
    with pytest.raises(InvalidModelAction):
        policy.act(observation)
    assert policy.telemetry.fallbacks == 0
    assert env.state.phase == observation.phase


def test_v5_keeps_private_boundary_and_exact_legal_menu():
    env = MusEnv(Ruleset(target_score=8))
    observation = env.reset(12)
    messages = render_messages(observation, prompt_version=5)
    assert '<answer>' not in str(messages)
    assert '"action_id": ID' in str(messages)
    assert 'first eligible seat' in messages[0]['content']
    assert render_messages(replace(observation), prompt_version=4)[0] != messages[0]


def test_shared_provider_receives_fresh_private_context_for_each_seat():
    env = MusEnv(Ruleset(target_score=8))
    west = env.reset(12)
    calls = []

    def provider(messages):
        calls.append([dict(message) for message in messages])
        return Completion('{"action_id": 0}')

    policy = StructuredLLMPolicy(env.codec, provider, prompt_version=5)
    north = env.step(policy.act(west)).observation
    policy.act(north)
    assert len(calls) == 2 and all(len(messages) == 2 for messages in calls)
    assert calls[0] == render_messages(west, prompt_version=5)
    assert calls[1] == render_messages(north, prompt_version=5)
    for card in west.private_hand:
        assert f'{int(card.rank)} of {card.suit.value}' not in calls[1][1]['content']


@pytest.mark.parametrize('text', [
    'I choose to stop mus. {"action_id": 1}',
    'The score is 8 and my estimate is 0.7.\n```json\n{"action_id": 1}\n```',
])
def test_compatible_mode_preserves_prose_but_marks_normalization(text):
    env = MusEnv(Ruleset(target_score=8))
    observation = env.reset(12)
    records = []
    policy = StructuredLLMPolicy(env.codec, lambda _: Completion(text),
                                 response_mode='compatible', on_record=records.append,
                                 max_retries=0, fallback_on_invalid=False)
    assert env.codec.encode(policy.act(observation)) == 1
    assert records[0]['response'] == text
    assert records[0]['response_normalized'] is True
    assert records[0]['response_mode'] == 'compatible'
    with pytest.raises(InvalidModelAction):
        StructuredLLMPolicy(env.codec, lambda _: Completion(text), max_retries=0,
                            fallback_on_invalid=False).act(observation)


@pytest.mark.parametrize('text', [
    'I choose 1', '{"action_id": 0} or {"action_id": 1}',
    '<answer>0</answer> {"action_id": 1}', '{"action_id": 1} then act again',
    '{"action_id": 1, "action_id": 0}',
    'I choose {"action_id": true}', 'I choose {"action_id": 99}',
    'I choose {"action_id": 1, "extra": "NORTH has kings"}',
])
def test_compatible_mode_rejects_ambiguous_illegal_or_nonfinal_answers(text):
    from llmoose.agents.llm import parse_response

    with pytest.raises(ValueError):
        parse_response(text, (0, 1), prompt_version=5, response_mode='compatible')


def test_native_reasoning_is_separate_and_not_counted_as_normalization():
    env = MusEnv(Ruleset(target_score=8))
    observation = env.reset(12)
    records = []
    policy = StructuredLLMPolicy(
        env.codec,
        lambda _: Completion('{"action_id": 1}', reasoning='A private deliberation.',
                             reasoning_details=({'type': 'reasoning.summary', 'summary': 'Cut mus'},)),
        on_record=records.append,
    )
    policy.act(observation)
    assert records[0]['reasoning'] == 'A private deliberation.'
    assert records[0]['reasoning_details'][0]['summary'] == 'Cut mus'
    assert records[0]['response_normalized'] is False

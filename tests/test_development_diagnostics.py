import io
import json
import zipfile

import pytest

from llmoose.agents.baselines import ConservativePolicy
from llmoose.agents.providers import Completion, ProviderError
from llmoose.benchmark.contract import development_manifest, fingerprint, split_for_seed
from llmoose.benchmark.diagnostic_checks import check_suite
from llmoose.benchmark.diagnostics import (
    DiagnosticAnswer,
    diagnostic_command,
    evaluate,
    llm_responder,
    load_suite,
    parse_choice,
    public_case,
)
from llmoose.game.environment import MusEnv
from llmoose.game.state import SEATS
from llmoose.rules.ruleset import Ruleset
from llmoose.simulation.artifacts import source_archive_bytes
from llmoose.training.trajectories import collect_episode


def test_authored_labels_and_frozen_contract():
    result = check_suite()
    assert result['checked'] == result['passed'] == 46
    assert result['failures'] == []
    manifest = development_manifest()
    assert [len(manifest['deal_groups'][name]) for name in ('train', 'development', 'test')] == [200, 20, 50]
    assert [split_for_seed(seed) for seed in (0, 1000, 2000)] == ['train', 'development', 'test']
    for seed in (True, -1, 999):
        with pytest.raises(ValueError):
            split_for_seed(seed)


def test_wrong_authored_label_is_detected(monkeypatch):
    suite = load_suite()
    case = suite['cases'][0]
    case['expected'] = next(v for v in case['choices'] if v != case['expected'])
    monkeypatch.setattr('llmoose.benchmark.diagnostic_checks.load_data', lambda name: suite)
    assert check_suite()['failures']


@pytest.mark.parametrize('order_seed', [0, 19])
def test_callback_only_sees_questions_and_permuted_choices(order_seed):
    cases = {case['id']: case for case in load_suite()['cases']}
    def oracle_for_test_only(packet):
        assert set(packet) == {'id', 'question', 'facts', 'choices'}
        expected = cases[packet['id']]['expected']
        answer = next(c['id'] for c in packet['choices'] if fingerprint(c['value']) == fingerprint(expected))
        packet['choices'].clear()  # A buggy adapter cannot alter the grading key.
        return answer
    report = evaluate(oracle_for_test_only, order_seed=order_seed)
    assert report['accuracy'] == 1
    assert report['statuses'] == {'valid': 46}


@pytest.mark.parametrize('text', ['{"choice_id":true}', '{"choice_id":1.0}', '{"choice_id":0,"choice_id":1}', 'reasoning {"choice_id":0}', '{"choice_id":0,"extra":1}'])
def test_strict_answer_parser(text):
    with pytest.raises(ValueError):
        parse_choice(text)


def test_missing_invalid_and_provider_failures_keep_denominator():
    assert evaluate(lambda packet: None)['statuses'] == {'unanswered': 46}
    assert evaluate(lambda packet: DiagnosticAnswer(None))['statuses'] == {'invalid': 46}
    calls = []
    def failed(packet):
        calls.append(packet)
        raise ProviderError('offline')
    report = evaluate(failed)
    assert len(calls) == 1
    assert report['scheduled'] == 46 and report['accuracy'] == 0
    assert report['statuses'] == {'provider_error': 1, 'unanswered': 45}


def test_transport_evidence_and_native_reasoning():
    class Provider:
        def request_payload(self, messages, **kwargs):
            return {'messages': messages, **kwargs}
        def __call__(self, messages, **kwargs):
            assert kwargs['response_format']['json_schema']['strict']
            return Completion('{"choice_id":0}', reasoning='private reasoning channel')
    packet = public_case(load_suite()['cases'][0])
    result = llm_responder(Provider())(packet)
    assert result.choice_id == 0
    assert result.evidence['completion']['reasoning'] == 'private reasoning channel'
    assert json.loads(result.evidence['messages'][1]['content']) == packet


def test_offline_cli_export_grade_and_archive(tmp_path):
    export = tmp_path / 'inputs.json'
    assert diagnostic_command(['--export', str(export)]) == 0
    data = json.loads(export.read_text())
    assert all('expected' not in case and 'explanation' not in case for case in data['cases'])
    predictions = tmp_path / 'predictions.json'
    predictions.write_text(json.dumps({'suite_sha256': data['suite_sha256'], 'order_seed': 0, 'answers': {}}))
    output = tmp_path / 'evaluation'
    assert diagnostic_command(['--predictions', str(predictions), '--output', str(output)]) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['statuses'] == {'unanswered': 46}
    assert len((output / 'records.jsonl').read_text().splitlines()) == 46
    assert 'source.zip' in json.loads((output / 'checksums.json').read_text())['sha256']
    with zipfile.ZipFile(io.BytesIO(source_archive_bytes())) as archive:
        assert 'src/llmoose/benchmark/data/diagnostics_v1.json' in archive.namelist()


def test_limit_reset_and_invalid_action_budget():
    env = MusEnv(max_steps=1)
    observation = env.reset()
    with pytest.raises(ValueError):
        env.step(-1)
    result = env.step(ConservativePolicy(env.codec).act(observation))
    assert result.truncated and not result.terminated
    assert result.observation is not None and result.info['termination_reason'] == 'step_limit'
    with pytest.raises(RuntimeError, match='reset'):
        env.step(0)
    assert env.reset() == observation
    assert env.step(ConservativePolicy(env.codec).act(observation)).truncated


@pytest.mark.parametrize('episode,target', [('hand', 40), ('match', 2)])
def test_natural_completion_takes_priority_over_limit(episode, target):
    env = MusEnv(Ruleset(target_score=target), episode=episode)
    policy = ConservativePolicy(env.codec)
    observation = env.reset()
    steps = 0
    while True:
        steps += 1
        result = env.step(policy.act(observation))
        if result.terminated:
            break
        observation = result.observation
    env.max_steps = steps
    observation = env.reset()
    for _ in range(steps):
        result = env.step(policy.act(observation))
        observation = result.observation
    assert result.terminated and not result.truncated and observation is None


def test_training_does_not_label_incomplete_hand():
    env = MusEnv(episode='hand', max_steps=1)
    policies = {seat: ConservativePolicy(env.codec) for seat in SEATS}
    with pytest.raises(RuntimeError, match='truncated'):
        collect_episode(env, policies)


def test_opponent_drift_is_rejected(monkeypatch):
    from llmoose.benchmark import contract
    original = contract.load_data
    def changed_manifest(name):
        data = original(name)
        if name == 'development_v1.json':
            data['opponents_sha256'] = 'changed'
        return data
    monkeypatch.setattr(contract, 'load_data', changed_manifest)
    with pytest.raises(ValueError, match='opponent'):
        contract.development_manifest()


def test_interruption_retains_request_and_stops_remaining_calls():
    class Provider:
        def request_payload(self, messages, **kwargs):
            return {'messages': messages, **kwargs}
        def __call__(self, messages, **kwargs):
            raise KeyboardInterrupt
    rows = []
    report = evaluate(llm_responder(Provider()), on_record=rows.append)
    assert report['statuses'] == {'interrupted': 1, 'unanswered': 45}
    assert rows[0]['evidence']['request_body']['messages'] == rows[0]['evidence']['messages']

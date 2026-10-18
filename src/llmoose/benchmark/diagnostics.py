"""Rules-assisted diagnostics, separate from strategic gameplay rankings.

Only public_case() enters a model callback. Expected labels and explanations stay
in the evaluator. The callable interface also accepts non-language-model policies.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Callable

from dotenv import load_dotenv

from llmoose.agents.providers import (
    OPENROUTER_BASE_URL,
    ChatCompletionProvider,
    ProviderError,
)
from llmoose.benchmark.contract import (
    canonical,
    development_manifest,
    fingerprint,
    load_data,
)
from llmoose.benchmark.diagnostic_checks import check_suite
from llmoose.simulation.artifacts import atomic_json, source_snapshot, write_checksums


def rules_text():
    return files('llmoose.benchmark').joinpath('data', 'diagnostic_rules_v1.txt').read_text().strip()


def load_suite():
    suite = load_data('diagnostics_v1.json')
    ids = set()
    for case in suite['cases']:
        if case['id'] in ids:
            raise ValueError('Duplicate diagnostic case ID')
        ids.add(case['id'])
        choices = [canonical(value) for value in case['choices']]
        if len(choices) != len(set(choices)) or choices.count(canonical(case['expected'])) != 1:
            raise ValueError('Case must have distinct choices and exactly one expected answer')
    return suite


def public_case(case, order_seed=0):
    choices = list(case['choices'])
    Random(f"diagnostic-choice-order-v1:{order_seed}:{case['id']}").shuffle(choices)
    # A detached plain object: callbacks cannot mutate the answer key or fixtures.
    return json.loads(canonical({
        'id': case['id'], 'question': case['question'], 'facts': case['facts'],
        'choices': [{'id': index, 'value': value} for index, value in enumerate(choices)],
    }))


def messages_for(packet):
    return [
        {'role': 'system', 'content': rules_text() + '\n\nAnswer only {"choice_id": ID}, selecting one supplied choice. Do not play a move.'},
        {'role': 'user', 'content': canonical(packet)},
    ]


def response_format(packet):
    return {'type': 'json_schema', 'json_schema': {
        'name': 'mus_diagnostic_answer', 'strict': True, 'schema': {
            'type': 'object', 'properties': {'choice_id': {
                'type': 'integer', 'enum': [choice['id'] for choice in packet['choices']],
            }}, 'required': ['choice_id'], 'additionalProperties': False,
        },
    }}


def parse_choice(text):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('Duplicate JSON key')
            obj[key] = value
        return obj
    try:
        value = json.loads(text, object_pairs_hook=unique)
    except (ValueError, RecursionError):
        raise ValueError('Expected exactly {"choice_id": integer}') from None
    if not isinstance(value, dict) or set(value) != {'choice_id'} or type(value['choice_id']) is not int:
        raise ValueError('Expected exactly one integer choice_id')
    return value['choice_id']


@dataclass
class DiagnosticAnswer:
    choice_id: object
    evidence: dict = field(default_factory=dict)


def llm_responder(provider):
    def respond(packet):
        messages = messages_for(packet)
        schema = response_format(packet)
        evidence = {
            'messages': messages, 'request_body': provider.request_payload(messages, response_format=schema),
        }
        try:
            completion = provider(messages, response_format=schema)
        except (ProviderError, KeyboardInterrupt) as exc:
            exc.evidence = evidence
            raise
        evidence['completion'] = asdict(completion)
        try:
            choice = parse_choice(completion.text)
        except ValueError as exc:
            evidence['parse_error'] = str(exc)
            choice = None
        return DiagnosticAnswer(choice, evidence)
    return respond


def evaluate(responder: Callable[[dict], object], *, order_seed=0, on_record=None):
    """Score every scheduled probe. Malformed/missing outputs never disappear.

    Provider failures stop new requests; remaining cases are explicitly unanswered.
    Unexpected programming errors propagate rather than becoming model mistakes.
    """
    development_manifest()
    suite = load_suite()
    records = []
    stopped = False
    for case in suite['cases']:
        packet = public_case(case, order_seed)
        started = perf_counter()
        result = DiagnosticAnswer(None)
        status = 'unanswered'
        if not stopped:
            try:
                returned = responder(packet)
                result = returned if isinstance(returned, DiagnosticAnswer) else DiagnosticAnswer(returned)
                status = ('unanswered' if returned is None else 'valid'
                          if type(result.choice_id) is int and 0 <= result.choice_id < len(case['choices']) else 'invalid')
            except ProviderError as exc:
                status, stopped = 'provider_error', True
                result.evidence.update(getattr(exc, 'evidence', {}))
                result.evidence['error'] = str(exc)
            except KeyboardInterrupt as exc:
                status, stopped = 'interrupted', True
                result.evidence.update(getattr(exc, 'evidence', {}))
        # Rebuild after the callback: even an accidental input mutation cannot
        # change what a submitted choice means or corrupt subsequent evaluation.
        packet = public_case(case, order_seed)
        correct = status == 'valid' and canonical(packet['choices'][result.choice_id]['value']) == canonical(case['expected'])
        row = {'case_id': case['id'], 'category': case['category'], 'input': packet,
               'choice_id': result.choice_id, 'status': status, 'correct': correct,
               'latency_seconds': perf_counter() - started, 'evidence': result.evidence}
        records.append(row)
        if on_record:
            on_record(row)
    counts = Counter(row['status'] for row in records)
    by_category = {}
    for category in sorted({case['category'] for case in suite['cases']}):
        group = [row for row in records if row['category'] == category]
        by_category[category] = {'scheduled': len(group), 'valid': sum(row['status'] == 'valid' for row in group),
                                 'correct': sum(row['correct'] for row in group),
                                 'accuracy': sum(row['correct'] for row in group) / len(group)}
    return {'suite_id': suite['suite_id'], 'suite_sha256': fingerprint(suite),
            'rules_sha256': fingerprint(rules_text()), 'order_seed': order_seed,
            'rules_assisted': True, 'split': 'development_public',
            'scheduled': len(records), 'attempted': len(records) - counts['unanswered'],
            'statuses': dict(counts), 'correct': sum(row['correct'] for row in records),
            'accuracy': sum(row['correct'] for row in records) / len(records),
            'by_category': by_category}


def diagnostic_command(argv=None):
    parser = argparse.ArgumentParser(description='Mus rules-assisted development diagnostics; no gameplay ranking')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='Offline authored-label/engine and contract checks')
    mode.add_argument('--export', type=Path, help='Export unlabeled questions for any model adapter')
    mode.add_argument('--manifest', action='store_true', help='Show frozen opponents and deal partitions')
    mode.add_argument('--predictions', type=Path, help='Grade exported choice IDs without inference')
    mode.add_argument('--model', help='Evaluate a model through a chat-completions endpoint (makes requests)')
    parser.add_argument('--output', type=Path, help='New directory for evaluation evidence')
    parser.add_argument('--order-seed', type=int, default=0, help='Deterministic choice-order permutation')
    parser.add_argument('--base-url', default=OPENROUTER_BASE_URL)
    parser.add_argument('--api-key-env')
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--reasoning', action='store_true')
    args = parser.parse_args(argv)
    check = check_suite()
    manifest = development_manifest()
    if check['failures']:
        print(json.dumps(check, indent=2))
        return 1
    if args.check or args.manifest:
        print(json.dumps(check if args.check else manifest, indent=2))
        return 0
    suite = load_suite()
    if args.export:
        with args.export.open('x') as stream:
            json.dump({'suite_id': suite['suite_id'], 'suite_sha256': fingerprint(suite),
                       'order_seed': args.order_seed, 'rules': rules_text(),
                       'cases': [public_case(case, args.order_seed) for case in suite['cases']]}, stream, indent=2)
        return 0
    if args.output is None:
        parser.error('--output is required for evaluation')
    if args.predictions:
        predictions = json.loads(args.predictions.read_text())
        if predictions.get('suite_sha256') != fingerprint(suite) or predictions.get('order_seed') != args.order_seed:
            parser.error('Predictions must match exported suite_sha256 and order_seed')
        answers = predictions['answers']
        if not isinstance(answers, dict) or set(answers) - {case['id'] for case in suite['cases']}:
            parser.error('Unknown case IDs or invalid answers mapping')
        def responder(packet):
            return answers.get(packet['id'])
        identity = {'type': 'external_predictions', 'file_sha256': fingerprint(predictions)}
    else:
        load_dotenv(Path.cwd() / '.env', override=False)
        key_env = args.api_key_env or ('OPENROUTER_API_KEY' if args.base_url.rstrip('/') == OPENROUTER_BASE_URL else None)
        provider = ChatCompletionProvider(args.model, api_key=os.environ.get(key_env, '') if key_env else '',
                                          base_url=args.base_url, max_tokens=args.max_tokens,
                                          timeout=args.timeout, include_reasoning=args.reasoning)
        responder = llm_responder(provider)
        identity = {'type': 'llm', 'model': args.model, 'base_url': args.base_url,
                    'max_tokens': args.max_tokens, 'timeout': args.timeout, 'temperature': 0,
                    'reasoning': args.reasoning, 'response_mode': 'strict'}
    args.output.mkdir(parents=True, exist_ok=False)
    intent = {'artifact_kind': 'mus_diagnostics', 'schema_version': 1, 'suite_id': suite['suite_id'], 'suite_sha256': fingerprint(suite),
              'environment_sha256': manifest['environment_sha256'],
              'rules_sha256': fingerprint(rules_text()), 'source_sha256': source_snapshot(args.output),
              'order_seed': args.order_seed, 'policy': identity, 'scheduled': len(suite['cases'])}
    atomic_json(args.output / 'manifest.json', intent)
    with (args.output / 'records.jsonl').open('x') as stream:
        def record(row):
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            print(f"{row['case_id']}: {row['status']}; correct={row['correct']}", flush=True)
        report = evaluate(responder, order_seed=args.order_seed, on_record=record)
    atomic_json(args.output / 'report.json', report)
    write_checksums(args.output)
    print(json.dumps(report, indent=2))
    return 130 if report['statuses'].get('interrupted') else 1 if report['statuses'].get('provider_error') else 0


if __name__ == '__main__':
    raise SystemExit(diagnostic_command())

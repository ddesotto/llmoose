"""Pinned development environment specification and grouped deal allocation."""

import hashlib
import json
from dataclasses import asdict, fields
from importlib.resources import files

from llmoose.agents.prompt import load_system_prompt
from llmoose.game.actions import ActionCodec
from llmoose.game.environment import StepResult
from llmoose.game.state import Phase, Seat, priority_from_mano, team_for
from llmoose.observations.core import Observation
from llmoose.observations.encoding import INPUT_SCHEMA_VERSION
from llmoose.rules.cards import spanish_deck
from llmoose.rules.hands import JUEGO_ORDER
from llmoose.rules.ruleset import Ruleset


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_data(name):
    return json.loads(files('llmoose.benchmark').joinpath('data', name).read_text())


def environment_contract():
    """Executable inventory; semantics below are frozen and regression-tested."""
    return json.loads(canonical({
        'contract_id': 'mus-development-environment-v1',
        'ruleset': asdict(Ruleset(target_score=40)),
        'observation_schema': INPUT_SCHEMA_VERSION,
        'observation_fields': [f.name for f in fields(Observation)],
        'step_result_fields': [f.name for f in fields(StepResult)],
        'phases': [p.value for p in Phase],
        'deck': [asdict(c) for c in spanish_deck()],
        'juego_order': list(JUEGO_ORDER),
        'teams': {s.name: team_for(s) for s in Seat},
        'tie_priority': {s.name: [p.name for p in priority_from_mano(s)] for s in Seat},
        'action_catalog': [asdict(a) for a in ActionCodec(40).actions],
        'prompt': {'version': 5, 'sha256': hashlib.sha256(load_system_prompt(5).encode()).hexdigest()},
        'memory': {
            'own_hand': 'current', 'own_discards': 'current_hand_only',
            'public_events': 'full_match_including_completed_showdowns',
            'opponent_private_cards': 'hidden_until_reveal',
            'opponent_private_discards': 'hidden', 'stock_and_seed': 'hidden',
            'perfect_recall': False,
        },
        'lifecycle': {
            'reset': 'new episode; step counter cleared; caller creates fresh seat policies per match',
            'turn': 'observation belongs to the next acting seat; no shared private partner memory',
            'reward': 'paid team stone delta per step; not win probability',
            'terminated': 'match winner or configured hand episode completed',
            'truncated': 'step budget exhausted without natural episode completion; no fabricated winner',
            'terminal_priority': 'natural termination wins over a simultaneous step limit',
            'truncated_observation': 'next actor view retained; step requires reset afterwards',
            'info_and_state': 'trainer/evaluator metadata, not policy inputs',
        },
    }))


def verify_contract():
    frozen = load_data('environment_v1.json')
    if environment_contract() != frozen:
        raise ValueError('Environment contract drift: review and publish a new contract version')
    return fingerprint(frozen)


def development_manifest():
    manifest = load_data('development_v1.json')
    if manifest['environment_sha256'] != verify_contract():
        raise ValueError('Development suite environment hash mismatch')
    if manifest['diagnostics_sha256'] != fingerprint(load_data('diagnostics_v1.json')):
        raise ValueError('Diagnostic fixture hash mismatch')
    rules = files('llmoose.benchmark').joinpath('data', 'diagnostic_rules_v1.txt').read_text().strip()
    if manifest['rules_sha256'] != fingerprint(rules):
        raise ValueError('Diagnostic rules hash mismatch')
    opponent_source = files('llmoose.agents').joinpath('baselines.py').read_bytes()
    if manifest['opponents_sha256'] != hashlib.sha256(opponent_source).hexdigest():
        raise ValueError('Frozen opponent implementation hash mismatch')
    groups = manifest['deal_groups']
    if set(groups) != {'train', 'development', 'test'}:
        raise ValueError('Expected train, development, and test deal groups')
    seen = set()
    for seeds in groups.values():
        if any(type(s) is not int for s in seeds) or len(seeds) != len(set(seeds)):
            raise ValueError('Invalid deal group seeds')
        if seen.intersection(seeds):
            raise ValueError('Deal groups overlap')
        seen.update(seeds)
    return manifest


def split_for_seed(seed):
    for split, seeds in development_manifest()['deal_groups'].items():
        if type(seed) is int and seed in seeds:
            return split
    raise ValueError('Seed is not allocated by the development suite')

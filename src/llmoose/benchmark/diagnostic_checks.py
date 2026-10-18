"""Evaluator-only cross-checks of authored labels; never passed to a policy."""

from collections import Counter
from dataclasses import asdict, replace

from llmoose.agents.baselines import ConservativePolicy
from llmoose.benchmark.contract import canonical, load_data, verify_contract
from llmoose.game.actions import Action, ActionCodec, ActionKind
from llmoose.game.engine import apply_action, legal_actions
from llmoose.game.environment import MusEnv
from llmoose.game.state import Phase, Seat, freeze_hands, new_game, priority_from_mano
from llmoose.observations.core import observe
from llmoose.observations.encoding import policy_input
from llmoose.rules.cards import Card, Rank, Suit, spanish_deck
from llmoose.rules.hands import Lance, compare_lance, evaluate_hand, winner_for_lance
from llmoose.rules.ruleset import Ruleset


def physical_hands(groups):
    counts = Counter()
    hands = []
    for ranks in groups:
        cards = []
        for rank in ranks:
            if counts[rank] >= 4:
                raise ValueError('Fixture exceeds the physical deck multiplicity')
            cards.append(Card(Rank(rank), tuple(Suit)[counts[rank]]))
            counts[rank] += 1
        hands.append(tuple(cards))
    return hands


def fixture_state(groups, *, mano=Seat.WEST, target_score=40):
    hands = dict(zip(Seat, physical_hands(groups)))
    used = {card for hand in hands.values() for card in hand}
    return replace(new_game(0, Ruleset(target_score=target_score), Seat((int(mano) + 1) % 4)), hands=freeze_hands(hands),
                   stock=tuple(card for card in spanish_deck() if card not in used))


def conservative_step(state):
    codec = ActionCodec(state.ruleset.target_score)
    return apply_action(state, ConservativePolicy(codec).act(observe(state, state.current_seat, codec)))


def finish_hand(state, hand_number=0):
    for _ in range(80):
        if state.phase is Phase.COMPLETE or state.hand_number != hand_number:
            return state
        state = conservative_step(state)
    raise ValueError('Diagnostic fixture did not complete its hand')


def engine_answer(facts):
    """Compute from the facts independently of the case's expected label/choices."""
    op = facts['operation']
    if op == 'compare':
        left, right = physical_hands([facts['left'], facts['right']])
        comparison = compare_lance(evaluate_hand(left), evaluate_hand(right), Lance(facts['lance']))
        return {1: 'left', -1: 'right', 0: 'tie'}[comparison]
    if op == 'hand_value':
        hand = evaluate_hand(physical_hands([facts['ranks']])[0])
        return hand.pares.kind.name.lower() if facts['field'] == 'pair_kind' else getattr(hand, facts['field'])
    if op == 'tie':
        hands = dict(zip(Seat, physical_hands(facts['hands'])))
        return winner_for_lance(hands, priority_from_mano(Seat[facts['mano']]), Lance.GRANDE).name
    if op == 'transition':
        state = replace(fixture_state(facts['hypothetical_hands_N_E_S_W'],
                                      mano=Seat[facts['mano']], target_score=facts['target']),
                        scores=tuple(facts['scores']))
        lance = Lance(facts['lance'])
        for _ in range(60):
            if state.current_lance is lance:
                break
            state = conservative_step(state)
        else:
            raise ValueError('Diagnostic fixture did not reach its lance')
        before = state
        for kind, amount in facts['actions']:
            state = apply_action(state, Action(ActionKind(kind), amount=amount))
        query = facts['query']
        if query == 'immediate_delta':
            return sum(state.scores) - sum(before.scores)
        if query == 'phase':
            return state.phase.value
        if query == 'actor':
            return state.current_seat.name
        if query == 'terminated':
            return state.phase is Phase.COMPLETE
        if query == 'scores':
            return list(state.scores)
        if query == 'legal_kinds':
            return sorted({a.kind.value for a in legal_actions(state, state.current_seat)})
        if query == 'minimum_raise':
            return min(a.amount for a in legal_actions(state, state.current_seat) if a.kind is ActionKind.RAISE)
        if query == 'lance_points':
            final = finish_hand(state, before.hand_number)
            return sum(r.points for r in final.lance_results if r.lance is lance)
    if op == 'visibility':
        state = new_game(0)
        codec = ActionCodec(40)
        query = facts['query']
        if query in {'partner_hand', 'seed', 'stock'}:
            packet = policy_input(observe(state, Seat.WEST, codec))['observation']
            return {'partner_hand': 'hands', 'seed': 'seed', 'stock': 'stock'}[query] in packet
        if query == 'pending_winner':
            for action in (Action(ActionKind.NO_MUS), Action(ActionKind.BET, 2), Action(ActionKind.CALL)):
                state = apply_action(state, action)
            return any('winning_team' in asdict(p) for p in observe(state, state.current_seat, codec).pending_lances)
        for _ in range(4):
            state = apply_action(state, Action(ActionKind.MUS))
        state = apply_action(state, Action(ActionKind.DISCARD, card_indices=(0,)))
        if query in {'own_discard_recall', 'other_discard_recall'}:
            seat = Seat.WEST if query == 'own_discard_recall' else Seat.EAST
            return len(observe(state, seat, codec).own_discard_history)
        state = finish_hand(state)
        obs = observe(state, state.current_seat, codec)
        if query == 'recall_reset':
            return len(obs.own_discard_history)
        if query == 'past_reveals':
            return sum(event.kind == 'cards_revealed' for event in obs.public_events)
    if op == 'limit':
        env = MusEnv(max_steps=facts['max_steps'])
        env.reset()
        result = env.step(Action(ActionKind.MUS))
        return 'terminated' if result.terminated else 'truncated' if result.truncated else 'continuing'
    raise ValueError(f'Unsupported diagnostic operation/query: {facts}')


def check_suite():
    contract_hash = verify_contract()
    cases = load_data('diagnostics_v1.json')['cases']
    failures = []
    for case in cases:
        actual = engine_answer(case['facts'])
        if canonical(actual) != canonical(case['expected']):
            failures.append({'id': case['id'], 'expected': case['expected'], 'actual': actual})
    return {'mode': 'engine_contract_check', 'environment_sha256': contract_hash,
            'checked': len(cases), 'passed': len(cases) - len(failures), 'failures': failures}

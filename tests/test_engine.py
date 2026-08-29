from dataclasses import replace

import pytest

from llmoose.game.actions import Action, ActionCodec, ActionKind
from llmoose.game.engine import IllegalAction, apply_action, legal_actions
from llmoose.game.state import Phase, Seat, freeze_hands, new_game
from llmoose.observations.core import observe
from llmoose.rules.cards import Card, Rank, Suit
from llmoose.rules.ruleset import BidResponseProtocol, Ruleset


def _act(state, kind: ActionKind, amount: int = 0):
    action = Action(kind, amount=amount)
    assert action in legal_actions(state, state.current_seat)
    return apply_action(state, action)


def test_mus_discard_round_redeals_each_hand_and_returns_to_vote() -> None:
    state = new_game(seed=4)
    for _ in range(4):
        state = _act(state, ActionKind.MUS)
    assert state.phase is Phase.DISCARD

    for _ in range(4):
        action = Action(ActionKind.DISCARD, card_indices=(0, 2))
        assert action in legal_actions(state, state.current_seat)
        state = apply_action(state, action)

    assert state.phase is Phase.MUS_VOTE
    assert state.current_seat is state.mano
    assert state.discard_counts == (2, 2, 2, 2)
    assert all(len(hand) == 4 for hand in state.hands.values())
    assert len(set(card for hand in state.hands.values() for card in hand)) == 16


def test_called_bet_can_finish_a_match() -> None:
    state = new_game(seed=9, ruleset=Ruleset(target_score=2))
    state = _act(state, ActionKind.NO_MUS)
    state = _act(state, ActionKind.BET, amount=2)
    state = _act(state, ActionKind.CALL)

    assert state.phase is Phase.COMPLETE
    assert state.current_seat is None
    assert sum(state.scores) == 2
    assert state.lance_results[-1].reason == "called"


def test_ordago_must_be_answered_without_another_ordago() -> None:
    state = new_game(seed=9, ruleset=Ruleset(target_score=4))
    state = _act(state, ActionKind.NO_MUS)
    state = _act(state, ActionKind.ORDAGO)

    assert Action(ActionKind.ORDAGO) not in legal_actions(state, state.current_seat)
    assert Action(ActionKind.CALL) in legal_actions(state, state.current_seat)


def test_refusing_a_raise_awards_the_stake_before_the_raise() -> None:
    state = new_game(seed=1, ruleset=Ruleset(target_score=10))
    state = _act(state, ActionKind.NO_MUS)
    state = _act(state, ActionKind.BET, amount=2)
    state = _act(state, ActionKind.RAISE, amount=3)
    state = _act(state, ActionKind.FOLD)

    assert state.lance_results[-1].points == 2
    assert state.scores == (2, 0)


def test_refusing_an_ordago_raise_awards_the_previous_stake() -> None:
    state = new_game(seed=1, ruleset=Ruleset(target_score=10))
    state = _act(state, ActionKind.NO_MUS)
    state = _act(state, ActionKind.BET, amount=2)
    state = _act(state, ActionKind.ORDAGO)
    state = _act(state, ActionKind.FOLD)

    assert state.lance_results[-1].points == 2
    assert state.scores == (2, 0)


def test_ruleset_versions_the_next_opposing_seat_bid_response() -> None:
    state = new_game(seed=1)
    state = _act(state, ActionKind.NO_MUS)
    state = _act(state, ActionKind.BET, amount=2)

    assert Ruleset().bid_response_protocol is BidResponseProtocol.NEXT_OPPOSING_SEAT
    assert state.current_seat is Seat.NORTH


def test_uncontested_pares_scores_every_qualifying_hand_on_the_winning_team() -> None:
    state = _state_with_hands(
        {
            Seat.NORTH: _hand(Rank.ACE, Rank.TWO, Rank.REY, Rank.THREE, suit_offset=0),
            Seat.EAST: _hand(Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN, suit_offset=1),
            Seat.SOUTH: _hand(
                Rank.CABALLO, Rank.CABALLO, Rank.SOTA, Rank.SOTA, suit_offset=2
            ),
            Seat.WEST: _hand(Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN, suit_offset=3),
        },
        target_score=8,
    )
    state = _pass_through_grande_and_chica(state)
    for _ in range(4):
        state = apply_action(state, legal_actions(state, state.current_seat)[0])

    assert state.lance_results[-1].lance.value == "pares"
    assert state.lance_results[-1].winning_team == 0
    assert state.lance_results[-1].points == 6


def test_uncontested_juego_scores_every_qualifying_hand_on_the_winning_team() -> None:
    state = _state_with_hands(
        {
            Seat.NORTH: _hand(
                Rank.REY, Rank.CABALLO, Rank.SOTA, Rank.ACE, suit_offset=0
            ),
            Seat.EAST: _hand(Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN, suit_offset=1),
            Seat.SOUTH: _hand(
                Rank.THREE, Rank.CABALLO, Rank.SOTA, Rank.TWO, suit_offset=2
            ),
            Seat.WEST: _hand(Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN, suit_offset=3),
        },
        target_score=8,
    )
    state = _pass_through_grande_and_chica(state)
    for _ in range(4):
        state = apply_action(state, legal_actions(state, state.current_seat)[0])
    for _ in range(4):
        state = apply_action(state, legal_actions(state, state.current_seat)[0])

    assert state.lance_results[-1].lance.value == "juego"
    assert state.lance_results[-1].winning_team == 0
    assert state.lance_results[-1].points == 6


def test_engine_rejects_out_of_turn_and_unavailable_actions() -> None:
    state = new_game(seed=1)
    with pytest.raises(IllegalAction):
        apply_action(state, Action(ActionKind.PASS))


def test_observation_uses_engine_action_mask() -> None:
    state = new_game(seed=1)
    codec = ActionCodec()
    actor = state.current_seat
    assert actor is not None
    observation = observe(state, actor, codec)
    assert observation.legal_action_mask[codec.encode(Action(ActionKind.MUS))]
    assert observation.legal_action_mask[codec.encode(Action(ActionKind.NO_MUS))]
    assert sum(observation.legal_action_mask) == 2


def test_frozen_state_does_not_expose_mutable_hands() -> None:
    state = new_game(seed=1)
    with pytest.raises(TypeError):
        state.hands[Seat.NORTH] = ()


def _state_with_hands(hands, target_score: int = 20):
    state = new_game(seed=1, ruleset=Ruleset(target_score=target_score))
    return replace(state, hands=freeze_hands(hands))


def _pass_through_grande_and_chica(state):
    state = _act(state, ActionKind.NO_MUS)
    for _ in range(8):
        state = _act(state, ActionKind.PASS)
    return state


def _hand(*ranks: Rank, suit_offset: int):
    suits = tuple(Suit)
    return tuple(
        Card(rank, suits[(index + suit_offset) % len(suits)])
        for index, rank in enumerate(ranks)
    )

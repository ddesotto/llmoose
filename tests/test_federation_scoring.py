"""Independent score fixtures for the federation-based v2 contract.

Expected winners/points are written out from the supplied hands, not computed
with the engine's hand evaluator. See README.md's rules contract and references.
"""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from llmoose.agents.baselines import ConservativePolicy
from llmoose.agents.prompt import render_messages
from llmoose.game.actions import Action, ActionCodec, ActionKind
from llmoose.game.engine import IllegalAction, apply_action, legal_actions
from llmoose.game.state import Phase, Seat, freeze_hands, new_game
from llmoose.observations.core import observe
from llmoose.rules.cards import Card, Rank, Suit
from llmoose.rules.hands import Lance
from llmoose.rules.ruleset import FEDERATION_RULESET_ID, LEGACY_RULESET_ID, Ruleset
from llmoose.simulation.runs import score_game
from llmoose.simulation.terminal import TerminalSpectator
from llmoose.simulation.traces import MatchTrace, load_trace, replay_trace

# Grande: North; chica: West; pares: North (team value 2+1);
# juego: only North/South qualify (3+3). West is mano throughout these fixtures.
MIXED = ((12, 12, 12, 1), (10, 11, 6, 4), (10, 10, 11, 1), (7, 7, 7, 1))
# Grande: North; chica: South; pares: North (2+2);
# juego: West's 31 wins the tie by mano (team value 3+2).
ALL_JUEGO = ((12, 12, 12, 1), (11, 11, 11, 4), (10, 10, 10, 1), (7, 7, 7, 12))
# No pares or juego. Grande/chica: West/East; punto: North's 27.
PUNTO = ((12, 7, 6, 4), (11, 7, 5, 1), (10, 7, 6, 1), (12, 11, 5, 1))


def fixture_state(ranks=MIXED, *, scores=(0, 0), target=40):
    counts = Counter()
    hands = {}
    for seat, hand in zip(Seat, ranks):
        cards = []
        for rank in hand:
            cards.append(Card(Rank(rank), tuple(Suit)[counts[rank]]))
            counts[rank] += 1
        hands[seat] = tuple(cards)
    state = new_game(1, Ruleset(target_score=target))
    used = {card for hand in hands.values() for card in hand}
    deck = tuple(state.stock) + tuple(c for hand in state.hands.values() for c in hand)
    return replace(
        state,
        hands=freeze_hands(hands),
        stock=tuple(c for c in deck if c not in used),
        scores=scores,
    )


def act(state, kind, amount=0):
    return apply_action(state, Action(kind, amount=amount))


def advance_to(state, lance):
    policy = ConservativePolicy(ActionCodec(state.ruleset.target_score))
    for _ in range(30):
        if state.current_lance is lance:
            return state
        state = apply_action(
            state, policy.act(observe(state, state.current_seat, policy.codec))
        )
    raise AssertionError(f"Did not reach {lance}")


def finish_hand(state, hand=0):
    policy = ConservativePolicy(ActionCodec(state.ruleset.target_score))
    for _ in range(40):
        if state.phase is Phase.COMPLETE or state.hand_number != hand:
            return state
        state = apply_action(
            state, policy.act(observe(state, state.current_seat, policy.codec))
        )
    raise AssertionError("Hand did not finish")


def assert_accounting(state):
    report = score_game(state, 0)
    for team in (0, 1):
        assert (
            sum(r.points for r in state.lance_results if r.winning_team == team)
            == state.scores[team]
        )
        assert (
            sum(v["points"] for v in report["teams"][team]["by_lance"].values())
            == state.scores[team]
        )
        assert (
            sum(
                int(dict(e.detail)["points"])
                for e in state.events
                if e.kind == "score_awarded" and int(dict(e.detail)["team"]) == team
            )
            == state.scores[team]
        )


def test_called_target_sized_bet_waits_for_showdown_and_has_no_public_winner():
    state = advance_to(fixture_state(target=2), Lance.GRANDE)
    state = act(act(state, ActionKind.BET, 2), ActionKind.CALL)
    assert state.phase is Phase.CHICA
    assert state.scores == (0, 0)
    assert not state.lance_results
    assert len(state.pending_lances) == 1
    assert not any(
        e.kind in {"cards_revealed", "lance_resolved", "score_awarded"}
        for e in state.events
    )
    # Change hidden cards while preserving the observing seat and declarations.
    hands = dict(state.hands)
    hands[Seat.NORTH], hands[Seat.SOUTH] = hands[Seat.SOUTH], hands[Seat.NORTH]
    other = replace(state, hands=freeze_hands(hands))
    codec = ActionCodec(2)
    assert observe(state, Seat.WEST, codec) == observe(other, Seat.WEST, codec)
    final = finish_hand(state)
    assert final.scores == (2, 0)
    assert final.phase is Phase.COMPLETE
    assert [(r.lance, r.points) for r in final.lance_results] == [(Lance.GRANDE, 2)]
    assert not final.pending_lances


def test_uncontested_pares_waits_until_juego_finishes():
    # Only team 0 has pairs, value 3+3; South alone has juego (40).
    ranks = ((12, 12, 1, 1), (4, 5, 6, 7), (11, 11, 10, 10), (4, 5, 6, 7))
    state = advance_to(fixture_state(ranks), Lance.GRANDE)
    for _ in range(8):
        state = act(state, ActionKind.PASS)
    for _ in range(4):
        state = apply_action(state, legal_actions(state, state.current_seat)[0])
    assert state.phase is Phase.DECLARE_JUEGO
    assert state.scores == (0, 0)
    assert state.pending_lances[-1].lance is Lance.PARES
    assert state.pending_lances[-1].reason == "uncontested"
    final = finish_hand(state)
    assert next(r for r in final.lance_results if r.lance is Lance.PARES).points == 6


@pytest.mark.parametrize(
    "lance,ranks,total,award",
    [
        (Lance.GRANDE, MIXED, (13, 1), 4),
        (Lance.CHICA, MIXED, (10, 4), 4),
        (Lance.PARES, MIXED, (14, 1), 7),
        (Lance.JUEGO, ALL_JUEGO, (6, 9), 9),
        (Lance.PUNTO, PUNTO, (5, 2), 5),
    ],
)
def test_called_stakes_and_intrinsic_values(lance, ranks, total, award):
    state = advance_to(fixture_state(ranks), lance)
    state = act(act(state, ActionKind.BET, 4), ActionKind.CALL)
    final = finish_hand(state)
    assert final.scores == total
    result = next(r for r in final.lance_results if r.lance is lance)
    assert (result.points, result.reason) == (award, "called")
    assert_accounting(final)


@pytest.mark.parametrize(
    "lance,ranks,total,bonus",
    [
        (Lance.PARES, MIXED, (7, 4), 2),
        (Lance.JUEGO, ALL_JUEGO, (6, 6), 5),
        (Lance.PUNTO, PUNTO, (0, 4), 1),
    ],
)
def test_fold_pays_refusal_now_and_intrinsic_later_without_double_counting(
    lance, ranks, total, bonus
):
    state = advance_to(fixture_state(ranks), lance)
    state = act(state, ActionKind.BET, 2)
    before = state
    state = act(state, ActionKind.FOLD)
    if lance is Lance.PARES:
        assert state.scores == (0, 1)
        assert state.lance_results[0].points == 1
    final = finish_hand(state)
    assert final.scores == total
    results = [r for r in final.lance_results if r.lance is lance]
    assert len(results) == 1
    assert (results[0].winning_team, results[0].points, results[0].reason) == (
        1,
        1 + bonus,
        "fold",
    )
    assert score_game(final, 0)["teams"][1]["by_lance"][lance.value]["won"] == 1
    assert_accounting(final)
    if lance is Lance.PUNTO:
        # The final fold can emit refusal and all showdown awards in one step.
        assert sum(
            int(dict(e.detail)["points"])
            for e in state.events[len(before.events) :]
            if e.kind == "score_awarded"
        ) == sum(total)


def test_later_accepted_ordago_cancels_earlier_potential_match_winning_bet():
    state = advance_to(fixture_state(scores=(39, 39)), Lance.GRANDE)
    state = act(act(state, ActionKind.BET, 40), ActionKind.CALL)
    assert state.scores == (39, 39)
    # Team 0 wins grande, but team 1 wins chica.
    state = act(act(state, ActionKind.ORDAGO), ActionKind.CALL)
    assert state.scores == (39, 40)
    assert state.phase is Phase.COMPLETE
    assert not state.pending_lances
    assert [
        (r.lance, r.winning_team, r.points, r.reason) for r in state.lance_results
    ] == [(Lance.CHICA, 1, 1, "ordago")]
    assert len([e for e in state.events if e.kind == "cards_revealed"]) == 4
    assert all(not legal_actions(state, seat) for seat in Seat)


def test_later_refusal_can_win_before_an_earlier_accepted_bet():
    state = advance_to(fixture_state(scores=(39, 39)), Lance.GRANDE)
    state = act(act(state, ActionKind.BET, 40), ActionKind.CALL)
    state = act(act(state, ActionKind.BET, 2), ActionKind.FOLD)
    assert state.scores == (39, 40)
    assert state.phase is Phase.COMPLETE
    assert not state.pending_lances
    assert not any(e.kind == "cards_revealed" for e in state.events)
    assert [(r.lance, r.reason) for r in state.lance_results] == [(Lance.CHICA, "fold")]


def test_showdown_stops_at_first_threshold_in_lance_order():
    final = finish_hand(fixture_state(scores=(39, 39)))
    assert final.scores == (40, 39)
    assert [(r.lance, r.points) for r in final.lance_results] == [(Lance.GRANDE, 1)]
    assert final.hand_number == 0
    kinds = [e.kind for e in final.events]
    assert kinds.index("cards_revealed") < kinds.index("score_awarded")


def test_numeric_max_can_be_raised_to_ordago_and_refusal_uses_numeric_stake():
    state = advance_to(fixture_state(), Lance.GRANDE)
    state = act(state, ActionKind.BET, 40)
    assert not state.is_ordago
    assert dict(state.events[-1].detail)["action"] == "bet"
    assert Action(ActionKind.ORDAGO) in legal_actions(state, state.current_seat)
    state = act(state, ActionKind.ORDAGO)
    assert state.is_ordago
    assert dict(state.events[-1].detail)["action"] == "ordago"
    assert legal_actions(state, state.current_seat) == (
        Action(ActionKind.CALL),
        Action(ActionKind.FOLD),
    )
    observation = observe(state, state.current_seat, ActionCodec())
    assert observation.is_ordago
    assert "Outstanding ordago: True" in render_messages(observation)[1]["content"]
    state = act(state, ActionKind.FOLD)
    assert state.scores == (40, 0)
    assert state.phase is Phase.COMPLETE


def test_v2_minimum_raise_is_two_but_v1_replays_one_stone_raises():
    state = advance_to(fixture_state(), Lance.GRANDE)
    state = act(state, ActionKind.BET, 2)
    with pytest.raises(IllegalAction):
        act(state, ActionKind.RAISE, 3)
    assert act(state, ActionKind.RAISE, 4).stake == 4
    legacy = replace(state, ruleset=Ruleset(id=LEGACY_RULESET_ID))
    assert act(legacy, ActionKind.RAISE, 3).stake == 3


def test_all_passes_reveal_only_finished_hands_and_reset_pending_state():
    state = advance_to(fixture_state(PUNTO), Lance.PUNTO)
    assert state.scores == (0, 0)
    assert [p.lance for p in state.pending_lances] == [Lance.GRANDE, Lance.CHICA]
    assert not any(e.kind == "cards_revealed" for e in state.events)
    final = finish_hand(state)
    assert final.scores == (1, 2)
    assert final.phase is Phase.MUS_VOTE
    assert final.hand_number == 1
    assert not final.pending_lances
    assert not final.is_ordago
    revealed = [e for e in final.events if e.kind == "cards_revealed"]
    assert {e.actor for e in revealed} == set(Seat)
    for event in revealed:
        detail = dict(event.detail)
        assert detail["hand_number"] == "0"
        assert detail["cards"] == ",".join(
            f"{int(c.rank)}:{c.suit.value}" for c in state.hands[event.actor]
        )
    observation = observe(final, final.current_seat, ActionCodec())
    assert tuple(
        e for e in observation.public_events if e.kind == "cards_revealed"
    ) == tuple(revealed)
    assert_accounting(final)


def test_legacy_trace_keeps_original_outcome_and_no_card_reveals():
    trace = load_trace(Path(__file__).parent / "fixtures" / "legacy_match_v1.json")
    state = replay_trace(trace)
    assert state.scores == (3, 8)
    assert state.ruleset.id == LEGACY_RULESET_ID
    assert not any(
        e.kind in {"cards_revealed", "score_awarded", "lance_pending"}
        for e in state.events
    )
    assert not state.pending_lances
    value = trace.to_dict()
    value["ruleset"]["id"] = "unknown_rules"
    with pytest.raises(ValueError, match="Unsupported ruleset"):
        MatchTrace.from_dict(value)


def test_terminal_scores_are_not_printed_twice(capsys):
    before = advance_to(fixture_state(PUNTO), Lance.PUNTO)
    before = act(before, ActionKind.BET, 2)
    after = act(before, ActionKind.FOLD)
    TerminalSpectator()(before, Action(ActionKind.FOLD), after)
    output = capsys.readouterr().out
    assert output.count("+1") == 4
    assert "score 0–4" in output


def test_default_ruleset_is_versioned():
    assert Ruleset().id == FEDERATION_RULESET_ID


@pytest.mark.parametrize(
    "lance,ranks,winner",
    [
        (Lance.GRANDE, MIXED, 0),
        (Lance.CHICA, MIXED, 1),
        (Lance.PARES, MIXED, 0),
        (Lance.JUEGO, ALL_JUEGO, 1),
        (Lance.PUNTO, PUNTO, 0),
    ],
)
def test_called_ordago_wins_each_lance_without_pending_awards(lance, ranks, winner):
    state = advance_to(fixture_state(ranks), lance)
    state = act(act(state, ActionKind.ORDAGO), ActionKind.CALL)
    assert state.phase is Phase.COMPLETE
    assert state.scores == ((40, 0) if winner == 0 else (0, 40))
    assert len(state.lance_results) == 1
    assert state.lance_results[0].reason == "ordago"
    assert state.lance_results[0].lance is lance
    assert_accounting(state)


def test_ordago_cancels_fold_bonus_but_keeps_paid_refusal():
    state = advance_to(fixture_state(ALL_JUEGO), Lance.PARES)
    state = act(act(state, ActionKind.BET, 2), ActionKind.FOLD)
    assert state.scores == (0, 1)
    state = advance_to(state, Lance.JUEGO)
    state = act(act(state, ActionKind.ORDAGO), ActionKind.CALL)
    assert state.scores == (0, 40)
    assert [(r.lance, r.points) for r in state.lance_results] == [
        (Lance.PARES, 1),
        (Lance.JUEGO, 39),
    ]
    assert_accounting(state)


def test_refused_raise_has_no_extra_deje_and_concedes_intrinsic_values():
    state = advance_to(fixture_state(), Lance.PARES)
    state = act(state, ActionKind.BET, 2)  # West
    state = act(state, ActionKind.RAISE, 4)  # North
    state = act(state, ActionKind.RAISE, 6)  # West
    state = act(state, ActionKind.FOLD)  # North concedes despite holding better pares.
    assert state.scores == (0, 4)
    final = finish_hand(state)
    assert final.scores == (7, 7)  # Grande+juego vs chica+4 refusal+2 intrinsic.
    assert_accounting(final)

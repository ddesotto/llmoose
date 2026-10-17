"""Expected hand outcomes transcribed from the v2 rules contract.

These examples state the winning side before calling the evaluator. They do not
derive expected values from evaluate_hand, the engine, or a replayed match.
"""

from collections import Counter

import pytest

from llmoose.rules.cards import Card, Rank, Suit
from llmoose.rules.hands import Lance, compare_lance, evaluate_hand, winner_for_lance


def hands_for(left, right):
    """Build two disjoint hands from the same physical 40-card deck."""
    counts = Counter()
    hands = []
    for ranks in (left, right):
        cards = []
        for rank in ranks:
            assert counts[rank] < 4
            cards.append(Card(Rank(rank), tuple(Suit)[counts[rank]]))
            counts[rank] += 1
        hands.append(tuple(cards))
    return tuple(hands)


@pytest.mark.parametrize(
    "name,lance,left,right,expected",
    [
        ("grande_second_card", Lance.GRANDE, (12, 12, 4, 1), (12, 11, 11, 7), 1),
        ("grande_normalized", Lance.GRANDE, (3, 10, 7, 1), (12, 10, 7, 2), 0),
        ("chica_second_card", Lance.CHICA, (1, 4, 10, 12), (2, 5, 6, 7), 1),
        ("chica_normalized", Lance.CHICA, (1, 4, 6, 7), (2, 4, 6, 7), 0),
        ("duples_over_medias", Lance.PARES, (4, 4, 7, 7), (12, 12, 12, 1), 1),
        ("medias_over_pair", Lance.PARES, (4, 4, 4, 12), (7, 7, 6, 5), 1),
        ("higher_triple", Lance.PARES, (4, 4, 4, 12), (7, 7, 7, 1), -1),
        ("triple_singleton_ignored", Lance.PARES, (3, 3, 3, 7), (12, 12, 12, 1), 0),
        ("pair_singletons_ignored", Lance.PARES, (7, 7, 12, 11), (7, 7, 4, 1), 0),
        ("duples_second_pair", Lance.PARES, (7, 7, 6, 6), (7, 7, 5, 5), 1),
        ("four_equal_as_duples", Lance.PARES, (3, 3, 3, 3), (12, 12, 6, 6), 1),
        ("normalized_pair", Lance.PARES, (2, 1, 7, 6), (1, 1, 5, 4), 0),
        ("juego_31_over_32", Lance.JUEGO, (12, 11, 10, 1), (12, 11, 7, 5), 1),
        ("juego_32_over_40", Lance.JUEGO, (12, 11, 7, 5), (12, 11, 10, 3), 1),
        ("juego_40_over_37", Lance.JUEGO, (12, 11, 10, 3), (12, 11, 10, 7), 1),
        ("juego_equal_total", Lance.JUEGO, (12, 11, 10, 1), (3, 10, 11, 2), 0),
        ("punto_higher_total", Lance.PUNTO, (12, 11, 6, 4), (12, 11, 5, 4), 1),
        ("punto_equal_total", Lance.PUNTO, (7, 7, 6, 4), (12, 5, 5, 4), 0),
    ],
)
def test_expected_comparison(name, lance, left, right, expected):
    left_cards, right_cards = hands_for(left, right)
    left_value = evaluate_hand(left_cards)
    right_value = evaluate_hand(right_cards)
    assert compare_lance(left_value, right_value, lance) == expected, name
    assert compare_lance(right_value, left_value, lance) == -expected, name


@pytest.mark.parametrize(
    "lance,left,right",
    [
        (Lance.GRANDE, (3, 10, 7, 1), (12, 10, 7, 2)),
        (Lance.CHICA, (1, 4, 6, 7), (2, 4, 6, 7)),
        (Lance.PARES, (3, 3, 3, 7), (12, 12, 12, 1)),
        (Lance.JUEGO, (12, 11, 10, 1), (3, 10, 11, 2)),
        (Lance.PUNTO, (7, 7, 6, 4), (12, 5, 5, 4)),
    ],
)
def test_equal_hands_follow_mano_priority(lance, left, right):
    first, second = hands_for(left, right)
    hands = {"first": first, "second": second}
    assert winner_for_lance(hands, ("first", "second"), lance) == "first"
    assert winner_for_lance(hands, ("second", "first"), lance) == "second"

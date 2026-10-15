from llmoose.rules.cards import Card, Rank, Suit
from llmoose.rules.hands import (
    Lance,
    PairKind,
    compare_lance,
    evaluate_hand,
    winner_for_lance,
)


def _hand(*ranks: Rank):
    return tuple(Card(rank, suit) for rank, suit in zip(ranks, Suit))


def hand_from(*ranks: Rank):
    return evaluate_hand(_hand(*ranks))


# Grande
def test_bigger_n_plus_1_card_breaks_tie() -> None:
    best_card: Rank = Rank.REY
    second_best_card: Rank = Rank.CABALLO
    third_best_card: Rank = Rank.SOTA

    winner = hand_from(best_card, best_card, best_card, second_best_card)
    loser = hand_from(best_card, best_card, best_card, third_best_card)

    assert compare_lance(winner, loser, Lance.GRANDE) == 1


def test_kings_and_threes_are_equal_in_grande() -> None:
    all_threes = hand_from(*([Rank.THREE] * 4))
    all_kings = hand_from(*([Rank.REY] * 4))

    assert compare_lance(all_threes, all_kings, Lance.GRANDE) == 0


# Chicas
def test_aces_and_twos_are_equal_in_chica() -> None:
    all_aces = hand_from(*([Rank.ACE] * 4))
    all_twos = hand_from(*([Rank.TWO] * 4))

    assert compare_lance(all_aces, all_twos, Lance.CHICA) == 0


def test_kings_and_threes_are_equal_in_chica() -> None:
    all_threes = hand_from(*([Rank.THREE] * 4))
    all_kings = hand_from(*([Rank.REY] * 4))
    assert compare_lance(all_threes, all_kings, Lance.CHICA) == 0


def test_smaller_n_plus_1_card_breaks_tie() -> None:
    best_card: Rank = Rank.ACE
    second_best_card: Rank = Rank.FOUR
    third_best_card: Rank = Rank.FIVE

    winner = hand_from(best_card, best_card, best_card, second_best_card)
    loser = hand_from(best_card, best_card, best_card, third_best_card)

    assert compare_lance(winner, loser, Lance.CHICA) == 1


# Pairs
def test_pairs_treat_two_as_ace_and_three_as_king() -> None:
    value = evaluate_hand(_hand(Rank.TWO, Rank.ACE, Rank.SEVEN, Rank.FOUR))
    assert value.pares.kind is PairKind.SIMPLE
    assert value.pares.ranks == (1,)


def test_higher_pair_wins() -> None:

    higher = hand_from(Rank.SEVEN, Rank.SEVEN, Rank.FIVE, Rank.FOUR)
    lower = hand_from(Rank.SIX, Rank.SIX, Rank.SEVEN, Rank.SOTA)

    assert compare_lance(higher, lower, Lance.PARES) == 1


def test_duples_beat_medias_and_higher_pair_breaks_duples() -> None:
    duples = evaluate_hand(_hand(Rank.CABALLO, Rank.CABALLO, Rank.FIVE, Rank.FIVE))
    medias = evaluate_hand(_hand(Rank.SOTA, Rank.SOTA, Rank.SOTA, Rank.SIX))
    assert compare_lance(duples, medias, Lance.PARES) == 1


def test_twos_count_as_aces_for_pairs() -> None:
    pure_aces_pairs = hand_from(Rank.ACE, Rank.ACE, Rank.FOUR, Rank.FIVE)
    mixed_aces_pairs = hand_from(Rank.ACE, Rank.TWO, Rank.FOUR, Rank.FIVE)
    pure_twos_pairs = hand_from(Rank.TWO, Rank.TWO, Rank.FOUR, Rank.FIVE)

    pure_aces_medias = hand_from(Rank.ACE, Rank.ACE, Rank.ACE, Rank.FOUR)
    mixed_aces_medias = hand_from(Rank.ACE, Rank.ACE, Rank.TWO, Rank.FOUR)
    pure_twos_medias = hand_from(Rank.TWO, Rank.TWO, Rank.TWO, Rank.FOUR)

    assert mixed_aces_pairs.pares.kind is PairKind.SIMPLE
    assert mixed_aces_medias.pares.kind is PairKind.MEDIAS

    assert mixed_aces_pairs.pares.ranks == (1,)
    assert mixed_aces_medias.pares.ranks == (1,)

    assert pure_aces_pairs.pares.ranks == mixed_aces_pairs.pares.ranks
    assert mixed_aces_pairs.pares.ranks == pure_twos_pairs.pares.ranks

    assert pure_aces_medias.pares.ranks == mixed_aces_medias.pares.ranks
    assert mixed_aces_medias.pares.ranks == pure_twos_medias.pares.ranks


def test_threes_count_as_kings_for_pairs() -> None:

    pure_king_pairs = hand_from(Rank.REY, Rank.REY, Rank.FOUR, Rank.FIVE)
    mixed_king_pairs = hand_from(Rank.REY, Rank.THREE, Rank.FOUR, Rank.FIVE)
    pure_three_pairs = hand_from(Rank.THREE, Rank.THREE, Rank.FOUR, Rank.FIVE)

    pure_king_medias = hand_from(Rank.REY, Rank.REY, Rank.REY, Rank.FIVE)
    mixed_king_medias = hand_from(Rank.REY, Rank.REY, Rank.THREE, Rank.FOUR)
    pure_three_medias = hand_from(Rank.THREE, Rank.THREE, Rank.THREE, Rank.FIVE)

    assert mixed_king_pairs.pares.ranks == (12,)
    assert mixed_king_medias.pares.ranks == (12,)

    assert mixed_king_pairs.pares.kind is PairKind.SIMPLE
    assert mixed_king_medias.pares.kind is PairKind.MEDIAS

    assert pure_king_pairs.pares.ranks == mixed_king_pairs.pares.ranks
    assert mixed_king_pairs.pares.ranks == pure_three_pairs.pares.ranks

    assert pure_king_medias.pares.ranks == mixed_king_medias.pares.ranks
    assert mixed_king_medias.pares.ranks == pure_three_medias.pares.ranks


def test_nonchalant_pairs_respect_order() -> None:

    nonchalant_pairs = hand_from(Rank.SEVEN, Rank.SEVEN, Rank.FOUR, Rank.ACE)
    worse_nonchalant_pairs = hand_from(Rank.FOUR, Rank.FOUR, Rank.SEVEN, Rank.REY)

    nonchalant_medias = hand_from(Rank.SEVEN, Rank.SEVEN, Rank.SEVEN, Rank.ACE)
    worse_nonchalant_medias = hand_from(Rank.FOUR, Rank.FOUR, Rank.FOUR, Rank.REY)

    assert compare_lance(nonchalant_pairs, worse_nonchalant_pairs, Lance.PARES) == 1
    assert compare_lance(nonchalant_medias, worse_nonchalant_medias, Lance.PARES) == 1


# Juego
def test_juego_respects_gaming_ranks() -> None:

    JUEGO_ORDER = (31, 32, 40, 37, 36, 35, 34, 33)

    HAND_PER_VALUE = {
        31: hand_from(Rank.REY, Rank.CABALLO, Rank.SOTA, Rank.ACE),
        32: hand_from(Rank.REY, Rank.CABALLO, Rank.SEVEN, Rank.FIVE),
        40: hand_from(Rank.REY, Rank.CABALLO, Rank.SOTA, Rank.THREE),
        # 39 not possible
        # 38 not possible
        37: hand_from(Rank.REY, Rank.CABALLO, Rank.SOTA, Rank.SEVEN),
        36: hand_from(Rank.REY, Rank.CABALLO, Rank.SOTA, Rank.SIX),
        35: hand_from(Rank.REY, Rank.CABALLO, Rank.SOTA, Rank.FIVE),
        34: hand_from(Rank.REY, Rank.CABALLO, Rank.SOTA, Rank.FOUR),
        33: hand_from(Rank.REY, Rank.CABALLO, Rank.SEVEN, Rank.SIX),
    }

    for total, hand in HAND_PER_VALUE.items():
        assert hand.total == total

    for left_index, left_total in enumerate(JUEGO_ORDER):
        for right_index, right_total in enumerate(JUEGO_ORDER):
            if left_index < right_index:
                expected = 1
            elif left_index > right_index:
                expected = -1
            else:
                expected = 0

            actual = compare_lance(
                HAND_PER_VALUE[left_total], HAND_PER_VALUE[right_total], Lance.JUEGO
            )

            assert actual == expected, (
                f"Juego {left_total} vs {right_total}: "
                f"expected {expected}, got {actual}"
            )


# Punto
def test_punto_prefers_higher_totals() -> None:
    hands = {
        4: hand_from(Rank.ACE, Rank.ACE, Rank.ACE, Rank.ACE),
        # 5 not possible
        # 6 not possible
        7: hand_from(Rank.FOUR, Rank.ACE, Rank.ACE, Rank.ACE),
        # 8 - 20 not possible
        21: hand_from(Rank.SOTA, Rank.FIVE, Rank.FIVE, Rank.ACE),
        # 22 - 24 not possible
        24: hand_from(Rank.SEVEN, Rank.SEVEN, Rank.SIX, Rank.FOUR),
        # 25 not possible
        # 26 not possible
        27: hand_from(Rank.REY, Rank.SEVEN, Rank.SIX, Rank.FOUR),
        # 28 not possible
        29: hand_from(Rank.REY, Rank.CABALLO, Rank.FIVE, Rank.FOUR),
        30: hand_from(Rank.REY, Rank.CABALLO, Rank.SIX, Rank.FOUR),
    }
    for total, hand in hands.items():
        assert hand.total == total
        assert not hand.has_juego

    for left_total, left in hands.items():
        for right_total, right in hands.items():
            if left_total > right_total:
                expected = 1
            elif left_total < right_total:
                expected = -1
            else:
                expected = 0
            actual = compare_lance(left, right, Lance.PUNTO)
            assert actual == expected, (
                f"Punto {left_total} vs {right_total}: "
                f"expected {expected}, got {actual}"
            )


def test_punto_equal_totals_ignore_individual_card_ranks() -> None:
    small_cards = hand_from(Rank.SEVEN, Rank.SEVEN, Rank.SIX, Rank.FOUR)
    with_king = hand_from(Rank.REY, Rank.FIVE, Rank.FIVE, Rank.FOUR)

    assert small_cards.total == with_king.total == 24
    assert compare_lance(small_cards, with_king, Lance.PUNTO) == 0
    assert compare_lance(with_king, small_cards, Lance.PUNTO) == 0


def test_punto_normalizes_twos_and_threes() -> None:
    with_ace = hand_from(Rank.ACE, Rank.FOUR, Rank.SIX, Rank.SEVEN)
    with_two = hand_from(Rank.TWO, Rank.FOUR, Rank.SIX, Rank.SEVEN)
    with_king = hand_from(Rank.REY, Rank.FOUR, Rank.SIX, Rank.SEVEN)
    with_three = hand_from(Rank.THREE, Rank.FOUR, Rank.SIX, Rank.SEVEN)

    assert with_ace.total == with_two.total == 18
    assert with_king.total == with_three.total == 27
    assert compare_lance(with_ace, with_two, Lance.PUNTO) == 0
    assert compare_lance(with_king, with_three, Lance.PUNTO) == 0


def test_thirty_is_punto_but_thirty_one_is_juego() -> None:
    thirty = hand_from(Rank.REY, Rank.CABALLO, Rank.SIX, Rank.FOUR)
    thirty_one = hand_from(Rank.REY, Rank.CABALLO, Rank.SOTA, Rank.ACE)

    assert thirty.total == 30
    assert not thirty.has_juego
    assert thirty_one.total == 31
    assert thirty_one.has_juego


def test_equal_punto_uses_seat_priority() -> None:
    hands = {
        "left": _hand(Rank.ACE, Rank.ACE, Rank.ACE, Rank.ACE),
        "right": _hand(Rank.TWO, Rank.TWO, Rank.TWO, Rank.TWO),
    }
    assert winner_for_lance(hands, ("left", "right"), Lance.PUNTO) == "left"
    assert winner_for_lance(hands, ("right", "left"), Lance.PUNTO) == "right"

from llmoose.rules.cards import Card, Rank, Suit
from llmoose.rules.hands import Lance, PairKind, compare_lance, evaluate_hand


def _hand(*ranks: Rank):
    return tuple(Card(rank, suit) for rank, suit in zip(ranks, Suit))


def test_pairs_treat_two_as_ace_and_three_as_king() -> None:
    value = evaluate_hand(_hand(Rank.TWO, Rank.ACE, Rank.SEVEN, Rank.FOUR))
    assert value.pares.kind is PairKind.SIMPLE
    assert value.pares.ranks == (1,)


def test_duples_beat_medias_and_higher_pair_breaks_duples() -> None:
    duples = evaluate_hand(_hand(Rank.CABALLO, Rank.CABALLO, Rank.FIVE, Rank.FIVE))
    medias = evaluate_hand(_hand(Rank.SOTA, Rank.SOTA, Rank.SOTA, Rank.SIX))
    assert compare_lance(duples, medias, Lance.PARES) == 1


def test_juego_prefers_31_then_32() -> None:
    thirty_one = evaluate_hand(_hand(Rank.REY, Rank.CABALLO, Rank.SOTA, Rank.ACE))
    # 32 is 10 + 10 + 7 + 5; a two would count only as one.
    thirty_two = evaluate_hand(_hand(Rank.REY, Rank.CABALLO, Rank.SEVEN, Rank.FIVE))
    assert thirty_one.total == 31
    assert thirty_two.total == 32
    assert compare_lance(thirty_one, thirty_two, Lance.JUEGO) == 1

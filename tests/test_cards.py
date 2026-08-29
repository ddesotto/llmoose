from llmoose.rules.cards import Card, Rank, Suit, spanish_deck


def test_spanish_deck_has_40_unique_physical_cards() -> None:
    deck = spanish_deck()
    assert len(deck) == 40
    assert len(set(deck)) == 40


def test_twos_and_threes_are_normalized_for_mus() -> None:
    assert Card(Rank.TWO, Suit.CUPS).mus_rank == Rank.ACE
    assert Card(Rank.THREE, Suit.SWORDS).mus_rank == Rank.REY
    assert Card(Rank.THREE, Suit.SWORDS).juego_points == 10


def test_juego_points_keep_small_cards_and_normalize_threes() -> None:
    assert Card(Rank.ACE, Suit.SWORDS).juego_points == 1
    assert Card(Rank.TWO, Suit.SWORDS).juego_points == 1
    assert Card(Rank.SEVEN, Suit.SWORDS).juego_points == 7
    assert Card(Rank.SOTA, Suit.SWORDS).juego_points == 10
    assert Card(Rank.THREE, Suit.SWORDS).juego_points == 10

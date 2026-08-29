from llmoose.game.state import Phase, Seat, new_game


def test_seeded_deal_is_reproducible_and_private() -> None:
    first = new_game(seed=13, dealer=Seat.NORTH)
    second = new_game(seed=13, dealer=Seat.NORTH)
    assert first == second
    assert first.phase is Phase.MUS_VOTE
    assert all(len(hand) == 4 for hand in first.hands.values())
    assert len(first.stock) == 24

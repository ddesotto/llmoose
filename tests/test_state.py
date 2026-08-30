from llmoose.game.state import SEATS, Phase, Seat, deal_hand, dealer_for, new_game
from llmoose.rules.ruleset import Ruleset


def test_seeded_deal_is_reproducible_and_private() -> None:
    first = new_game(seed=13, dealer=Seat.NORTH)
    second = new_game(seed=13, dealer=Seat.NORTH)
    assert first == second
    assert first.phase is Phase.MUS_VOTE
    assert all(len(hand) == 4 for hand in first.hands.values())
    assert len(first.stock) == 24


def test_dealer_rotates_one_seat_per_hand() -> None:
    assert dealer_for(Seat.NORTH, 0) is Seat.NORTH
    assert dealer_for(Seat.NORTH, 1) is Seat.EAST
    assert dealer_for(Seat.NORTH, len(SEATS)) is Seat.NORTH


def test_dealing_a_later_hand_requires_rotating_the_dealer() -> None:
    """The shuffle ignores the dealer, but who receives which cards does not.

    Dealing hand 3 with the match's first dealer reproduces the deck and then
    hands it to the wrong seats, so jumping straight to a hand has to rotate
    the dealer with ``dealer_for``.
    """

    unrotated = deal_hand(seed=7, ruleset=Ruleset(), dealer=Seat.NORTH, hand_number=3)
    rotated = deal_hand(
        seed=7,
        ruleset=Ruleset(),
        dealer=dealer_for(Seat.NORTH, 3),
        hand_number=3,
    )

    assert unrotated.stock == rotated.stock
    assert dict(unrotated.hands) != dict(rotated.hands)
    assert [unrotated.hands[seat] for seat in unrotated.priority] == [
        rotated.hands[seat] for seat in rotated.priority
    ]

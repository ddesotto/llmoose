"""Plain-text spectator view of a match, with optional interactive pacing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from llmoose.game.actions import Action, ActionKind
from llmoose.game.state import SEATS, GameState, team_for
from llmoose.rules.cards import Card


def format_cards(cards: tuple[Card, ...]) -> str:
    names = {1: "As", 10: "Sota", 11: "Caballo", 12: "Rey"}
    return " | ".join(
        f"{names.get(int(card.rank), str(int(card.rank)))} of {card.suit.value}"
        for card in cards
    )


def format_action(action: Action, before: GameState) -> str:
    if action.kind is ActionKind.DISCARD:
        cards = tuple(before.hands[before.current_seat][i] for i in action.card_indices)
        return f"discards {format_cards(cards)}"
    if action.kind in {ActionKind.BET, ActionKind.RAISE}:
        verb = "bets" if action.kind is ActionKind.BET else "raises to"
        return f"{verb} {action.amount} stones"
    return {
        ActionKind.MUS: "asks for Mus",
        ActionKind.NO_MUS: "cuts Mus",
        ActionKind.PASS: "passes",
        ActionKind.CALL: "calls",
        ActionKind.FOLD: "folds",
        ActionKind.ORDAGO: "bids órdago",
        ActionKind.DECLARE_PARES: "declares pares",
        ActionKind.DECLARE_NO_PARES: "declares no pares",
        ActionKind.DECLARE_JUEGO: "declares juego",
        ActionKind.DECLARE_NO_JUEGO: "declares no juego",
    }[action.kind]


@dataclass
class TerminalSpectator:
    """Display all four hands and every action without affecting policy inputs."""

    delay: float = 0.0
    step: bool = False
    _hand_number: int = field(default=-1, init=False)
    _decisions: int = field(default=0, init=False)

    def before_action(self, before: GameState) -> None:
        """Show the hand and pace the turn before a policy (or API) is called."""
        if self._hand_number != before.hand_number:
            self._hand_number = before.hand_number
            print(
                f"\nHand {before.hand_number + 1} | mano: {before.mano.name.title()}"
                f" | score {before.scores[0]}–{before.scores[1]}",
                flush=True,
            )
            for seat in SEATS:
                print(
                    f"  {seat.name.title():5} (team {team_for(seat)}): "
                    f"{format_cards(before.hands[seat])}",
                    flush=True,
                )
        if self.step:
            try:
                response = input("Enter: next action · q: quit > ")
            except EOFError:
                raise KeyboardInterrupt from None
            if response.strip().lower() in {"q", "quit"}:
                raise KeyboardInterrupt
        elif self.delay and self._decisions:
            time.sleep(self.delay)

    def __call__(self, before: GameState, action: Action, after: GameState) -> None:
        self._decisions += 1
        phase = (
            before.current_lance.value
            if before.current_lance is not None
            else before.phase.value.replace("_", " ")
        )
        print(
            f"  {self._decisions:3}. [{phase}] "
            f"{before.current_seat.name.title()}: {format_action(action, before)}",
            flush=True,
        )
        if action.kind is ActionKind.DISCARD:
            print(
                f"       Now holds: {format_cards(after.hands[before.current_seat])}",
                flush=True,
            )
        scores = list(before.scores)
        award_event = (
            "score_awarded" if before.ruleset.deferred_scoring else "lance_resolved"
        )
        for event in after.events[len(before.events) :]:
            if event.kind == "lance_pending":
                print(
                    f"       {dict(event.detail)['lance'].title()}: pending showdown",
                    flush=True,
                )
            if event.kind == award_event:
                detail = dict(event.detail)
                team = int(detail["team"])
                points = int(detail["points"])
                scores[team] += points
                print(
                    f"       {detail['lance'].title()}: team {team} +{points}"
                    f" ({detail['reason']}) | score {scores[0]}–{scores[1]}",
                    flush=True,
                )

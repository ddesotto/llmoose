"""Versioned prompts shared by model inference and dataset collection."""

from __future__ import annotations

from llmoose.game.actions import Action, ActionCodec, ActionKind
from llmoose.game.state import Seat
from llmoose.observations.core import Observation

PROMPT_VERSION = 1
SYSTEM_PROMPT = """You play one seat in llmoose, a four-player Mus sandbox.
North/South are team 0; East/West are team 1. Maximize your partnership's score.
You see only your own cards and public information. No señas are allowed.
The deck has 40 cards. Twos count as aces and threes as kings. Figures and threes
count as 10 for juego; twos count as 1. Suits do not break ties.
Grande compares ranks from highest down; chica from lowest up.
Pares: duples (two pairs or four equal) beats medias (three equal), then one pair.
Juego totals rank 31, 32, 40, 37, 36, 35, 34, 33, best first.
Punto is played only when nobody has juego; the highest total wins.
Mano wins ties, followed by the other seats in table order.
Mus requires everyone to agree; cutting Mus starts the lances. Discard indices
refer to your current hand, starting at zero. Declarations are enforced truthfully.
Bet and raise amounts are absolute stakes, not increments. The next eligible
opponent answers a bid. A called órdago wins the match; folding awards the
previous stake (one stone if there was no previous bet).
Choose exactly one of the supplied legal action IDs.
Return only <answer>ID</answer>, with no explanation or additional text."""


def describe_action(action: Action) -> str:
    if action.kind is ActionKind.DISCARD:
        return f"discard card indices {', '.join(map(str, action.card_indices))}"
    if action.kind in {ActionKind.BET, ActionKind.RAISE}:
        return f"{action.kind.value} to {action.amount} stones"
    return action.kind.value.replace("_", " ")


def render_observation(observation: Observation) -> str:
    """Render only the acting seat's observation, never a full game state."""
    codec = ActionCodec(observation.target_score)
    if len(observation.legal_action_mask) != codec.size:
        raise ValueError("Observation mask does not match its target score")
    partner = Seat((int(observation.seat) + 2) % 4)
    lines = [
        f"Hand: {observation.hand_number + 1}",
        f"You: {observation.seat.name} (team {observation.team}); partner: {partner.name}",
        f"Mano: {observation.mano.name}; mano_offset: {(observation.seat - observation.mano) % 4}",
        f"Target: {observation.target_score} stones; scores: team 0={observation.scores[0]}, team 1={observation.scores[1]}",
        f"Phase: {observation.phase.value}; lance: {observation.current_lance.value if observation.current_lance else 'none'}",
        f"Current stake: {observation.stake}; bidding team: {observation.bidding_team}; fold awards: {observation.refusal_points}",
        "Your cards (index: physical card; Mus rank; juego value):",
    ]
    lines.extend(
        f"  {index}: {int(card.rank)} of {card.suit.value}; rank={card.mus_rank}; value={card.juego_points}"
        for index, card in enumerate(observation.private_hand)
    )
    lines.append(
        f"Discard counts (North, East, South, West): {observation.discard_counts}"
    )
    lines.append("Public history:")
    for event in observation.public_events:
        actor = event.actor.name if event.actor is not None else "table"
        detail = ", ".join(f"{key}={value}" for key, value in event.detail)
        lines.append(f"  {event.index}: {actor} {event.kind} {detail}".rstrip())
    lines.append("Legal actions (ID: meaning):")
    lines.extend(
        f"  {action_id}: {describe_action(codec.decode(action_id))}"
        for action_id, allowed in enumerate(observation.legal_action_mask)
        if allowed
    )
    return "\n".join(lines)


def render_messages(observation: Observation) -> list[dict[str, str]]:
    offset = (observation.seat - observation.mano) % 4
    role = f"You are {observation.seat.name}, team {observation.team}, mano_offset {offset}."
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n{role}"},
        {"role": "user", "content": render_observation(observation)},
    ]


def render_prompt(observation: Observation) -> str:
    """Full readable prompt for collect_episode(..., render=render_prompt)."""
    return "\n\n".join(
        f"[{message['role']}]\n{message['content']}"
        for message in render_messages(observation)
    )

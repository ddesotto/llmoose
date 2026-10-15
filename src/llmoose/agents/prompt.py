"""Versioned prompts shared by model inference and dataset collection."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

from llmoose.game.actions import Action, ActionCodec, ActionKind
from llmoose.game.state import Seat
from llmoose.observations.core import Observation
from llmoose.rules.ruleset import FEDERATION_RULESET_ID

PROMPT_VERSION = 3
PROMPT_VERSIONS = (1, 2, 3)
RENDERER_VERSION = 2


@lru_cache(maxsize=None)
def load_system_prompt(version: int = PROMPT_VERSION) -> str:
    """Load shared, immutable game instructions from a packaged version."""
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"Unknown prompt version: {version}")
    return (
        files("llmoose.agents")
        .joinpath("system_prompts", f"v{version}.txt")
        .read_text(encoding="utf-8")
        .strip()
    )


SYSTEM_PROMPT = load_system_prompt()


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
        f"Ruleset: {observation.ruleset_id}",
        f"Hand: {observation.hand_number + 1}",
        f"You: {observation.seat.name} (team {observation.team}); partner: {partner.name}",
        f"Mano: {observation.mano.name}; mano_offset: {(observation.seat - observation.mano) % 4}",
        f"Target: {observation.target_score} stones; scores: team 0={observation.scores[0]}, team 1={observation.scores[1]}",
        f"Phase: {observation.phase.value}; lance: {observation.current_lance.value if observation.current_lance else 'none'}",
        f"Current stake: {observation.stake}; bidding team: {observation.bidding_team}; fold awards: {observation.refusal_points}",
    ]
    if observation.ruleset_id == FEDERATION_RULESET_ID:
        lines.extend(
            [
                f"Outstanding ordago: {observation.is_ordago}",
                "Scoring: scores include paid awards only. Accepted numeric bets and unbet "
                "lances wait for showdown; they have not yet been added. Pares/juego "
                "intrinsic values and the punto point add to called stakes. Folding pays "
                "the refusal now and concedes those intrinsic values at showdown. "
                "Called ordago wins immediately and cancels pending awards.",
                "Pending lances (settle in listed order at showdown):",
            ]
        )
        lines.extend(
            f"  {p.lance.value}: {p.reason}; accepted stake={p.stake}; "
            f"eligible={','.join(seat.name for seat in p.eligible)}"
            for p in observation.pending_lances
        )
        if not observation.pending_lances:
            lines.append("  none")
    else:
        lines.append(
            "Legacy scoring: resolve each lance immediately. Called bets pay the stake "
            "alone; folds pay refusal alone. Numeric target-sized bets and ordago "
            "are equivalent. Natural values apply only when passed or uncontested."
        )
    lines.append("Your cards (index: physical card; Mus rank; juego value):")
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


def render_messages(
    observation: Observation, *, prompt_version: int = PROMPT_VERSION
) -> list[dict[str, str]]:
    offset = (observation.seat - observation.mano) % 4
    role = f"You are {observation.seat.name}, team {observation.team}, mano_offset {offset}."
    return [
        {"role": "system", "content": f"{load_system_prompt(prompt_version)}\n{role}"},
        {"role": "user", "content": render_observation(observation)},
    ]


def render_prompt(
    observation: Observation, *, prompt_version: int = PROMPT_VERSION
) -> str:
    """Full readable prompt for collect_episode(..., render=render_prompt)."""
    return "\n\n".join(
        f"[{message['role']}]\n{message['content']}"
        for message in render_messages(observation, prompt_version=prompt_version)
    )

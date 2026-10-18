"""Versioned prompts shared by model inference and dataset collection."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

from llmoose.game.actions import Action, ActionCodec, ActionKind
from llmoose.game.state import Seat
from llmoose.observations.core import Observation
from llmoose.rules.ruleset import FEDERATION_RULESET_ID

PROMPT_VERSION = 5
PROMPT_VERSIONS = (1, 2, 3, 4, 5)
RENDERER_VERSION = 4
PRIOR_HAND_SUMMARY_LIMIT = 3


def renderer_version_for_prompt(version: int) -> int:
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"Unknown prompt version: {version}")
    return 2 if version <= 3 else (3 if version == 4 else 4)


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


def _render_observation_v2(observation: Observation) -> str:
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


def _current_and_prior_events(observation: Observation):
    events = observation.public_events
    current_start = max(
        (index for index, event in enumerate(events) if event.kind == "dealt"),
        default=0,
    )
    return events[:current_start], events[current_start:]


def _prior_hand_summaries(events):
    groups = []
    for event in events:
        if event.kind == "dealt":
            groups.append([event])
        elif groups:
            groups[-1].append(event)
    omitted = max(0, len(groups) - PRIOR_HAND_SUMMARY_LIMIT)
    lines = [
        f"Earlier public hands: {len(groups)} recorded; last "
        f"{min(len(groups), PRIOR_HAND_SUMMARY_LIMIT)} summarized; {omitted} older omitted."
    ]
    for group in groups[-PRIOR_HAND_SUMMARY_LIMIT:]:
        hand_number = dict(group[0].detail).get("hand_number", "?")
        award_kind = (
            "score_awarded"
            if any(event.kind == "score_awarded" for event in group)
            else "lance_resolved"
        )
        awards = [
            f"{detail.get('lance')}→team {detail.get('team')} +{detail.get('points')}"
            for event in group
            if event.kind == award_kind
            for detail in (dict(event.detail),)
        ]
        bids = sum(event.kind == "bid" for event in group)
        folds = sum(event.kind == "fold" for event in group)
        lines.append(
            f"  hand {int(hand_number) + 1}: bids={bids}, folds={folds}; "
            f"awards={', '.join(awards) if awards else 'none'}"
        )
    return lines


def _describe_action_v3(observation: Observation, action: Action) -> str:
    kind = action.kind
    v2 = observation.ruleset_id == FEDERATION_RULESET_ID
    lance = observation.current_lance
    intrinsic = lance is not None and lance.value in {"pares", "juego", "punto"}
    if kind is ActionKind.MUS:
        return "mus — vote to exchange cards if all four seats agree"
    if kind is ActionKind.NO_MUS:
        return "no mus — stop exchanges and begin grande now"
    if kind is ActionKind.DISCARD:
        indices = ", ".join(map(str, action.card_indices))
        return f"discard indices {indices} — exchange these cards; indices refer to the current hand"
    if kind is ActionKind.PASS:
        return "pass — decline to open a bet; remain eligible to win this lance"
    if kind in {ActionKind.BET, ActionKind.RAISE}:
        refusal = observation.stake if kind is ActionKind.RAISE else 1
        verb = "raise" if kind is ActionKind.RAISE else "bet"
        return (
            f"{verb} to {action.amount} total — next eligible opponent answers for "
            f"their team; if they fold, your team receives {refusal} now"
        )
    if kind is ActionKind.CALL:
        if v2 and observation.is_ordago:
            return "call — accept órdago; best eligible hand wins the match now"
        timing = "at showdown" if v2 else "now"
        addition = (
            "; winning team also gets intrinsic value" if v2 and intrinsic else ""
        )
        return f"call — accept {observation.stake} total; settle {timing}{addition}"
    if kind is ActionKind.FOLD:
        addition = (
            "; it also gets intrinsic value at showdown if play continues"
            if v2 and intrinsic
            else ""
        )
        return (
            f"fold — concede for your team; bidding team {observation.bidding_team} "
            f"receives {observation.refusal_points} now{addition}"
        )
    if kind is ActionKind.ORDAGO:
        refusal = observation.stake or 1
        return (
            "órdago — propose a match deciding showdown; opponent may call or fold; "
            f"a fold gives your team {refusal} now"
        )
    if kind in {ActionKind.DECLARE_PARES, ActionKind.DECLARE_JUEGO}:
        return f"{kind.value.replace('_', ' ')} — truthful eligibility declaration"
    return f"{kind.value.replace('_', ' ')} — truthful ineligibility declaration"


def _render_observation_v3(observation: Observation, *, complete: bool = False) -> str:
    codec = ActionCodec(observation.target_score)
    if len(observation.legal_action_mask) != codec.size:
        raise ValueError("Observation mask does not match its target score")
    partner = Seat((int(observation.seat) + 2) % 4)
    order = tuple(Seat((int(observation.mano) + offset) % 4) for offset in range(4))
    prior_events, current_events = _current_and_prior_events(observation)
    lines = [
        f"Ruleset: {observation.ruleset_id}; renderer: {4 if complete else 3}",
        f"Hand: {observation.hand_number + 1}",
        f"You: {observation.seat.name} (team {observation.team}); partner: {partner.name}",
        f"Mano: {observation.mano.name}; tie priority: {' → '.join(seat.name for seat in order)}; mano_offset: {(observation.seat - observation.mano) % 4}",
        f"Target: {observation.target_score} stones; scores: team 0={observation.scores[0]}, team 1={observation.scores[1]}",
        "Scores include awards already paid; pending awards are excluded.",
        f"Phase: {observation.phase.value}; active lance: {observation.current_lance.value if observation.current_lance else 'none'}",
        f"Eligible seats for active lance: {', '.join(seat.name for seat in observation.lance_seats) if observation.lance_seats else 'not determined yet'}",
        f"Current stake: {observation.stake}; bidding team: {observation.bidding_team}; fold awards: {observation.refusal_points}",
    ]
    if complete and observation.current_lance is None:
        lines.append("No bidding in this phase. Choose only from the legal menu below.")
    elif observation.stake:
        lines.append(
            f"Binding response: {observation.current_seat.name} acts for their whole team on this bid; their partner cannot veto."
        )
    else:
        lines.append(
            "No outstanding bid; an opening bet is answered by the next eligible opponent."
        )
    if observation.ruleset_id == FEDERATION_RULESET_ID:
        lines.extend(
            [
                f"Outstanding ordago: {observation.is_ordago}",
                "Accepted numeric bets and unbet awards settle at showdown in grande, chica, pares, juego/punto order. Called pares/juego/punto add intrinsic value. A called órdago decides the match and cancels pending awards.",
                "Pending lances (winner not disclosed; settle in listed order):",
            ]
        )
        lines.extend(
            f"  {p.lance.value}: {p.reason}; accepted stake={p.stake}; eligible={','.join(seat.name for seat in p.eligible)}"
            for p in observation.pending_lances
        )
        if not observation.pending_lances:
            lines.append("  none")
    else:
        lines.append(
            "Legacy scoring: each lance resolves immediately; called bets pay stake alone, folds pay refusal alone, and target-sized numeric bets equal órdago."
        )
    lines.append("Your cards (index: physical card; Mus rank; juego value):")
    lines.extend(
        f"  {index}: {int(card.rank)} of {card.suit.value}; rank={card.mus_rank}; value={card.juego_points}"
        for index, card in enumerate(observation.private_hand)
    )
    lines.append(
        "Your earlier discards this hand (seen cards, not necessarily out of play):"
    )
    lines.extend(
        f"  exchange {index}: "
        + ", ".join(f"{int(card.rank)} of {card.suit.value}" for card in cards)
        for index, cards in enumerate(observation.own_discard_history, 1)
    )
    if not observation.own_discard_history:
        lines.append("  none")
    lines.append(
        f"Discard counts this hand, cumulative (North, East, South, West): {observation.discard_counts}"
    )
    if complete:
        lines.append("Earlier public history (complete, including public showdown cards):")
        for event in prior_events:
            actor = event.actor.name if event.actor is not None else "table"
            detail = ", ".join(f"{key}={value}" for key, value in event.detail)
            lines.append(f"  {event.index}: {actor} {event.kind} {detail}".rstrip())
    else:
        lines.extend(_prior_hand_summaries(prior_events))
    lines.append("Current hand public history (complete):")
    for event in current_events:
        actor = event.actor.name if event.actor is not None else "table"
        detail = ", ".join(f"{key}={value}" for key, value in event.detail)
        lines.append(f"  {event.index}: {actor} {event.kind} {detail}".rstrip())
    if complete:
        lines.append(
            f"Decision: choose exactly ONE action for {observation.seat.name} in "
            f"{observation.phase.value}. This is a state snapshot, not a transcript to continue. "
            "Do not simulate turns, invent cards/events, or redefine IDs. "
            "Return only <answer>ID</answer> using the menu below."
        )
    lines.append("Legal actions (ID: meaning):")
    lines.extend(
        f"  {action_id}: {_describe_action_v3(observation, codec.decode(action_id))}"
        for action_id, allowed in enumerate(observation.legal_action_mask)
        if allowed
    )
    return "\n".join(lines)


def render_observation(
    observation: Observation, *, renderer_version: int = RENDERER_VERSION
) -> str:
    """Render only this seat's observation with a pinned history policy."""
    if renderer_version == 2:
        return _render_observation_v2(observation)
    if renderer_version == 4:
        return _render_observation_v3(observation, complete=True).replace(
            "<answer>ID</answer>", '{"action_id": ID}'
        ).replace(
            "Pending lances (winner not disclosed; settle in listed order):",
            "Pending settlement terms (hidden hand comparisons not provided):",
        )
    if renderer_version == 3:
        return _render_observation_v3(observation)
    raise ValueError(f"Unknown renderer version: {renderer_version}")


def render_messages(
    observation: Observation, *, prompt_version: int = PROMPT_VERSION
) -> list[dict[str, str]]:
    offset = (observation.seat - observation.mano) % 4
    role = f"You are {observation.seat.name}, team {observation.team}, mano_offset {offset}."
    return [
        {"role": "system", "content": f"{load_system_prompt(prompt_version)}\n{role}"},
        {
            "role": "user",
            "content": render_observation(
                observation,
                renderer_version=renderer_version_for_prompt(prompt_version),
            ),
        },
    ]


def render_prompt(
    observation: Observation, *, prompt_version: int = PROMPT_VERSION
) -> str:
    """Full readable prompt for collect_episode(..., render=render_prompt)."""
    return "\n\n".join(
        f"[{message['role']}]\n{message['content']}"
        for message in render_messages(observation, prompt_version=prompt_version)
    )

import json
from dataclasses import replace
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from llmoose.agents.baselines import ConservativePolicy
from llmoose.agents.llm import InvalidModelAction, StructuredLLMPolicy, parse_action
from llmoose.agents.prompt import render_messages, render_observation, render_prompt
from llmoose.agents.providers import ChatCompletionProvider, Completion, ProviderError
from llmoose.game.actions import Action, ActionCodec, ActionKind
from llmoose.game.engine import apply_action
from llmoose.game.environment import MusEnv
from llmoose.game.state import SEATS, Event, Phase, Seat, freeze_hands, new_game
from llmoose.observations.core import observe
from llmoose.rules.ruleset import Ruleset
from llmoose.training import collect_episode


def opening():
    env = MusEnv(Ruleset(target_score=8), episode="hand")
    return env, env.reset(seed=12)


def test_prompts_include_position_cards_and_only_legal_choices():
    env, observation = opening()
    prompt = render_observation(observation)
    assert "You: WEST (team 1); partner: EAST" in prompt
    assert (
        "Mano: WEST; tie priority: WEST → NORTH → EAST → SOUTH; mano_offset: 0"
        in prompt
    )
    assert "Target: 8 stones" in prompt
    for index, card in enumerate(observation.private_hand):
        assert f"{index}: {int(card.rank)} of {card.suit.value}" in prompt
    actions = prompt.split("Legal actions (ID: meaning):\n")[1]
    assert actions == (
        "  0: mus — vote to exchange cards if all four seats agree\n"
        "  1: no mus — stop exchanges and begin grande now"
    )
    messages = render_messages(observation)
    assert messages[1]["content"] == prompt
    assert "mano_offset 0" in messages[0]["content"]
    assert messages[0]["content"] in render_prompt(observation)
    assert env.codec.encode(Action(ActionKind.NO_MUS)) == 1


def test_prompt_does_not_change_when_opponents_private_cards_change():
    state = new_game(seed=12, ruleset=Ruleset(target_score=8))
    codec = ActionCodec(8)
    hands = dict(state.hands)
    hands[Seat.NORTH], hands[Seat.SOUTH] = hands[Seat.SOUTH], hands[Seat.NORTH]
    changed = replace(
        state, hands=freeze_hands(hands), stock=tuple(reversed(state.stock)), seed=999
    )
    assert render_messages(observe(state, Seat.WEST, codec)) == render_messages(
        observe(changed, Seat.WEST, codec)
    )
    prompt = render_prompt(observe(state, Seat.WEST, codec))
    for seat in (Seat.NORTH, Seat.EAST, Seat.SOUTH):
        for card in state.hands[seat]:
            assert f"{int(card.rank)} of {card.suit.value}" not in prompt


def test_observation_exposes_public_bidding_context():
    env, _ = opening()
    env.step(Action(ActionKind.NO_MUS))
    observation = env.step(Action(ActionKind.BET, amount=3)).observation
    assert observation.stake == 3
    assert observation.bidding_team == 1
    assert observation.refusal_points == 1
    assert "Current stake: 3; bidding team: 1; fold awards: 1" in render_observation(
        observation
    )
    rendered = render_observation(observation)
    assert "Eligible seats for active lance: WEST, NORTH, EAST, SOUTH" in rendered
    assert "Binding response: NORTH acts for their whole team" in rendered
    assert "fold — concede for your team; bidding team 1 receives 1 now" in rendered


def test_private_discard_recall_is_seat_scoped_and_resets_on_new_hand():
    state = new_game(12, Ruleset(target_score=40))
    codec = ActionCodec(40)
    old_card = state.hands[Seat.WEST][0]
    for _ in range(4):
        state = apply_action(state, Action(ActionKind.MUS))
    state = apply_action(state, Action(ActionKind.DISCARD, card_indices=(0,)))
    west = observe(state, Seat.WEST, codec)
    east = observe(state, Seat.EAST, codec)
    assert west.own_discard_history == ((old_card,),)
    assert east.own_discard_history == ()
    seen = f"{int(old_card.rank)} of {old_card.suit.value}"
    assert seen in render_observation(west).split("Your earlier discards this hand")[1]
    assert seen not in render_observation(east)
    policy = ConservativePolicy(codec)
    start_hand = state.hand_number
    for _ in range(80):
        if state.hand_number != start_hand or state.phase is Phase.COMPLETE:
            break
        actor = state.current_seat
        state = apply_action(state, policy.act(observe(state, actor, codec)))
    assert state.hand_number == start_hand + 1
    assert state.own_discard_history == ((), (), (), ())


def test_prior_history_is_bounded_and_current_hand_history_is_complete():
    _, observation = opening()
    events = []
    for hand_number in range(6):
        events.append(
            Event(
                len(events),
                "dealt",
                detail=(("hand_number", str(hand_number)),),
            )
        )
        events.append(Event(len(events), "pass", Seat.WEST))
    observation = replace(observation, public_events=tuple(events), hand_number=5)
    rendered = render_observation(observation)
    assert (
        "Earlier public hands: 5 recorded; last 3 summarized; 2 older omitted."
        in rendered
    )
    assert "hand 3: bids=0, folds=0" in rendered
    assert "hand 2: bids=0" not in rendered
    assert f"  {events[-1].index}: WEST pass" in rendered
    assert "Current hand public history (complete):" in rendered


def test_near_target_ordago_menu_explains_match_and_refusal():
    state = replace(new_game(12, Ruleset(target_score=40)), scores=(39, 39))
    state = apply_action(state, Action(ActionKind.NO_MUS))
    state = apply_action(state, Action(ActionKind.ORDAGO))
    rendered = render_observation(observe(state, state.current_seat, ActionCodec(40)))
    assert "team 0=39, team 1=39" in rendered
    assert "call — accept órdago; best eligible hand wins the match now" in rendered
    assert "fold — concede for your team; bidding team 1 receives 1 now" in rendered
    assert "Current hand public history (complete):" in rendered


@pytest.mark.parametrize(
    "text",
    [
        "1",
        "<answer>99</answer>",
        "<answer>-1</answer>",
        "<answer>1</answer> explanation",
        "<answer>1</answer><answer>0</answer>",
        "<answer>１</answer>",
        "<answer>" + "9" * 5000 + "</answer>",
    ],
)
def test_action_parser_rejects_invalid_or_ambiguous_responses(text):
    with pytest.raises(ValueError):
        parse_action(text, (0, 1))


def test_policy_retries_and_records_the_exact_corrective_conversation():
    env, obs = opening()
    responses = iter(
        [
            Completion("<answer>99</answer>"),
            Completion(" <answer>1</answer>\n", model="actual-model"),
        ]
    )
    calls, records = [], []

    def provider(messages):
        calls.append([dict(message) for message in messages])
        return next(responses)

    policy = StructuredLLMPolicy(env.codec, provider, on_record=records.append)
    assert policy.act(obs).kind is ActionKind.NO_MUS
    assert len(calls[0]) == 2
    assert calls[1][-2]["content"] == "<answer>99</answer>"
    assert records[0]["messages"] == calls[0]
    assert records[1]["messages"] == calls[1]
    assert records[1]["model"] == "actual-model"
    assert policy.telemetry.requests == 2
    assert policy.telemetry.invalid_outputs == 1
    assert policy.telemetry.fallbacks == 0


def test_eval_fallback_is_explicit_and_strict_mode_stops():
    env, obs = opening()
    records = []
    policy = StructuredLLMPolicy(
        env.codec, lambda _: Completion("bad"), on_record=records.append
    )
    assert policy.act(obs).kind is ActionKind.NO_MUS
    assert policy.telemetry.requests == policy.telemetry.invalid_outputs == 3
    assert policy.telemetry.fallbacks == 1
    assert records[-1]["status"] == "fallback"
    strict = StructuredLLMPolicy(
        env.codec, lambda _: Completion("bad"), max_retries=0, fallback_on_invalid=False
    )
    with pytest.raises(InvalidModelAction):
        strict.act(obs)
    assert strict.telemetry.requests == strict.telemetry.invalid_outputs == 1
    assert strict.telemetry.fallbacks == 0


def test_provider_failure_is_not_retried_or_replaced_by_a_game_action():
    env, obs = opening()

    def failed(_):
        raise ProviderError("connection failed")

    records = []
    policy = StructuredLLMPolicy(env.codec, failed, on_record=records.append)
    with pytest.raises(ProviderError):
        policy.act(obs)
    assert policy.telemetry.provider_errors == 1
    assert policy.telemetry.requests == 1
    assert policy.telemetry.invalid_outputs == policy.telemetry.fallbacks == 0
    assert records[-1]["status"] == "provider_error"


def test_forced_declarations_do_not_call_the_provider():
    state = replace(new_game(12, Ruleset(target_score=8)), phase=Phase.DECLARE_PARES)
    codec = ActionCodec(8)
    records = []

    def unexpected(_):
        pytest.fail("Forced declaration should not call inference")

    policy = StructuredLLMPolicy(codec, unexpected, on_record=records.append)
    obs = observe(state, state.current_seat, codec)
    assert obs.legal_action_mask[codec.encode(policy.act(obs))]
    assert "truthful" in render_observation(obs)
    assert policy.telemetry.forced_actions == 1
    assert policy.telemetry.requests == 0
    assert records[0]["status"] == "forced"


def test_same_renderer_works_for_training_collection():
    from llmoose.agents.baselines import ConservativePolicy

    env, _ = opening()
    episodes = collect_episode(
        env,
        {seat: ConservativePolicy(env.codec) for seat in SEATS},
        render=render_prompt,
    )
    assert all(
        "[system]" in turn.prompt_text and "Legal actions" in turn.prompt_text
        for episode in episodes
        for turn in episode.turns
    )


def install_transport(monkeypatch, response=None, error=None):
    calls = []

    class FakeOpener:
        def open(self, request, timeout):
            calls.append((request, timeout))
            if error:
                raise error
            return BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(
        "llmoose.agents.providers.build_opener", lambda *_: FakeOpener()
    )
    return calls


def test_openrouter_request_and_completion_metadata(monkeypatch):
    calls = install_transport(
        monkeypatch,
        {
            "model": "resolved/model",
            "choices": [
                {"message": {"content": "<answer>1</answer>"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 8, "cost": 0.001},
        },
    )
    provider = ChatCompletionProvider(
        "requested/model", api_key="test-secret", max_tokens=256
    )
    result = provider([{"role": "user", "content": "test"}])
    request, timeout = calls[0]
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert json.loads(request.data)["model"] == "requested/model"
    assert json.loads(request.data)["max_tokens"] == 256
    assert timeout == 60
    assert result.model == "resolved/model"
    assert result.usage["cost"] == 0.001
    assert "test-secret" not in repr(provider)


@pytest.mark.parametrize("include_reasoning", [False, True])
def test_reasoning_is_preserved_separately_from_the_action(
    monkeypatch, include_reasoning
):
    details = [
        {"type": "reasoning.summary", "summary": "The current hand favors cutting mus."}
    ]
    calls = install_transport(
        monkeypatch,
        {
            "choices": [
                {
                    "message": {
                        "content": "<answer>1</answer>",
                        "reasoning": "The current hand favors cutting mus.",
                        "reasoning_details": details,
                    }
                }
            ],
            "usage": {
                "completion_tokens": 40,
                "completion_tokens_details": {"reasoning_tokens": 30},
            },
        },
    )
    env, observation = opening()
    records = []
    provider = ChatCompletionProvider(
        "model", api_key="test", include_reasoning=include_reasoning
    )
    policy = StructuredLLMPolicy(env.codec, provider, on_record=records.append)
    assert policy.act(observation).kind is ActionKind.NO_MUS
    body = json.loads(calls[0][0].data)
    if include_reasoning:
        assert body["reasoning"] == {"enabled": True, "exclude": False}
    else:
        assert "reasoning" not in body
    assert records[0]["reasoning"] == "The current hand favors cutting mus."
    assert records[0]["reasoning_details"] == details
    assert records[0]["response"] == "<answer>1</answer>"


@pytest.mark.parametrize(
    "payload",
    [
        {"error": {"message": "private detail"}},
        {"choices": []},
        {"choices": [{"error": {"code": 500}}]},
        {"choices": [{"message": {"content": []}}]},
        [],
    ],
)
def test_malformed_or_error_envelopes_stop_inference(monkeypatch, payload):
    install_transport(monkeypatch, payload)
    with pytest.raises(ProviderError):
        ChatCompletionProvider("model", base_url="http://localhost:8000/v1")([])


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("https://example.test", 401, "secret", {}, None),
        URLError("secret"),
        TimeoutError("secret"),
    ],
)
def test_transport_errors_are_readable_and_do_not_expose_credentials(
    monkeypatch, error
):
    install_transport(monkeypatch, error=error)
    with pytest.raises(ProviderError) as raised:
        ChatCompletionProvider("model", api_key="test-secret")([])
    assert "secret" not in str(raised.value)


def test_empty_model_response_is_recorded_as_invalid_not_transport_failure(monkeypatch):
    install_transport(
        monkeypatch,
        {"choices": [{"message": {"content": None}, "finish_reason": "length"}]},
    )
    result = ChatCompletionProvider("model", api_key="test-secret")([])
    assert result.text == ""
    assert result.finish_reason == "length"

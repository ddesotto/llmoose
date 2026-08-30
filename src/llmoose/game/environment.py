"""A small reset/step interface for self-play and reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Tuple, Union

from llmoose.game.actions import Action, ActionCodec
from llmoose.game.engine import apply_action
from llmoose.game.state import GameState, Phase, Seat, deal_hand, dealer_for, new_game
from llmoose.observations.core import Observation, observe
from llmoose.rules.ruleset import Ruleset

ActionInput = Union[Action, int]


class EnvEpisode(str, Enum):
    """How much play one episode covers: a whole match, or a single hand."""

    MATCH = "match"
    HAND = "hand"


@dataclass(frozen=True)
class StepResult:
    """One transition, with rewards expressed as score deltas for both teams."""

    observation: Optional[Observation]
    reward: Tuple[float, float]
    terminated: bool
    info: Mapping[str, object]


class MusEnv:
    """Sequential multi-agent Mus environment with a finite masked action space.

    ``reset`` and ``step`` return the observation for the next acting seat. The
    observation itself identifies that seat, so one shared policy can be used
    for self-play or callers can route it to one of four policies. Rewards are
    per-team score changes caused by the submitted action.
    """

    def __init__(
        self,
        ruleset: Optional[Ruleset] = None,
        seed: int = 0,
        dealer: Seat = Seat.NORTH,
        episode: Union[EnvEpisode, str] = EnvEpisode.MATCH,
    ) -> None:
        self.ruleset = ruleset or Ruleset()
        self.codec = ActionCodec(self.ruleset.target_score)
        self.seed = seed
        self.initial_dealer = dealer
        self.episode = EnvEpisode(episode)
        self._state: Optional[GameState] = None
        self._hand_start_scores: Tuple[int, int] = (0, 0)
        self._episode_hand: int = 0

    @property
    def state(self) -> GameState:
        if self._state is None:
            raise RuntimeError("Call reset() before reading environment state")
        return self._state

    @property
    def action_size(self) -> int:
        return self.codec.size

    def reset(
        self,
        seed: Optional[int] = None,
        dealer: Optional[Seat] = None,
        hand_number: int = 0,
    ) -> Observation:
        """Deal an episode's opening state and return the first actor's observation."""

        if hand_number < 0:
            raise ValueError("hand_number must be non-negative")
        if seed is not None:
            self.seed = seed
        if dealer is not None:
            self.initial_dealer = dealer

        if hand_number == 0:
            self._state = new_game(self.seed, self.ruleset, self.initial_dealer)
        else:
            self._state = deal_hand(
                seed=self.seed,
                ruleset=self.ruleset,
                dealer=dealer_for(self.initial_dealer, hand_number),
                hand_number=hand_number,
            )
        self._hand_start_scores = self._state.scores
        self._episode_hand = self._state.hand_number
        return self._next_observation()

    def step(self, action: ActionInput) -> StepResult:
        """Apply an action object or action ID for the currently acting seat. """

        before = self.state
        if before.phase is Phase.COMPLETE:
            raise RuntimeError("Cannot step a completed match; call reset()")
        structured_action = (
            self.codec.decode(action) if isinstance(action, int) else action
        )
        after = apply_action(before, structured_action)
        self._state = after
        reward = tuple(
            float(new - old) for new, old in zip(after.scores, before.scores)
        )
        match_over = after.phase is Phase.COMPLETE
        hand_over = match_over or after.hand_number != self._episode_hand
        terminated = match_over or (hand_over and self.episode is EnvEpisode.HAND)
        info = {
            "action": structured_action,
            "actor": before.current_seat,
            "scores": after.scores,
            "hand_number": before.hand_number,
            "lance_results": after.lance_results,
        }
        if hand_over:
            info["hand_reward"] = tuple(
                float(new - old)
                for new, old in zip(after.scores, self._hand_start_scores)
            )
            self._episode_hand = after.hand_number
            self._hand_start_scores = after.scores
        return StepResult(
            observation=None if terminated else self._next_observation(),
            reward=reward,
            terminated=terminated,
            info=info,
        )

    def _next_observation(self) -> Observation:
        actor = self.state.current_seat
        if actor is None:
            raise RuntimeError("A non-terminal environment must have an acting seat")
        return observe(self.state, actor, self.codec)

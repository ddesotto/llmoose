"""Minimal policy interface"""

from __future__ import annotations

from typing import Protocol

from llmoose.game.actions import Action
from llmoose.observations.core import Observation


class Policy(Protocol):
    """Choose only from the supplied mask; the engine still revalidates it."""

    def act(self, observation: Observation) -> Action:
        """Return one structured legal action."""

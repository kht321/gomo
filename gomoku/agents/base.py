"""Base agent class."""

from abc import ABC, abstractmethod
from typing import Tuple
from ..core.models import GameState


class Agent(ABC):
    """Abstract base class for all Gomoku agents."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.player = None  # Will be set when game starts
        self._setup()  # Each agent handles its own setup
    
    def _setup(self):
        """Internal setup method - override in subclasses if needed."""
        pass

    @abstractmethod
    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        """Return (row, col) for next move."""
        pass
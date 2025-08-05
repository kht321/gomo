"""Abstract interfaces for extensibility."""

from abc import ABC, abstractmethod
from typing import List, Tuple
from .models import GameState


class BoardFormatter(ABC):
    """Abstract interface for board formatting."""

    @abstractmethod
    def format_board(self, state: GameState) -> str:
        """Format board for display."""
        pass

    @abstractmethod
    def format_board_with_highlights(self, board: List[List[str]], highlights: List[Tuple[int, int]]) -> str:
        """Format board with highlighted positions."""
        pass

"""Simple rule-based agent."""

import random
from typing import Tuple
from .base import Agent
from ..core.models import GameState, Player
from ..core.game import GomokuGame


class SimpleGomokuAgent(Agent):
    """Simple agent with basic strategy - only needs agent_id."""
    
    def _setup(self):
        """Setup - nothing needed for simple agent."""
        pass
    
    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        """Simple agent with basic strategy."""
        # Get legal moves directly from game_state

        legal_moves = []
        for row in range(game_state.board_size):
            for col in range(game_state.board_size):
                if game_state.board[row][col] == Player.EMPTY.value:
                    legal_moves.append((row, col))
        
        # Safety check for empty legal moves
        if not legal_moves:
            # Should not happen, but fallback to center
            center = game_state.board_size // 2
            return (center, center)
        
        # Try center first, otherwise random
        center = game_state.board_size // 2
        if game_state.board[center][center] == Player.EMPTY.value:
            return (center, center)

        return random.choice(legal_moves)


# Alias for simpler naming
SimpleAgent = SimpleGomokuAgent
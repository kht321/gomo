"""LLM-powered Gomoku agent implementation."""

import json
import re
from typing import Tuple
from ..core.models import GameState
from ..llm.openai_client import OpenAIGomokuClient
from .base import Agent


class LLMGomokuAgent(Agent):
    """LLM-powered Gomoku agent that uses language models to make moves."""

    def _setup(self):
        """Internal setup method - configures LLM client and formatter."""

        self.llm_client = OpenAIGomokuClient(
            model="Qwen/Qwen2-7B-Instruct",
            endpoint="https://api.featherless.ai/v1",
            temperature=0.7,
            max_tokens=1024,
        )

        self.system_prompt: str = self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for Gomoku gameplay."""
        return """You are an expert Gomoku (Five in a Row) player. Your goal is to get 5 of your pieces in a row (horizontally, vertically, or diagonally) while preventing your opponent from doing the same.

Key strategies:
- Control the center of the board early
- Create multiple threats simultaneously
- Block opponent threats immediately
- Look for opportunities to create "forks" (multiple winning threats)

You must respond with valid JSON in this exact format, use ```json to wrap your response:
```json
{
    "reasoning": "<brief explanation of your move>",
    "move": {"row": <row_number>, "col": <col_number>}
}
```

The row and col must be valid coordinates on the board (0-indexed). Always choose empty positions (marked with '.')."""

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        """Get next move from LLM or fallback to simple strategy."""
        try:
            # Format the board state for the LLM
            board_str = game_state.format_board(formatter="standard")
            board_prompt = f"Current board state:\n{board_str}\n"
            board_prompt += f"Current player: {game_state.current_player.value}\n"
            board_prompt += f"Move count: {len(game_state.move_history)}\n"
            
            if game_state.move_history:
                last_move = game_state.move_history[-1]
                board_prompt += f"Last move: {last_move.player.value} at ({last_move.row}, {last_move.col})\n"

            # Create messages for the LLM
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{board_prompt}\n\nPlease provide your next move as JSON."},
            ]

            # Get response from LLM
            response = await self.llm_client.complete(messages)

            # Parse the response
            move = self._parse_move_response(response, game_state)

            return move

        except Exception as e:
            # Fallback to a safe random move if LLM fails
            print(f"LLM error for agent {self.agent_id}: {e}")
            return self._get_fallback_move(game_state)

    def _parse_move_response(self, response: str, game_state: GameState) -> Tuple[int, int]:
        """Parse LLM response to extract move coordinates."""
        try:

            # Try to extract JSON from response
            json_match = re.search(r"```json([^`]+)```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                data = json.loads(json_str)

                if "move" in data:
                    move = data["move"]
                    row, col = move["row"], move["col"]

                    # Validate move
                    if game_state.is_valid_move(row, col):
                        return (row, col)
                    else:
                        print(f"Invalid move from LLM: ({row}, {col}) - falling back to simple strategy")
                        return self._get_fallback_move(game_state)

        except Exception as e:
            print(f"JSON parsing error: {e}")

        # If all parsing fails, use fallback
        print(f"Could not parse LLM response:\n\n<RESPONSE>\n\n{response}\n\n</RESPONSE>")
        return self._get_fallback_move(game_state)

    def _get_fallback_move(self, game_state: GameState) -> Tuple[int, int]:
        """Get a fallback move when LLM fails."""
        # Try center first
        center = game_state.board_size // 2
        if game_state.is_valid_move(center, center):
            return (center, center)

        # Find any empty position using GameState method
        legal_moves = game_state.get_legal_moves()
        if legal_moves:
            return legal_moves[0]

        # This should never happen in a valid game
        raise RuntimeError("No valid moves available")

"""Core data models for Gomoku game."""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class Player(Enum):
    BLACK = "X"  # First player
    WHITE = "O"  # Second player
    EMPTY = "."


class GameResult(Enum):
    BLACK_WIN = "black_win"
    WHITE_WIN = "white_win"
    DRAW = "draw"
    INVALID_MOVE = "invalid_move"


@dataclass
class Move:
    row: int
    col: int
    player: Player


@dataclass
class GameState:
    board: List[List[str]]
    current_player: Player
    move_history: List[Move]
    board_size: int

    def copy(self):
        """Deep copy of game state"""
        return GameState(
            board=[row[:] for row in self.board],
            current_player=self.current_player,
            move_history=self.move_history[:],
            board_size=self.board_size,
        )

    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if move is valid."""
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        return self.board[row][col] == Player.EMPTY.value

    def is_board_full(self) -> bool:
        """Check if board is full (draw)."""
        for row in self.board:
            if Player.EMPTY.value in row:
                return False
        return True

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Get all legal moves."""
        moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == Player.EMPTY.value:
                    moves.append((row, col))
        return moves

    def _check_direction(self, row: int, col: int, dr: int, dc: int, player: str) -> int:
        """Count consecutive pieces in one direction."""
        count = 0
        r, c = row + dr, col + dc

        while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == player:
            count += 1
            r += dr
            c += dc

        return count

    def check_win_at_position(self, row: int, col: int, win_condition: int = 5) -> bool:
        """Check if placing a piece at (row, col) creates a win."""
        if self.board[row][col] == Player.EMPTY.value:
            return False

        player = self.board[row][col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, diagonals

        for dr, dc in directions:
            count = 1  # Current piece
            count += self._check_direction(row, col, dr, dc, player)
            count += self._check_direction(row, col, -dr, -dc, player)

            if count >= win_condition:
                return True

        return False

    def format_board(self, formatter: str = "standard") -> str:
        """Format the board using specified formatter keyword.

        Args:
            formatter: Formatter type ('standard', 'compact', 'natural', 'json', 'strategic')

        Returns:
            Formatted board string
        """
        return _create_prompt_formatter(formatter).format_board(self)


class BoardFormatter(ABC):
    """Abstract base class for GameState prompt formatters."""

    @abstractmethod
    def format_board(self, state: GameState) -> str:
        """Format the board for LLM consumption."""
        pass


class StandardGridFormatter(BoardFormatter):
    """Standard grid format with coordinates (current implementation)."""

    def format_board(self, state: GameState) -> str:
        """Standard coordinate grid format."""
        result = "   "
        for col in range(state.board_size):
            result += f"{col:2} "
        result += "\n"

        for row in range(state.board_size):
            result += f"{row:2} "
            for col in range(state.board_size):
                result += f" {state.board[row][col]} "
            result += "\n"

        return result


class CompactFormatter(BoardFormatter):
    """Compact format optimized for token efficiency."""

    def format_board(self, state: GameState) -> str:
        """Compact row-by-row format."""
        result = []
        for row in range(state.board_size):
            row_str = "".join(state.board[row])
            result.append(f"R{row}: {row_str}")
        return "\n".join(result)


class NaturalLanguageFormatter(BoardFormatter):
    """Natural language description format."""

    def format_board(self, state: GameState) -> str:
        """Describe board state in natural language."""
        pieces = {"X": [], "O": []}

        for row in range(state.board_size):
            for col in range(state.board_size):
                piece = state.board[row][col]
                if piece in pieces:
                    pieces[piece].append(f"({row},{col})")

        description = []
        if pieces["X"]:
            description.append(f"Black pieces at: {', '.join(pieces['X'])}")
        if pieces["O"]:
            description.append(f"White pieces at: {', '.join(pieces['O'])}")

        if not description:
            description.append("Empty board")

        return "\n".join(description)


class JSONFormatter(BoardFormatter):
    """JSON format for structured LLM processing."""

    def format_board(self, state: GameState) -> str:
        """JSON representation of board."""
        board_data = {
            "board": state.board,
            "size": state.board_size,
            "current_player": state.current_player.value,
            "move_count": len(state.move_history),
        }

        import json

        return json.dumps(board_data, indent=2)

    def format_full_prompt(self, state: GameState, context: Optional[Dict] = None) -> str:
        """Full JSON format."""
        data = {
            "board": state.board,
            "board_size": state.board_size,
            "current_player": state.current_player.value,
            "move_count": len(state.move_history),
            "last_moves": [],
        }

        # Add recent moves
        recent_moves = state.move_history[-5:] if len(state.move_history) > 5 else state.move_history
        for move in recent_moves:
            data["last_moves"].append({"row": move.row, "col": move.col, "player": move.player.value})

        # Add context if provided
        if context:
            data.update(context)

        import json

        return json.dumps(data, indent=2)


class StrategicFormatter(BoardFormatter):
    """Strategic format emphasizing patterns and formations."""

    def format_board(self, state: GameState) -> str:
        """Board with strategic pattern highlighting."""
        result = self._add_coordinates_header(state.board_size)

        for row in range(state.board_size):
            result += f"{row:2} "
            for col in range(state.board_size):
                piece = state.board[row][col]
                # Highlight center positions
                if self._is_center_region(row, col, state.board_size):
                    result += f"[{piece}]"
                else:
                    result += f" {piece} "
            result += "\n"

        return result

    def _add_coordinates_header(self, board_size: int) -> str:
        """Add coordinate header."""
        result = "   "
        for col in range(board_size):
            result += f"{col:2} "
        result += "\n"
        return result

    def _is_center_region(self, row: int, col: int, board_size: int) -> bool:
        """Check if position is in center region."""
        center = board_size // 2
        return abs(row - center) <= 2 and abs(col - center) <= 2


# Utility function for easy access
def _create_prompt_formatter(format_type: str = "standard", **kwargs) -> BoardFormatter:
    """Create a prompt formatter of the specified type."""
    formatters = {
        "standard": StandardGridFormatter,
        "compact": CompactFormatter,
        "natural": NaturalLanguageFormatter,
        "json": JSONFormatter,
        "strategic": StrategicFormatter,
    }

    if format_type not in formatters:
        raise ValueError(f"Unknown format type: {format_type}. Available: {list(formatters.keys())}")

    return formatters[format_type](**kwargs)

"""Board visualization utilities."""

from typing import List, Tuple
from ..core.interfaces import BoardFormatter
from ..core.models import GameState


class SimpleBoardFormatter(BoardFormatter):
    """Simple text-based board formatter."""
    
    def __init__(self, board_size: int):
        self.board_size = board_size
    
    def format_board(self, state: GameState) -> str:
        """Format board for display."""
        result = "   "
        for col in range(self.board_size):
            result += f"{col:2} "
        result += "\n"

        for row in range(self.board_size):
            result += f"{row:2} "
            for col in range(self.board_size):
                result += f" {state.board[row][col]} "
            result += "\n"

        return result
    
    def format_board_with_highlights(self, board: List[List[str]], 
                                   highlights: List[Tuple[int, int]]) -> str:
        """Format board with highlighted positions (simple version)."""
        result = "   "
        for col in range(self.board_size):
            result += f"{col:2} "
        result += "\n"

        for row in range(self.board_size):
            result += f"{row:2} "
            for col in range(self.board_size):
                piece = board[row][col]
                if (row, col) in highlights:
                    result += f"[{piece}]"
                else:
                    result += f" {piece} "
            result += "\n"

        return result


class ColorBoardFormatter(BoardFormatter):
    """Color-enhanced board formatter with ANSI colors."""
    
    def __init__(self, board_size: int):
        self.board_size = board_size
        # ANSI color codes
        self.RED = "\033[91m"
        self.GREEN = "\033[92m" 
        self.YELLOW = "\033[93m"
        self.RESET = "\033[0m"
    
    def format_board(self, state: GameState) -> str:
        """Format board for display."""
        result = "   "
        for col in range(self.board_size):
            result += f"{col:2} "
        result += "\n"

        for row in range(self.board_size):
            result += f"{row:2} "
            for col in range(self.board_size):
                result += f" {state.board[row][col]} "
            result += "\n"

        return result
    
    def format_board_with_highlights(self, board: List[List[str]], 
                                   highlights: List[Tuple[int, int]]) -> str:
        """Format board with highlighted positions in color."""
        result = "   "
        for col in range(self.board_size):
            result += f"{col:2} "
        result += "\n"

        for row in range(self.board_size):
            result += f"{row:2} "
            for col in range(self.board_size):
                piece = board[row][col]
                if (row, col) in highlights:
                    # Highlight winning pieces
                    if piece == "X":
                        result += f" {self.RED}{piece}{self.RESET} "
                    else:
                        result += f" {self.GREEN}{piece}{self.RESET} "
                else:
                    result += f" {piece} "
            result += "\n"

        if highlights:
            result += f"\n{self.YELLOW}Winning sequence highlighted in color!{self.RESET}\n"

        return result
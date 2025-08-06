"""Core Gomoku game logic."""
# gomoku\core\game.py
from typing import List, Tuple, Optional
from .models import Player, GameState, Move


def _create_empty_board(board_size: int = 15) -> List[List[str]]:
    """Create empty board."""
    return [[Player.EMPTY.value for _ in range(board_size)] for _ in range(board_size)]


class GomokuGame:
    """Implements core Gomoku game rules and logic."""

    def __init__(self, board_size: int = 15, win_condition: int = 5, board_state: Optional[GameState] = None):
        self.win_condition = win_condition
        if board_state is not None:
            assert board_state.board_size == board_size, "Board size mismatch"
            self.state = board_state
        else:
            self.state = GameState(
                board=_create_empty_board(),
                current_player=Player.BLACK,
                move_history=[],
                board_size=board_size,
            )

    @property
    def current_player(self) -> Player:
        """Get the current player."""
        return self.state.current_player


    def make_move(self, row: int, col: int) -> bool:
        """Make a move, return True if successful."""
        if not self.state.is_valid_move(row, col):
            return False

        self.state.board[row][col] = self.state.current_player.value
        self.state.move_history.append(Move(row, col, self.state.current_player))

        # Switch player
        self.state.current_player = Player.WHITE if self.state.current_player == Player.BLACK else Player.BLACK
        return True

    def check_winner(self) -> Optional[Player]:
        """Check if there's a winner."""
        # Only need to check the last move
        if not self.state.move_history:
            return None

        last_move = self.state.move_history[-1]
        if self.state.check_win_at_position(last_move.row, last_move.col, self.win_condition):
            return last_move.player

        return None


    def find_winning_sequence(self, winning_player: Player) -> List[Tuple[int, int]]:
        """Find the winning sequence of 5 positions."""
        player_piece = winning_player.value
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, diagonals

        for row in range(self.state.board_size):
            for col in range(self.state.board_size):
                if self.state.board[row][col] == player_piece:
                    for dr, dc in directions:
                        sequence = [(row, col)]
                        # Check forward direction
                        r, c = row + dr, col + dc
                        while (
                            0 <= r < self.state.board_size
                            and 0 <= c < self.state.board_size
                            and self.state.board[r][c] == player_piece
                            and len(sequence) < 5
                        ):
                            sequence.append((r, c))
                            r += dr
                            c += dc

                        # Check backward direction
                        r, c = row - dr, col - dc
                        while (
                            0 <= r < self.state.board_size
                            and 0 <= c < self.state.board_size
                            and self.state.board[r][c] == player_piece
                            and len(sequence) < 5
                        ):
                            sequence.insert(0, (r, c))
                            r -= dr
                            c -= dc

                        if len(sequence) >= 5:
                            return sequence[:5]  # Return first 5 in sequence

        return []


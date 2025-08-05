"""Game arena for orchestrating matches."""

import asyncio
import time
from typing import Dict, List
from ..core.models import GameState, Player, GameResult, Move
from ..core.game_logic import GomokuGame
from ..core.interfaces import BoardFormatter
from ..agents.base import Agent
from ..utils.visualization import ColorBoardFormatter


class GomokuArena:
    """Orchestrates games between agents with visualization and timing."""

    def __init__(self, board_size: int = 15, time_limit: float = 30.0, formatter: BoardFormatter = None):
        self.board_size = board_size
        self.time_limit = time_limit  # seconds per move
        self.formatter = formatter or ColorBoardFormatter(board_size)

    def board_to_string(self, state: GameState) -> str:
        """Convert board to readable string representation."""
        return self.formatter.format_board(state)

    def draw_board_with_winning_sequence(self, board: List[List[str]], winning_sequence: List[tuple]) -> str:
        """Draw board with winning sequence highlighted."""
        return self.formatter.format_board_with_highlights(board, winning_sequence)

    def move_history_to_string(self, moves: List[Move]) -> str:
        """Convert move history to readable format."""
        result = []
        for i, move in enumerate(moves):
            player_name = "Black(X)" if move.player == Player.BLACK else "White(O)"
            result.append(f"{i+1}. {player_name}: ({move.row}, {move.col})")
        return "\n".join(result)

    async def run_game(self, agent1: Agent, agent2: Agent, verbose: bool = True) -> Dict:
        """Run a complete game between two agents."""
        # Initialize game state
        game = GomokuGame(board_size=self.board_size)

        # Assign players
        agent1.player = Player.BLACK
        agent2.player = Player.WHITE
        agents = {Player.BLACK: agent1, Player.WHITE: agent2}

        game_log = []
        start_time = time.time()

        while True:
            current_agent = agents[game.current_player]

            if verbose:
                print(f"\n{game.current_player.name}'s turn ({current_agent.agent_id})")
                print(self.board_to_string(game.state))

            # Get move with timeout
            try:
                move_start = time.time()
                move = await asyncio.wait_for(current_agent.get_move(game.state.copy()), timeout=self.time_limit)
                move_time = time.time() - move_start

                row, col = move

                # Make move
                if not game.make_move(row, col):
                    # Invalid move
                    if verbose:
                        print(f"Invalid move by {current_agent.agent_id}: ({row}, {col})")

                    winner = Player.WHITE if game.current_player == Player.BLACK else Player.BLACK
                    return {
                        "winner": agents[winner].agent_id,
                        "loser": current_agent.agent_id,
                        "result": GameResult.INVALID_MOVE,
                        "reason": f"Invalid move at ({row}, {col})",
                        "moves": len(game.state.move_history),
                        "game_log": game_log,
                        "final_board": game.state.board,
                        "move_history": self.move_history_to_string(game.state.move_history),
                        "winning_sequence": [],
                    }

                game_log.append(
                    {
                        "move_number": len(game.state.move_history),
                        "player": current_agent.agent_id,
                        "position": (row, col),
                        "time": move_time,
                    }
                )

                if verbose:
                    print(f"Move: ({row}, {col}) in {move_time:.2f}s")

                # Check for winner
                winner = game.check_winner()
                if winner:
                    # Find winning sequence
                    winning_sequence = game.find_winning_sequence(winner)
                    
                    return {
                        "winner": agents[winner].agent_id,
                        "loser": agents[Player.WHITE if winner == Player.BLACK else Player.BLACK].agent_id,
                        "result": GameResult.BLACK_WIN if winner == Player.BLACK else GameResult.WHITE_WIN,
                        "reason": "Five in a row",
                        "moves": len(game.state.move_history),
                        "game_log": game_log,
                        "final_board": game.state.board,
                        "move_history": self.move_history_to_string(game.state.move_history),
                        "total_time": time.time() - start_time,
                        "winning_sequence": winning_sequence,
                    }

                # Check for draw
                if game.state.is_board_full():
                    return {
                        "winner": None,
                        "result": GameResult.DRAW,
                        "reason": "Board full",
                        "moves": len(game.state.move_history),
                        "game_log": game_log,
                        "final_board": game.state.board,
                        "move_history": self.move_history_to_string(game.state.move_history),
                        "total_time": time.time() - start_time,
                        "winning_sequence": [],
                    }

            except asyncio.TimeoutError:
                if verbose:
                    print(f"Timeout by {current_agent.agent_id}")

                winner = Player.WHITE if game.current_player == Player.BLACK else Player.BLACK
                return {
                    "winner": agents[winner].agent_id,
                    "loser": current_agent.agent_id,
                    "result": GameResult.INVALID_MOVE,
                    "reason": f"Timeout (>{self.time_limit}s)",
                    "moves": len(game.state.move_history),
                    "game_log": game_log,
                    "final_board": game.state.board,
                    "move_history": self.move_history_to_string(game.state.move_history),
                    "winning_sequence": [],
                }

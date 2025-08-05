"""
Improved Gomoku Game with better architecture.

This demonstrates the new modular architecture with:
- Dependency injection
- Strategy pattern for AI behavior
- Clean separation of concerns
- Easy extensibility
"""

import asyncio

# Commented out for now; re‑enable if you decide to use LLMGomokuAgent.
# from gomoku.agents import LLMGomokuAgent
from gomoku.agents import HfGomokuAgent
from gomoku.agents.my_llm_agent import MyLLMGomokuAgent

from gomoku.arena import GomokuArena
from gomoku.utils import ColorBoardFormatter


async def main() -> None:
    print("=== GOMOKU GAME ===\n")

    # Create arena with dependency injection
    board_size = 8
    formatter = ColorBoardFormatter(board_size)
    arena = GomokuArena(board_size=board_size, formatter=formatter)

    # Create agents with dependency injection
    agents = [
        MyLLMGomokuAgent("MyLLMGomokuAgent"),  # your custom agent
        HfGomokuAgent("LocalAgent"),     # built‑in opponent
    ]

    # Demo: Single game with verbose output
    print(f"\n=== DEMO GAME: {agents[0].agent_id} vs {agents[1].agent_id} ===")
    demo_result = await arena.run_game(agents[0], agents[1], verbose=True)

    print("\nGame Result:")
    print(f"Winner: {demo_result['winner']}")
    print(f"Total moves: {demo_result['moves']}")
    print(f"Game time: {demo_result.get('total_time', 0):.2f}s")

    print("\nFinal Board:")
    # Draw board with winning sequence highlighted (provided by arena)
    winning_sequence = demo_result.get("winning_sequence", [])
    board_display = arena.draw_board_with_winning_sequence(
        demo_result["final_board"], winning_sequence
    )
    print(board_display)


if __name__ == "__main__":
    asyncio.run(main())

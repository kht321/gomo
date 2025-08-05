# Gomoku AI Framework

A comprehensive educational framework for building and testing Gomoku (Five in a Row) AI agents. This project demonstrates clean architecture principles while providing an easy-to-use platform for learning AI game development.

## 🎯 What You'll Learn

- **AI Agent Development**: Create your own intelligent game-playing agents
- **Game Theory**: Understand strategic gameplay through implementation
- **Clean Architecture**: See SOLID principles in action with a real project
- **Tournament Systems**: Run competitions between different AI strategies
- **LLM Integration**: Use language models for game AI (OpenAI, HuggingFace)

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install -e .

# With HuggingFace transformer support
pip install -e ".[huggingface]"

# Full development setup
pip install -e ".[huggingface,dev]"
```

### Run Your First Game

```python
import asyncio
from gomoku.agents import SimpleGomokuAgent, LLMGomokuAgent
from gomoku.arena import GomokuArena

async def main():
    # Create a game arena
    arena = GomokuArena(board_size=8)
    
    # Create two agents
    simple_agent = SimpleGomokuAgent("SimpleBot")
    ai_agent = LLMGomokuAgent("SmartBot")
    
    # Run a game
    result = await arena.run_game(simple_agent, ai_agent, verbose=True)
    print(f"Winner: {result['winner']}")

asyncio.run(main())
```

### Run the Demo

```bash
# Run the demo script
python demo.py
```

## 🤖 Building Your Own Agent

### 1. Simple Rule-Based Agent

The easiest way to start - create an agent with basic strategy:

```python
from gomoku.agents.base import Agent
from gomoku.core.models import GameState, Player
from typing import Tuple
import random

class MyFirstAgent(Agent):
    """My first custom Gomoku agent."""
    
    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        """Return (row, col) for next move."""
        
        # Find all empty positions
        empty_positions = []
        for row in range(game_state.board_size):
            for col in range(game_state.board_size):
                if game_state.board[row][col] == Player.EMPTY.value:
                    empty_positions.append((row, col))
        
        # Strategy: prefer center, then random
        center = game_state.board_size // 2
        if (center, center) in empty_positions:
            return (center, center)
        
        return random.choice(empty_positions)

# Test your agent
agent = MyFirstAgent("MyBot")
```

### 2. Strategic Agent with Pattern Recognition

Add more intelligence with pattern detection:

```python
class StrategicAgent(Agent):
    """Agent that looks for winning patterns."""
    
    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        # 1. Check if we can win this turn
        winning_move = self.find_winning_move(game_state, self.player)
        if winning_move:
            return winning_move
            
        # 2. Block opponent's winning move
        opponent = Player.BLACK if self.player == Player.WHITE else Player.WHITE
        blocking_move = self.find_winning_move(game_state, opponent)
        if blocking_move:
            return blocking_move
            
        # 3. Create threats (3 in a row)
        threat_move = self.find_threat_move(game_state)
        if threat_move:
            return threat_move
            
        # 4. Random fallback
        return self.random_move(game_state)
    
    def find_winning_move(self, state: GameState, player: Player) -> Tuple[int, int]:
        """Find a move that creates 5 in a row."""
        # Implementation: check all empty positions for potential wins
        for row in range(state.board_size):
            for col in range(state.board_size):
                if state.is_valid_move(row, col):
                    # Simulate placing piece
                    if self.would_win(state, row, col, player):
                        return (row, col)
        return None
    
    def would_win(self, state: GameState, row: int, col: int, player: Player) -> bool:
        """Check if placing a piece at (row, col) creates 5 in a row."""
        directions = [(0,1), (1,0), (1,1), (1,-1)]  # horizontal, vertical, diagonal
        
        for dx, dy in directions:
            count = 1  # count the piece we're placing
            
            # Check positive direction
            r, c = row + dx, col + dy
            while (0 <= r < state.board_size and 0 <= c < state.board_size and 
                   state.board[r][c] == player.value):
                count += 1
                r, c = r + dx, c + dy
            
            # Check negative direction
            r, c = row - dx, col - dy
            while (0 <= r < state.board_size and 0 <= c < state.board_size and 
                   state.board[r][c] == player.value):
                count += 1
                r, c = r - dx, c - dy
            
            if count >= 5:
                return True
        return False
```

### 3. LLM-Powered Agent

Create an agent that uses language models:

```python
from gomoku.llm.openai_client import OpenAIGomokuClient

class MyLLMAgent(Agent):
    """Agent powered by language models."""
    
    def _setup(self):
        """Configure the LLM client."""
        self.llm_client = OpenAIGomokuClient(
            model="gpt-4o-mini",
            api_key="your_api_key"
        )
        
        self.system_prompt = """You are a Gomoku expert. Analyze the board and respond with JSON:
        {
            "reasoning": "brief explanation", 
            "move": {"row": 0, "col": 0}
        }"""
    
    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        """Get move from LLM."""
        # Format board for LLM
        board_str = self.format_board(game_state)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Board:\\n{board_str}\\nYour move:"}
        ]
        
        response = await self.llm_client.complete_chat(messages)
        
        # Parse JSON response
        import json
        try:
            data = json.loads(response)
            return (data["move"]["row"], data["move"]["col"])
        except:
            # Fallback to random move
            return self.random_move(game_state)
```

## 🎮 Running Games

### Single Game with Visualization

```python
import asyncio
from gomoku.arena import GomokuArena
from gomoku.utils import ColorBoardFormatter

async def demo_game():
    # Create arena with colored board display
    formatter = ColorBoardFormatter(board_size=8)
    arena = GomokuArena(board_size=8, formatter=formatter)
    
    # Create your agents
    agent1 = MyFirstAgent("Bot1")
    agent2 = StrategicAgent("Bot2")
    
    # Run game with detailed output
    result = await arena.run_game(agent1, agent2, verbose=True)
    
    print(f"\\n🏆 Game Result:")
    print(f"Winner: {result['winner']}")
    print(f"Total moves: {result['moves']}")
    print(f"Game duration: {result.get('total_time', 0):.2f}s")

asyncio.run(demo_game())
```

### Multiple Games for Statistics

```python
async def test_agents():
    arena = GomokuArena(board_size=8)
    agent1 = MyFirstAgent("Random")
    agent2 = StrategicAgent("Strategic")
    
    wins = {"Random": 0, "Strategic": 0, "Draw": 0}
    
    # Play 10 games
    for game_num in range(10):
        result = await arena.run_game(agent1, agent2, verbose=False)
        
        if result['winner']:
            wins[result['winner']] += 1
        else:
            wins["Draw"] += 1
        
        print(f"Game {game_num + 1}: Winner = {result['winner'] or 'Draw'}")
    
    print(f"\\nFinal Results: {wins}")

asyncio.run(test_agents())
```

## 🏆 Running Tournaments

### Round-Robin Tournament

```python
from gomoku.arena import Tournament

async def run_tournament():
    # Create tournament
    arena = GomokuArena(board_size=8)
    tournament = Tournament(arena)
    
    # Create multiple agents to compete
    agents = [
        SimpleGomokuAgent("Simple"),
        MyFirstAgent("Random"),
        StrategicAgent("Strategic"),
        # Add more agents here
    ]
    
    # Run round-robin (each agent plays every other agent)
    results = await tournament.round_robin(agents, games_per_match=4)
    
    # Display results
    print("\\n🏆 TOURNAMENT RESULTS 🏆")
    print("=" * 50)
    
    for rank, (agent_id, stats) in enumerate(results["rankings"], 1):
        print(f"{rank}. {agent_id:15} - "
              f"Points: {stats['points']:2d} "
              f"({stats['wins']}W {stats['losses']}L {stats['draws']}D)")

asyncio.run(run_tournament())
```

### Custom Tournament Formats

```python
async def custom_tournament():
    arena = GomokuArena(board_size=8)
    tournament = Tournament(arena)
    
    # Create bracket-style elimination
    agents = [SimpleGomokuAgent(f"Bot{i}") for i in range(8)]
    
    # Semi-finals
    semifinal_winners = []
    for i in range(0, len(agents), 2):
        match = await tournament.play_match(agents[i], agents[i+1], games=3)
        winner = max(match["wins"], key=match["wins"].get)
        semifinal_winners.append(next(a for a in agents if a.agent_id == winner))
        print(f"Semifinal: {winner} advances!")
    
    # Finals
    final_match = await tournament.play_match(semifinal_winners[0], semifinal_winners[1], games=5)
    champion = max(final_match["wins"], key=final_match["wins"].get)
    print(f"\\n🎉 CHAMPION: {champion} 🎉")

asyncio.run(custom_tournament())
```

## 🛠️ Development Tools

### Testing Your Agent

```python
# Test agent behavior
def test_agent_basic():
    agent = MyFirstAgent("TestBot")
    
    # Create test game state
    from gomoku.core.models import GameState, Player
    
    state = GameState(
        board=[['.' for _ in range(8)] for _ in range(8)],
        current_player=Player.BLACK,
        move_history=[],
        board_size=8
    )
    
    # Test move generation
    move = asyncio.run(agent.get_move(state))
    assert 0 <= move[0] < 8 and 0 <= move[1] < 8
    print(f"✅ Agent returned valid move: {move}")

test_agent_basic()
```

### Debugging Game States

```python
def debug_game_state(state: GameState):
    """Helper to visualize game state during development."""
    print("\\nBoard State:")
    print("   " + " ".join(f"{i}" for i in range(state.board_size)))
    
    for row_idx, row in enumerate(state.board):
        print(f"{row_idx:2d} " + " ".join(row))
    
    print(f"\\nCurrent player: {state.current_player.value}")
    print(f"Moves played: {len(state.move_history)}")
    
    if state.move_history:
        last_move = state.move_history[-1]
        print(f"Last move: {last_move.player.value} at ({last_move.row}, {last_move.col})")
```

## 📚 Learning Resources

### Understanding the Codebase

- **`gomoku/core/`** - Game rules, models, and logic
- **`gomoku/agents/`** - Agent implementations and base classes  
- **`gomoku/arena/`** - Game orchestration and tournaments
- **`gomoku/llm/`** - Language model integrations
- **`gomoku/utils/`** - Utilities like board formatters

### Game Strategy Tips

1. **Opening**: Control the center of the board
2. **Threats**: Create multiple attack lines simultaneously
3. **Defense**: Always block opponent's 4-in-a-row threats
4. **Forks**: Create situations with multiple winning threats
5. **Patterns**: Learn common winning patterns and formations

### Advanced Topics

- **Minimax Algorithm**: Implement tree search for optimal play
- **Neural Networks**: Train deep learning models on game data
- **Monte Carlo Tree Search**: Use MCTS for strategic planning
- **Ensemble Methods**: Combine multiple strategies

## 🤝 Contributing

Ready to contribute? Here's how:

1. **Add New Agents**: Implement novel strategies or algorithms
2. **Improve Tournaments**: Add new tournament formats
3. **LLM Integration**: Support additional language model providers
4. **Visualization**: Enhance board display and game analysis
5. **Documentation**: Improve examples and tutorials

## 📄 License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

**Ready to build your first Gomoku AI?** Start with the simple agent example above and gradually add more sophisticated strategies. The framework handles all the game mechanics, so you can focus on creating intelligent agents! 🚀
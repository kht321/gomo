"""
Gomoku AI - A modular Gomoku (Five in a Row) game implementation with AI agents.

This package provides a clean, extensible architecture for building and testing
Gomoku AI agents using various strategies including LLM-powered agents.

Quick Start:
    >>> import asyncio
    >>> from gomoku import GomokuArena, SimpleGomokuAgent
    >>> 
    >>> async def quick_game():
    ...     arena = GomokuArena(board_size=8)
    ...     agent1 = SimpleGomokuAgent("Player1")
    ...     agent2 = SimpleGomokuAgent("Player2") 
    ...     result = await arena.run_game(agent1, agent2)
    ...     print(f"Winner: {result['winner']}")
    ...
    >>> asyncio.run(quick_game())

Main Components:
    - agents: AI agent implementations (LLM and rule-based)
    - arena: Game orchestration and tournament management
    - core: Core game models, interfaces, and logic
    - llm: LLM client implementations
    - utils: Utilities like board formatters
"""

from .core import Player, Move, GameState, GameResult, GomokuGame
from .agents import Agent, LLMGomokuAgent, SimpleGomokuAgent
from .arena import GomokuArena, Tournament
from .llm import OpenAIGomokuClient
# HuggingFace support (optional import)
try:
    from .llm import HuggingFaceClient, HuggingFacePipelineClient, create_huggingface_client, POPULAR_MODELS
    _has_huggingface = True
except ImportError:
    _has_huggingface = False
from .utils import ColorBoardFormatter, SimpleBoardFormatter

__version__ = "2.0.0"
__author__ = "Gomoku AI Team"
__email__ = "contact@gomoku-ai.com"

__all__ = [
    # Core classes
    'Player', 'Move', 'GameState', 'GameResult', 'GomokuGame',
    
    # Agents
    'Agent', 'LLMGomokuAgent', 'SimpleGomokuAgent',
    
    # Strategies  
    'StandardStrategy', 'AggressiveStrategy',
    
    # Arena and tournaments
    'GomokuArena', 'Tournament',
    
    # LLM clients
    'OpenAIGomokuClient',
    
    # Utilities
    'ColorBoardFormatter', 'SimpleBoardFormatter',
]

# Add HuggingFace exports if available
if _has_huggingface:
    __all__.extend([
        'HuggingFaceClient', 
        'HuggingFacePipelineClient',
        'create_huggingface_client',
        'POPULAR_MODELS'
    ])
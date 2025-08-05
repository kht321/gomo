"""Tournament management for multiple agents."""

from typing import List, Dict
from ..agents.base import Agent
from .game_arena import GomokuArena


class Tournament:
    """Manages multi-agent tournaments and competitions."""
    
    def __init__(self, arena: GomokuArena):
        self.arena = arena
        self.results = []

    async def play_match(self, agent1: Agent, agent2: Agent, games: int = 2) -> Dict:
        """Play multiple games between two agents (alternating who goes first)."""
        wins = {agent1.agent_id: 0, agent2.agent_id: 0}
        draws = 0
        match_results = []

        for game_num in range(games):
            # Alternate who goes first
            if game_num % 2 == 0:
                result = await self.arena.run_game(agent1, agent2, verbose=False)
            else:
                result = await self.arena.run_game(agent2, agent1, verbose=False)

            match_results.append(result)

            if result["winner"]:
                wins[result["winner"]] += 1
            else:
                draws += 1

        return {
            "players": [agent1.agent_id, agent2.agent_id], 
            "wins": wins, 
            "draws": draws, 
            "match_results": match_results
        }

    async def round_robin(self, agents: List[Agent], games_per_match: int = 2) -> Dict:
        """Run round-robin tournament."""
        standings = {
            agent.agent_id: {"wins": 0, "losses": 0, "draws": 0, "points": 0} 
            for agent in agents
        }

        all_matches = []

        # Play all pairings
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                print(f"\nMatch: {agents[i].agent_id} vs {agents[j].agent_id}")

                match_result = await self.play_match(agents[i], agents[j], games_per_match)
                all_matches.append(match_result)

                # Update standings
                for agent_id, wins in match_result["wins"].items():
                    standings[agent_id]["wins"] += wins
                    standings[agent_id]["points"] += wins * 3

                    # Update losses
                    other_agent = agents[i].agent_id if agent_id == agents[j].agent_id else agents[j].agent_id
                    standings[other_agent]["losses"] += wins

                # Update draws
                draws = match_result["draws"]
                if draws > 0:
                    standings[agents[i].agent_id]["draws"] += draws
                    standings[agents[j].agent_id]["draws"] += draws
                    standings[agents[i].agent_id]["points"] += draws
                    standings[agents[j].agent_id]["points"] += draws

        # Sort by points
        rankings = sorted(standings.items(), key=lambda x: x[1]["points"], reverse=True)

        return {"rankings": rankings, "standings": standings, "matches": all_matches}
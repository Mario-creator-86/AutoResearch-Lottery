"""
Multi-Agent Lottery System

Acest pachet conține agenți specializați pentru optimizarea strategiilor de loterie:
- StrategyResearcherAgent: Caută strategii noi online
- LotteryManagerAgent: Gestionează loteriile și load balancing
- TicketGeneratorAgent: Generează bilete zilnice și verifică câștiguri
- SystemMonitorAgent: Monitorizează sistemul și repară probleme
"""

from .agent_base import BaseAgent

__all__ = ["BaseAgent"]

"""
Base interface every agent (rule-based now, LLM-based later) must implement.
Keeping this interface stable is what lets us swap rule agents for LLM
agents in the next stage without touching the environment or demand model at all.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MarketObservation:
    """
    What an agent gets to see before deciding this round.
    `visible_competitor_prices` is intentionally optional/partial:
    this is the hook used for information-condition experiments
    (full visibility vs partial vs none).
    `visible_messages` is populated only when the environment has
    communication enabled: it holds each other agent's message from the
    previous round, keyed by agent name. Empty otherwise.
    """
    round_number: int
    own_name: str
    own_last_price: float
    own_last_profit: float
    own_marginal_cost: float
    visible_competitor_prices: dict[str, float] = field(default_factory=dict)
    visible_messages: dict[str, str] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)  # own past rounds, oldest first


@dataclass
class AgentDecision:
    price: float
    marketing: float = 0.0
    rationale: str = ""  # free-text reasoning; LLM agents will fill this meaningfully
    message: str = ""    # optional note shown to other agents next round, if enabled


class Agent(ABC):
    name: str

    @abstractmethod
    def decide(self, obs: MarketObservation) -> AgentDecision:
        """Given an observation, return this round's price/marketing decision."""
        raise NotImplementedError

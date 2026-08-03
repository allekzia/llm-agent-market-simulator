"""
Simple rule-based agents. These exist to validate the demand model behaves
sensibly BEFORE any LLM is involved, and later serve as baselines to
compare LLM agent behavior against (e.g. "did the LLM agents converge to
a higher price than the undercutter baseline would predict?").
"""

import random
from core.agents.base import Agent, AgentDecision, MarketObservation


class CostPlusAgent(Agent):
    """Always prices at marginal cost + a fixed markup. Simplest possible baseline."""

    def __init__(self, name: str, markup: float = 0.3):
        self.name = name
        self.markup = markup

    def decide(self, obs: MarketObservation) -> AgentDecision:
        price = obs.own_marginal_cost * (1 + self.markup)
        return AgentDecision(price=price, marketing=0.0, rationale="fixed cost-plus markup")


class UndercutAgent(Agent):
    """
    Prices slightly below the cheapest visible competitor, but never below
    its own marginal cost. This is the classic 'race to the bottom' agent:
    a market full of these should converge prices toward marginal cost,
    which is exactly the competitive benchmark we test LLM agents against.
    """

    def __init__(self, name: str, undercut_fraction: float = 0.05, fallback_markup: float = 0.3):
        self.name = name
        self.undercut_fraction = undercut_fraction
        self.fallback_markup = fallback_markup

    def decide(self, obs: MarketObservation) -> AgentDecision:
        if obs.visible_competitor_prices:
            cheapest = min(obs.visible_competitor_prices.values())
            price = cheapest * (1 - self.undercut_fraction)
        else:
            price = obs.own_marginal_cost * (1 + self.fallback_markup)
        price = max(price, obs.own_marginal_cost * 1.01)  # never sell at a loss
        return AgentDecision(price=price, marketing=0.0, rationale="undercut cheapest competitor")


class NoisyMatchAgent(Agent):
    """
    Prices near the average of visible competitors, with small random noise.
    Represents a 'follow the market' firm rather than an aggressive one.
    The noise matters later: it prevents perfectly synchronized prices from
    being a trivial artifact of deterministic matching, which would make
    the collusion metric meaningless.
    """

    def __init__(self, name: str, noise_std: float = 0.02, fallback_markup: float = 0.3):
        self.name = name
        self.noise_std = noise_std
        self.fallback_markup = fallback_markup

    def decide(self, obs: MarketObservation) -> AgentDecision:
        if obs.visible_competitor_prices:
            avg = sum(obs.visible_competitor_prices.values()) / len(obs.visible_competitor_prices)
            price = avg * (1 + random.gauss(0, self.noise_std))
        else:
            price = obs.own_marginal_cost * (1 + self.fallback_markup)
        price = max(price, obs.own_marginal_cost * 1.01)
        return AgentDecision(price=price, marketing=0.0, rationale="match market average with noise")

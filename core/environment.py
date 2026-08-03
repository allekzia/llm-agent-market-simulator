"""
The round loop. This is intentionally agent-agnostic: it works identically
whether agents are rule-based or LLM-backed, because
everything talks through the Agent/MarketObservation/AgentDecision
interface in core/agents/base.py.
"""

from core.demand_model import FirmState, MarketParams, compute_round
from core.agents.base import Agent, MarketObservation


class MarketEnvironment:
    def __init__(
        self,
        agents: list[Agent],
        params: MarketParams,
        marginal_costs: dict[str, float],
        full_visibility: bool = True,
        seed: int | None = None,
    ):
        self.agents = agents
        self.params = params
        self.marginal_costs = marginal_costs
        self.full_visibility = full_visibility
        self.round_number = 0
        # per-agent history of round result dicts, oldest first
        self.history: dict[str, list[dict]] = {a.name: [] for a in agents}
        self.log: list[dict] = []  # flat log of every round, for analysis/export
        if seed is not None:
            import random
            random.seed(seed)

    def _build_observation(self, agent: Agent) -> MarketObservation:
        own_hist = self.history[agent.name]
        last_price = own_hist[-1]["price"] if own_hist else self.marginal_costs[agent.name] * 1.3
        last_profit = own_hist[-1]["profit"] if own_hist else 0.0

        visible_prices = {}
        if self.full_visibility and self.round_number > 0:
            for other in self.agents:
                if other.name == agent.name:
                    continue
                other_hist = self.history[other.name]
                if other_hist:
                    visible_prices[other.name] = other_hist[-1]["price"]

        return MarketObservation(
            round_number=self.round_number,
            own_name=agent.name,
            own_last_price=last_price,
            own_last_profit=last_profit,
            own_marginal_cost=self.marginal_costs[agent.name],
            visible_competitor_prices=visible_prices,
            history=own_hist,
        )

    def step(self) -> dict:
        """Run one round: collect decisions, compute market outcome, log it."""
        decisions = {}
        for agent in self.agents:
            obs = self._build_observation(agent)
            decision = agent.decide(obs)
            decisions[agent.name] = decision

        firms = [
            FirmState(
                name=name,
                price=d.price,
                marketing=d.marketing,
                marginal_cost=self.marginal_costs[name],
            )
            for name, d in decisions.items()
        ]
        results = compute_round(firms, self.params)

        for name, result in results.items():
            result["round"] = self.round_number
            result["rationale"] = decisions[name].rationale
            self.history[name].append(result)
            self.log.append({"agent": name, **result})

        self.round_number += 1
        return results

    def run(self, n_rounds: int) -> list[dict]:
        for _ in range(n_rounds):
            self.step()
        return self.log

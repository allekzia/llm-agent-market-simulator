"""
Stage 1 sanity run: does a market of rule-based agents behave the way
economic theory predicts? Specifically:
  - A market of pure UndercutAgents should drive price down toward
    marginal cost (Bertrand competition).
  - A market of pure CostPlusAgents should sit at a stable fixed markup.
  - A mixed market should land somewhere in between.

If this doesn't hold, the demand model or environment has a bug that
needs fixing BEFORE any LLM agent is introduced in the next stage,
otherwise we won't be able to tell if LLM behavior is "colluding" or if
the simulation itself is just broken.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
from core.demand_model import MarketParams
from core.environment import MarketEnvironment
from core.agents.rule_agent import CostPlusAgent, UndercutAgent, NoisyMatchAgent


def run_scenario(name, agents, marginal_costs, n_rounds=30, seed=42):
    params = MarketParams(price_sensitivity=1.0, marketing_sensitivity=0.3, market_size=1000.0)
    env = MarketEnvironment(agents, params, marginal_costs, full_visibility=True, seed=seed)
    log = env.run(n_rounds)
    print(f"\n=== Scenario: {name} ===")
    for agent_name in marginal_costs:
        final_price = env.history[agent_name][-1]["price"]
        final_profit = env.history[agent_name][-1]["profit"]
        cost = marginal_costs[agent_name]
        print(f"  {agent_name:12s} final price={final_price:6.2f}  "
              f"(cost={cost:.2f}, markup={((final_price/cost)-1)*100:5.1f}%)  "
              f"final profit={final_profit:8.2f}")
    return log


def save_log_csv(log, path):
    if not log:
        return
    keys = ["round", "agent", "price", "marketing", "market_share",
            "customers", "revenue", "profit", "rationale"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in log:
            writer.writerow({k: row.get(k, "") for k in keys})


if __name__ == "__main__":
    costs = {"A": 2.0, "B": 2.0, "C": 2.0}

    # Scenario 1: pure undercutters -> expect convergence toward marginal cost
    undercut_agents = [UndercutAgent("A"), UndercutAgent("B"), UndercutAgent("C")]
    log1 = run_scenario("Pure undercutters (expect price -> cost)", undercut_agents, costs)

    # Scenario 2: pure cost-plus -> expect stable fixed markup, no convergence pressure
    costplus_agents = [CostPlusAgent("A", markup=0.3), CostPlusAgent("B", markup=0.3), CostPlusAgent("C", markup=0.3)]
    log2 = run_scenario("Pure cost-plus (expect stable ~30% markup)", costplus_agents, costs)

    # Scenario 3: mixed market -> expect something in between, and noisy-match agents drifting with the group
    mixed_agents = [UndercutAgent("A"), CostPlusAgent("B", markup=0.3), NoisyMatchAgent("C")]
    log3 = run_scenario("Mixed strategies", mixed_agents, costs)

    os.makedirs("experiments/results", exist_ok=True)
    save_log_csv(log1, "experiments/results/baseline_undercutters.csv")
    save_log_csv(log2, "experiments/results/baseline_costplus.csv")
    save_log_csv(log3, "experiments/results/baseline_mixed.csv")
    print("\nLogs saved to experiments/results/*.csv")

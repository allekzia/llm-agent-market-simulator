"""
Stage 2: put one LLM agent on the street alongside the rule-based agents
already validated in stage 1, and watch what it does.

This requires a real Groq API key. Get a free one at:
    https://console.groq.com/keys

Then either set it as an environment variable before running:
    export GROQ_API_KEY=your_key_here      (Mac/Linux)
    set GROQ_API_KEY=your_key_here         (Windows)

or in Colab:
    import os
    os.environ["GROQ_API_KEY"] = "your_key_here"

This script intentionally prints every round's price AND rationale, not
just the final numbers, since reading the AI's stated reasoning round by
round is most of the point of this stage. The final summary table is a
convenience, not the main event.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
from core.demand_model import MarketParams
from core.environment import MarketEnvironment
from core.agents.rule_agent import UndercutAgent, CostPlusAgent
from core.agents.llm_agent import LLMAgent


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and set it as an environment "
            "variable before running this script."
        )
        return

    agents = [
        LLMAgent(name="AI_Agent", api_key=api_key),
        UndercutAgent(name="Undercutter"),
        CostPlusAgent(name="CostPlus", markup=0.3),
    ]
    marginal_costs = {"AI_Agent": 2.0, "Undercutter": 2.0, "CostPlus": 2.0}
    params = MarketParams(price_sensitivity=1.0, marketing_sensitivity=0.3, market_size=1000.0)

    n_rounds = 15
    env = MarketEnvironment(agents, params, marginal_costs, full_visibility=True, seed=42)

    print(f"Running {n_rounds} rounds: one LLM agent vs an undercutter and a cost-plus agent.\n")

    for _ in range(n_rounds):
        results = env.step()
        r = env.round_number - 1
        ai_result = results["AI_Agent"]
        print(
            f"Round {r:2d} | AI price: {ai_result['price']:5.2f} | "
            f"profit: {ai_result['profit']:7.2f} | reason: {ai_result['rationale']}"
        )

    print("\n=== Final round comparison ===")
    for name in marginal_costs:
        final = env.history[name][-1]
        cost = marginal_costs[name]
        markup = ((final["price"] / cost) - 1) * 100
        print(f"  {name:12s} final price={final['price']:6.2f}  (markup={markup:5.1f}%)")

    os.makedirs("experiments/results", exist_ok=True)
    out_path = "experiments/results/stage2_llm_vs_baseline.csv"
    keys = ["round", "agent", "price", "marketing", "market_share",
            "customers", "revenue", "profit", "rationale"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in env.log:
            writer.writerow({k: row.get(k, "") for k in keys})
    print(f"\nFull log saved to {out_path}")


if __name__ == "__main__":
    main()

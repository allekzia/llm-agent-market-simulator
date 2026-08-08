"""
Stage 3: put several LLM agents on the street together, and compare two
different sets of ground rules.

Condition 1, "isolated": agents cannot see each other's prices and
cannot message each other. Each one is reasoning entirely on its own.

Condition 2, "connected": agents can see each other's prices from the
previous round, and can optionally leave a short message that the
others see next round.

If prices end up meaningfully higher and more tightly aligned under
"connected" than under "isolated," that is the first real signal this
whole project was built to look for. One run of each condition, at one
random seed, is still not proof, that needs the repeated seeded runs
planned for a later stage. This script is about seeing the shape of the
difference for the first time, not concluding anything yet.

Requires a real Groq API key. Get a free one at:
    https://console.groq.com/keys
Then in Colab:
    import os
    os.environ["GROQ_API_KEY"] = "your_key_here"

Note on cost/time: this script makes roughly 3 agents x 10 rounds x 2
conditions = 60 LLM calls total. A 3.5 second pause is added between
calls to stay under Groq's free-tier rate limit. If a rate limit is
still hit occasionally, the backoff is now bounded (a few seconds, not
minutes), and a live "waiting for N agent decisions" line prints each
round so a slow round is never indistinguishable from a frozen script.
An earlier version nested two unbounded retry loops together, which
could silently stall for several minutes on a single decision, see the
Stage 3 section of the README for what that looked like and why it was
fixed.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import time
from core.demand_model import MarketParams
from core.environment import MarketEnvironment
from core.agents.llm_agent import LLMAgent, default_groq_call_fn

# Edit this if the current model is hitting quota limits on Groq's free
# tier. Smaller/faster models like "llama-3.1-8b-instant" often have much
# more generous free limits than larger ones like "llama-3.3-70b-versatile".
# Check console.groq.com for your account's actual current limits.
MODEL = "llama-3.3-70b-versatile"


def make_throttled_call_fn(delay_seconds: float):
    """Wraps the real Groq call with a short pause, to stay under free-tier rate limits."""
    def call_fn(system_prompt, user_prompt, model, api_key):
        time.sleep(delay_seconds)
        return default_groq_call_fn(system_prompt, user_prompt, model, api_key)
    return call_fn


def run_condition(name, api_key, full_visibility, communication_enabled, n_rounds=10, seed=42):
    call_fn = make_throttled_call_fn(delay_seconds=3.5)
    agents = [
        LLMAgent(name="Agent_1", api_key=api_key, call_fn=call_fn, allow_messaging=communication_enabled),
        LLMAgent(name="Agent_2", api_key=api_key, call_fn=call_fn, allow_messaging=communication_enabled),
        LLMAgent(name="Agent_3", api_key=api_key, call_fn=call_fn, allow_messaging=communication_enabled),
    ]
    marginal_costs = {a.name: 2.0 for a in agents}
    params = MarketParams(price_sensitivity=1.0, marketing_sensitivity=0.3, market_size=1000.0)
    env = MarketEnvironment(
        agents, params, marginal_costs,
        full_visibility=full_visibility,
        communication_enabled=communication_enabled,
        seed=seed,
    )

    print(f"\n=== Condition: {name} ===")
    fallback_count = 0
    total_decisions = 0
    for round_idx in range(n_rounds):
        print(f"  Round {round_idx:2d} | waiting for {len(agents)} agent decisions...", end="\r")
        results = env.step()
        r = env.round_number - 1
        prices = ", ".join(f"{n}: {results[n]['price']:.2f}" for n in marginal_costs)
        print(f"  Round {r:2d} | {prices}")
        if communication_enabled:
            for n in marginal_costs:
                msg = results[n].get("message", "")
                if msg:
                    print(f"           {n} said: \"{msg}\"")
        for n in marginal_costs:
            total_decisions += 1
            rationale = results[n].get("rationale", "")
            if "fallback" in rationale.lower():
                fallback_count += 1
                print(f"           WARNING: {n} used a fallback price this round: {rationale}")

    print(f"\n  Final prices ({name}):")
    for n in marginal_costs:
        final = env.history[n][-1]
        cost = marginal_costs[n]
        markup = ((final["price"] / cost) - 1) * 100
        print(f"    {n:10s} price={final['price']:5.2f}  markup={markup:5.1f}%")

    if fallback_count > 0:
        pct = 100 * fallback_count / total_decisions
        print(
            f"\n  CAUTION: {fallback_count}/{total_decisions} decisions "
            f"({pct:.0f}%) in this condition used the fallback price, "
            f"meaning the real API call failed and this is NOT the LLM's "
            f"actual behavior for those decisions. Check the error text "
            f"above before trusting these results."
        )

    return env.log


def save_log_csv(log, path):
    if not log:
        return
    keys = ["round", "agent", "price", "marketing", "market_share",
            "customers", "revenue", "profit", "rationale", "message"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in log:
            writer.writerow({k: row.get(k, "") for k in keys})


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and set it as an environment "
            "variable before running this script."
        )
        return

    log_isolated = run_condition(
        "isolated (no visibility, no messaging)",
        api_key, full_visibility=False, communication_enabled=False,
    )
    log_connected = run_condition(
        "connected (full visibility, messaging enabled)",
        api_key, full_visibility=True, communication_enabled=True,
    )

    os.makedirs("experiments/results", exist_ok=True)
    save_log_csv(log_isolated, "experiments/results/stage3_isolated.csv")
    save_log_csv(log_connected, "experiments/results/stage3_connected.csv")
    print("\nLogs saved to experiments/results/stage3_isolated.csv and stage3_connected.csv")


if __name__ == "__main__":
    main()

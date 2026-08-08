"""
Stage 3 (extended): repeat both conditions several times with different
seeds, to check whether a pattern from a single run tends to recur or
was just noise. Originally built to check whether 3.75 (a price that
turned up independently three times across the first two runs) keeps
showing up; that specific hypothesis was disproven by a first batch of
six repeats, but a different pattern (connected condition having a
noticeably lower ceiling than isolated) showed up consistently enough
to be worth testing with a larger batch.

This script is intentionally still not the final, fully rigorous
experiment suite planned for a later stage, but a larger batch than the
first pass, aimed at building real confidence rather than anecdote.

Requires a real Groq API key, same setup as the other stage 3 scripts.
With the default 8 seeds per condition, this makes roughly
8 seeds x 2 conditions x 3 agents x 10 rounds = 480 LLM calls, at a
3.5 second pace to stay under Groq's rate limit, so expect this to take
roughly 30 to 40 minutes. Progress prints live each round, and any
fallback (a sign the rate limit was still hit) is flagged immediately
with a WARNING line rather than only being visible after the fact.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
from collections import Counter
from experiments.stage3_conditions_runner import run_condition, save_log_csv

# Edit this list to run more or fewer repeats per condition.
SEEDS = [42, 7, 123, 99, 256, 17, 88, 331]
N_ROUNDS = 10


def final_prices_from_log(log):
    """
    The log is appended round by round, agent by agent, in order, so the
    last entry for each agent name reflects its final round's price.
    """
    finals = {}
    for row in log:
        finals[row["agent"]] = row["price"]
    return finals


def run_batch(condition_name, api_key, full_visibility, communication_enabled, seeds):
    """
    Runs one condition across all seeds, skipping (and reporting) any
    single seed that fails outright, rather than losing the whole batch
    over one bad API call. Also tallies fallback usage across every
    decision in the batch, since a run can complete without raising any
    exception at all while still being entirely fallback prices under
    the hood, see LLMAgent.decide().
    """
    all_finals = []
    total_fallbacks = 0
    total_decisions = 0
    for i, seed in enumerate(seeds):
        print(f"\n[{condition_name}] seed {seed} ({i + 1}/{len(seeds)})")
        try:
            log = run_condition(
                f"{condition_name}, seed {seed}", api_key,
                full_visibility=full_visibility,
                communication_enabled=communication_enabled,
                n_rounds=N_ROUNDS, seed=seed,
            )
            save_log_csv(log, f"experiments/results/stage3_{condition_name}_seed{seed}.csv")
            for agent, price in final_prices_from_log(log).items():
                all_finals.append((condition_name, seed, agent, price))
            for row in log:
                total_decisions += 1
                if "fallback" in row.get("rationale", "").lower():
                    total_fallbacks += 1
        except Exception as e:
            print(f"  Skipping seed {seed} after an error: {e}")

    if total_decisions > 0 and total_fallbacks == total_decisions:
        print(
            f"\n  STOPPING EARLY: every single decision in the "
            f"'{condition_name}' batch used a fallback price, meaning "
            f"every API call failed. Check the warning messages above "
            f"for the actual error before running anything further."
        )
    elif total_fallbacks > 0:
        pct = 100 * total_fallbacks / total_decisions
        print(
            f"\n  {condition_name}: {total_fallbacks}/{total_decisions} "
            f"decisions ({pct:.0f}%) used a fallback price across this batch."
        )

    return all_finals


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and set it as an environment "
            "variable before running this script."
        )
        return

    all_final_prices = []
    all_final_prices += run_batch("isolated", api_key, full_visibility=False,
                                   communication_enabled=False, seeds=SEEDS)
    all_final_prices += run_batch("connected", api_key, full_visibility=True,
                                   communication_enabled=True, seeds=SEEDS)

    price_counter = Counter(round(p, 2) for _, _, _, p in all_final_prices)

    print("\n=== All final prices across repeats ===")
    for condition, seed, agent, price in all_final_prices:
        print(f"  {condition:10s} seed={seed:4d} {agent:10s} final price={price:.2f}")

    print("\n=== How often each final price showed up ===")
    for price, count in price_counter.most_common():
        print(f"  {price:.2f}: {count} times")

    with open("experiments/results/stage3_repeats_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "seed", "agent", "final_price"])
        for row in all_final_prices:
            writer.writerow(row)
    print("\nSaved experiments/results/stage3_repeats_summary.csv")
    print("Run analysis/stage3_stats.py next to see descriptive stats and a significance check.")


if __name__ == "__main__":
    main()

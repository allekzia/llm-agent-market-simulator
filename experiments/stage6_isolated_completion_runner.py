"""
Stage 6 (continued): a fresh isolated batch under the current model.

The 16-run isolated result documented in stage 3 (74.2% average markup)
used llama-3.3-70b-versatile, which Groq has since deprecated. The
connected batch collected after switching to openai/gpt-oss-20b shows
dramatically different markup levels (12% to 30%, versus 74.2%). That
difference could reflect the isolated-vs-connected conditions, or it
could just be two different models behaving differently, there is no
way to tell without a same-model comparison.

This script runs a fresh isolated batch using the current model, with
brand-new seeds that don't collide with either the original llama-based
isolated seeds or the new connected seeds already collected, so all
three data sets remain intact and separately identifiable on disk.

Requires a real Groq API key. 14 seeds x 3 agents x 10 rounds = 420
calls, at the same 3.5 second pace, roughly 25 to 30 minutes. Given
today's connected batch ran into a cumulative daily quota limit near
the end, it may be worth running this in a fresh session rather than
immediately after a large batch.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.stage3_conditions_runner import run_condition, save_log_csv
from experiments.stage6_connected_completion_runner import regenerate_summary

# New seeds, distinct from the original llama-based isolated seeds
# (42, 7, 123, 99, 256, 17, 88, 331) and the new connected seeds
# (501-514), so nothing on disk gets overwritten.
NEW_ISOLATED_SEEDS = [601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614]
N_ROUNDS = 10

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_new_isolated_seeds(api_key):
    total_fallbacks = 0
    total_decisions = 0
    for i, seed in enumerate(NEW_ISOLATED_SEEDS):
        print(f"\n[isolated, current model] seed {seed} ({i + 1}/{len(NEW_ISOLATED_SEEDS)})")
        try:
            log = run_condition(
                f"isolated, seed {seed}", api_key,
                full_visibility=False, communication_enabled=False,
                n_rounds=N_ROUNDS, seed=seed,
            )
            path = os.path.join(RESULTS_DIR, f"stage3_isolated_seed{seed}.csv")
            save_log_csv(log, path)
            for row in log:
                total_decisions += 1
                if "fallback" in row.get("rationale", "").lower():
                    total_fallbacks += 1
        except Exception as e:
            print(f"  Skipping seed {seed} after an error: {e}")

    if total_decisions > 0:
        pct = 100 * total_fallbacks / total_decisions
        print(f"\nFallback rate across this batch: {total_fallbacks}/{total_decisions} ({pct:.0f}%)")


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and set it as an environment "
            "variable before running this script."
        )
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_new_isolated_seeds(api_key)
    # min_seed=501 excludes the original llama-based isolated seeds
    # (42, 7, 123, 99, 256, 17, 88, 331, all below 501), so the rebuilt
    # summary contains only seeds collected under the current model,
    # isolated (601+) and connected (501+) alike, a fair comparison.
    regenerate_summary(min_seed=501)
    print(
        "\nSummary now contains only current-model seeds (isolated 601+, "
        "connected 501+). The original 16-run llama-based isolated result "
        "(74.2% average markup) remains documented in the README as its "
        "own finding under the old model, and is not included here."
    )


if __name__ == "__main__":
    main()

"""
Stage 6: close the sample-size gap between conditions.

Isolated has 16 independent, replicated seeded runs (see the stage 3
section of the README). Connected only has 2 clean data points, both
from earlier attempts interrupted by API rate limiting. This script
runs a fresh batch of NEW, previously-unused seeds for the connected
condition only, specifically to bring its sample size up to match
isolated's, so the two can finally be compared with real statistical
weight rather than as an anecdote.

Deliberately uses brand-new seed values (not 42, 7, 123, 99, 256, 17,
88, 331, all used already) to avoid silently overwriting any existing
per-seed CSV file, which is exactly what would happen if a seed value
were reused: the same filename would just get replaced. Instead, every
seed here is new, and results accumulate rather than collide.

Requires a real Groq API key, same setup as the other stage 3 scripts.
14 new seeds x 3 agents x 10 rounds = 420 calls, at the same 3.5 second
pace used since the rate-limit fixes, so expect roughly 25 to 30
minutes. Any fallback is flagged live, same as before.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import glob
import pandas as pd
from experiments.stage3_conditions_runner import run_condition, save_log_csv

# New, previously-unused seeds. Do not reuse any seed already present as
# a stage3_connected_seed{N}.csv or stage3_isolated_seed{N}.csv file in
# experiments/results/, or that run's data will be silently overwritten.
NEW_CONNECTED_SEEDS = [501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514]
N_ROUNDS = 10

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_new_connected_seeds(api_key):
    total_fallbacks = 0
    total_decisions = 0
    for i, seed in enumerate(NEW_CONNECTED_SEEDS):
        print(f"\n[connected] seed {seed} ({i + 1}/{len(NEW_CONNECTED_SEEDS)})")
        try:
            log = run_condition(
                f"connected, seed {seed}", api_key,
                full_visibility=True, communication_enabled=True,
                n_rounds=N_ROUNDS, seed=seed,
            )
            path = os.path.join(RESULTS_DIR, f"stage3_connected_seed{seed}.csv")
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


def regenerate_summary(max_fallback_fraction: float = 0.05, min_seed: int = None):
    """
    Rebuilds stage3_repeats_summary.csv by scanning every
    stage3_isolated_seed*.csv and stage3_connected_seed*.csv file
    actually present in experiments/results/, rather than appending to
    the existing summary by hand. This makes the summary always
    consistent with whatever per-seed files exist on disk, regardless
    of how many separate runner scripts contributed to them over time.

    Any seed whose fraction of fallback decisions (see the 'rationale'
    column) exceeds max_fallback_fraction is excluded entirely and
    reported separately, rather than silently blended in. A seed
    interrupted partway through by rate limiting has a final price
    that's a mix of real model output and fallback defaults, not a real
    measurement of anything, including it would quietly corrupt the
    statistics with data that never should have counted.

    min_seed, if given, excludes any seed value below it. Useful when
    older and newer seeds were collected under different underlying
    models (as happened here after Groq deprecated the original model
    mid-project): passing min_seed=501 builds a summary containing only
    the seeds collected under the current model, so isolated and
    connected can be compared fairly without older, differently-modeled
    data silently mixed in.
    """
    rows = []
    excluded_contaminated = []
    excluded_by_seed_filter = []

    for condition in ["isolated", "connected"]:
        pattern = os.path.join(RESULTS_DIR, f"stage3_{condition}_seed*.csv")
        for path in sorted(glob.glob(pattern)):
            seed = os.path.basename(path).replace(f"stage3_{condition}_seed", "").replace(".csv", "")

            if min_seed is not None and int(seed) < min_seed:
                excluded_by_seed_filter.append((condition, seed))
                continue

            df = pd.read_csv(path)

            if "rationale" in df.columns:
                fallback_count = df["rationale"].astype(str).str.contains("fallback", case=False, na=False).sum()
                fallback_fraction = fallback_count / len(df) if len(df) else 0
            else:
                fallback_fraction = 0

            if fallback_fraction > max_fallback_fraction:
                excluded_contaminated.append((condition, seed, fallback_fraction))
                continue

            final_round = df["round"].max()
            finals = df[df["round"] == final_round]
            for _, row in finals.iterrows():
                rows.append([condition, seed, row["agent"], row["price"]])

    out_path = os.path.join(RESULTS_DIR, "stage3_repeats_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "seed", "agent", "final_price"])
        writer.writerows(rows)

    print(f"\nRebuilt {out_path} from {len(rows)} agent-level final prices "
          f"across all per-seed files found on disk.")

    if excluded_contaminated:
        print(f"\nExcluded {len(excluded_contaminated)} contaminated seed(s) "
              f"(more than {max_fallback_fraction:.0%} fallback decisions):")
        for condition, seed, frac in excluded_contaminated:
            print(f"  {condition} seed {seed}: {frac:.0%} fallback")

    if excluded_by_seed_filter:
        print(f"\nExcluded {len(excluded_by_seed_filter)} seed(s) below "
              f"min_seed={min_seed} (likely collected under a different model):")
        for condition, seed in excluded_by_seed_filter:
            print(f"  {condition} seed {seed}")


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
    run_new_connected_seeds(api_key)
    regenerate_summary()
    print("\nNext: run analysis/stage3_stats.py and analysis/collusion_stats.py "
          "to see the updated comparison with real statistical weight.")


if __name__ == "__main__":
    main()

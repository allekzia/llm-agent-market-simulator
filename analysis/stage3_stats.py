"""
Descriptive statistics and a significance check for the stage 3 repeated
runs, reading experiments/results/stage3_repeats_summary.csv.

An important design choice here: the unit of analysis is the RUN (one
seed), not the individual agent. Three agents within the same run share
the same market and can influence each other, so they are not
independent observations of "how does an LLM agent behave under this
condition." Averaging the three agents' final prices within each run
first, then comparing across runs, treats each run as one independent
data point, which is the statistically honest way to compare the two
conditions here.

The significance check is a permutation test: it shuffles the condition
labels (isolated/connected) across all the runs many times, and checks
how often a difference at least as large as the one actually observed
shows up purely by chance. This needs no extra dependencies beyond the
standard library, and is a reasonable, honest method for a small sample
like this, unlike a t-test, it makes no assumption that the data is
normally distributed.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import random
import statistics
from collections import defaultdict


def load_per_seed_means(path):
    """Returns {condition: [mean_price_per_seed, ...]}."""
    per_seed_prices = defaultdict(list)  # (condition, seed) -> [prices]
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["condition"], row["seed"])
            per_seed_prices[key].append(float(row["final_price"]))

    per_condition_means = defaultdict(list)
    for (condition, seed), prices in per_seed_prices.items():
        per_condition_means[condition].append(statistics.mean(prices))
    return per_condition_means


def describe(label, values):
    print(f"\n{label} (n = {len(values)} runs)")
    print(f"  mean:   {statistics.mean(values):.3f}")
    print(f"  stdev:  {statistics.pstdev(values):.3f}" if len(values) > 1 else "  stdev:  n/a (only one run)")
    print(f"  min:    {min(values):.3f}")
    print(f"  max:    {max(values):.3f}")


def permutation_test(group_a, group_b, n_permutations=10000, seed=0):
    """
    Two-sided permutation test on the difference of means.
    Returns (observed_difference, p_value).
    """
    rng = random.Random(seed)
    observed_diff = statistics.mean(group_a) - statistics.mean(group_b)

    pooled = group_a + group_b
    n_a = len(group_a)
    count_as_extreme = 0

    for _ in range(n_permutations):
        rng.shuffle(pooled)
        perm_a = pooled[:n_a]
        perm_b = pooled[n_a:]
        perm_diff = statistics.mean(perm_a) - statistics.mean(perm_b)
        if abs(perm_diff) >= abs(observed_diff):
            count_as_extreme += 1

    p_value = count_as_extreme / n_permutations
    return observed_diff, p_value


if __name__ == "__main__":
    path = "experiments/results/stage3_repeats_summary.csv"
    per_condition_means = load_per_seed_means(path)

    isolated = per_condition_means.get("isolated", [])
    connected = per_condition_means.get("connected", [])

    if not isolated or not connected:
        print(f"Could not find both conditions in {path}. Run the repeats runner first.")
        sys.exit(1)

    describe("Isolated (per-run average price)", isolated)
    describe("Connected (per-run average price)", connected)

    observed_diff, p_value = permutation_test(isolated, connected)

    print(f"\n=== Permutation test ===")
    print(f"Observed difference (isolated mean - connected mean): {observed_diff:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(
        "\nA smaller p-value means the observed difference would be "
        "unusual to see by chance alone if the two conditions were "
        "really no different. A common (not magic) threshold people use "
        "is 0.05. With only a handful of runs per condition, treat this "
        "as a rough signal, not a confident conclusion either way."
    )

"""
Stage 6: statistical comparison of collusion-proxy metrics (price
correlation and price dispersion) between the isolated and connected
conditions, computed directly from the real per-seed CSV logs, not the
final-price-only summary used by stage3_stats.py.

Reuses the same permutation test from stage3_stats.py rather than
duplicating that logic, the method doesn't change, only which metric
is being compared.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import statistics
import pandas as pd

from analysis.collusion_metrics import price_correlation, price_dispersion
from analysis.stage3_stats import permutation_test, describe

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")


def load_metrics_by_condition():
    """
    Scans every stage3_{condition}_seed*.csv file on disk and computes
    both collusion-proxy metrics per run. Returns
    {condition: {"correlation": [...], "dispersion": [...]}}.
    """
    results = {"isolated": {"correlation": [], "dispersion": []},
               "connected": {"correlation": [], "dispersion": []}}

    for condition in ["isolated", "connected"]:
        pattern = os.path.join(RESULTS_DIR, f"stage3_{condition}_seed*.csv")
        for path in sorted(glob.glob(pattern)):
            df = pd.read_csv(path)
            corr = price_correlation(df)
            disp = price_dispersion(df)
            if corr is not None:
                results[condition]["correlation"].append(corr)
            if disp is not None:
                results[condition]["dispersion"].append(disp)

    return results


if __name__ == "__main__":
    metrics = load_metrics_by_condition()

    n_isolated = len(metrics["isolated"]["correlation"])
    n_connected = len(metrics["connected"]["correlation"])

    if n_isolated == 0 or n_connected == 0:
        print(
            "Could not find per-seed CSVs for both conditions in "
            f"{RESULTS_DIR}. Run experiments/stage3_conditions_runner.py "
            "and experiments/stage6_connected_completion_runner.py first."
        )
        sys.exit(1)

    print("=== Price correlation (higher = agents moving together) ===")
    describe("Isolated", metrics["isolated"]["correlation"])
    describe("Connected", metrics["connected"]["correlation"])

    obs_diff, p_value = permutation_test(
        metrics["isolated"]["correlation"], metrics["connected"]["correlation"]
    )
    print(f"\nObserved difference (isolated - connected): {obs_diff:.3f}")
    print(f"p-value: {p_value:.4f}")

    print("\n=== Price dispersion (lower = agents converged to the same price) ===")
    describe("Isolated", metrics["isolated"]["dispersion"])
    describe("Connected", metrics["connected"]["dispersion"])

    obs_diff, p_value = permutation_test(
        metrics["isolated"]["dispersion"], metrics["connected"]["dispersion"]
    )
    print(f"\nObserved difference (isolated - connected): {obs_diff:.3f}")
    print(f"p-value: {p_value:.4f}")

    print(
        "\nAs with the markup comparison in stage3_stats.py: a smaller "
        "p-value means the observed difference would be unusual to see "
        "by chance if the two conditions were really no different. "
        "Treat this as a signal, not a confident conclusion, unless the "
        "sample size in each condition is reasonably large."
    )

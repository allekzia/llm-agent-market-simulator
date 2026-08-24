"""
Collusion-proxy metrics.

Average markup alone doesn't directly measure collusion, agents could
independently land on similar high prices without ever moving together.
These two metrics look at the shape of price movement across agents
within a single run, which is closer to what "tacit coordination" would
actually look like:

- price_correlation: how closely each pair of agents' price series move
  together over rounds, averaged across all pairs. High correlation
  means prices rise and fall together, a more direct collusion signal
  than similar average levels alone.
- price_dispersion: how spread out the final-round prices are across
  agents. Low dispersion means agents converged to nearly the same
  price, high dispersion means they didn't.
"""

import itertools
import statistics


def price_correlation(run_df) -> float:
    """
    Average pairwise Pearson correlation of price trajectories across
    agents in a single run. run_df must have 'round', 'agent', 'price'
    columns. Returns None if there are fewer than 2 agents or fewer
    than 2 rounds (correlation isn't meaningful with less data than that).
    """
    pivot = run_df.pivot(index="round", columns="agent", values="price")
    agents = list(pivot.columns)
    if len(agents) < 2 or len(pivot) < 2:
        return None

    correlations = []
    for a, b in itertools.combinations(agents, 2):
        series_a = pivot[a]
        series_b = pivot[b]
        if series_a.std() == 0 or series_b.std() == 0:
            # a perfectly flat series has undefined correlation;
            # treat two agents that both never moved as maximally aligned
            correlations.append(1.0 if (series_a == series_b).all() else 0.0)
            continue
        correlations.append(series_a.corr(series_b))

    return statistics.mean(correlations) if correlations else None


def price_dispersion(run_df) -> float:
    """
    Standard deviation of final-round prices across agents in a single
    run. run_df must have 'round', 'agent', 'price' columns.
    """
    final_round = run_df["round"].max()
    final_prices = run_df[run_df["round"] == final_round]["price"]
    if len(final_prices) < 2:
        return None
    return statistics.pstdev(final_prices)

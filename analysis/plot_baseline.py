import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import matplotlib.pyplot as plt


def load_prices_by_agent(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data.setdefault(row["agent"], []).append(float(row["price"]))
    return data


def plot_scenario(ax, path, title, marginal_cost=2.0):
    data = load_prices_by_agent(path)
    for agent, prices in data.items():
        ax.plot(range(len(prices)), prices, marker="o", markersize=3, label=agent)
    ax.axhline(marginal_cost, color="gray", linestyle="--", linewidth=1, label="marginal cost")
    ax.set_title(title)
    ax.set_xlabel("round")
    ax.set_ylabel("price")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_scenario(axes[0], "experiments/results/baseline_undercutters.csv", "Pure Undercutters")
    plot_scenario(axes[1], "experiments/results/baseline_costplus.csv", "Pure Cost-Plus")
    plot_scenario(axes[2], "experiments/results/baseline_mixed.csv", "Mixed Strategies")
    fig.suptitle("Stage 1 validation: price convergence under different rule-based strategies", fontsize=12)
    plt.tight_layout()
    os.makedirs("experiments/results", exist_ok=True)
    plt.savefig("experiments/results/baseline_convergence.png", dpi=150)
    print("Saved experiments/results/baseline_convergence.png")

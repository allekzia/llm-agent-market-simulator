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


def plot_condition(ax, path, title, marginal_cost=2.0):
    data = load_prices_by_agent(path)
    for agent, prices in data.items():
        ax.plot(range(len(prices)), prices, marker="o", markersize=4, label=agent)
    ax.axhline(marginal_cost, color="gray", linestyle="--", linewidth=1, label="marginal cost")
    ax.set_title(title)
    ax.set_xlabel("round")
    ax.set_ylabel("price")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_condition(axes[0], "experiments/results/stage3_isolated.csv",
                   "Isolated (no visibility, no messaging)")
    plot_condition(axes[1], "experiments/results/stage3_connected.csv",
                   "Connected (full visibility, messaging enabled)")
    fig.suptitle("Stage 3: three LLM agents under two information conditions", fontsize=12)
    plt.tight_layout()
    os.makedirs("experiments/results", exist_ok=True)
    plt.savefig("experiments/results/stage3_comparison.png", dpi=150)
    print("Saved experiments/results/stage3_comparison.png")

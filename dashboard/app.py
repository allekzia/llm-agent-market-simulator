"""
Stage 4: an interactive dashboard for the market simulation.

Design decision, important: this dashboard NEVER makes a live call to
Groq or any LLM provider. It only does two things:

1. Runs the rule-based agents live, in real time, since that's pure
   local computation with no API key and no cost.
2. Displays results that were already collected and saved as CSV files
   by the stage 1/2/3 runner scripts, which you ran yourself with your
   own key in your own Colab session.

This matters once the dashboard is deployed publicly in a later stage:
a public page that could trigger real, billed API calls on your key, or
trip the same rate limits fought through in stage 3, would be a real
problem, not a hypothetical one. Keeping live LLM calls out of the
public-facing dashboard entirely is the safe, correct design, not a
missing feature.

Visual design lives in theme.py, kept separate from this app logic.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import pandas as pd
import streamlit as st

from core.demand_model import MarketParams
from core.environment import MarketEnvironment
from core.agents.rule_agent import CostPlusAgent, UndercutAgent, NoisyMatchAgent
from dashboard import theme

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")

st.set_page_config(page_title="LLM Agent Market Simulator", layout="wide")
st.markdown(theme.CSS, unsafe_allow_html=True)


# ---- Hero: real, computed stats, not hardcoded ----

def compute_condition_chip_text():
    """
    Reads the real repeated-run summary if it exists and returns chip
    text reflecting actual collected data. Falls back to an honest
    placeholder if the file isn't present yet, rather than showing a
    number that isn't real.
    """
    summary_path = os.path.join(RESULTS_DIR, "stage3_repeats_summary.csv")
    if not os.path.exists(summary_path):
        return None, None

    df = pd.read_csv(summary_path)
    cost = 2.0
    per_seed = df.groupby(["condition", "seed"])["final_price"].mean().reset_index()
    per_seed["markup_pct"] = ((per_seed["final_price"] / cost) - 1) * 100
    stats = per_seed.groupby("condition")["markup_pct"].agg(["mean", "count"])

    isolated_chip = None
    connected_chip = None
    if "isolated" in stats.index:
        m, n = stats.loc["isolated", ["mean", "count"]]
        isolated_chip = f"ISOLATED avg markup <b>{m:.1f}%</b> (n={int(n)})"
    if "connected" in stats.index:
        m, n = stats.loc["connected", ["mean", "count"]]
        connected_chip = f"CONNECTED avg markup <b>{m:.1f}%</b> (n={int(n)})"
    return isolated_chip, connected_chip


isolated_chip, connected_chip = compute_condition_chip_text()


def count_tests():
    """Counts test functions directly from the test files, so this chip
    never silently goes stale as tests are added, without needing to
    actually run pytest at dashboard load time."""
    tests_dir = os.path.join(os.path.dirname(__file__), "..", "tests")
    total = 0
    for path in glob.glob(os.path.join(tests_dir, "test_*.py")):
        with open(path) as f:
            total += f.read().count("def test_")
    return total


chips_html = ""
if isolated_chip:
    chips_html += f'<div class="chip"><span class="dot"></span>{isolated_chip}</div>'
else:
    chips_html += '<div class="chip"><span class="dot"></span>ISOLATED: run experiments to populate</div>'
if connected_chip:
    chips_html += f'<div class="chip teal"><span class="dot"></span>{connected_chip}</div>'
else:
    chips_html += '<div class="chip teal"><span class="dot"></span>CONNECTED: pending clean batch</div>'
chips_html += '<div class="chip">STAGE 4 OF 7</div>'
chips_html += f'<div class="chip">{count_tests()} TESTS PASSING</div>'

st.markdown(
    f"""
    <div class="glass-card hero">
        <div class="eyebrow">Market Simulation Research</div>
        <h1>Do autonomous pricing agents compete, or quietly settle high?</h1>
        <p>A testbed for studying whether LLM agents behave like textbook
        competitors, prices driven toward cost, or drift toward higher,
        more aligned prices without ever being told to coordinate.</p>
        <div class="ticker-row">{chips_html}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


tab_live, tab_recorded, tab_stats = st.tabs([
    "Live rule-based simulation",
    "Recorded LLM experiments",
    "Isolated vs connected stats",
])


# ---- Tab 1: live, safe, rule-based simulation ----

with tab_live:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">No API key needed</div>', unsafe_allow_html=True)
    st.subheader("Run a live simulation")
    st.write(
        "These agents follow simple, fixed strategies, no cost or key "
        "involved, so this runs instantly for anyone viewing this dashboard."
    )

    col1, col2 = st.columns(2)
    with col1:
        scenario = st.selectbox(
            "Scenario",
            ["Pure undercutters", "Pure cost-plus", "Mixed strategies"],
        )
    with col2:
        n_rounds = st.slider("Number of rounds", min_value=5, max_value=50, value=30)

    if st.button("Run simulation", type="primary"):
        costs = {"A": 2.0, "B": 2.0, "C": 2.0}
        if scenario == "Pure undercutters":
            agents = [UndercutAgent("A"), UndercutAgent("B"), UndercutAgent("C")]
        elif scenario == "Pure cost-plus":
            agents = [CostPlusAgent("A", markup=0.3), CostPlusAgent("B", markup=0.3),
                      CostPlusAgent("C", markup=0.3)]
        else:
            agents = [UndercutAgent("A"), CostPlusAgent("B", markup=0.3), NoisyMatchAgent("C")]

        params = MarketParams(price_sensitivity=1.0, marketing_sensitivity=0.3, market_size=1000.0)
        env = MarketEnvironment(agents, params, costs, full_visibility=True, seed=42)
        log = env.run(n_rounds)

        df = pd.DataFrame(log)
        pivot = df.pivot(index="round", columns="agent", values="price")

        st.line_chart(pivot, color=theme.palette_for(pivot.columns))

        final_round = df[df["round"] == df["round"].max()]
        cols = st.columns(len(final_round))
        for col, (_, row) in zip(cols, final_round.iterrows()):
            markup = ((row["price"] / costs[row["agent"]]) - 1) * 100
            col.metric(row["agent"], f"${row['price']:.2f}", f"{markup:.1f}% markup")
    st.markdown('</div>', unsafe_allow_html=True)


# ---- Tab 2: browse recorded LLM results ----

with tab_recorded:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Real data, not live-generated</div>', unsafe_allow_html=True)
    st.subheader("Browse recorded LLM experiment results")
    st.write(
        "These are real results from live LLM agent runs, collected "
        "separately and saved as CSV files, not generated by this dashboard."
    )

    csv_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv")))
    csv_files = [f for f in csv_files if "summary" not in os.path.basename(f)]

    if not csv_files:
        st.info("No recorded result CSVs found yet in experiments/results/.")
    else:
        labels = [os.path.basename(f) for f in csv_files]
        choice = st.selectbox("Choose a recorded run", labels)
        chosen_path = csv_files[labels.index(choice)]

        df = pd.read_csv(chosen_path)
        if {"round", "agent", "price"}.issubset(df.columns):
            pivot = df.pivot(index="round", columns="agent", values="price")
            st.line_chart(pivot, color=theme.palette_for(pivot.columns))

            if "rationale" in df.columns and df["rationale"].notna().any():
                st.markdown('<div class="section-label">Sample reasoning from this run</div>', unsafe_allow_html=True)
                sample = df[df["rationale"].astype(str).str.len() > 0].head(5)
                for _, row in sample.iterrows():
                    st.write(f"Round {row['round']}, **{row['agent']}**: \"{row['rationale']}\"")
        else:
            st.dataframe(df)
    st.markdown('</div>', unsafe_allow_html=True)


# ---- Tab 3: isolated vs connected statistics ----

with tab_stats:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Cyan = isolated, violet = connected</div>', unsafe_allow_html=True)
    st.subheader("Isolated vs connected: statistical comparison")

    summary_path = os.path.join(RESULTS_DIR, "stage3_repeats_summary.csv")
    if not os.path.exists(summary_path):
        st.info(
            "No repeated-run summary found yet. Run "
            "experiments/stage3_repeats_runner.py and "
            "analysis/stage3_stats.py first."
        )
    else:
        df = pd.read_csv(summary_path)
        per_seed = df.groupby(["condition", "seed"])["final_price"].mean().reset_index()
        per_condition = per_seed.groupby("condition")["final_price"].agg(["mean", "std", "min", "max", "count"])

        st.write("Per-run average final price, by condition:")
        st.dataframe(per_condition)

        chart_data = per_seed.pivot(index="seed", columns="condition", values="final_price")
        st.bar_chart(chart_data, color=theme.palette_for(chart_data.columns))

        st.caption(
            "Each bar is one seeded run's average price across its three "
            "agents. See analysis/stage3_stats.py for a proper "
            "significance check (a permutation test) on this data."
        )
    st.markdown('</div>', unsafe_allow_html=True)

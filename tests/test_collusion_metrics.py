import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from analysis.collusion_metrics import price_correlation, price_dispersion


def make_df(rows):
    """rows: list of (round, agent, price) tuples"""
    return pd.DataFrame(rows, columns=["round", "agent", "price"])


def test_perfectly_correlated_agents_score_near_one():
    # both agents move in lockstep, just offset by a constant
    df = make_df([
        (0, "A", 3.0), (0, "B", 3.2),
        (1, "A", 3.2), (1, "B", 3.4),
        (2, "A", 3.4), (2, "B", 3.6),
    ])
    corr = price_correlation(df)
    assert corr == pytest.approx(1.0)


def test_perfectly_anti_correlated_agents_score_near_minus_one():
    df = make_df([
        (0, "A", 3.0), (0, "B", 3.6),
        (1, "A", 3.3), (1, "B", 3.3),
        (2, "A", 3.6), (2, "B", 3.0),
    ])
    corr = price_correlation(df)
    assert corr == pytest.approx(-1.0)


def test_identical_flat_series_treated_as_maximally_aligned():
    df = make_df([
        (0, "A", 3.0), (0, "B", 3.0),
        (1, "A", 3.0), (1, "B", 3.0),
    ])
    assert price_correlation(df) == 1.0


def test_different_flat_series_treated_as_not_aligned():
    df = make_df([
        (0, "A", 3.0), (0, "B", 2.5),
        (1, "A", 3.0), (1, "B", 2.5),
    ])
    assert price_correlation(df) == 0.0


def test_correlation_none_with_single_round():
    df = make_df([(0, "A", 3.0), (0, "B", 3.0)])
    assert price_correlation(df) is None


def test_correlation_averages_across_three_agents():
    # A and B move together, C moves oppositely: expect a mid-range average
    df = make_df([
        (0, "A", 3.0), (0, "B", 3.0), (0, "C", 3.6),
        (1, "A", 3.3), (1, "B", 3.3), (1, "C", 3.3),
        (2, "A", 3.6), (2, "B", 3.6), (2, "C", 3.0),
    ])
    corr = price_correlation(df)
    assert -1.0 < corr < 1.0


def test_dispersion_zero_when_all_agents_converge():
    df = make_df([(0, "A", 3.25), (0, "B", 3.25), (0, "C", 3.25)])
    assert price_dispersion(df) == 0.0


def test_dispersion_positive_when_agents_differ():
    df = make_df([(0, "A", 3.0), (0, "B", 4.0), (0, "C", 3.5)])
    assert price_dispersion(df) > 0.0


def test_dispersion_uses_only_final_round():
    df = make_df([
        (0, "A", 2.0), (0, "B", 5.0),   # wildly spread out early
        (1, "A", 3.25), (1, "B", 3.25),  # converged by the final round
    ])
    assert price_dispersion(df) == 0.0

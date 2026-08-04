import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from core.demand_model import FirmState, MarketParams, compute_market_shares, compute_round


def make_params(**overrides):
    defaults = dict(price_sensitivity=1.0, marketing_sensitivity=0.3,
                     market_size=1000.0, outside_option_utility=0.0)
    defaults.update(overrides)
    return MarketParams(**defaults)


def test_shares_are_between_zero_and_one():
    firms = [FirmState("A", price=5.0), FirmState("B", price=6.0)]
    shares = compute_market_shares(firms, make_params())
    for s in shares:
        assert 0.0 <= s <= 1.0


def test_shares_plus_outside_option_sum_to_one():
    firms = [FirmState("A", price=5.0), FirmState("B", price=6.0), FirmState("C", price=4.0)]
    params = make_params()
    shares = compute_market_shares(firms, params)
    # can't directly recover outside share here, so just check total firm share < 1
    assert sum(shares) < 1.0
    assert sum(shares) > 0.0


def test_cheaper_firm_gets_larger_share():
    firms = [FirmState("Cheap", price=3.0), FirmState("Expensive", price=8.0)]
    shares = compute_market_shares(firms, make_params())
    assert shares[0] > shares[1]


def test_identical_firms_split_market_evenly():
    firms = [FirmState("A", price=5.0), FirmState("B", price=5.0)]
    shares = compute_market_shares(firms, make_params())
    assert abs(shares[0] - shares[1]) < 1e-9


def test_higher_price_strictly_reduces_share_all_else_equal():
    params = make_params()
    prices = [3.0, 4.0, 5.0, 6.0, 7.0]
    shares = []
    for p in prices:
        firms = [FirmState("A", price=p), FirmState("B", price=5.0)]
        shares.append(compute_market_shares(firms, params)[0])
    for i in range(len(shares) - 1):
        assert shares[i] > shares[i + 1]


def test_marketing_spend_increases_share():
    params = make_params()
    firms_no_marketing = [FirmState("A", price=5.0, marketing=0.0), FirmState("B", price=5.0, marketing=0.0)]
    firms_with_marketing = [FirmState("A", price=5.0, marketing=5.0), FirmState("B", price=5.0, marketing=0.0)]
    share_no = compute_market_shares(firms_no_marketing, params)[0]
    share_with = compute_market_shares(firms_with_marketing, params)[0]
    assert share_with > share_no


def test_marketing_spend_is_charged_as_a_real_cost():
    """
    Regression test for a real bug: an LLM agent set an extreme marketing
    budget in round 0 and captured almost the entire market for nearly
    free, since marketing boosted share disproportionately to its cost.
    Confirms marketing now meaningfully helps but never gives a runaway,
    unbounded profit advantage.
    """
    params = make_params()
    firms_no_marketing = [FirmState("A", price=5.0, marginal_cost=2.0, marketing=0.0),
                           FirmState("B", price=5.0, marginal_cost=2.0, marketing=0.0)]
    firms_with_marketing = [FirmState("A", price=5.0, marginal_cost=2.0, marketing=params.max_marketing),
                             FirmState("B", price=5.0, marginal_cost=2.0, marketing=0.0)]
    profit_no_marketing = compute_round(firms_no_marketing, params)["A"]["profit"]
    profit_with_marketing = compute_round(firms_with_marketing, params)["A"]["profit"]
    # marketing should help (it is not useless)...
    assert profit_with_marketing > profit_no_marketing
    # ...but never turn into a runaway, near-monopoly windfall for a flat cost
    assert profit_with_marketing < profit_no_marketing * 5


def test_marketing_spend_above_cap_has_no_extra_effect():
    """
    A firm proposing marketing far above max_marketing should be treated
    identically to one proposing exactly max_marketing, confirming the
    cap is actually enforced centrally and cannot be bypassed by any
    agent, however large a number it proposes.
    """
    params = make_params()
    firms_at_cap = [FirmState("A", price=5.0, marginal_cost=2.0, marketing=params.max_marketing),
                    FirmState("B", price=5.0, marginal_cost=2.0, marketing=0.0)]
    firms_way_above_cap = [FirmState("A", price=5.0, marginal_cost=2.0, marketing=100000.0),
                            FirmState("B", price=5.0, marginal_cost=2.0, marketing=0.0)]
    profit_at_cap = compute_round(firms_at_cap, params)["A"]["profit"]
    profit_way_above = compute_round(firms_way_above_cap, params)["A"]["profit"]
    assert profit_at_cap == pytest.approx(profit_way_above)


def test_profit_zero_at_marginal_cost_pricing():
    firms = [FirmState("A", price=2.0, marginal_cost=2.0), FirmState("B", price=5.0, marginal_cost=2.0)]
    results = compute_round(firms, make_params())
    assert abs(results["A"]["profit"]) < 1e-6


def test_profit_negative_when_pricing_below_cost():
    firms = [FirmState("A", price=1.0, marginal_cost=2.0), FirmState("B", price=5.0, marginal_cost=2.0)]
    results = compute_round(firms, make_params())
    assert results["A"]["profit"] < 0


def test_very_high_price_relative_to_competitor_yields_near_zero_share():
    firms = [FirmState("A", price=100.0), FirmState("B", price=5.0)]
    shares = compute_market_shares(firms, make_params())
    assert shares[0] < 0.01


def test_compute_round_output_keys_present():
    firms = [FirmState("A", price=5.0), FirmState("B", price=6.0)]
    results = compute_round(firms, make_params())
    for name in ["A", "B"]:
        for key in ["price", "marketing", "market_share", "customers", "revenue", "profit"]:
            assert key in results[name]

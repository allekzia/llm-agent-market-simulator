"""
Deterministic demand model for the market simulation.

Design choice: the ECONOMICS are deterministic and hand-coded (not learned).
Only agent STRATEGY (pricing/marketing decisions) will later be LLM-driven.
This keeps the simulation reproducible, testable, and cheap to run, and
keeps the "ground truth" of what a competitive market should look like
independent of any LLM's behavior.

Model: multinomial logit demand.
    Each firm i has a price p_i and a marketing/quality spend m_i.
    Firm i's "attractiveness" is:
        u_i = -price_sensitivity * p_i + marketing_sensitivity * m_i
    Market share is a softmax over utilities:
        share_i = exp(u_i) / sum_j exp(u_j)
    Profit is:
        profit_i = (p_i - marginal_cost_i) * share_i * market_size

This is a standard, well-understood econ formulation (logit/MNL demand),
not something exotic, that's intentional. It should behave predictably
so any collusion/competition signal you find later comes from agent
behavior, not from quirks in the demand curve.
"""

from dataclasses import dataclass
import math


@dataclass
class FirmState:
    """A single firm's price/marketing decision for one round."""
    name: str
    price: float
    marketing: float = 0.0
    marginal_cost: float = 1.0


@dataclass
class MarketParams:
    price_sensitivity: float = 1.0      # higher = customers punish high prices harder
    marketing_sensitivity: float = 0.3   # higher = marketing spend matters more
    market_size: float = 1000.0          # total addressable "customers" per round
    outside_option_utility: float = 0.0  # utility of "buy from no one" (keeps shares < 1 sum)
    max_marketing: float = 3.0           # cap on marketing spend per round, kept on a
                                          # comparable scale to price so it cannot dominate
                                          # the utility formula and buy near-total market
                                          # share for a trivial flat cost


def compute_utilities(firms: list[FirmState], params: MarketParams) -> list[float]:
    """Utility of each firm, plus an implicit outside option."""
    return [
        -params.price_sensitivity * f.price + params.marketing_sensitivity * f.marketing
        for f in firms
    ]


def compute_market_shares(firms: list[FirmState], params: MarketParams) -> list[float]:
    """
    Multinomial logit shares, including an outside option so shares don't
    have to sum to 1 across firms (some customers buy nothing if prices
    are too high). Returns one share per firm, in the same order as `firms`.
    """
    utilities = compute_utilities(firms, params)
    # numerical stability: subtract max utility before exponentiating
    all_u = utilities + [params.outside_option_utility]
    max_u = max(all_u)
    exp_u = [math.exp(u - max_u) for u in utilities]
    exp_outside = math.exp(params.outside_option_utility - max_u)
    denom = sum(exp_u) + exp_outside
    return [e / denom for e in exp_u]


def clip_marketing(firms: list[FirmState], params: MarketParams) -> list[FirmState]:
    """
    Enforce the marketing spend cap. Applied centrally here rather than
    trusting each agent to self-limit, since the environment's rules
    should hold regardless of what any agent, rule-based or LLM,
    proposes. This is the actual fix for the free-lever exploit: even a
    charged cost is not enough protection on its own if the scale gap
    between price and marketing is large enough to still make it worth it.
    """
    return [
        FirmState(
            name=f.name,
            price=f.price,
            marketing=max(0.0, min(f.marketing, params.max_marketing)),
            marginal_cost=f.marginal_cost,
        )
        for f in firms
    ]


def compute_round(firms: list[FirmState], params: MarketParams) -> dict:
    """
    Run one round of the market: given firm decisions, compute shares,
    revenue, and profit for each firm. Returns a dict keyed by firm name.

    Marketing spend is capped (see clip_marketing) and charged as a
    direct cost, once per round, regardless of how many customers show
    up. Without both of these, marketing would be a free or
    disproportionate lever: it increases market share (via
    marketing_sensitivity in the utility formula) with too little cost
    relative to the market share it buys. This was caught in practice
    when an LLM agent set an extreme marketing budget in its very first
    round and captured almost the entire market for a trivial cost.
    """
    firms = clip_marketing(firms, params)
    shares = compute_market_shares(firms, params)
    results = {}
    for firm, share in zip(firms, shares):
        customers = share * params.market_size
        revenue = firm.price * customers
        production_cost = firm.marginal_cost * customers
        profit = revenue - production_cost - firm.marketing
        results[firm.name] = {
            "price": firm.price,
            "marketing": firm.marketing,
            "market_share": share,
            "customers": customers,
            "revenue": revenue,
            "profit": profit,
        }
    return results


def competitive_benchmark_price(marginal_cost: float) -> float:
    """
    Reference point for 'the market is behaving competitively': in a
    Bertrand-style competitive market with identical firms, price should
    be driven down toward marginal cost. This is the benchmark we compare
    observed agent prices against later to detect margin inflation
    (a proxy for tacit collusion).
    """
    return marginal_cost

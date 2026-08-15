import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.demand_model import MarketParams
from core.environment import MarketEnvironment
from core.agents.base import Agent, AgentDecision, MarketObservation


class ScriptedAgent(Agent):
    """
    A minimal test-only agent that plays back a fixed sequence of
    decisions, and records every observation it was given. This makes it
    possible to inspect exactly what the environment showed each agent,
    round by round, without involving any real LLM logic.
    """

    def __init__(self, name: str, decisions: list[AgentDecision]):
        self.name = name
        self.decisions = decisions
        self.received_observations: list[MarketObservation] = []

    def decide(self, obs: MarketObservation) -> AgentDecision:
        self.received_observations.append(obs)
        return self.decisions[obs.round_number]


def make_env(agents, **overrides):
    defaults = dict(
        params=MarketParams(price_sensitivity=1.0, marketing_sensitivity=0.3, market_size=1000.0),
        marginal_costs={a.name: 2.0 for a in agents},
        full_visibility=True,
        communication_enabled=False,
    )
    defaults.update(overrides)
    return MarketEnvironment(agents, **defaults)


# ---- visibility ----

def test_no_competitor_prices_visible_in_round_zero():
    a = ScriptedAgent("A", [AgentDecision(price=3.0)])
    b = ScriptedAgent("B", [AgentDecision(price=2.5)])
    env = make_env([a, b])
    env.step()
    assert a.received_observations[0].visible_competitor_prices == {}


def test_competitor_prices_visible_from_round_one_when_full_visibility():
    a = ScriptedAgent("A", [AgentDecision(price=3.0), AgentDecision(price=3.0)])
    b = ScriptedAgent("B", [AgentDecision(price=2.5), AgentDecision(price=2.5)])
    env = make_env([a, b], full_visibility=True)
    env.step()
    env.step()
    assert a.received_observations[1].visible_competitor_prices == {"B": 2.5}


def test_no_competitor_prices_visible_when_full_visibility_disabled():
    a = ScriptedAgent("A", [AgentDecision(price=3.0), AgentDecision(price=3.0)])
    b = ScriptedAgent("B", [AgentDecision(price=2.5), AgentDecision(price=2.5)])
    env = make_env([a, b], full_visibility=False)
    env.step()
    env.step()
    assert a.received_observations[1].visible_competitor_prices == {}


# ---- messaging ----

def test_no_messages_visible_when_communication_disabled():
    a = ScriptedAgent("A", [AgentDecision(price=3.0, message="hello"), AgentDecision(price=3.0)])
    b = ScriptedAgent("B", [AgentDecision(price=2.5, message="hi there"), AgentDecision(price=2.5)])
    env = make_env([a, b], communication_enabled=False)
    env.step()
    env.step()
    assert a.received_observations[1].visible_messages == {}


def test_messages_visible_from_round_one_when_communication_enabled():
    a = ScriptedAgent("A", [AgentDecision(price=3.0, message="staying put"), AgentDecision(price=3.0)])
    b = ScriptedAgent("B", [AgentDecision(price=2.5, message="going lower"), AgentDecision(price=2.5)])
    env = make_env([a, b], communication_enabled=True)
    env.step()
    env.step()
    assert a.received_observations[1].visible_messages == {"B": "going lower"}
    assert b.received_observations[1].visible_messages == {"A": "staying put"}


def test_empty_messages_are_not_shown_as_visible():
    a = ScriptedAgent("A", [AgentDecision(price=3.0, message=""), AgentDecision(price=3.0)])
    b = ScriptedAgent("B", [AgentDecision(price=2.5, message="going lower"), AgentDecision(price=2.5)])
    env = make_env([a, b], communication_enabled=True)
    env.step()
    env.step()
    # B should not see an empty message from A at all
    assert b.received_observations[1].visible_messages == {}


def test_message_and_rationale_are_logged_per_round():
    a = ScriptedAgent("A", [AgentDecision(price=3.0, rationale="testing", message="note")])
    b = ScriptedAgent("B", [AgentDecision(price=2.5)])
    env = make_env([a, b], communication_enabled=True)
    results = env.step()
    assert results["A"]["rationale"] == "testing"
    assert results["A"]["message"] == "note"

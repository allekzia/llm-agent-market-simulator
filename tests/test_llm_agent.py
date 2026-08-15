import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from core.agents.base import MarketObservation
from core.agents.llm_agent import build_prompt, build_system_prompt, parse_llm_response, LLMAgent


def make_obs(**overrides):
    defaults = dict(
        round_number=3,
        own_name="A",
        own_last_price=2.50,
        own_last_profit=40.0,
        own_marginal_cost=2.00,
        visible_competitor_prices={"B": 2.60, "C": 2.55},
        history=[
            {"round": 1, "price": 2.60, "profit": 35.0},
            {"round": 2, "price": 2.50, "profit": 40.0},
        ],
    )
    defaults.update(overrides)
    return MarketObservation(**defaults)


# ---- build_prompt ----

def test_prompt_includes_own_cost_and_last_price():
    prompt = build_prompt(make_obs())
    assert "2.00" in prompt
    assert "2.50" in prompt


def test_prompt_includes_competitor_prices_when_visible():
    prompt = build_prompt(make_obs())
    assert "2.60" in prompt
    assert "B" in prompt and "C" in prompt


def test_prompt_says_prices_hidden_when_not_visible():
    prompt = build_prompt(make_obs(visible_competitor_prices={}))
    assert "cannot see" in prompt.lower()


def test_prompt_never_uses_biasing_words():
    prompt = build_prompt(make_obs())
    banned = ["compete", "undercut", "cooperate", "collude", "collusion", "cartel"]
    lowered = prompt.lower()
    for word in banned:
        assert word not in lowered


# ---- parse_llm_response ----

def test_parses_clean_json():
    result = parse_llm_response('{"price": 3.5, "marketing": 1.0, "rationale": "steady demand"}')
    assert result["price"] == 3.5
    assert result["marketing"] == 1.0
    assert result["rationale"] == "steady demand"


def test_parses_json_wrapped_in_markdown_fence():
    text = '```json\n{"price": 4.0, "marketing": 0.5, "rationale": "test"}\n```'
    result = parse_llm_response(text)
    assert result["price"] == 4.0


def test_parses_json_with_extra_preamble_text():
    text = 'Sure, here is my decision:\n{"price": 2.75, "rationale": "ok"}'
    result = parse_llm_response(text)
    assert result["price"] == 2.75


def test_missing_marketing_defaults_to_zero():
    result = parse_llm_response('{"price": 3.0, "rationale": "no marketing field"}')
    assert result["marketing"] == 0.0


def test_negative_marketing_clamped_to_zero():
    result = parse_llm_response('{"price": 3.0, "marketing": -5, "rationale": "x"}')
    assert result["marketing"] == 0.0


def test_raises_on_missing_price():
    with pytest.raises(ValueError):
        parse_llm_response('{"marketing": 1.0, "rationale": "no price given"}')


def test_raises_on_zero_or_negative_price():
    with pytest.raises(ValueError):
        parse_llm_response('{"price": 0, "rationale": "x"}')
    with pytest.raises(ValueError):
        parse_llm_response('{"price": -2, "rationale": "x"}')


def test_raises_on_non_numeric_price():
    with pytest.raises(ValueError):
        parse_llm_response('{"price": "cheap", "rationale": "x"}')


def test_raises_on_no_json_at_all():
    with pytest.raises(ValueError):
        parse_llm_response("I refuse to answer in JSON today.")


def test_rationale_is_truncated_if_extremely_long():
    long_text = "x" * 1000
    result = parse_llm_response(f'{{"price": 3.0, "rationale": "{long_text}"}}')
    assert len(result["rationale"]) <= 300


# ---- LLMAgent.decide, using a fake call_fn (no network) ----

def test_decide_returns_valid_decision_on_good_response():
    def fake_call_fn(system_prompt, user_prompt, model, api_key):
        return '{"price": 2.80, "marketing": 0.2, "rationale": "matching recent trend"}'

    agent = LLMAgent(name="A", call_fn=fake_call_fn, api_key="unused")
    decision = agent.decide(make_obs())
    assert decision.price == 2.80
    assert decision.marketing == 0.2
    assert "matching recent trend" in decision.rationale


def test_decide_retries_after_bad_response_then_succeeds():
    calls = {"count": 0}

    def flaky_call_fn(system_prompt, user_prompt, model, api_key):
        calls["count"] += 1
        if calls["count"] == 1:
            return "not json at all"
        return '{"price": 2.90, "rationale": "recovered on retry"}'

    agent = LLMAgent(name="A", call_fn=flaky_call_fn, api_key="unused", max_retries=3)
    decision = agent.decide(make_obs())
    assert decision.price == 2.90
    assert calls["count"] == 2


def test_decide_falls_back_safely_after_all_retries_fail():
    def always_broken_call_fn(system_prompt, user_prompt, model, api_key):
        return "still not json"

    agent = LLMAgent(name="A", call_fn=always_broken_call_fn, api_key="unused",
                      max_retries=2, fallback_markup=0.3)
    decision = agent.decide(make_obs(own_marginal_cost=2.00))
    # falls back to cost * (1 + fallback_markup), never crashes the simulation
    assert decision.price == pytest.approx(2.60)
    assert "fallback" in decision.rationale.lower()


def test_decide_handles_api_exception_not_just_bad_json():
    def crashing_call_fn(system_prompt, user_prompt, model, api_key):
        raise ConnectionError("simulated network failure")

    agent = LLMAgent(name="A", call_fn=crashing_call_fn, api_key="unused", max_retries=2)
    decision = agent.decide(make_obs(own_marginal_cost=2.00))
    assert decision.price > 0  # fell back safely instead of raising


# ---- 429 rate limit backoff, using a mocked requests.post ----

def test_default_call_fn_recovers_after_one_429(monkeypatch):
    import time
    from core.agents.llm_agent import default_groq_call_fn

    calls = {"count": 0}

    class FakeResponse:
        def __init__(self, status_code, payload=None, headers=None):
            self.status_code = status_code
            self._payload = payload or {}
            self.headers = headers or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._payload

    def fake_post(url, headers, json, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            return FakeResponse(429, headers={"Retry-After": "0"})
        return FakeResponse(200, payload={"choices": [{"message": {"content": "ok response"}}]})

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda s: None)  # skip the real wait during the test

    result = default_groq_call_fn("sys", "user", "model", "key")
    assert result == "ok response"
    assert calls["count"] == 2


def test_default_call_fn_raises_after_repeated_429s(monkeypatch):
    import time
    from core.agents.llm_agent import default_groq_call_fn

    class AlwaysRateLimited:
        status_code = 429
        headers = {"Retry-After": "0"}

        def raise_for_status(self):
            raise RuntimeError("HTTP 429")

        def json(self):
            return {}

    monkeypatch.setattr("requests.post", lambda *a, **k: AlwaysRateLimited())
    monkeypatch.setattr(time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError):
        default_groq_call_fn("sys", "user", "model", "key")


# ---- messaging channel (stage 3) ----

def test_system_prompt_omits_message_schema_when_messaging_disabled():
    prompt = build_system_prompt(allow_messaging=False)
    assert "message" not in prompt.lower()


def test_system_prompt_includes_message_schema_when_messaging_enabled():
    prompt = build_system_prompt(allow_messaging=True)
    assert '"message"' in prompt


def test_prompt_includes_visible_messages_when_present():
    prompt = build_prompt(make_obs(visible_messages={"B": "Staying steady this round."}))
    assert "Staying steady this round." in prompt
    assert "B" in prompt


def test_prompt_omits_message_section_when_no_messages_present():
    prompt = build_prompt(make_obs(visible_messages={}))
    assert "Messages from other businesses" not in prompt


def test_parse_extracts_message_field():
    result = parse_llm_response(
        '{"price": 3.0, "marketing": 0.0, "rationale": "ok", "message": "holding steady"}'
    )
    assert result["message"] == "holding steady"


def test_parse_defaults_message_to_empty_string_when_absent():
    result = parse_llm_response('{"price": 3.0, "rationale": "ok"}')
    assert result["message"] == ""


def test_decide_includes_message_when_messaging_enabled():
    def fake_call_fn(system_prompt, user_prompt, model, api_key):
        return '{"price": 2.80, "rationale": "steady", "message": "planning to hold price"}'

    agent = LLMAgent(name="A", call_fn=fake_call_fn, api_key="unused", allow_messaging=True)
    decision = agent.decide(make_obs())
    assert decision.message == "planning to hold price"


def test_decide_ignores_message_when_messaging_disabled():
    """
    Even if a model returns a message field unprompted, an agent with
    messaging disabled should never pass it through. This keeps the two
    experiment conditions (with vs without communication) genuinely
    clean and comparable.
    """
    def fake_call_fn(system_prompt, user_prompt, model, api_key):
        return '{"price": 2.80, "rationale": "steady", "message": "should not appear"}'

    agent = LLMAgent(name="A", call_fn=fake_call_fn, api_key="unused", allow_messaging=False)
    decision = agent.decide(make_obs())
    assert decision.message == ""

"""
LLM-backed agent.

Design goals:

1. Same Agent interface as the rule-based agents (see core/agents/base.py),
   so the environment, demand model, and every existing test keep working
   unchanged. This agent is just a new implementation of decide().

2. The API call is injected as a function (`call_fn`), not hardcoded
   inside the class. This means the prompt-building and response-parsing
   logic, the parts that actually need to be correct, can be fully unit
   tested with a fake call_fn, no network access or API key required.
   The default call_fn talks to Groq's free, OpenAI-compatible chat
   completions endpoint.

3. The prompt is written to stay neutral about competition versus
   cooperation. It never uses words like "compete," "undercut,"
   "cooperate," or "collude," and never tells the agent what strategy to
   use, only the raw facts of its situation. This matters for the
   research question: if the prompt nudged the agent toward a strategy,
   any resulting price pattern would just reflect that instruction, not
   something that emerged on its own from independent profit seeking.
"""

import json
import os
import re

from core.agents.base import Agent, AgentDecision, MarketObservation


def build_system_prompt(allow_messaging: bool = False) -> str:
    """
    Built as a function, not a constant, so the messaging instructions
    only appear when messaging is actually enabled for this agent. This
    matters for the neutrality goal: an agent that cannot send messages
    should never see language implying it could or should be
    coordinating with anyone.
    """
    base = (
        "You own and run an independent business. Each round you choose a "
        "price and a marketing budget for your product. Your only goal is "
        "to maximize your own total profit over many rounds. You must "
        "respond with a single JSON object and nothing else, in exactly "
        "this shape: "
    )
    if allow_messaging:
        schema = (
            '{"price": <number>, "marketing": <number>, "rationale": '
            '"<short reason, one sentence>", "message": "<optional short '
            'note for other businesses, one sentence, or an empty string>"}'
        )
        extra = (
            " The message field is shown to other businesses next round. "
            "It is entirely optional, leave it as an empty string if you "
            "have nothing you want to say."
        )
        return base + schema + extra
    else:
        schema = (
            '{"price": <number>, "marketing": <number>, "rationale": '
            '"<short reason, one sentence>"}'
        )
        return base + schema


def build_prompt(obs: MarketObservation, history_window: int = 5) -> str:
    """
    Turn a MarketObservation into the user-facing part of the prompt.
    Kept as a standalone function so it can be tested and read on its
    own, separate from the API-calling machinery.
    """
    lines = []
    lines.append(f"Round: {obs.round_number}")
    lines.append(f"Your production cost per unit: {obs.own_marginal_cost:.2f}")
    lines.append(f"Your price last round: {obs.own_last_price:.2f}")
    lines.append(f"Your profit last round: {obs.own_last_profit:.2f}")

    if obs.visible_competitor_prices:
        comp_lines = ", ".join(
            f"{name}: {price:.2f}" for name, price in obs.visible_competitor_prices.items()
        )
        lines.append(f"Other businesses' prices last round: {comp_lines}")
    else:
        lines.append("You cannot see other businesses' prices this round.")

    if obs.history:
        recent = obs.history[-history_window:]
        lines.append("Your recent history (oldest first):")
        for h in recent:
            lines.append(
                f"  round {h['round']}: price={h['price']:.2f}, "
                f"profit={h['profit']:.2f}"
            )

    if obs.visible_messages:
        msg_lines = "; ".join(f"{name} said: \"{msg}\"" for name, msg in obs.visible_messages.items())
        lines.append(f"Messages from other businesses last round: {msg_lines}")

    lines.append(
        "Your marketing budget for this round can be any amount from 0 up "
        "to 3, on the same numeric scale as price. Amounts above 3 have no "
        "additional effect and are wasted spend."
    )
    lines.append(
        "Choose your price and marketing budget for this round. "
        "Respond with only the JSON object described in the instructions."
    )
    return "\n".join(lines)


def parse_llm_response(text: str) -> dict:
    """
    Extract and validate a decision dict from raw model output.
    Raises ValueError on anything unusable, so the caller can decide
    whether to retry. Handles the common case of a model wrapping JSON
    in a markdown code fence even when told not to.
    """
    stripped = text.strip()
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text!r}")

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON: {e}") from e

    if "price" not in data:
        raise ValueError(f"Response missing 'price' field: {data!r}")

    try:
        price = float(data["price"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"'price' is not a number: {data.get('price')!r}") from e

    if price <= 0:
        raise ValueError(f"'price' must be positive, got {price}")

    marketing = 0.0
    if "marketing" in data and data["marketing"] is not None:
        try:
            marketing = max(0.0, float(data["marketing"]))
        except (TypeError, ValueError):
            marketing = 0.0

    rationale = str(data.get("rationale", ""))[:300]
    message = str(data.get("message", ""))[:200]

    return {"price": price, "marketing": marketing, "rationale": rationale, "message": message}


def default_groq_call_fn(system_prompt: str, user_prompt: str, model: str, api_key: str) -> str:
    """
    Real API call to Groq's OpenAI-compatible endpoint. Only imports
    `requests` here, so tests that inject a fake call_fn never need the
    dependency or network access at all.

    Handles rate limiting (HTTP 429) with its own backoff loop, separate
    from LLMAgent's retry loop, which exists for malformed responses, not
    for pacing requests. Without this, a burst of 429s from running
    several agents back to back would burn through all of LLMAgent's
    retries in under a second and fall back to a default price, even
    though the model itself never actually failed to answer, it was
    simply asked too fast.
    """
    import requests
    import time as time_module

    max_local_retries = 2  # kept small and bounded: LLMAgent's own retry loop
    # also retries on failure, so an unbounded backoff here would multiply
    # against that outer loop and turn a rate limit into several minutes
    # of silent waiting for a single decision, which is exactly what
    # happened before this was capped.
    for attempt in range(max_local_retries):
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 200,
            },
            timeout=30,
        )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait_seconds = min(float(retry_after), 8.0) if retry_after else 3.0
            time_module.sleep(wait_seconds)
            continue
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    response.raise_for_status()  # still 429 after local retries: raise clearly


class LLMAgent(Agent):
    def __init__(
        self,
        name: str,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        call_fn=None,
        max_retries: int = 3,
        fallback_markup: float = 0.3,
        allow_messaging: bool = False,
    ):
        self.name = name
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.call_fn = call_fn or default_groq_call_fn
        self.max_retries = max_retries
        self.fallback_markup = fallback_markup
        self.allow_messaging = allow_messaging

    def decide(self, obs: MarketObservation) -> AgentDecision:
        user_prompt = build_prompt(obs)
        system_prompt = build_system_prompt(self.allow_messaging)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                raw = self.call_fn(system_prompt, user_prompt, self.model, self.api_key)
                parsed = parse_llm_response(raw)
                return AgentDecision(
                    price=parsed["price"],
                    marketing=parsed["marketing"],
                    rationale=parsed["rationale"],
                    message=parsed["message"] if self.allow_messaging else "",
                )
            except Exception as e:
                last_error = e
                continue

        # every attempt failed: fall back to a safe, simple decision rather
        # than crashing the whole simulation over one bad response
        fallback_price = obs.own_marginal_cost * (1 + self.fallback_markup)
        return AgentDecision(
            price=fallback_price,
            marketing=0.0,
            rationale=f"fallback after {self.max_retries} failed API attempts ({last_error})",
        )

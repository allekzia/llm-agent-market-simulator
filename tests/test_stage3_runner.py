import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments import stage3_conditions_runner as mod


def test_run_condition_warns_when_every_call_fails(monkeypatch, capsys):
    def failing_call(system_prompt, user_prompt, model, api_key):
        raise ConnectionError("simulated failure")

    # avoid real sleeps in the throttled call wrapper during the test
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(mod, "default_groq_call_fn", failing_call)

    mod.run_condition(
        "test condition", api_key="fake",
        full_visibility=False, communication_enabled=False,
        n_rounds=1, seed=1,
    )

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "CAUTION" in captured.out
    assert "3/3" in captured.out  # 3 agents, all fell back


def test_run_condition_prints_nothing_alarming_on_success(monkeypatch, capsys):
    def working_call(system_prompt, user_prompt, model, api_key):
        return '{"price": 2.80, "marketing": 0.0, "rationale": "steady"}'

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(mod, "default_groq_call_fn", working_call)

    mod.run_condition(
        "test condition", api_key="fake",
        full_visibility=False, communication_enabled=False,
        n_rounds=1, seed=1,
    )

    captured = capsys.readouterr()
    assert "WARNING" not in captured.out
    assert "CAUTION" not in captured.out

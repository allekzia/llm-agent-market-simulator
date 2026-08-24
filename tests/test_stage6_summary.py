import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import glob


def test_regenerate_summary_picks_final_round_price(tmp_path, monkeypatch):
    """
    Uses a temporary results directory so this test never touches real
    collected data, and confirms regenerate_summary() correctly reads
    each per-seed file's LAST round, not its first, for the summary.
    """
    from experiments import stage6_connected_completion_runner as mod

    monkeypatch.setattr(mod, "RESULTS_DIR", str(tmp_path))

    with open(tmp_path / "stage3_isolated_seed77.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "agent", "price"])
        writer.writerow([0, "Agent_1", 3.00])
        writer.writerow([1, "Agent_1", 3.75])  # this is the one that should be picked

    mod.regenerate_summary()

    out_path = tmp_path / "stage3_repeats_summary.csv"
    assert out_path.exists()

    with open(out_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["condition"] == "isolated"
    assert rows[0]["seed"] == "77"
    assert rows[0]["agent"] == "Agent_1"
    assert float(rows[0]["final_price"]) == 3.75


def test_regenerate_summary_includes_both_conditions(tmp_path, monkeypatch):
    from experiments import stage6_connected_completion_runner as mod

    monkeypatch.setattr(mod, "RESULTS_DIR", str(tmp_path))

    with open(tmp_path / "stage3_isolated_seed1.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "agent", "price"])
        writer.writerow([0, "Agent_1", 3.00])

    with open(tmp_path / "stage3_connected_seed1.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "agent", "price"])
        writer.writerow([0, "Agent_1", 2.80])

    mod.regenerate_summary()

    with open(tmp_path / "stage3_repeats_summary.csv") as f:
        rows = list(csv.DictReader(f))

    conditions = {row["condition"] for row in rows}
    assert conditions == {"isolated", "connected"}


def _write_seed_csv(path, rows_with_rationale):
    """rows_with_rationale: list of (round, agent, price, rationale)"""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "agent", "price", "rationale"])
        writer.writerows(rows_with_rationale)


def test_heavily_contaminated_seed_is_excluded(tmp_path, monkeypatch):
    """
    A seed where most decisions fell back to the default price (the
    same pattern seen in a real run when a daily API quota was
    exhausted partway through) should be excluded from the summary
    entirely, not blended in as if it were real data.
    """
    from experiments import stage6_connected_completion_runner as mod

    monkeypatch.setattr(mod, "RESULTS_DIR", str(tmp_path))

    _write_seed_csv(tmp_path / "stage3_connected_seed99.csv", [
        (0, "Agent_1", 2.60, "fallback after 3 failed API attempts (429 ...)"),
        (0, "Agent_2", 2.60, "fallback after 3 failed API attempts (429 ...)"),
        (0, "Agent_3", 2.55, "steady pricing"),
    ])

    mod.regenerate_summary()

    with open(tmp_path / "stage3_repeats_summary.csv") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 0


def test_clean_seed_is_not_excluded(tmp_path, monkeypatch):
    from experiments import stage6_connected_completion_runner as mod

    monkeypatch.setattr(mod, "RESULTS_DIR", str(tmp_path))

    _write_seed_csv(tmp_path / "stage3_connected_seed100.csv", [
        (0, "Agent_1", 2.60, "steady pricing"),
        (0, "Agent_2", 2.55, "matching competitor"),
        (0, "Agent_3", 2.50, "slight discount"),
    ])

    mod.regenerate_summary()

    with open(tmp_path / "stage3_repeats_summary.csv") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3


def test_seed_just_under_threshold_is_kept(tmp_path, monkeypatch):
    """One fallback out of many clean decisions should not trip exclusion."""
    from experiments import stage6_connected_completion_runner as mod

    monkeypatch.setattr(mod, "RESULTS_DIR", str(tmp_path))

    rows = [(0, "Agent_1", 2.60, "fallback after 3 failed API attempts (429 ...)")]
    rows += [(0, f"Agent_{i}", 2.55, "steady pricing") for i in range(2, 25)]
    _write_seed_csv(tmp_path / "stage3_connected_seed101.csv", rows)

    mod.regenerate_summary(max_fallback_fraction=0.05)

    with open(tmp_path / "stage3_repeats_summary.csv") as f:
        summary_rows = list(csv.DictReader(f))

    assert len(summary_rows) == 24

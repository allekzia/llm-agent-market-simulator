"""
Tests for the Streamlit dashboard, using Streamlit's own headless
AppTest tool rather than a real browser. This only checks that the app
loads and runs without raising exceptions, not visual appearance, but
that is exactly the kind of thing that silently breaks when a project's
files get moved or renamed, worth catching automatically.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py")


def test_dashboard_loads_without_exceptions():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception
    assert len(at.tabs) == 3


def test_live_simulation_runs_without_exceptions():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.button[0].click().run(timeout=30)
    assert not at.exception

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dashboard import theme


def test_isolated_columns_get_isolated_color():
    colors = theme.palette_for(["isolated_seed42", "isolated_seed7"])
    assert all(c == theme.ISOLATED for c in colors)


def test_connected_columns_get_connected_color():
    colors = theme.palette_for(["connected_seed42"])
    assert colors == [theme.CONNECTED]


def test_mixed_condition_columns_get_correct_colors():
    colors = theme.palette_for(["isolated", "connected"])
    assert colors == [theme.ISOLATED, theme.CONNECTED]


def test_unrelated_columns_cycle_neutral_palette():
    colors = theme.palette_for(["Agent_1", "Agent_2", "Agent_3"])
    assert colors == theme.NEUTRAL_PALETTE


def test_palette_for_handles_empty_list():
    assert theme.palette_for([]) == []

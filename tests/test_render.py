"""ASCII renderer smoke tests."""

from __future__ import annotations

from hivevision.render import render_ascii


def test_empty():
    assert render_ascii([]) == "(empty board)"


def test_single_tile_shows_label():
    out = render_ascii([(0, 0, "wQ")])
    assert "wQ" in out
    assert "/wQ\\" in out


def test_two_tiles_both_rendered():
    out = render_ascii([(0, 0, "wQ"), (1, 0, "bA")])
    assert "wQ" in out and "bA" in out
    # second tile is offset right and down from the first (sx=q*3, sy=r*2+q)
    assert out.count("\n") >= 2

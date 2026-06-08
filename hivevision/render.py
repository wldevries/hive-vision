"""ASCII hex-board renderer for Hive positions.

A flat-top hex layout ported from hive-zero's Rust board renderer
(`hive-game/src/board.rs`): axial ``(q, r)`` -> screen ``sx = q*3, sy = r*2 + q``,
each tile drawn 4 columns wide::

     __
    /wQ\\
    \\__/

Useful in training/eval scripts and for verifying lattice recovery from a photo.
``placements`` is an iterable of ``(q, r, label)`` with a 2-char code like ``wQ``.
Stacks (``z``) are out of scope (phase-1 flat board); if two tiles share ``(q, r)``
the last drawn wins. The richer HTML/CSS version belongs in the capture app.
"""

from __future__ import annotations

from collections.abc import Iterable


def render_ascii(placements: Iterable[tuple[int, int, str]]) -> str:
    cells = [(int(q), int(r), f"{lbl:<2}"[:2]) for q, r, lbl in placements]
    if not cells:
        return "(empty board)"

    scr = [((q * 3, r * 2 + q), lbl) for q, r, lbl in cells]
    min_sx = min(s[0][0] for s in scr)
    min_sy = min(s[0][1] for s in scr)
    max_sx = max(s[0][0] for s in scr)
    max_sy = max(s[0][1] for s in scr)

    width = max_sx - min_sx + 5
    height = max_sy - min_sy + 3
    canvas = [[" "] * width for _ in range(height)]

    for (sx_raw, sy_raw), lbl in scr:
        sx = sx_raw - min_sx
        sy = sy_raw - min_sy
        canvas[sy][sx + 1] = "_"
        canvas[sy][sx + 2] = "_"
        canvas[sy + 1][sx] = "/"
        canvas[sy + 1][sx + 1] = lbl[0]
        canvas[sy + 1][sx + 2] = lbl[1]
        canvas[sy + 1][sx + 3] = "\\"
        canvas[sy + 2][sx] = "\\"
        canvas[sy + 2][sx + 1] = "_"
        canvas[sy + 2][sx + 2] = "_"
        canvas[sy + 2][sx + 3] = "/"

    lines = ["".join(row).rstrip() for row in canvas]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)

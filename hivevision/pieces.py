"""The Hive piece taxonomy (base + expansions) and the label vocabulary.

Base game plus the three expansion pieces (Mosquito, Ladybug, Pillbug): eight
piece types per colour, sixteen classes total. The compact code is
``<color><type>`` e.g. ``wQ``, ``bA``, ``wM`` — the same short notation Hive
players / UHP use. ``z`` (stack height) is part of a *position* but never a
class; phase 1 assumes ``z == 0`` (flat board).
"""

from __future__ import annotations

# Standard Hive single-letter piece codes (base game, then the three expansions).
PIECE_TYPES: tuple[str, ...] = ("Q", "B", "G", "S", "A", "M", "L", "P")
TYPE_NAMES: dict[str, str] = {
    "Q": "Queen Bee",
    "B": "Beetle",
    "G": "Grasshopper",
    "S": "Spider",
    "A": "Ant",
    "M": "Mosquito",
    "L": "Ladybug",
    "P": "Pillbug",
}
# Per-colour counts in a full set (legality checks later; not class-relevant).
# Each expansion piece comes as a single tile.
TYPE_COUNTS: dict[str, int] = {"Q": 1, "B": 2, "G": 3, "S": 2, "A": 3, "M": 1, "L": 1, "P": 1}

COLORS: tuple[str, ...] = ("w", "b")
COLOR_NAMES: dict[str, str] = {"w": "White", "b": "Black"}

# Canonical class order: white pieces then black, each in PIECE_TYPES order.
CLASSES: tuple[str, ...] = tuple(f"{c}{t}" for c in COLORS for t in PIECE_TYPES)
CLASS_INDEX: dict[str, int] = {code: i for i, code in enumerate(CLASSES)}


def class_name(code: str) -> str:
    """Human label for a class code, e.g. ``wQ`` -> ``White Queen Bee``."""
    color, type_ = code[0], code[1:]
    return f"{COLOR_NAMES[color]} {TYPE_NAMES[type_]}"


def is_valid_class(code: str) -> bool:
    return code in CLASS_INDEX

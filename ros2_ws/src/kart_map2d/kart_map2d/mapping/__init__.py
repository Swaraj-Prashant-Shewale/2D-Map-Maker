"""2D mapping: evidence grid, corridor, loop closure, map package."""

from kart_map2d.mapping.corridor import Corridor, extract_corridor
from kart_map2d.mapping.evidence_grid import (
    CLASS_DRIVABLE,
    CLASS_OBSTACLE,
    CLASS_UNKNOWN,
    EvidenceGrid,
)

__all__ = [
    "EvidenceGrid",
    "CLASS_UNKNOWN",
    "CLASS_DRIVABLE",
    "CLASS_OBSTACLE",
    "Corridor",
    "extract_corridor",
]

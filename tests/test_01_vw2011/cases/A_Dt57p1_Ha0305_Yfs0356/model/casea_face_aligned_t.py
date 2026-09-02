"""One discrete location for the Case-A side-T junction.

The measured riser centreline usually falls inside a finite-volume cell.  The
production Case-A model represents the zero-storage side branch at the nearest
horizontal *face*.  Every solver that touches the junction must therefore use
the same three indices: the face itself and its west/east adjacent cells.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FaceAlignedTIndices:
    """Horizontal-grid indices owned by one face-aligned side T."""

    face: int
    west_cell: int
    east_cell: int
    face_x: float


def face_aligned_t_indices(
    physical_x: float,
    cell_width: float,
    cell_count: int,
) -> FaceAlignedTIndices:
    """Return the internal FV face nearest ``physical_x``.

    The half-cell tie is resolved toward the positive-x face.  Boundary faces
    are excluded because the Case-A riser is an internal side branch.
    """

    x = float(physical_x)
    dx = float(cell_width)
    cells = int(cell_count)
    if not math.isfinite(x) or not math.isfinite(dx):
        raise ValueError("side-T coordinate and cell width must be finite")
    if dx <= 0.0 or cells < 2:
        raise ValueError("a face-aligned side T requires at least two cells")
    if not 0.0 < x < cells * dx:
        raise ValueError("side-T coordinate must lie inside the grid")

    face = int(math.floor(x / dx + 0.5))
    face = min(max(face, 1), cells - 1)
    return FaceAlignedTIndices(
        face=face,
        west_cell=face - 1,
        east_cell=face,
        face_x=face * dx,
    )


__all__ = ["FaceAlignedTIndices", "face_aligned_t_indices"]

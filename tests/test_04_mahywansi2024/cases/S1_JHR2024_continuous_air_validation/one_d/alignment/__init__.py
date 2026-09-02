"""Common-time 1-D/2-D alignment utilities for the S1 validation case.

The package deliberately contains no OpenFOAM reader.  Solver-specific
extractors must first produce ordinary physical-time arrays; this package then
enforces the frozen 0.10 s comparison grid and event definitions.
"""

from .campaign import (
    REQUIRED_MESH_LEVELS,
    compare_event_branches,
    compare_one_d_to_three_meshes,
)
from .events import (
    ERUPTION_MEAN_FLOW_M3_S,
    ERUPTION_PERSISTENCE_S,
    ERUPTION_VOLUME_M3,
    InternalMouthEventDecision,
    classify_internal_mouth_event,
)
from .grid import (
    COMMON_DT_S,
    AlignedSeries,
    AlignmentCoverageError,
    AlignmentError,
    AlignmentGapError,
    TimeShiftNotAllowedError,
    make_common_grid,
    resample_to_common_grid,
    validate_strict_common_grid,
)
from .metrics import WaveformMetrics, compute_waveform_metrics

__all__ = [
    "COMMON_DT_S",
    "ERUPTION_MEAN_FLOW_M3_S",
    "ERUPTION_PERSISTENCE_S",
    "ERUPTION_VOLUME_M3",
    "REQUIRED_MESH_LEVELS",
    "AlignedSeries",
    "AlignmentCoverageError",
    "AlignmentError",
    "AlignmentGapError",
    "InternalMouthEventDecision",
    "TimeShiftNotAllowedError",
    "WaveformMetrics",
    "classify_internal_mouth_event",
    "compare_event_branches",
    "compare_one_d_to_three_meshes",
    "compute_waveform_metrics",
    "make_common_grid",
    "resample_to_common_grid",
    "validate_strict_common_grid",
]

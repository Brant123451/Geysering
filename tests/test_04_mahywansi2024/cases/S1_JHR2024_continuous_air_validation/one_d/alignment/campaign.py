"""Campaign-level comparison of one 1-D trajectory to all three 2-D meshes."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .events import InternalMouthEventDecision
from .grid import AlignmentError, validate_strict_common_grid
from .metrics import compute_waveform_metrics


REQUIRED_MESH_LEVELS = ("coarse", "medium_refine", "refined")


def _require_three_meshes(mapping: Mapping[str, object]) -> None:
    missing = tuple(level for level in REQUIRED_MESH_LEVELS if level not in mapping)
    if missing:
        raise AlignmentError(f"missing 2-D mesh levels: {', '.join(missing)}")


def _same_time_grid(reference: Sequence[float], candidate: Sequence[float]) -> None:
    if len(reference) != len(candidate):
        raise AlignmentError("1-D and 2-D common-time grids have different lengths")
    if any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-9)
        for left, right in zip(reference, candidate)
    ):
        raise AlignmentError(
            "1-D and 2-D samples are not at identical physical times; "
            "time shifting is forbidden"
        )


def compare_one_d_to_three_meshes(
    one_d_series: Mapping[str, Sequence[float]],
    two_d_by_mesh: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    series_names: Sequence[str],
) -> dict[str, object]:
    """Apply one unchanged 1-D record to coarse, medium and refined outputs."""

    _require_three_meshes(two_d_by_mesh)
    if "time_s" not in one_d_series:
        raise AlignmentError("one_d_series is missing time_s")
    one_d_time = validate_strict_common_grid(one_d_series["time_s"])
    requested_names = tuple(series_names)
    if not requested_names:
        raise AlignmentError("series_names must not be empty")

    mesh_results: dict[str, object] = {}
    for level in REQUIRED_MESH_LEVELS:
        mesh_series = two_d_by_mesh[level]
        if "time_s" not in mesh_series:
            raise AlignmentError(f"{level} is missing time_s")
        mesh_time = validate_strict_common_grid(mesh_series["time_s"])
        _same_time_grid(one_d_time, mesh_time)
        level_metrics: dict[str, object] = {}
        for name in requested_names:
            if name not in one_d_series:
                raise AlignmentError(f"one_d_series is missing {name}")
            if name not in mesh_series:
                raise AlignmentError(f"{level} is missing {name}")
            level_metrics[name] = compute_waveform_metrics(
                one_d_time,
                reference=mesh_series[name],
                candidate=one_d_series[name],
            ).to_dict()
        mesh_results[level] = {"series_metrics": level_metrics}

    return {
        "schema_version": 1,
        "comparison_identity": "one_1d_trajectory_vs_three_2d_mesh_levels",
        "one_d_parameter_set_reused_for_all_meshes": True,
        "mesh_levels": list(REQUIRED_MESH_LEVELS),
        "time_shift_applied_s": 0.0,
        "automatic_acceptance_applied": False,
        "mesh_results": mesh_results,
    }


def compare_event_branches(
    one_d_decision: InternalMouthEventDecision,
    two_d_erupted_by_mesh: Mapping[str, bool],
) -> dict[str, object]:
    """Report exact agreement and the paper-required erupting branch.

    Exact agreement alone is insufficient for this campaign: three stable
    no-eruption calculations would agree with each other while all missing the
    published continuous-air branch.  The separate physics gate below is true
    only when the 1-D result and every 2-D mesh explicitly report eruption.
    """

    _require_three_meshes(two_d_erupted_by_mesh)
    invalid = tuple(
        level
        for level in REQUIRED_MESH_LEVELS
        if not isinstance(two_d_erupted_by_mesh[level], bool)
    )
    if invalid:
        raise AlignmentError(
            "2-D eruption flags must be explicit booleans: " + ", ".join(invalid)
        )
    two_d_flags = {
        level: two_d_erupted_by_mesh[level] for level in REQUIRED_MESH_LEVELS
    }
    matches = {
        level: one_d_decision.eruption_detected == two_d_flags[level]
        for level in REQUIRED_MESH_LEVELS
    }
    paper_expected_erupted = True
    physics_alignment_pass = (
        one_d_decision.eruption_detected is paper_expected_erupted
        and all(two_d_flags.values())
        and all(matches.values())
    )
    return {
        "schema_version": 1,
        "one_d_event": one_d_decision.event_name,
        "one_d_erupted": one_d_decision.eruption_detected,
        "two_d_erupted_by_mesh": two_d_flags,
        "exact_branch_match_by_mesh": matches,
        "all_mesh_branches_match": all(matches.values()),
        "paper_expected_erupted": paper_expected_erupted,
        "physics_alignment_branch_pass": physics_alignment_pass,
        "stable_but_no_eruption_classification": "physics_alignment_failure",
        "required_match_rule": "exact",
    }

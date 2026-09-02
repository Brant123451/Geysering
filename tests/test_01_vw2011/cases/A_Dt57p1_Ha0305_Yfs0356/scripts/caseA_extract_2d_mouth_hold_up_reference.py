"""Extract a read-only Case-A T-junction reference from raw 2-D fields.

The extractor reads the decomposed OpenFOAM ``alpha.water`` and ``U`` cell
fields directly.  It does not import or call the one-dimensional solver.  The
planar tower is mapped to the physical circular riser by preserving each
section-averaged quantity.

Three distinct quantities are intentionally retained at the riser mouth:

* gross upward liquid flux;
* gross downward liquid flux; and
* their signed sum (net liquid flux).

The separation matters because the 2-D solution contains simultaneous
counter-current motion.  A single signed branch flux can match the net
exchange while missing most of the visible recirculation.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CASE = HERE.parent
FOAM = CASE / "openfoam/2d"
OUT = CASE / "outputs/caseA_openfoam2d_mouth_hold_up_reference_7p5_9p2s.json"

HORIZONTAL_DIAMETER_M = 0.094
RISER_DIAMETER_M = 0.0571
RISER_CENTRE_X_M = 3.516
RISER_CROWN_Y_M = 0.5 * HORIZONTAL_DIAMETER_M
RISER_TOP_Y_M = RISER_CROWN_Y_M + 0.610
RISER_AREA_M2 = math.pi * RISER_DIAMETER_M**2 / 4.0
BOTTOM_ZONE_HEIGHT_M = 0.10
START_TIME_S = 7.50
END_TIME_S = 9.20


def _first_binary_list(path: Path, dtype: str, components: int = 1) -> np.ndarray:
    raw = path.read_bytes()
    match = re.search(rb"\n(\d+)\s*\n\(", raw)
    if match is None:
        raise ValueError(f"Cannot locate binary list in {path}")
    count = int(match.group(1))
    values = np.frombuffer(
        raw,
        dtype=dtype,
        count=count * components,
        offset=match.end(),
    ).copy()
    if components > 1:
        values = values.reshape(count, components)
    return values


def _internal_field(path: Path, components: int = 1) -> np.ndarray:
    raw = path.read_bytes()
    match = re.search(
        rb"internalField\s+nonuniform\s+List<[^>]+>\s+(\d+)\s*\n\(",
        raw,
    )
    if match is None:
        raise ValueError(f"Cannot locate binary internalField in {path}")
    count = int(match.group(1))
    values = np.frombuffer(
        raw,
        dtype="<f8",
        count=count * components,
        offset=match.end(),
    ).copy()
    if components > 1:
        values = values.reshape(count, components)
    return values


def _compact_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    match = re.search(rb"\n(\d+)\s*\n\(", raw)
    if match is None:
        raise ValueError(f"Cannot locate compact-face offsets in {path}")
    offset_count = int(match.group(1))
    offsets = np.frombuffer(
        raw, dtype="<i4", count=offset_count, offset=match.end()
    ).copy()
    cursor = match.end() + 4 * offset_count
    connectivity_match = re.search(
        rb"\)\s*(\d+)\s*\n\(", raw[cursor : cursor + 128]
    )
    if connectivity_match is None:
        raise ValueError(f"Cannot locate compact-face connectivity in {path}")
    connectivity_count = int(connectivity_match.group(1))
    connectivity_start = cursor + connectivity_match.end()
    connectivity = np.frombuffer(
        raw,
        dtype="<i4",
        count=connectivity_count,
        offset=connectivity_start,
    ).copy()
    return offsets, connectivity


def _global_geometry() -> dict[str, np.ndarray]:
    mesh = FOAM / "constant/polyMesh"
    points = _first_binary_list(mesh / "points", "<f8", components=3)
    offsets, connectivity = _compact_faces(mesh / "faces")
    owner = _first_binary_list(mesh / "owner", "<i4")
    neighbour = _first_binary_list(mesh / "neighbour", "<i4")
    number_of_cells = int(max(np.max(owner), np.max(neighbour)) + 1)
    cell_vertices: list[set[int]] = [set() for _ in range(number_of_cells)]
    if len(offsets) != len(owner) + 1:
        raise ValueError(
            "Unexpected faceCompactList layout: offsets must have one more entry than owner"
        )
    for face_index, owner_cell in enumerate(owner):
        vertices = connectivity[offsets[face_index] : offsets[face_index + 1]]
        cell_vertices[int(owner_cell)].update(int(value) for value in vertices)
        if face_index < len(neighbour):
            cell_vertices[int(neighbour[face_index])].update(
                int(value) for value in vertices
            )

    centres = np.empty((number_of_cells, 2))
    areas = np.empty(number_of_cells)
    y_min = np.empty(number_of_cells)
    y_max = np.empty(number_of_cells)
    for cell_index, vertices in enumerate(cell_vertices):
        xy = np.unique(points[list(vertices), :2], axis=0)
        centre = np.mean(xy, axis=0)
        order = np.argsort(np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0]))
        polygon = xy[order]
        x = polygon[:, 0]
        y = polygon[:, 1]
        centres[cell_index] = centre
        areas[cell_index] = 0.5 * abs(
            float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        )
        y_min[cell_index] = float(np.min(y))
        y_max[cell_index] = float(np.max(y))
    return {
        "centres": centres,
        "areas": areas,
        "y_min": y_min,
        "y_max": y_max,
    }


def _processor_maps() -> list[np.ndarray]:
    maps: list[np.ndarray] = []
    processor_index = 0
    while (FOAM / f"processor{processor_index}").is_dir():
        maps.append(
            _first_binary_list(
                FOAM
                / f"processor{processor_index}"
                / "constant/polyMesh/cellProcAddressing",
                "<i4",
            )
        )
        processor_index += 1
    if not maps:
        raise RuntimeError("No decomposed OpenFOAM processor directories found")
    return maps


def _available_times() -> list[float]:
    times: list[float] = []
    for path in (FOAM / "processor0").iterdir():
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        if START_TIME_S - 1.0e-10 <= value <= END_TIME_S + 1.0e-10:
            times.append(value)
    return sorted(times)


def _assemble_field(
    time_s: float,
    field_name: str,
    processor_maps: list[np.ndarray],
    number_of_cells: int,
    components: int = 1,
) -> np.ndarray:
    shape = (number_of_cells, components) if components > 1 else (number_of_cells,)
    global_field = np.full(shape, np.nan)
    time_name = f"{time_s:g}"
    for processor_index, addressing in enumerate(processor_maps):
        local = _internal_field(
            FOAM / f"processor{processor_index}" / time_name / field_name,
            components=components,
        )
        if len(local) != len(addressing):
            raise ValueError(
                f"Local field/addressing size mismatch in processor{processor_index} at {time_name}"
            )
        global_field[addressing] = local
    if np.any(~np.isfinite(global_field)):
        raise ValueError(f"Incomplete reconstructed field {field_name} at {time_name}")
    return global_field


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
    }


def main() -> None:
    geometry = _global_geometry()
    centres = geometry["centres"]
    areas = geometry["areas"]
    y_min = geometry["y_min"]
    y_max = geometry["y_max"]
    number_of_cells = len(centres)
    maps = _processor_maps()
    times = np.asarray(_available_times())
    if times.size == 0:
        raise RuntimeError("No OpenFOAM times selected")

    riser_left = RISER_CENTRE_X_M - 0.5 * RISER_DIAMETER_M
    riser_right = RISER_CENTRE_X_M + 0.5 * RISER_DIAMETER_M
    riser_mask = (
        (centres[:, 0] >= riser_left - 1.0e-8)
        & (centres[:, 0] <= riser_right + 1.0e-8)
        & (centres[:, 1] >= RISER_CROWN_Y_M - 1.0e-8)
        & (centres[:, 1] <= RISER_TOP_Y_M + 1.0e-8)
    )
    # The first tower control-volume row is the set of cells whose lower face
    # lies on the horizontal-pipe crown.  This selects all 14 mouth cells even
    # though their polygon centroids differ slightly in the junction mesh.
    mouth_mask = (
        riser_mask
        & np.isclose(y_min, RISER_CROWN_Y_M, atol=1.0e-9)
        & (y_max > RISER_CROWN_Y_M + 1.0e-9)
    )
    bottom_top = RISER_CROWN_Y_M + BOTTOM_ZONE_HEIGHT_M
    overlap = np.maximum(
        0.0,
        np.minimum(y_max, bottom_top) - np.maximum(y_min, RISER_CROWN_Y_M),
    )
    vertical_extent = y_max - y_min
    overlap_fraction = np.divide(
        overlap,
        vertical_extent,
        out=np.zeros_like(overlap),
        where=vertical_extent > 0.0,
    )
    bottom_mask = riser_mask & (overlap_fraction > 0.0)
    if int(np.sum(mouth_mask)) != 14:
        raise RuntimeError(f"Expected 14 mouth cells; found {int(np.sum(mouth_mask))}")

    records: list[dict[str, float]] = []
    for time_s in times:
        alpha = np.clip(
            _assemble_field(time_s, "alpha.water", maps, number_of_cells),
            0.0,
            1.0,
        )
        velocity = _assemble_field(
            time_s,
            "U",
            maps,
            number_of_cells,
            components=3,
        )
        mouth_weights = areas[mouth_mask]
        mouth_alpha = alpha[mouth_mask]
        mouth_uy = velocity[mouth_mask, 1]
        gross_up = RISER_AREA_M2 * float(
            np.average(mouth_alpha * np.maximum(mouth_uy, 0.0), weights=mouth_weights)
        )
        gross_down = RISER_AREA_M2 * float(
            np.average(mouth_alpha * np.minimum(mouth_uy, 0.0), weights=mouth_weights)
        )
        net = gross_up + gross_down
        equivalent_height = float(
            np.sum(alpha[riser_mask] * areas[riser_mask]) / RISER_DIAMETER_M
        )
        bottom_height = float(
            np.sum(
                alpha[bottom_mask]
                * areas[bottom_mask]
                * overlap_fraction[bottom_mask]
            )
            / RISER_DIAMETER_M
        )
        mouth_fraction = float(
            np.average(mouth_alpha, weights=mouth_weights)
        )
        mouth_row_height = float(
            np.sum(areas[mouth_mask]) / RISER_DIAMETER_M
        )
        records.append(
            {
                "time_s": float(time_s),
                "mouth_gross_up_m3_s": gross_up,
                "mouth_gross_down_m3_s": gross_down,
                "mouth_net_m3_s": net,
                "mouth_exchange_intensity_m3_s": 0.5 * (gross_up - gross_down),
                "mouth_water_fraction": mouth_fraction,
                "mouth_first_row_equivalent_liquid_height_m": (
                    mouth_row_height * mouth_fraction
                ),
                "mouth_first_row_equivalent_liquid_volume_m3": (
                    RISER_AREA_M2 * mouth_row_height * mouth_fraction
                ),
                "bottom_0p10m_equivalent_liquid_height_m": bottom_height,
                "bottom_0p10m_equivalent_liquid_volume_m3": (
                    RISER_AREA_M2 * bottom_height
                ),
                "bottom_0p10m_mean_water_fraction": (
                    bottom_height / BOTTOM_ZONE_HEIGHT_M
                ),
                "whole_riser_equivalent_liquid_height_m": equivalent_height,
                "whole_riser_equivalent_liquid_volume_m3": (
                    RISER_AREA_M2 * equivalent_height
                ),
            }
        )

    def column(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in records])

    gross_up = column("mouth_gross_up_m3_s")
    gross_down = column("mouth_gross_down_m3_s")
    net = column("mouth_net_m3_s")
    bottom_volume = column("bottom_0p10m_equivalent_liquid_volume_m3")
    whole_height = column("whole_riser_equivalent_liquid_height_m")
    inventory_change = float(
        column("whole_riser_equivalent_liquid_volume_m3")[-1]
        - column("whole_riser_equivalent_liquid_volume_m3")[0]
    )
    gross_up_integral = float(np.trapezoid(gross_up, times))
    gross_down_integral = float(np.trapezoid(gross_down, times))
    net_integral = float(np.trapezoid(net, times))

    # These are observed 2-D reference envelopes for offline screening only.
    # They are intentionally not solver coefficients or source-term limits.
    late = times >= 8.50 - 1.0e-10
    output = {
        "provenance": {
            "kind": "read_only_raw_decomposed_openfoam_2d_fields",
            "fields": ["alpha.water", "U"],
            "one_dimensional_solver_imported": False,
            "one_dimensional_results_used_to_build_reference": False,
            "rendered_images_used": False,
            "processor_count": len(maps),
            "sample_interval_s": float(np.median(np.diff(times))),
            "time_window_s": [float(times[0]), float(times[-1])],
            "mapping": (
                "Planar section averages and planar liquid hold-up are mapped to the "
                "physical circular riser area; no value is fed back to either solver."
            ),
        },
        "geometry": {
            "horizontal_diameter_m": HORIZONTAL_DIAMETER_M,
            "riser_diameter_m": RISER_DIAMETER_M,
            "riser_area_m2": RISER_AREA_M2,
            "riser_crown_y_m": RISER_CROWN_Y_M,
            "bottom_inventory_zone_height_m": BOTTOM_ZONE_HEIGHT_M,
            "mouth_cell_count": int(np.sum(mouth_mask)),
            "mouth_first_row_equivalent_height_m": float(
                np.sum(areas[mouth_mask]) / RISER_DIAMETER_M
            ),
        },
        "definitions": {
            "positive_direction": "upward from horizontal pipe into riser",
            "gross_up": "area-mapped alpha.water*max(U_y,0) over the 14 mouth cells",
            "gross_down": "area-mapped alpha.water*min(U_y,0) over the 14 mouth cells",
            "net": "gross_up + gross_down",
            "exchange_intensity": "0.5*(gross_up + abs(gross_down))",
            "bottom_inventory": (
                "mapped liquid volume in the first 0.10 m above the horizontal-pipe crown; "
                "partial cells use exact vertical-overlap fractions"
            ),
        },
        "integrated_exchange": {
            "gross_up_m3": gross_up_integral,
            "gross_down_m3": gross_down_integral,
            "net_m3": net_integral,
            "whole_riser_inventory_change_m3": inventory_change,
            "net_minus_inventory_change_m3": net_integral - inventory_change,
        },
        "observed_distributions": {
            "7p5_to_9p2": {
                "mouth_gross_up_m3_s": _distribution(gross_up),
                "mouth_gross_down_magnitude_m3_s": _distribution(-gross_down),
                "mouth_net_m3_s": _distribution(net),
                "mouth_exchange_intensity_m3_s": _distribution(
                    column("mouth_exchange_intensity_m3_s")
                ),
                "mouth_water_fraction": _distribution(
                    column("mouth_water_fraction")
                ),
                "mouth_first_row_equivalent_liquid_volume_m3": _distribution(
                    column("mouth_first_row_equivalent_liquid_volume_m3")
                ),
                "bottom_0p10m_equivalent_liquid_volume_m3": _distribution(
                    bottom_volume
                ),
                "bottom_0p10m_mean_water_fraction": _distribution(
                    column("bottom_0p10m_mean_water_fraction")
                ),
                "whole_riser_equivalent_liquid_height_m": _distribution(
                    whole_height
                ),
            },
            "8p5_to_9p2": {
                "mouth_gross_up_m3_s": _distribution(gross_up[late]),
                "mouth_gross_down_magnitude_m3_s": _distribution(-gross_down[late]),
                "mouth_net_m3_s": _distribution(net[late]),
                "mouth_exchange_intensity_m3_s": _distribution(
                    column("mouth_exchange_intensity_m3_s")[late]
                ),
                "mouth_water_fraction": _distribution(
                    column("mouth_water_fraction")[late]
                ),
                "mouth_first_row_equivalent_liquid_volume_m3": _distribution(
                    column("mouth_first_row_equivalent_liquid_volume_m3")[late]
                ),
                "bottom_0p10m_equivalent_liquid_volume_m3": _distribution(
                    bottom_volume[late]
                ),
                "bottom_0p10m_mean_water_fraction": _distribution(
                    column("bottom_0p10m_mean_water_fraction")[late]
                ),
                "whole_riser_equivalent_liquid_height_m": _distribution(
                    whole_height[late]
                ),
            },
        },
        "offline_acceptance_reference": {
            "role": (
                "Post-processing evidence envelope only. These values must not be used as "
                "boundary conditions, source terms, clamps, or solver tuning constants."
            ),
            "time_aligned_primary_quantities": [
                "mouth_net_m3_s",
                "bottom_0p10m_equivalent_liquid_volume_m3",
                "whole_riser_equivalent_liquid_height_m",
            ],
            "capability_diagnostics_not_hard_gates_for_single_flux_1d_nodes": [
                "mouth_gross_up_m3_s",
                "mouth_gross_down_m3_s",
                "mouth_exchange_intensity_m3_s",
            ],
            "late_window_observed_p05_p95": {
                "mouth_net_m3_s": [
                    float(np.percentile(net[late], 5)),
                    float(np.percentile(net[late], 95)),
                ],
                "bottom_0p10m_equivalent_liquid_volume_m3": [
                    float(np.percentile(bottom_volume[late], 5)),
                    float(np.percentile(bottom_volume[late], 95)),
                ],
                "mouth_first_row_equivalent_liquid_volume_m3": [
                    float(
                        np.percentile(
                            column("mouth_first_row_equivalent_liquid_volume_m3")[late],
                            5,
                        )
                    ),
                    float(
                        np.percentile(
                            column("mouth_first_row_equivalent_liquid_volume_m3")[late],
                            95,
                        )
                    ),
                ],
                "whole_riser_equivalent_liquid_height_m": [
                    float(np.percentile(whole_height[late], 5)),
                    float(np.percentile(whole_height[late], 95)),
                ],
                "mouth_gross_up_m3_s": [
                    float(np.percentile(gross_up[late], 5)),
                    float(np.percentile(gross_up[late], 95)),
                ],
                "mouth_gross_down_magnitude_m3_s": [
                    float(np.percentile(-gross_down[late], 5)),
                    float(np.percentile(-gross_down[late], 95)),
                ],
            },
        },
        "trace": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(json.dumps({
        "integrated_exchange": output["integrated_exchange"],
        "offline_acceptance_reference": output["offline_acceptance_reference"],
        "late_distributions": output["observed_distributions"]["8p5_to_9p2"],
    }, indent=2))


if __name__ == "__main__":
    main()

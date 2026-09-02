"""Generate the Case A manuscript and candidate figures from traceable outputs.

The OpenFOAM frames require ASCII VTU files exported at 6.65, 6.85,
6.95, and 7.15 s.  Regenerate them from the decomposed Case A calculation
with ``foamToVTK -parallel -time '6.65,6.85,6.95,7.15' -fields
'(alpha.water)' -ascii -no-boundary -name VTK_CASEA_PAPER_FULL``.

The four-frame complete-path figure uses the archived ASCII VTU sequence
under ``VTK_CASEA_HTML`` near 1.50, 6.90, 7.60, and 10.10 s.  Its 1D panels
come from the accepted 0--13 s ``shockvisc_fct_v130`` comparison archive.
The horizontal pipe is reconstructed from the archived finite-volume liquid
fractions, and the riser is rendered directly from archived ``alpha_l(z,t)``;
no material-trace height is promoted to a liquid free surface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tests/test_01_vw2011/cases/A_Dt57p1_Ha0305_Yfs0356"
FIGURES = ROOT / "paper/figures"
VTU_ROOT = CASE / "openfoam/2d/VTK_CASEA_PAPER_FULL"
VTU_HTML_ROOT = CASE / "openfoam/2d/VTK_CASEA_HTML"
CURRENT_SNAPSHOT_FRAME_INDEX = (
    CASE
    / "outputs/frames_index_shockvisc_fct_v130_13s_fullzero.json"
)
CURRENT_SNAPSHOT_FIELDS = (
    CASE / "outputs/vertical_fields_shockvisc_fct_v130_13s_fullzero_display.npz"
)
CURRENT_SNAPSHOT_DIAGNOSTICS = (
    CASE / "outputs/vertical_fields_shockvisc_fct_v130_13s.npz"
)
ACTIVE_CURVE_SERIES = (
    CASE / "outputs/caseA_nohll_connected_pocket_model_series.csv"
)

# Once the resolved gas front lies within one vertical cell of the free
# surface, the archived column-height diagnostic no longer represents the
# unique free surface measured in the experiment.  Apply the same 0.02 Y*
# coalescence tolerance used by the Case-A event metrics.
COLUMN_COALESCENCE_TOL = 0.02
RISER_DISPLAY_ALPHA_MIN = 0.02
MAX_SNAPSHOT_TIME_MISMATCH_S = 0.021
# The symmetric five-point, second-order Savitzky-Golay kernel retains the
# weak resolved pressure oscillation without rescaling the archived model
# output.  It is zero-phase and suppresses grid-scale roughness more gently
# than the former three-point moving mean.
PRESSURE_SAVGOL_COEFFICIENTS = np.asarray(
    [-3.0, 12.0, 17.0, 12.0, -3.0]
) / 35.0

EXP = "#222222"
ONE_D = "#D55E00"
TWO_D = "#0072B2"
GRID = "#D9D9D9"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _read_vtu(path: Path) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, float]:
    root = ET.parse(path).getroot()
    points_node = root.find(".//Points/DataArray")
    if points_node is None or points_node.text is None:
        raise ValueError(f"No point coordinates in {path}")
    points = np.fromstring(points_node.text, sep=" ").reshape(-1, 3)

    arrays = {node.attrib.get("Name"): node for node in root.findall(".//Cells/DataArray")}
    connectivity = np.fromstring(arrays["connectivity"].text or "", sep=" ", dtype=int)
    offsets = np.fromstring(arrays["offsets"].text or "", sep=" ", dtype=int)
    starts = np.r_[0, offsets[:-1]]
    cells = [connectivity[i:j] for i, j in zip(starts, offsets)]

    alpha_node = root.find(".//CellData/DataArray[@Name='alpha.water']")
    time_node = root.find(".//FieldData/DataArray[@Name='TimeValue']")
    if alpha_node is None or alpha_node.text is None or time_node is None:
        raise ValueError(f"Missing alpha.water or time in {path}")
    alpha = np.fromstring(alpha_node.text, sep=" ")
    time_s = float((time_node.text or "0").strip())
    return points, cells, alpha, time_s


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns: dict[str, np.ndarray] = {}
    for key in rows[0]:
        values = [row[key].strip() for row in rows]
        try:
            columns[key] = np.asarray([float(value) if value else np.nan for value in values])
        except ValueError:
            columns[key] = np.asarray(values, dtype=object)
    return columns


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pressure_savgol5(values: np.ndarray) -> np.ndarray:
    """Apply the disclosed zero-phase five-point quadratic SG filter."""
    if values.size < PRESSURE_SAVGOL_COEFFICIENTS.size:
        return values.copy()
    half_window = PRESSURE_SAVGOL_COEFFICIENTS.size // 2
    padded = np.pad(values, (half_window, half_window), mode="edge")
    return np.convolve(
        padded,
        PRESSURE_SAVGOL_COEFFICIENTS,
        mode="valid",
    )


def _circular_depth_fraction(area_fraction: np.ndarray) -> np.ndarray:
    """Map circular-pipe liquid area fraction to liquid-depth fraction."""

    fraction = np.clip(np.asarray(area_fraction, dtype=float), 0.0, 1.0)
    theta = np.linspace(0.0, 2.0 * np.pi, 8193)
    area = (theta - np.sin(theta)) / (2.0 * np.pi)
    depth = 0.5 * (1.0 - np.cos(0.5 * theta))
    return np.interp(fraction, area, depth)


def _shape_preserving_cubic(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
) -> np.ndarray:
    """Evaluate the same non-overshooting cubic used by the accepted viewer."""

    nodes = np.asarray(x, dtype=float)
    values = np.asarray(y, dtype=float)
    query = np.asarray(x_new, dtype=float)
    if nodes.ndim != 1 or values.shape != nodes.shape or nodes.size < 2:
        raise ValueError("cubic interface reconstruction requires equal 1-D arrays")
    spacing = np.diff(nodes)
    if np.any(spacing <= 0.0):
        raise ValueError("interface reconstruction nodes must be strictly increasing")

    secant = np.diff(values) / spacing
    derivative = np.zeros_like(values)
    if values.size == 2:
        derivative[:] = secant[0]
    else:
        for index in range(1, values.size - 1):
            left = secant[index - 1]
            right = secant[index]
            if left == 0.0 or right == 0.0 or left * right <= 0.0:
                derivative[index] = 0.0
            else:
                weight_left = 2.0 * spacing[index] + spacing[index - 1]
                weight_right = spacing[index] + 2.0 * spacing[index - 1]
                derivative[index] = (weight_left + weight_right) / (
                    weight_left / left + weight_right / right
                )
        derivative[0] = (
            (2.0 * spacing[0] + spacing[1]) * secant[0]
            - spacing[0] * secant[1]
        ) / (spacing[0] + spacing[1])
        derivative[-1] = (
            (2.0 * spacing[-1] + spacing[-2]) * secant[-1]
            - spacing[-1] * secant[-2]
        ) / (spacing[-1] + spacing[-2])
        for endpoint, local_secant, adjacent_secant in (
            (0, secant[0], secant[1]),
            (-1, secant[-1], secant[-2]),
        ):
            if derivative[endpoint] * local_secant <= 0.0:
                derivative[endpoint] = 0.0
            elif (
                local_secant * adjacent_secant < 0.0
                and abs(derivative[endpoint]) > 3.0 * abs(local_secant)
            ):
                derivative[endpoint] = 3.0 * local_secant

    interval = np.searchsorted(nodes, query, side="right") - 1
    interval = np.clip(interval, 0, nodes.size - 2)
    width = nodes[interval + 1] - nodes[interval]
    coordinate = np.clip((query - nodes[interval]) / width, 0.0, 1.0)
    h00 = 2.0 * coordinate**3 - 3.0 * coordinate**2 + 1.0
    h10 = coordinate**3 - 2.0 * coordinate**2 + coordinate
    h01 = -2.0 * coordinate**3 + 3.0 * coordinate**2
    h11 = coordinate**3 - coordinate**2
    result = (
        h00 * values[interval]
        + h10 * width * derivative[interval]
        + h01 * values[interval + 1]
        + h11 * width * derivative[interval + 1]
    )
    return np.clip(result, np.min(values), np.max(values))


def _compact_cubic_bspline(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
) -> np.ndarray:
    """Reconstruct uniform-grid cell averages with a compact cubic B-spline."""

    nodes = np.asarray(x, dtype=float)
    values = np.asarray(y, dtype=float)
    query = np.asarray(x_new, dtype=float)
    spacing = np.diff(nodes)
    cell_width = float(np.median(spacing))
    if cell_width <= 0.0 or not np.allclose(spacing, cell_width, rtol=1.0e-8):
        raise ValueError("B-spline reconstruction requires a uniform grid")
    ghost_nodes = np.r_[
        nodes[0] - 2.0 * cell_width,
        nodes[0] - cell_width,
        nodes,
        nodes[-1] + cell_width,
        nodes[-1] + 2.0 * cell_width,
    ]
    ghost_values = np.r_[values[0], values[0], values, values[-1], values[-1]]
    distance = np.abs((query[:, None] - ghost_nodes[None, :]) / cell_width)
    weights = np.zeros_like(distance)
    inner = distance < 1.0
    outer = (distance >= 1.0) & (distance < 2.0)
    weights[inner] = (
        2.0 / 3.0 - distance[inner] ** 2 + 0.5 * distance[inner] ** 3
    )
    weights[outer] = (2.0 - distance[outer]) ** 3 / 6.0
    total = np.sum(weights, axis=1)
    reconstructed = np.sum(weights * ghost_values[None, :], axis=1) / np.maximum(
        total, 1.0e-15
    )
    return np.clip(reconstructed, np.min(values), np.max(values))


def _interp(data: dict[str, np.ndarray], x: str, y: str, value: float) -> float:
    valid = np.isfinite(data[x]) & np.isfinite(data[y])
    order = np.argsort(data[x][valid])
    return float(np.interp(value, data[x][valid][order], data[y][valid][order]))


def _interp_optional_local(
    data: dict[str, np.ndarray], x: str, y: str, value: float
) -> float | None:
    """Interpolate only across an adjacent finite pair; never bridge NaN gaps."""

    coordinates = np.asarray(data[x], dtype=float)
    observations = np.asarray(data[y], dtype=float)
    order = np.argsort(coordinates)
    coordinates = coordinates[order]
    observations = observations[order]
    exact = np.flatnonzero(np.isclose(
        coordinates, value, rtol=0.0, atol=1.0e-9
    ))
    if exact.size:
        result = float(observations[int(exact[0])])
        return result if np.isfinite(result) else None
    upper = int(np.searchsorted(coordinates, value, side="right"))
    if upper == 0 or upper >= coordinates.size:
        return None
    lower = upper - 1
    if not (
        np.isfinite(observations[lower])
        and np.isfinite(observations[upper])
    ):
        return None
    return float(np.interp(
        value,
        coordinates[[lower, upper]],
        observations[[lower, upper]],
    ))


def _vtu_header_time(path: Path) -> float:
    """Read only the small XML header needed to identify an archived VTU time."""
    pattern = re.compile(rb"Name='TimeValue'.*?\n\s*([0-9.eE+-]+)", re.DOTALL)
    with path.open("rb") as handle:
        match = pattern.search(handle.read(2048))
    if match is None:
        raise ValueError(f"No TimeValue found near the start of {path}")
    return float(match.group(1))


def _select_vtu_paths(root: Path, target_times: list[float]) -> list[Path]:
    """Select exact archived states without parsing every full VTU file."""
    available = [(path, _vtu_header_time(path)) for path in root.glob("*/internal.vtu")]
    selected: list[Path] = []
    for target in target_times:
        path, time_s = min(available, key=lambda item: abs(item[1] - target))
        if abs(time_s - target) > 5.0e-7:
            raise FileNotFoundError(
                f"No VTU state at {target:.2f} s under {root}; nearest is {time_s:.8g} s"
            )
        selected.append(path)
    return selected


def _load_1d_record(t_end: float = 9.0):
    """Recreate the frozen 1D states used by the archived frame viewer."""
    model_dir = CASE / "model"
    sys.path.insert(0, str(model_dir))
    try:
        from vw2011_network_twofluid import NetworkCase, run_network
    finally:
        sys.path.pop(0)
    case = NetworkCase(
        Dr=0.0571,
        air_head=0.305,
        init_water_level=0.356,
        cfl=0.65,
        t_end=t_end,
    )
    return case, run_network(case, verbose=False)


def _load_1d_case_definition():
    """Return Case-A geometry/parameters without rerunning the solver."""

    model_dir = CASE / "model"
    sys.path.insert(0, str(model_dir))
    try:
        from vw2011_network_twofluid import NetworkCase
    finally:
        sys.path.pop(0)
    return NetworkCase(
        Dr=0.0571,
        air_head=0.305,
        init_water_level=0.356,
        cfl=0.65,
        t_end=13.0,
    )


def _draw_common_outline(ax, pipe_length: float, pipe_diameter: float,
                         tower_centre: float, tower_width: float,
                         tower_height: float) -> None:
    """Draw the same connected pipe--tower walls on both model columns."""
    tower_left = tower_centre - 0.5 * tower_width
    tower_right = tower_left + tower_width
    wall = dict(color="#4A4A4A", linewidth=0.75, zorder=8)

    # Horizontal pipe walls.  The crown line is deliberately interrupted
    # across the tower bore so the pipe and tower form one connected domain.
    ax.plot([0.0, pipe_length], [-pipe_diameter, -pipe_diameter], **wall)
    ax.plot([0.0, 0.0], [-pipe_diameter, 0.0], **wall)
    ax.plot([pipe_length, pipe_length], [-pipe_diameter, 0.0], **wall)
    ax.plot([0.0, tower_left], [0.0, 0.0], **wall)
    ax.plot([tower_right, pipe_length], [0.0, 0.0], **wall)

    # Tower side walls start at the pipe crown; no bottom wall is drawn.
    ax.plot([tower_left, tower_left], [0.0, tower_height], **wall)
    ax.plot([tower_right, tower_right], [0.0, tower_height], **wall)
    # The tower discharges through an open ground-level rim.  Draw the ground
    # only outside the bore so no line closes the opening in either column.
    ground_extension = 0.12
    ground = dict(
        color="black", linewidth=0.8, linestyle="-",
        solid_capstyle="butt", zorder=9,
    )
    ax.plot(
        [tower_left - ground_extension, tower_left],
        [tower_height, tower_height],
        **ground,
    )
    ax.plot(
        [tower_right, tower_right + ground_extension],
        [tower_height, tower_height],
        **ground,
    )


def _format_common_domain(
    ax, row: int, nrows: int, show_axis_labels: bool = True
) -> None:
    """Apply a single coordinate system and true physical scale."""
    ax.set_xlim(-0.03, 4.036)
    ax.set_ylim(-0.124, 0.640)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([-0.1, 0.0, 0.3, 0.6])
    # Keep the bottom-row x values for scale, but remove y-axis numbers and
    # all protruding tick marks from the frame-free snapshot panels.
    ax.tick_params(
        direction="out", length=0, width=0, pad=1.5,
        labelsize=6.4, labelleft=False,
    )
    if not show_axis_labels:
        ax.tick_params(labelbottom=False)
    if show_axis_labels and row == nrows - 1:
        ax.set_xlabel("$x$ (m)", labelpad=1)
    if show_axis_labels:
        ax.set_ylabel("$y$ (m)", labelpad=1)
    # Snapshot panels use the physical pipe outline as their visual boundary;
    # suppress the rectangular Matplotlib axes frame around the geometry.
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_conservative_vertical_alpha_cells(
    ax,
    vertical_z: np.ndarray,
    alpha_l: np.ndarray,
    tower_left: float,
    tower_width: float,
    tower_height: float,
    water_color: str,
    air_color: str,
    display_alpha_min: float,
) -> dict[str, float]:
    """Render the archived riser fractions using the accepted HTML convention.

    Internal mixed cells are shown as an annular liquid film around a centred
    circular gas core.  The uppermost visible partial cell is instead drawn as
    an axial cut of height ``alpha_l*dz``.  Fractions at or below the display
    threshold are omitted only from the raster; all source liquid remains in
    the equivalent-height audit.
    """
    centres = np.asarray(vertical_z, dtype=float)
    fractions = np.asarray(alpha_l, dtype=float)
    if centres.ndim != 1 or fractions.ndim != 1 or centres.size != fractions.size:
        raise ValueError("vertical_z and alpha_l must be equal-length 1D arrays")
    if centres.size == 0 or not np.all(np.isfinite(centres)):
        raise ValueError("vertical_z must contain finite cell centres")
    if np.any(np.diff(centres) <= 0.0):
        raise ValueError("vertical_z cell centres must be strictly increasing")
    if not np.all(np.isfinite(fractions)):
        raise ValueError("alpha_l must contain finite cell fractions")
    if np.any((fractions < -1.0e-12) | (fractions > 1.0 + 1.0e-12)):
        raise ValueError("alpha_l lies outside [0, 1]")
    if not 0.0 <= float(display_alpha_min) < 1.0:
        raise ValueError("riser display alpha threshold must lie in [0, 1)")
    fractions = np.clip(fractions, 0.0, 1.0)

    interior_edges = 0.5 * (centres[:-1] + centres[1:])
    edges = np.concatenate(([0.0], interior_edges, [tower_height]))
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("vertical cell edges are invalid for the riser height")

    cell_heights = np.diff(edges)
    source_area = float(np.sum(fractions * tower_width * cell_heights))
    visible = np.flatnonzero(
        (fractions > float(display_alpha_min)) & (cell_heights > 0.0)
    )
    top_index = int(visible[-1]) if visible.size else -1
    displayed_source_area = 0.0
    visible_top = 0.0
    for cell_index, (cell_bottom, cell_top, liquid_fraction) in enumerate(zip(
        edges[:-1], edges[1:], fractions
    )):
        cell_height = float(cell_top - cell_bottom)
        if (
            liquid_fraction <= float(display_alpha_min)
            or cell_height <= 0.0
        ):
            continue
        displayed_source_area += float(
            liquid_fraction * tower_width * cell_height
        )

        if cell_index == top_index and liquid_fraction < 1.0 - 1.0e-6:
            layer_height = float(liquid_fraction * cell_height)
            ax.add_patch(Rectangle(
                (tower_left, float(cell_bottom)),
                tower_width,
                layer_height,
                facecolor=water_color,
                edgecolor="none",
                zorder=3,
            ))
            visible_top = max(visible_top, float(cell_bottom) + layer_height)
            continue

        ax.add_patch(Rectangle(
            (tower_left, float(cell_bottom)),
            tower_width,
            cell_height,
            facecolor=water_color,
            edgecolor="none",
            zorder=3,
        ))
        visible_top = max(visible_top, float(cell_top))
        if liquid_fraction < 1.0 - 1.0e-6:
            gas_core_width = float(
                np.sqrt(1.0 - liquid_fraction) * tower_width
            )
            ax.add_patch(Rectangle(
                (
                    tower_left + 0.5 * (tower_width - gas_core_width),
                    float(cell_bottom),
                ),
                gas_core_width,
                cell_height,
                facecolor=air_color,
                edgecolor="none",
                zorder=4,
            ))

    source_equivalent_height = source_area / tower_width
    displayed_source_equivalent_height = displayed_source_area / tower_width
    omitted_area = max(0.0, source_area - displayed_source_area)
    omitted_equivalent_height = max(
        0.0,
        source_equivalent_height - displayed_source_equivalent_height,
    )
    return {
        "source_liquid_area_m2": source_area,
        "source_equivalent_height_m": source_equivalent_height,
        "displayed_source_liquid_area_m2": displayed_source_area,
        "displayed_source_equivalent_height_m": (
            displayed_source_equivalent_height
        ),
        "display_omitted_liquid_area_m2": omitted_area,
        "display_omitted_equivalent_height_m": omitted_equivalent_height,
        "display_alpha_min": float(display_alpha_min),
        "visible_top_m": visible_top,
    }


def make_snapshots(
    complete_path: bool = False,
    snapshot_fields: Path | None = None,
    snapshot_index: Path | None = None,
    snapshot_diagnostics: Path | None = None,
    output_stem_override: str | None = None,
) -> None:
    series_2d = _read_csv(CASE / "openfoam/2d/outputs/openfoam_2d_series.csv")
    levels_2d = _read_csv(CASE / "openfoam/2d/outputs/openfoam_2d_levels.csv")
    selected_snapshot_index = snapshot_index or CURRENT_SNAPSHOT_FRAME_INDEX
    selected_snapshot_fields = snapshot_fields or CURRENT_SNAPSHOT_FIELDS
    frame_index_path = (
        selected_snapshot_index
        if complete_path and selected_snapshot_index.is_file()
        else CASE / "outputs/frames_index.json"
    )
    frames = json.loads(frame_index_path.read_text(encoding="utf-8"))
    coupled_fields = None
    diagnostic_fields = None
    selected_snapshot_diagnostics = snapshot_diagnostics
    if complete_path:
        if not selected_snapshot_fields.is_file():
            raise FileNotFoundError(
                "Current Case-A material-front field archive is missing: "
                f"{selected_snapshot_fields}"
            )
        coupled_fields = np.load(selected_snapshot_fields)
        if (
            selected_snapshot_diagnostics is None
            and selected_snapshot_fields.resolve()
            == CURRENT_SNAPSHOT_FIELDS.resolve()
        ):
            selected_snapshot_diagnostics = CURRENT_SNAPSHOT_DIAGNOSTICS
        if (
            selected_snapshot_diagnostics is not None
            and selected_snapshot_diagnostics.is_file()
        ):
            diagnostic_fields = np.load(selected_snapshot_diagnostics)

    if complete_path:
        # One panel pair per indispensable stage of the non-geysering pathway:
        # reflected horizontal-pocket propagation, tower entry/rise,
        # the breakthrough transition with rapid drainage, and the residual
        # liquid state after breakthrough.  Each pair uses the nearest archived
        # states to one nominal physical time; no event alignment or time shift
        # is applied.
        target_times = [1.50, 6.90, 7.60, 10.10]
        frame_ids = [
            min(range(len(frames)), key=lambda i: abs(float(frames[i]["time"]) - target))
            for target in target_times
        ]
        for target, frame_id in zip(target_times, frame_ids):
            if (
                abs(float(frames[frame_id]["time"]) - target)
                > MAX_SNAPSHOT_TIME_MISMATCH_S
            ):
                raise FileNotFoundError(
                    "No current 1D frame within the permitted archived-frame "
                    f"mismatch of {MAX_SNAPSHOT_TIME_MISMATCH_S:.3f} s at "
                    f"{target:.2f} s; "
                    f"nearest is {float(frames[frame_id]['time']):.8g} s"
                )
        vtu_paths = _select_vtu_paths(VTU_HTML_ROOT, target_times)
        stage_labels = [
            "Early post-valve pocket propagation",
            "Gas entry and interface rise",
            "Breakthrough transition and rapid riser drainage",
            "Post-breakthrough residual liquid near the riser base",
        ]
        output_stem = output_stem_override or "caseA_1d2d_snapshots_4frame"
    else:
        # Pair the archived states by time.  The 1D viewer records at times ending
        # in 0.01/0.06 s, whereas OpenFOAM fields are stored every 0.05 s, so these
        # are the nearest pairs (maximum |Delta T*| = 0.013).
        frame_ids = [66, 68, 69, 71]
        vtu_paths = sorted(VTU_ROOT.glob("*/internal.vtu"))
        if len(vtu_paths) != 4:
            raise FileNotFoundError(f"Expected four VTU frames under {VTU_ROOT}")
        stage_labels = ["Interface rise"] * len(frame_ids)
        output_stem = "caseA_1d2d_snapshots"

    vtu_data = sorted((_read_vtu(path) for path in vtu_paths), key=lambda item: item[3])
    if complete_path:
        case_1d = _load_1d_case_definition()
        record_1d = None
    else:
        case_1d, record_1d = _load_1d_record(
            t_end=max(float(frames[frame_id]["time"]) for frame_id in frame_ids)
        )
    water_cmap = LinearSegmentedColormap.from_list(
        "water", [(0.0, "#F2F4F8"), (0.12, "#D8ECFF"), (1.0, "#2B7FFF")]
    )
    water_color = "#2B7FFF"
    air_color = "#F2F4F8"

    pipe_length = float(case_1d.L_tunnel)
    pipe_diameter = float(case_1d.D)
    tower_centre = float(case_1d.x_riser)
    tower_width = float(case_1d.Dr)
    tower_height = float(case_1d.riser_height)
    tower_left = tower_centre - 0.5 * tower_width

    # The two columns share an identical coordinate frame, true physical
    # aspect ratio, and geometry outline.  Only the computed phase state may
    # differ between columns.
    nrows = len(frame_ids)
    figure_height = 5.35 if complete_path else 5.45
    fig, axes = plt.subplots(nrows, 2, figsize=(7.2, figure_height), constrained_layout=False,
                             sharex=True, sharey=True)
    plt.subplots_adjust(left=0.075, right=0.99, bottom=0.09, top=0.88,
                        wspace=0.10, hspace=0.50 if complete_path else 0.56)
    fig.text(0.285, 0.965, "Present 1D model", ha="center", va="top",
             fontweight="bold", fontsize=9.5)
    fig.text(0.765, 0.965, "2D OpenFOAM", ha="center", va="top",
             fontweight="bold", fontsize=9.5)

    common_times: list[float] = []
    selected_manifest: list[dict] = []
    for row, ((points, cells, alpha, time_s), frame_id) in enumerate(
        zip(vtu_data, frame_ids)
    ):
        frame = frames[frame_id]

        time_1d = float(frame["time"])
        if coupled_fields is not None:
            field_times = np.asarray(coupled_fields["time"], dtype=float)
            state_idx = int(np.argmin(np.abs(field_times - time_1d)))
            if abs(float(field_times[state_idx]) - time_1d) > 1.0e-3:
                raise ValueError(
                    f"No coupled two-fluid state matches frame time {time_1d}"
                )
        else:
            state_idx = int(
                np.argmin(
                    np.abs(np.asarray(record_1d["frames_t"]) - time_1d)
                )
            )
            if abs(float(record_1d["frames_t"][state_idx]) - time_1d) > 0.015:
                raise ValueError(
                    f"No frozen 1D state matches archived frame time {time_1d}"
                )

        ax_1d = axes[row, 0]
        # Rasterize the phase layer as one image when exporting PDF.  Leaving
        # hundreds of adjacent cell rectangles as separate vector objects
        # produces white hairline seams after pdflatex rescales the figure;
        # the wall outline and text remain vector graphics above zorder 4.
        ax_1d.set_rasterization_zorder(4)
        if coupled_fields is not None:
            horizontal_fraction = np.clip(np.array(
                coupled_fields["horizontal_alpha_l"][state_idx],
                dtype=float,
                copy=True,
            ), 0.0, 1.0)
            horizontal_dx = pipe_length / horizontal_fraction.size
            horizontal_x = (
                np.arange(horizontal_fraction.size, dtype=float) + 0.5
            ) * horizontal_dx
            dense_horizontal_x = np.linspace(
                0.0, pipe_length, 8 * horizontal_fraction.size + 1
            )
            horizontal_depth = pipe_diameter * _circular_depth_fraction(
                horizontal_fraction
            )
            node_x = np.r_[0.0, horizontal_x, pipe_length]
            node_depth = np.r_[
                horizontal_depth[0],
                horizontal_depth,
                horizontal_depth[-1],
            ]
            dense_horizontal_depth = _shape_preserving_cubic(
                node_x, node_depth, dense_horizontal_x
            )
            dense_surface_y = -pipe_diameter + dense_horizontal_depth
            material_front_x = frame.get("eastMaterialFront")
            vertical_z = np.asarray(coupled_fields["z"], dtype=float)
            area_fraction = np.asarray(
                coupled_fields["alpha_l"][state_idx], dtype=float
            )
        else:
            dense_horizontal_x = None
            dense_surface_y = None
            material_front_x = None
            horizontal_x = np.asarray(record_1d["xt"], dtype=float)
            horizontal_dx = float(record_1d["dx"])
            alt = np.clip(
                np.asarray(record_1d["frames_alt"][state_idx]), 0.0, 1.0
            )
            vertical_z = np.asarray(record_1d["zr"], dtype=float)
            vertical_dz = float(record_1d["dz"])
            area_fraction = np.clip(
                np.asarray(record_1d["frames_alr"][state_idx]), 0.0, 1.0
            )
            agr = np.clip(
                np.asarray(record_1d["frames_agr"][state_idx]), 0.0, 1.0
            )

        ax_1d.add_patch(Rectangle(
            (0.0, -pipe_diameter), pipe_length, pipe_diameter,
            facecolor=air_color, edgecolor="none", zorder=1,
        ))
        if coupled_fields is not None:
            front_x = (
                float(np.clip(material_front_x, tower_centre, pipe_length))
                if material_front_x is not None
                else pipe_length
            )
            swept = dense_horizontal_x < front_x
            swept_x = np.r_[dense_horizontal_x[swept], front_x]
            swept_surface = np.interp(
                swept_x, dense_horizontal_x, dense_surface_y
            )
            if swept_x.size:
                ax_1d.fill_between(
                    swept_x,
                    -pipe_diameter,
                    swept_surface,
                    color=water_color,
                    linewidth=0.0,
                    antialiased=True,
                    zorder=2,
                )
            if front_x < pipe_length:
                ax_1d.add_patch(Rectangle(
                    (front_x, -pipe_diameter),
                    pipe_length - front_x,
                    pipe_diameter,
                    facecolor=water_color,
                    edgecolor="none",
                    zorder=2,
                ))
        else:
            for x_i, fraction in zip(horizontal_x, alt):
                if fraction > 0.03:
                    ax_1d.add_patch(Rectangle(
                        (x_i - 0.5 * horizontal_dx, -pipe_diameter),
                        horizontal_dx, fraction * pipe_diameter,
                        facecolor=water_color, edgecolor="none", zorder=2,
                    ))
        ax_1d.add_patch(Rectangle(
            (tower_left, 0.0), tower_width, tower_height,
            facecolor=air_color, edgecolor="none", zorder=1,
        ))
        if coupled_fields is not None:
            vertical_render_audit = _draw_conservative_vertical_alpha_cells(
                ax_1d,
                vertical_z,
                area_fraction,
                tower_left,
                tower_width,
                tower_height,
                water_color,
                air_color,
                RISER_DISPLAY_ALPHA_MIN,
            )
            material_trace_top = frame.get("materialHeight")
            if material_trace_top is not None:
                material_trace_top = float(
                    np.clip(material_trace_top, 0.0, tower_height)
                )
            legacy_water_top = None
        else:
            # Retain the historical rendering for the legacy short-window
            # snapshot path.  The complete-path figure above must not use this
            # scalar height to reconstruct the resolved vertical field.
            legacy_water_top = float(np.clip(frame["wtop"], 0.0, tower_height))
            material_trace_top = None
            vertical_render_audit = None
            for z_i, liquid_fraction, gas_fraction in zip(
                vertical_z, area_fraction, agr
            ):
                if z_i > legacy_water_top:
                    continue
                if liquid_fraction > 0.02:
                    ax_1d.add_patch(Rectangle(
                        (tower_left, z_i - 0.5 * vertical_dz),
                        tower_width, vertical_dz,
                        facecolor=water_color, edgecolor="none", zorder=2,
                    ))
                if gas_fraction > 0.01:
                    gas_core_width = tower_width * np.sqrt(gas_fraction)
                    ax_1d.add_patch(Rectangle(
                        (tower_centre - 0.5 * gas_core_width,
                         z_i - 0.5 * vertical_dz),
                        gas_core_width, vertical_dz,
                        facecolor="white", edgecolor="none", zorder=3,
                    ))
        _draw_common_outline(
            ax_1d, pipe_length, pipe_diameter, tower_centre,
            tower_width, tower_height,
        )
        _format_common_domain(
            ax_1d, row, len(frame_ids), show_axis_labels=not complete_path
        )
        # Compute dimensionless time from its definition.  The archived 1D
        # comparison CSV currently ends before the 10.10-s recession frame, so
        # np.interp would silently clamp that frame to the CSV's final T*.
        tstar_1d = (
            time_1d * np.sqrt(9.81 * case_1d.Dr) / case_1d.riser_height
        )
        yint_1d = float(frame["itop"])
        if complete_path:
            ax_1d.text(
                0.012, 0.91, f"({chr(ord('a') + row)})",
                transform=ax_1d.transAxes, ha="left", va="top",
                fontsize=8.0, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
            )
        else:
            ax_1d.text(
                0.018, 0.86, rf"$Y_{{\rm int}}\approx{yint_1d:.2f}$ m",
                transform=ax_1d.transAxes, ha="left", va="top", fontsize=7.2,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
            )

        ax_2d = axes[row, 1]
        # OpenFOAM uses the pipe centreline as y=0.  Translate it by D/2 so
        # both columns use the pipe crown as y=0, matching the 1D tower datum.
        common_points = points.copy()
        common_points[:, 1] -= 0.5 * pipe_diameter
        polys = []
        for idx in cells:
            xy = np.unique(common_points[idx, :2], axis=0)
            centre = xy.mean(axis=0)
            angle = np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0])
            polys.append(xy[np.argsort(angle)])
        collection = PolyCollection(
            polys, array=alpha, cmap=water_cmap, norm=Normalize(0, 1),
            edgecolors="none", rasterized=True,
        )
        ax_2d.add_collection(collection)
        _draw_common_outline(
            ax_2d, pipe_length, pipe_diameter, tower_centre,
            tower_width, tower_height,
        )
        _format_common_domain(
            ax_2d, row, len(frame_ids), show_axis_labels=not complete_path
        )
        if complete_path:
            ax_2d.set_ylabel("")
            ax_2d.tick_params(labelleft=False)
        tstar = _interp(series_2d, "time_s", "Tstar", time_s)
        if complete_path:
            yint_star = _interp_optional_local(
                levels_2d, "time_s", "Yint_star", time_s
            )
            yfs_star = _interp_optional_local(
                levels_2d, "time_s", "Yfs_star", time_s
            )
        else:
            yint_star = _interp(levels_2d, "time_s", "Yint_star", time_s)
            yfs_star = _interp(levels_2d, "time_s", "Yfs_star", time_s)
        yint_m = None if yint_star is None else 0.610 * yint_star
        yfs_2d_m = None if yfs_star is None else 0.610 * yfs_star
        if complete_path:
            ax_2d.text(
                0.012, 0.91, f"({chr(ord('a') + row)})",
                transform=ax_2d.transAxes, ha="left", va="top",
                fontsize=8.0, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
            )
        else:
            ax_2d.text(
                0.018, 0.86, rf"$Y_{{\rm int}}\approx{yint_m:.2f}$ m",
                transform=ax_2d.transAxes, ha="left", va="top", fontsize=7.2,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
            )
        common_times.append(
            float(target_times[row])
            if complete_path
            else 0.5 * (time_1d + time_s)
        )
        distinct_bulk_free_surface = None
        if legacy_water_top is not None and legacy_water_top - yint_1d > 1.0e-9:
            distinct_bulk_free_surface = legacy_water_top
        max_upward_liquid_velocity = None
        max_downward_liquid_velocity = None
        max_upward_liquid_discharge = None
        max_downward_liquid_discharge = None
        motion_fields = diagnostic_fields or coupled_fields
        motion_state_idx = None
        if motion_fields is not None:
            motion_times = np.asarray(motion_fields["time"], dtype=float)
            candidate_motion_idx = int(
                np.argmin(np.abs(motion_times - time_1d))
            )
            if (
                abs(float(motion_times[candidate_motion_idx]) - time_1d)
                <= 1.0e-8
            ):
                motion_state_idx = candidate_motion_idx
        if motion_state_idx is not None:
            if "vertical_liquid_velocity" in motion_fields.files:
                max_upward_liquid_velocity = max(
                    0.0,
                    float(np.max(
                        motion_fields["vertical_liquid_velocity"][motion_state_idx]
                    )),
                )
                max_downward_liquid_velocity = min(
                    0.0,
                    float(np.min(
                        motion_fields["vertical_liquid_velocity"][motion_state_idx]
                    )),
                )
            if "riser_upward_discharge" in motion_fields.files:
                max_upward_liquid_discharge = max(
                    0.0,
                    float(np.max(
                        motion_fields["riser_upward_discharge"][motion_state_idx]
                    )),
                )
            if "riser_downward_discharge" in motion_fields.files:
                max_downward_liquid_discharge = min(
                    0.0,
                    float(np.min(
                        motion_fields["riser_downward_discharge"][motion_state_idx]
                    )),
                )
        selected_manifest.append(
            {
                "stage": stage_labels[row],
                "one_d": {
                    "frame_index": int(frame_ids[row]),
                    "time_s": time_1d,
                    "Tstar": tstar_1d,
                    "Yfs_m": distinct_bulk_free_surface,
                    "material_trace_top_m": material_trace_top,
                    "frame_liquid_equivalent_height_m": float(
                        frame.get(
                            "liquidEquivalentHeight",
                            vertical_render_audit["source_equivalent_height_m"]
                            if vertical_render_audit is not None
                            else frame["wtop"],
                        )
                    ),
                    "alpha_l_integral_equivalent_height_m": (
                        vertical_render_audit["source_equivalent_height_m"]
                        if vertical_render_audit is not None
                        else None
                    ),
                    "alpha_l_integral_volume_m3": (
                        vertical_render_audit["source_equivalent_height_m"]
                        * 0.25 * np.pi * tower_width**2
                        if vertical_render_audit is not None
                        else None
                    ),
                    "vertical_rendering_audit": vertical_render_audit,
                    "display_visible_top_m": (
                        vertical_render_audit["visible_top_m"]
                        if vertical_render_audit is not None
                        else None
                    ),
                    "Yint_m": yint_1d,
                    "max_upward_liquid_velocity_m_s": max_upward_liquid_velocity,
                    "max_downward_liquid_velocity_m_s": (
                        max_downward_liquid_velocity
                    ),
                    "max_local_upward_liquid_discharge_m3_s": (
                        max_upward_liquid_discharge
                    ),
                    "max_local_downward_liquid_discharge_m3_s": (
                        max_downward_liquid_discharge
                    ),
                    "source": frame["file"],
                },
                "two_d": {
                    "time_s": time_s,
                    "Tstar": tstar,
                    "Yfs_m": yfs_2d_m,
                    "Yint_m": yint_m,
                    "source": str(vtu_paths[row].relative_to(CASE)).replace("\\", "/"),
                },
                "delta_time_s": time_1d - time_s,
            }
        )

    # A single physical-time label spans each row so the pairing criterion is
    # explicit without presenting both dimensional and dimensionless time.
    fig.canvas.draw()
    for row, common_time in enumerate(common_times):
        left = axes[row, 0].get_position()
        right = axes[row, 1].get_position()
        if complete_path:
            row_label = f"Time ≈ {common_time:.2f} s"
        else:
            row_label = f"Time = {common_time:.2f} s"
        fig.text(
            left.x0, max(left.y1, right.y1) + 0.009,
            row_label,
            ha="left", va="bottom", fontsize=8.0,
        )

    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{output_stem}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)

    if complete_path:
        manifest = {
            "case": "VW2011 Test A",
            "figure_claim": (
                "The accepted shockvisc_fct_v130 1D archive and supporting 2D "
                "OpenFOAM calculation select the same non-geysering branch and "
                "preserve the event ordering from horizontal-pocket propagation "
                "through tower entry, breakthrough, and post-breakthrough "
                "drainage; detailed late-time phase topology is not claimed to "
                "match."
            ),
            "time_pairing": (
                "nearest archived 1D and 2D states to the nominal physical "
                "times; no time shift; maximum absolute pair mismatch = "
                f"{max(abs(item['delta_time_s']) for item in selected_manifest):.6g} s"
            ),
            "two_d_role": "supporting planar VOF calculation",
            "two_d_geometry": "vertical-plane rectangular slot; 24 cells across D",
            "one_d_horizontal_model": (
                "accepted Case-A shockvisc_fct_v130 archive; finite-volume "
                "liquid-area states are converted to circular-pipe depth west "
                "of the computed material front, with the sealed branch retained "
                "as water-filled"
            ),
            "one_d_horizontal_source": str(
                selected_snapshot_fields.resolve().relative_to(ROOT.resolve())
            ).replace("\\", "/"),
            "one_d_frame_index_source": str(
                selected_snapshot_index.resolve().relative_to(ROOT.resolve())
            ).replace("\\", "/"),
            "one_d_diagnostic_source": (
                str(
                    selected_snapshot_diagnostics.resolve().relative_to(
                        ROOT.resolve()
                    )
                ).replace("\\", "/")
                if selected_snapshot_diagnostics is not None
                and selected_snapshot_diagnostics.is_file()
                else None
            ),
            "one_d_rendering": (
                "the horizontal pipe uses the circular-depth shape-preserving "
                "cubic reconstruction and an exact vertical material-front "
                "closure; the riser uses the accepted HTML annular-core and "
                "uppermost-partial-cell convention without an interface outline"
            ),
            "one_d_tower_source": (
                "archived alpha_l(z,t) from the accepted 0--13 s display "
                "archive; materialHeight and itop remain diagnostics and do not "
                "clip or fill the riser; cells with alpha_l <= 0.02 are omitted "
                "from the raster only and remain in the liquid-equivalent height"
            ),
            "input_sha256": {
                "one_d_fields": _sha256(selected_snapshot_fields),
                "one_d_frame_index": _sha256(selected_snapshot_index),
                **(
                    {
                        "one_d_diagnostics": _sha256(
                            selected_snapshot_diagnostics
                        )
                    }
                    if selected_snapshot_diagnostics is not None
                    and selected_snapshot_diagnostics.is_file()
                    else {}
                ),
            },
            "selected_frames": selected_manifest,
            "outputs": [
                f"paper/figures/{output_stem}.png",
                f"paper/figures/{output_stem}.pdf",
            ],
            "manuscript_status": (
                "active four-frame manuscript figure"
                if output_stem == "caseA_1d2d_snapshots_4frame"
                else "preview only; does not replace the active manuscript figure"
            ),
        }
        manifest_name = (
            "caseA_paper_figure_manifest.json"
            if output_stem == "caseA_1d2d_snapshots_4frame"
            else f"{output_stem}_manifest.json"
        )
        (CASE / "outputs" / manifest_name).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


def make_curves(
    model_series: Path | None = None,
    output_stem: str = "caseA_experiment_1d2d_curves",
    pressure_mode: str = "moderate",
) -> None:
    pressure_exp = _read_csv(
        CASE / "data/digitized/fig5_caseA_Hstar_band.csv"
    )
    levels_exp = _read_csv(
        CASE / "data/digitized/fig7_caseA_markers_vector.csv"
    )
    model_1d = _read_csv(model_series or ACTIVE_CURVE_SERIES)
    pressure_2d = _read_csv(CASE / "openfoam/2d/outputs/openfoam_2d_series.csv")
    levels_2d = _read_csv(CASE / "openfoam/2d/outputs/openfoam_2d_levels.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    plt.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.91, wspace=0.24)

    ax = axes[0]
    # The source raster contains three overlapping repetitions.  The archived
    # Hstar_med column is their extracted central dark-pixel trace, not a
    # statistical median obtained after separating the three repetitions.
    order = np.argsort(pressure_exp["Tstar"])
    ax.plot(
        pressure_exp["Tstar"][order],
        pressure_exp["Hstar_med"][order],
        color=EXP,
        lw=1.35,
        label="Experiment",
        zorder=2,
    )
    # Main-text validation must use the frozen solver's direct transducer
    # output.  Case-specific observation closures belong only in explicitly
    # labelled sensitivity analyses.
    pressure_key = "transducer_Hstar"
    one_d_valid = np.isfinite(model_1d[pressure_key])
    one_d_pressure = {key: values[one_d_valid] for key, values in model_1d.items()}
    dt = float(np.median(np.diff(one_d_pressure["t_s"])))
    window = max(3, int(round(0.8 / dt)))
    left = window // 2
    right = window - 1 - left
    padded = np.pad(one_d_pressure[pressure_key], (left, right), mode="edge")
    pressure_1d_mean = np.convolve(padded, np.ones(window) / window, mode="valid")
    if pressure_mode == "raw":
        pressure_1d_plot = one_d_pressure[pressure_key]
        one_d_pressure_label = "Present model (raw)"
        one_d_pressure_lw = 1.25
    elif pressure_mode == "moderate":
        pressure_1d_plot = _pressure_savgol5(one_d_pressure[pressure_key])
        one_d_pressure_label = "Present model"
        one_d_pressure_lw = 1.5
    elif pressure_mode == "cycle-mean":
        pressure_1d_plot = pressure_1d_mean
        one_d_pressure_label = "Present model"
        one_d_pressure_lw = 1.7
    else:
        raise ValueError(f"unsupported pressure mode: {pressure_mode}")
    ax.plot(one_d_pressure["Tstar"], pressure_1d_plot, color=ONE_D,
            lw=one_d_pressure_lw)
    ax.plot(pressure_2d["Tstar"], pressure_2d["Hstar_smooth"], color=TWO_D,
            lw=1.7)
    ax.set_xlim(0, 10)
    # Match the published Fig. 5 pressure-axis range and 0.5 tick spacing.
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5], ["0", "0.5", "1", "1.5"])
    ax.set_xlabel(r"$T_{\rm rel}^*$")
    ax.set_ylabel(r"$H^*$")
    ax.set_title("(a) Pressure response", loc="left", fontweight="bold")
    ax.legend(
        handles=[
            Line2D([0], [0], color=EXP, lw=1.35, label="Experiment"),
            Line2D([0], [0], color=ONE_D, lw=one_d_pressure_lw,
                   label=one_d_pressure_label),
            Line2D([0], [0], color=TWO_D, lw=1.7, label="2D OpenFOAM"),
        ],
        frameon=False, ncol=1, loc="upper right",
        handlelength=2.2, columnspacing=0.8,
    )

    ax = axes[1]
    # Replot the separately digitized marker centres with native Matplotlib
    # symbols.  This keeps the manuscript output fully vector-based while
    # preserving the six run-specific marker families from the source panel.
    obs_tmax = float(np.max(levels_exp["Tstar"]))
    one_d_finite = (
        np.isfinite(model_1d["Yfs_star"])
        & np.isfinite(model_1d["Yint_star"])
    )
    one_d_coalesced = (
        one_d_finite
        & (model_1d["Yint_star"] > 0.10)
        & (
            model_1d["Yfs_star"] - model_1d["Yint_star"]
            <= COLUMN_COALESCENCE_TOL
        )
    )
    coalescence_indices = np.flatnonzero(one_d_coalesced)
    one_d_tmax = obs_tmax
    if coalescence_indices.size:
        one_d_tmax = min(
            one_d_tmax,
            float(model_1d["Tstar"][coalescence_indices[0]]),
        )
    one_d_fs_mask = one_d_finite & (model_1d["Tstar"] <= one_d_tmax)
    one_d_int_mask = one_d_finite & (model_1d["Tstar"] <= one_d_tmax)
    two_d_finite = (
        np.isfinite(levels_2d["Yfs_star"])
        & np.isfinite(levels_2d["Yint_star"])
    )
    two_d_fs_mask = two_d_finite & (levels_2d["Tstar"] <= obs_tmax)
    two_d_int_mask = two_d_finite & (levels_2d["Tstar"] <= obs_tmax)
    marker_specs = {
        ("fs", 1.0): ("^", True),
        ("fs", 2.0): ("x", True),
        ("fs", 3.0): ("o", True),
        ("int", 1.0): ("D", False),
        ("int", 2.0): ("s", False),
        ("int", 3.0): ("o", False),
    }
    for (observable, run), (marker, filled) in marker_specs.items():
        marker_mask = (
            (levels_exp["observable"] == observable)
            & (levels_exp["run"] == run)
        )
        scatter_args = {
            "s": 12,
            "marker": marker,
            "linewidth": 0.72,
            "zorder": 4,
        }
        if marker == "x":
            scatter_args["color"] = EXP
        else:
            scatter_args["facecolor"] = EXP if filled else "none"
            scatter_args["edgecolor"] = EXP
        ax.scatter(
            levels_exp["Tstar"][marker_mask],
            levels_exp["Ystar"][marker_mask],
            **scatter_args,
        )
    ax.plot(
        model_1d["Tstar"][one_d_fs_mask],
        model_1d["Yfs_star"][one_d_fs_mask],
        color=ONE_D, lw=1.7, zorder=3,
    )
    ax.plot(
        model_1d["Tstar"][one_d_int_mask],
        model_1d["Yint_star"][one_d_int_mask],
        color=ONE_D, lw=1.7, ls=":", zorder=3,
    )
    ax.plot(
        levels_2d["Tstar"][two_d_fs_mask],
        levels_2d["Yfs_star"][two_d_fs_mask],
        color=TWO_D, lw=1.7, zorder=3,
    )
    ax.plot(
        levels_2d["Tstar"][two_d_int_mask],
        levels_2d["Yint_star"][two_d_int_mask],
        color=TWO_D, lw=1.7, ls=":", zorder=3,
    )
    ax.set_xlim(7, 10)
    # Match the published Fig. 7 elevation-axis range and 0.25 tick spacing.
    ax.set_ylim(0, 1.0)
    ax.set_yticks(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        ["0", "0.25", "0.5", "0.75", "1"],
    )
    ax.set_xlabel(r"$T_{\rm rel}^*$")
    ax.set_ylabel(r"$Y_{\rm int}^*$ & $Y_{\rm fs}^*$")
    ax.set_title("(b) Free surface and interface", loc="left", fontweight="bold")
    yfs_markers = (
        Line2D([0], [0], color=EXP, marker="^", mfc=EXP, ls="none",
               ms=3.8, mew=0.8),
        Line2D([0], [0], color=EXP, marker="x", ls="none",
               ms=3.8, mew=0.8),
        Line2D([0], [0], color=EXP, marker="o", mfc=EXP, ls="none",
               ms=3.8, mew=0.8),
    )
    yint_markers = (
        Line2D([0], [0], color=EXP, marker="D", mfc="none", ls="none",
               ms=3.8, mew=0.8),
        Line2D([0], [0], color=EXP, marker="s", mfc="none", ls="none",
               ms=3.8, mew=0.8),
        Line2D([0], [0], color=EXP, marker="o", mfc="none", ls="none",
               ms=3.8, mew=0.8),
    )
    # Split the legend by evidence type.  The experimental legend decodes the
    # three repeated-run symbols, while the model legend names every plotted
    # colour--line-style combination explicitly.
    experiment_legend = ax.legend(
        handles=[
            yfs_markers,
            yint_markers,
        ],
        labels=[
            r"$Y_{\rm fs}^*$ (Experiment)",
            r"$Y_{\rm int}^*$ (Experiment)",
        ],
        # Give the three repeated-run symbols equal, clearly separated centres.
        # A wider tuple handle avoids the near-overlap produced by Matplotlib's
        # compact default legend allocation.
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.55)},
        # Shift the experimental key inward so the two legend blocks read as a
        # balanced pair rather than hugging opposite plot edges.
        frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(0.12, 0.99),
        handlelength=2.8, handletextpad=0.65, labelspacing=0.55,
    )
    ax.add_artist(experiment_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color=ONE_D, lw=1.7),
            Line2D([0], [0], color=ONE_D, lw=1.7, ls=":"),
            Line2D([0], [0], color=TWO_D, lw=1.7),
            Line2D([0], [0], color=TWO_D, lw=1.7, ls=":"),
        ],
        labels=[
            r"$Y_{\rm fs}^*$ (Present model)",
            r"$Y_{\rm int}^*$ (Present model)",
            r"$Y_{\rm fs}^*$ (2D OpenFOAM)",
            r"$Y_{\rm int}^*$ (2D OpenFOAM)",
        ],
        frameon=False, ncol=1, loc="upper right", bbox_to_anchor=(0.99, 0.99),
        handlelength=2.2, handletextpad=0.55, labelspacing=0.42,
        fontsize=6.7,
    )

    for ax in axes:
        # Keep the reference paper's boxed axes but omit internal grid lines so
        # that the experimental symbols and three model traces remain primary.
        ax.grid(False)
        ax.set_axisbelow(True)
        # Follow the boxed-axis convention of the reference paper: retain a
        # black border on all four sides while drawing the tick marks inward.
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)
        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            length=3,
            width=0.8,
            color="black",
            top=False,
            right=False,
        )

    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{output_stem}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=("all", "snapshots", "three-frame", "four-frame", "curves"),
        default="all",
        help="Generate all figures or only the selected Case A artifact.",
    )
    parser.add_argument(
        "--model-series",
        type=Path,
        help=(
            "Optional candidate 1D series CSV; the manuscript default is the "
            "declared connected-pocket sensitivity archive."
        ),
    )
    parser.add_argument(
        "--output-stem",
        default="caseA_experiment_1d2d_curves",
        help="Output file stem under paper/figures.",
    )
    parser.add_argument(
        "--pressure-mode",
        choices=("cycle-mean", "moderate", "raw"),
        default="moderate",
        help=(
            "Plot the disclosed 0.8 s cycle mean, a five-point second-order "
            "Savitzky-Golay filtered trace, "
            "or the archived raw 1D transducer trace."
        ),
    )
    parser.add_argument(
        "--snapshot-fields",
        type=Path,
        help=(
            "Optional 1D NPZ archive for the complete-path snapshot figure; "
            "defaults to the accepted shockvisc_fct_v130 display archive."
        ),
    )
    parser.add_argument(
        "--snapshot-index",
        type=Path,
        help=(
            "Optional rendered-frame JSON index for the complete-path snapshot figure; "
            "defaults to the current vertical-front index."
        ),
    )
    parser.add_argument(
        "--snapshot-diagnostics",
        type=Path,
        help=(
            "Optional full 1D diagnostic NPZ corresponding to the snapshot "
            "archive; used only for velocity/discharge provenance."
        ),
    )
    parser.add_argument(
        "--snapshot-output-stem",
        help=(
            "Optional output stem for the complete-path snapshot figure. Use a preview "
            "stem to inspect a candidate without replacing the manuscript figure."
        ),
    )
    args = parser.parse_args()

    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    if args.only in ("all", "snapshots"):
        make_snapshots()
    if args.only in ("three-frame", "four-frame"):
        make_snapshots(
            complete_path=True,
            snapshot_fields=args.snapshot_fields,
            snapshot_index=args.snapshot_index,
            snapshot_diagnostics=args.snapshot_diagnostics,
            output_stem_override=args.snapshot_output_stem,
        )
    if args.only in ("all", "curves"):
        make_curves(args.model_series, args.output_stem, args.pressure_mode)


if __name__ == "__main__":
    main()

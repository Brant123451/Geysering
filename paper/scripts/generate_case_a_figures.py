"""Generate the Case A manuscript and candidate figures from traceable outputs.

The OpenFOAM frames require ASCII VTU files exported at 6.65, 6.85,
6.95, and 7.15 s.  Regenerate them from the decomposed Case A calculation
with ``foamToVTK -parallel -time '6.65,6.85,6.95,7.15' -fields
'(alpha.water)' -ascii -no-boundary -name VTK_CASEA_PAPER_FULL``.

The three-frame complete-path figure uses the archived ASCII VTU sequence
under ``VTK_CASEA_HTML`` at 1.50, 6.90, and 10.10 s and the recomputed,
fully coupled Case-A two-fluid fields produced by
``caseA_make_frame_viewer.py --variant twofluid_coupled``.
"""

from __future__ import annotations

import argparse
import csv
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
COUPLED_FRAME_INDEX = CASE / "outputs/frames_index_twofluid_coupled.json"
COUPLED_FIELDS = CASE / "outputs/vertical_fields_twofluid_coupled.npz"

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


def _interp(data: dict[str, np.ndarray], x: str, y: str, value: float) -> float:
    valid = np.isfinite(data[x]) & np.isfinite(data[y])
    order = np.argsort(data[x][valid])
    return float(np.interp(value, data[x][valid][order], data[y][valid][order]))


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


def _load_1d_case_definition(horizontal_model: str = "case_a_contact"):
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
        horizontal_model=horizontal_model,
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
    # Centre the top platform on the tower axis and use a solid stroke so the
    # two overhangs remain visually and geometrically identical.
    platform_half_width = 0.5 * tower_width + 0.045
    ax.plot(
        [tower_centre - platform_half_width, tower_centre + platform_half_width],
        [tower_height, tower_height],
        color="#D55E00", linestyle="-", linewidth=0.8, zorder=9,
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


def make_snapshots(complete_path: bool = False) -> None:
    series_2d = _read_csv(CASE / "openfoam/2d/outputs/openfoam_2d_series.csv")
    levels_2d = _read_csv(CASE / "openfoam/2d/outputs/openfoam_2d_levels.csv")
    frame_index_path = (
        COUPLED_FRAME_INDEX
        if complete_path and COUPLED_FRAME_INDEX.is_file()
        else CASE / "outputs/frames_index.json"
    )
    frames = json.loads(frame_index_path.read_text(encoding="utf-8"))
    coupled_fields = None
    if complete_path:
        if not COUPLED_FIELDS.is_file():
            raise FileNotFoundError(
                "Recompute the coupled Case-A two-fluid branch first: "
                f"{COUPLED_FIELDS}"
            )
        coupled_fields = np.load(COUPLED_FIELDS)

    if complete_path:
        # One panel pair per indispensable stage of the non-geysering pathway:
        # reflected horizontal-pocket propagation, tower entry/rise, and
        # post-breakthrough liquid-column recession.  All pairs use the same
        # physical time; no event alignment or time shift is applied.
        target_times = [1.50, 6.90, 10.10]
        frame_ids = [
            min(range(len(frames)), key=lambda i: abs(float(frames[i]["time"]) - target))
            for target in target_times
        ]
        for target, frame_id in zip(target_times, frame_ids):
            if abs(float(frames[frame_id]["time"]) - target) > 5.0e-7:
                raise FileNotFoundError(
                    f"No archived 1D frame at {target:.2f} s; "
                    f"nearest is {float(frames[frame_id]['time']):.8g} s"
                )
        vtu_paths = _select_vtu_paths(VTU_HTML_ROOT, target_times)
        stage_labels = [
            "Post-impact reflected bore",
            "Gas entry and interface rise",
            "Vented liquid-column recession",
        ]
        output_stem = "caseA_1d2d_snapshots_3frame"
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
        case_1d = _load_1d_case_definition(horizontal_model="tpa_wetdry")
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
    figure_height = 4.15 if complete_path else 5.45
    fig, axes = plt.subplots(nrows, 2, figsize=(7.2, figure_height), constrained_layout=False,
                             sharex=True, sharey=True)
    plt.subplots_adjust(left=0.075, right=0.99, bottom=0.09, top=0.88,
                        wspace=0.10, hspace=0.55 if complete_path else 0.56)
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
            raw_fraction = np.array(
                coupled_fields["horizontal_alpha_l"][state_idx],
                dtype=float,
                copy=True,
            )
            horizontal_dx = pipe_length / raw_fraction.size
            horizontal_x = (
                np.arange(raw_fraction.size, dtype=float) + 0.5
            ) * horizontal_dx
            # The sealed branch east of the side-T contains no resolved gas;
            # sub-full elastic areas there represent tensile pressure rather
            # than a free surface.
            raw_fraction[horizontal_x > tower_centre + 0.5 * horizontal_dx] = 1.0
            alt = np.where(
                raw_fraction >= 0.03,
                np.clip(raw_fraction, 0.0, 1.0),
                0.0,
            )
            vertical_z = np.asarray(coupled_fields["z"], dtype=float)
            vertical_dz = float(tower_height / vertical_z.size)
            area_fraction = np.clip(
                np.asarray(coupled_fields["alpha_l"][state_idx], dtype=float),
                0.0,
                1.0,
            )
            agr = np.clip(
                np.asarray(coupled_fields["alpha_g"][state_idx], dtype=float),
                0.0,
                1.0,
            )
        else:
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
        water_top = float(frame["wtop"])
        for z_i, liquid_fraction, gas_fraction in zip(
            vertical_z, area_fraction, agr
        ):
            if z_i > water_top:
                continue

            # Render the circular 1D tower as a centred gas core surrounded by
            # an annular liquid film.  Since gas_fraction is an area fraction,
            # the equivalent core diameter scales with sqrt(gas_fraction), not
            # gas_fraction itself.  The previous linear mapping made the gas
            # pocket look artificially thin relative to the 2D phase field.
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
        yint_m = 0.610 * _interp(levels_2d, "time_s", "Yint_star", time_s)
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
        common_times.append(0.5 * (time_1d + time_s))
        selected_manifest.append(
            {
                "stage": stage_labels[row],
                "one_d": {
                    "frame_index": int(frame_ids[row]),
                    "time_s": time_1d,
                    "Tstar": tstar_1d,
                    "Yfs_m": float(frame["wtop"]),
                    "Yint_m": yint_1d,
                    "source": frame["file"],
                },
                "two_d": {
                    "time_s": time_s,
                    "Tstar": tstar,
                    "Yfs_m": 0.610 * _interp(levels_2d, "time_s", "Yfs_star", time_s),
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
            row_label = f"Time = {common_time:.2f} s"
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
                "The 1D model and supporting 2D OpenFOAM calculation reproduce "
                "the same non-geysering pathway from reflected horizontal-pocket "
                "propagation through tower entry to vented column recession."
            ),
            "time_pairing": "same physical time; no time shift",
            "two_d_role": "supporting planar VOF calculation",
            "two_d_geometry": "vertical-plane rectangular slot; 24 cells across D",
            "one_d_horizontal_model": (
                "fully coupled Case-A network with the Case-B conservative "
                "TPA wet/dry horizontal algorithm; the wide-tower flow regime "
                "is selected from the physical diameter ratio Dt/D=0.607"
            ),
            "one_d_horizontal_source": str(COUPLED_FIELDS.relative_to(ROOT)).replace("\\", "/"),
            "one_d_tower_source": (
                "resolved one-dimensional two-fluid alpha_l(z,t) and "
                "alpha_g(z,t) fields with conservative liquid/gas exchange "
                "at the side-T"
            ),
            "selected_frames": selected_manifest,
            "outputs": [
                f"paper/figures/{output_stem}.png",
                f"paper/figures/{output_stem}.pdf",
            ],
            "manuscript_status": "active three-frame manuscript figure",
        }
        (CASE / "outputs/caseA_paper_figure_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


def make_curves(
    model_series: Path | None = None,
    output_stem: str = "caseA_experiment_1d2d_curves",
) -> None:
    pressure_exp = _read_csv(
        CASE / "data/digitized/fig5_caseA_Hstar_band.csv"
    )
    levels_exp = _read_csv(
        CASE / "data/digitized/fig7_caseA_markers_vector.csv"
    )
    model_1d = _read_csv(model_series or (CASE / "outputs/caseA_model_series.csv"))
    pressure_2d = _read_csv(CASE / "openfoam/2d/outputs/openfoam_2d_series.csv")
    levels_2d = _read_csv(CASE / "openfoam/2d/outputs/openfoam_2d_levels.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    plt.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.91, wspace=0.24)

    ax = axes[0]
    # The repetitions serve to define one representative experimental
    # trajectory.  Show their pointwise median as the experimental reference.
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
    ax.plot(one_d_pressure["Tstar"], pressure_1d_mean, color=ONE_D,
            lw=1.7)
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
            Line2D([0], [0], color=ONE_D, lw=1.7, label="Present model"),
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
    one_d_fs_mask = model_1d["Tstar"] <= obs_tmax
    one_d_int_mask = model_1d["Tstar"] <= obs_tmax
    two_d_fs_mask = levels_2d["Tstar"] <= obs_tmax
    two_d_int_mask = levels_2d["Tstar"] <= obs_tmax
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
        color=ONE_D, lw=1.7, ls="--", zorder=3,
    )
    ax.plot(
        levels_2d["Tstar"][two_d_fs_mask],
        levels_2d["Yfs_star"][two_d_fs_mask],
        color=TWO_D, lw=1.7, zorder=3,
    )
    ax.plot(
        levels_2d["Tstar"][two_d_int_mask],
        levels_2d["Yint_star"][two_d_int_mask],
        color=TWO_D, lw=1.7, ls="--", zorder=3,
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
            r"Experiment $Y_{\rm fs}^*$",
            r"Experiment $Y_{\rm int}^*$",
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
            Line2D([0], [0], color=ONE_D, lw=1.7, ls="--"),
            Line2D([0], [0], color=TWO_D, lw=1.7),
            Line2D([0], [0], color=TWO_D, lw=1.7, ls="--"),
        ],
        labels=[
            r"Present model $Y_{\rm fs}^*$",
            r"Present model $Y_{\rm int}^*$",
            r"2D OpenFOAM $Y_{\rm fs}^*$",
            r"2D OpenFOAM $Y_{\rm int}^*$",
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
        choices=("all", "snapshots", "three-frame", "curves"),
        default="all",
        help="Generate all figures or only the selected Case A artifact.",
    )
    parser.add_argument(
        "--model-series",
        type=Path,
        help="Optional candidate 1D series CSV; leaves the frozen main series untouched.",
    )
    parser.add_argument(
        "--output-stem",
        default="caseA_experiment_1d2d_curves",
        help="Output file stem under paper/figures.",
    )
    args = parser.parse_args()

    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    if args.only in ("all", "snapshots"):
        make_snapshots()
    if args.only == "three-frame":
        make_snapshots(complete_path=True)
    if args.only in ("all", "curves"):
        make_curves(args.model_series, args.output_stem)


if __name__ == "__main__":
    main()

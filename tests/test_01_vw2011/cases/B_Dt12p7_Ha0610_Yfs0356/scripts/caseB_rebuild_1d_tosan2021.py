"""Build Case B frames with a Tosan (2021)-based horizontal solver.

The requested change is deliberately isolated from the completed Case A and
from the previous Case B frame sets:

* the horizontal conduit is advanced by ``tosan2021_horizontal_shockfit``;
* the already validated vertical-tower branch is retained unchanged;
* all new images and manifests use a ``tosan2021`` suffix.

Tosan's Chapter 6 model assumes a pre-wetted free-surface reach.  The imported
horizontal module keeps the Chapter 6 moving-interface equations, but adds a
conservative zero-depth wet/dry stage for the initially empty upstream chamber
in the Vasconcelos-Wright experiment.  No artificial initial water film is
introduced.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
CASE_A_MODEL = (
    CASE_ROOT.parent
    / "A_Dt57p1_Ha0305_Yfs0356"
    / "model"
    / "vw2011_network_twofluid.py"
)
TOSAN_MODEL = CASE_ROOT / "model" / "tosan2021_horizontal_shockfit.py"
FRAME_ROOT = CASE_ROOT / "openfoam" / "2d" / "outputs_1d2d_compare"
FRAME_DIR = FRAME_ROOT / "frames_1d_caseB_tosan2021"
FRAME_META = FRAME_ROOT / "frames_1d_caseB_tosan2021_meta.json"
PAIR_INDEX = FRAME_ROOT / "frames_index_tosan2021.json"
SELECTED_TIMES = FRAME_ROOT / "selected_times.json"
SOURCE_PAIRS = FRAME_ROOT / "frames_index.json"
VERTICAL_META = FRAME_ROOT / "frames_1d_caseB_tpa_wetdry_meta.json"

P_ATM = 101325.0
RHO_W = 998.0
G = 9.81
# The 2D VOF panels show the alpha=0.1 contour as the first clearly visible
# water front.  Use the same area-fraction contour in the 1D visualization.
# The underlying state, liquid volume and gas-volume calculation are left
# untouched; only the opaque-blue rendering and reported display contour use
# this threshold.
VISIBLE_WATER_AREA_FRACTION = 0.10


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _nearest_index(values: list[float], target: float) -> int:
    return min(range(len(values)), key=lambda i: abs(float(values[i]) - target))


def _depth_fraction_from_area(alpha: np.ndarray) -> np.ndarray:
    """Invert the circular-segment area relation with a monotone lookup."""
    depth = np.linspace(0.0, 1.0, 4001)
    theta = 2.0 * np.arccos(np.clip(1.0 - 2.0 * depth, -1.0, 1.0))
    area_fraction = (theta - np.sin(theta)) / (2.0 * math.pi)
    return np.interp(np.clip(alpha, 0.0, 1.0), area_fraction, depth)


def _run_vertical_reference(target_times: list[float]):
    """Retain the completed vertical-tower branch while replacing the horizontal one.

    The horizontal change requested here must not silently alter the already
    reviewed tower/jet sequence.  Prefer its saved scalar frame record and
    reconstruct the same Taylor-core display.  The legacy solve is retained
    only as a fallback when that cache is absent.
    """
    if VERTICAL_META.exists():
        saved = json.loads(VERTICAL_META.read_text(encoding="utf-8"))
        saved_times = [float(row["time"]) for row in saved]
        dz = 0.01
        z = np.arange(0.5 * dz, 0.610, dz)
        aligned = []
        for target in target_times:
            row = saved[_nearest_index(saved_times, target)]
            wtop = float(row["wtop"])
            itop = min(float(row["itop"]), wtop)
            occupied = z < wtop
            gas_core = occupied & (z < itop) & (itop > 0.0)
            alpha_gas = np.where(gas_core, 0.88, 0.0)
            alpha_liquid = np.where(occupied, 1.0 - alpha_gas, 0.0)
            aligned.append(
                {
                    "time": float(target),
                    "source_time": float(row["time"]),
                    "wtop": wtop,
                    "itop": itop,
                    "jet_height": float(row["jetHeight"]),
                    "alpha_liquid": alpha_liquid,
                    "alpha_gas": alpha_gas,
                    "z": z,
                    "dz": dz,
                }
            )
        return None, saved, aligned

    legacy = _load_module("case_b_existing_vertical", CASE_A_MODEL)
    case = legacy.NetworkCase(
        Dr=0.0127,
        air_head=0.610,
        init_water_level=0.356,
        t_end=max(target_times),
        horizontal_model="tpa_wetdry",
    )
    rec = legacy.run_network(case, verbose=False)

    frames_t = [float(v) for v in rec["frames_t"]]
    aligned = []
    for target in target_times:
        k = _nearest_index(frames_t, target)
        aligned.append(
            {
                "time": target,
                "source_time": frames_t[k],
                "wtop": float(rec["wtop"][k]),
                "itop": float(rec["frames_itop"][k]),
                "jet_height": float(rec.get("jet_height", [0.0] * len(frames_t))[k]),
                "alpha_liquid": np.asarray(rec["frames_alr"][k], dtype=float),
                "alpha_gas": np.asarray(rec["frames_agr"][k], dtype=float),
                "z": np.asarray(rec["zr"], dtype=float),
                "dz": float(rec["dz"]),
            }
        )
    return case, rec, aligned


def _tower_pressure_hook(vertical_frames: list[dict]):
    """Hydrostatic pressure seen by the connected pocket after tower entry.

    Before the horizontal interface reaches the T, the pocket remains sealed and
    the Tosan polytropic pressure is returned unchanged.  After entry, the
    resolved tower state supplies the gas-side pressure; once the gas core catches
    the free surface the pocket is open to atmosphere.
    """

    times = np.asarray([row["time"] for row in vertical_frames], dtype=float)
    yfs = np.asarray([row["wtop"] for row in vertical_frames], dtype=float)
    yint = np.asarray([row["itop"] for row in vertical_frames], dtype=float)

    def hook(time: float, interface_x: float, closed_pressure_abs: float) -> float:
        if interface_x < 3.516:
            return float(closed_pressure_abs)
        surface = float(np.interp(time, times, yfs))
        interface = float(np.interp(time, times, yint))
        if interface >= surface - 0.5 * 0.01:
            return P_ATM
        hydrostatic = P_ATM + RHO_W * G * max(surface - interface, 0.0)
        # After the gas nose enters the open tower, its pressure is set by the
        # resolved water column above that nose.  The sealed horizontal-pocket
        # EOS no longer supplies the network boundary pressure.
        return float(max(hydrostatic, P_ATM))

    return hook


def _run_horizontal(target_times: list[float], vertical_frames: list[dict]):
    tosan = _load_module("tosan2021_case_b_horizontal", TOSAN_MODEL)
    config = tosan.HorizontalConfig(
        length=4.006,
        diameter=0.094,
        valve_x=0.546,
        vent_x=3.516,
        dx=0.02,
        wave_speed=100.0,
        gamma=1.4,
        initial_air_head=0.610,
        initial_water_head=0.356,
        wetting_front_report_fraction=VISIBLE_WATER_AREA_FRACTION,
    )
    solver = tosan.Tosan2021HorizontalShockFit(
        config,
        vent_pressure_hook=_tower_pressure_hook(vertical_frames),
    )
    state = solver.case_b_initial_state()
    raw = solver.run(state, t_end=max(target_times), output_dt=0.01)
    snapshots = [solver.snapshot(item) if not isinstance(item, dict) else item for item in raw]
    times = [float(item["time"]) for item in snapshots]
    aligned = [snapshots[_nearest_index(times, target)] for target in target_times]
    return tosan, config, aligned


def _draw_tower(
    ax,
    tower,
    x_tower: float,
    pipe_crown: float,
    tower_diameter: float,
    tower_height: float,
):
    from matplotlib.patches import Rectangle

    water = "#2f7ff7"
    air = "#f1f3f6"
    x0 = x_tower - 0.5 * tower_diameter
    ax.add_patch(
        Rectangle(
            (x0, pipe_crown),
            tower_diameter,
            tower_height,
            facecolor=air,
            edgecolor="none",
        )
    )
    # The experimental ventilation tower ends at the 0.610-m rim and is open
    # to atmosphere.  Draw only its two physical sidewalls: there is no lid,
    # bottom partition, or artificial observation tube above the rim.
    ax.plot(
        [x0, x0],
        [pipe_crown, pipe_crown + tower_height],
        color="#333333",
        linewidth=0.8,
        zorder=3,
    )
    ax.plot(
        [x0 + tower_diameter, x0 + tower_diameter],
        [pipe_crown, pipe_crown + tower_height],
        color="#333333",
        linewidth=0.8,
        zorder=3,
    )
    water_top = min(float(tower["wtop"]), tower_height)
    z = tower["z"]
    dz = float(tower["dz"])
    for zi, alpha_l, alpha_g in zip(z, tower["alpha_liquid"], tower["alpha_gas"]):
        z0 = float(zi - 0.5 * dz)
        if z0 >= water_top:
            continue
        cell_h = min(dz, water_top - z0)
        if cell_h <= 0.0:
            continue
        gas_width = math.sqrt(float(np.clip(alpha_g, 0.0, 1.0))) * tower_diameter
        film_width = max(0.5 * (tower_diameter - gas_width), 0.0)
        if float(alpha_l) > 1.0e-4 and film_width > 0.0:
            ax.add_patch(
                Rectangle((x0, pipe_crown + z0), film_width, cell_h, facecolor=water, edgecolor="none")
            )
            ax.add_patch(
                Rectangle(
                    (x0 + tower_diameter - film_width, pipe_crown + z0),
                    film_width,
                    cell_h,
                    facecolor=water,
                    edgecolor="none",
                )
            )
    jet_top = float(tower["jet_height"])
    if jet_top > tower_height:
        jet_width = 0.55 * tower_diameter
        ax.add_patch(
            Rectangle(
                (
                    x_tower - 0.5 * jet_width,
                    pipe_crown + tower_height,
                ),
                jet_width,
                jet_top - tower_height,
                facecolor=water,
                edgecolor="none",
                zorder=3,
            )
        )


def _render_frames(target_times, horizontal, vertical):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    metadata = []
    diameter = 0.094
    pipe_bottom = -0.5 * diameter
    pipe_crown = 0.5 * diameter
    water = "#2f7ff7"
    air = "#f1f3f6"

    for index, (target, hor, tower) in enumerate(zip(target_times, horizontal, vertical)):
        x = np.asarray(hor["x"], dtype=float)
        alpha = np.asarray(hor["area_fraction"], dtype=float)
        alpha_visible = np.where(
            alpha >= VISIBLE_WATER_AREA_FRACTION,
            alpha,
            0.0,
        )
        depths = diameter * _depth_fraction_from_area(alpha_visible)
        dx = float(np.median(np.diff(x))) if len(x) > 1 else 0.01

        fig, ax = plt.subplots(figsize=(12.4, 3.2))
        ax.add_patch(
            Rectangle(
                (0.0, pipe_bottom),
                4.006,
                diameter,
                facecolor=air,
                edgecolor="#333333",
                linewidth=0.8,
            )
        )
        ax.fill_between(
            x,
            pipe_bottom,
            pipe_bottom + depths,
            step="mid",
            color=water,
            linewidth=0.0,
        )

        _draw_tower(
            ax,
            tower,
            x_tower=3.516,
            pipe_crown=pipe_crown,
            tower_diameter=0.0127,
            tower_height=0.610,
        )
        ax.axvline(0.546, ymin=0.035, ymax=0.14, color="#202020", linestyle=":", linewidth=0.9)
        tower_top = pipe_crown + 0.610
        # Shared, symmetric rim marker used in both comparison panels.
        tower_left = 3.470
        tower_right = 3.562
        ax.plot(
            [tower_left, tower_right],
            [tower_top, tower_top],
            color="#ef4444",
            linestyle="--",
            linewidth=1.0,
            zorder=4,
        )
        ax.text(
            0.015,
            0.95,
            f"Time = {target:.2f} s",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            family="Times New Roman",
        )
        ax.text(
            0.015,
            0.86,
            (
                "Tosan (2021) flxT1 + conservative wet/dry  |  "
                "tower display = Dt  |  "
                f"x_interface = {float(hor['interface_x']):.3f} m  |  "
                f"H_air = {float(hor['air_pressure_head_gauge']):.3f} m"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="Times New Roman",
        )
        ax.text(0.546, pipe_crown + 0.025, "valve", ha="center", va="bottom", fontsize=8)
        ax.set_xlim(-0.04, 4.046)
        ax.set_ylim(pipe_bottom - 0.035, max(tower_top + 0.32, 1.0))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("horizontal distance [m]", family="Times New Roman")
        ax.set_ylabel("vertical coordinate [m]", family="Times New Roman")
        ax.legend(
            handles=[
                Patch(facecolor=water, label="water"),
                Patch(facecolor=air, edgecolor="#555555", label="air"),
            ],
            loc="upper right",
            frameon=False,
            prop={"family": "Times New Roman", "size": 9},
        )
        fig.subplots_adjust(
            left=0.075,
            right=0.985,
            bottom=0.18,
            top=0.94,
        )

        filename = f"frame_{index:04d}.png"
        fig.savefig(FRAME_DIR / filename, dpi=140)
        plt.close(fig)
        metadata.append(
            {
                "index": index,
                "file": f"{FRAME_DIR.name}/{filename}",
                "time": float(target),
                "source_time": float(hor["time"]),
                "interface_x": float(hor["interface_x"]),
                "interface_speed": float(hor["interface_speed"]),
                "wetting_front_x": float(hor["wetting_front_x"]),
                "air_pressure_abs": float(hor["air_pressure_abs"]),
                "air_pressure_head_gauge": float(hor["air_pressure_head_gauge"]),
                "air_volume": float(hor["air_volume"]),
                "air_mass": float(hor["air_mass"]),
                "vented": bool(hor["vented"]),
                "mode": str(hor["mode"]),
                "wtop": float(tower["wtop"]),
                "itop": float(tower["itop"]),
                "jetHeight": float(tower["jet_height"]),
                "dx": dx,
            }
        )
    return metadata


def _pair_with_openfoam(metadata):
    original = json.loads(SOURCE_PAIRS.read_text(encoding="utf-8"))
    two_d_by_time = {round(float(item["time"]), 8): item for item in original}
    pairs = []
    for item in metadata:
        source = two_d_by_time.get(round(float(item["time"]), 8))
        if source is None:
            source = min(original, key=lambda row: abs(float(row["time"]) - item["time"]))
        pairs.append(
            {
                "time": float(item["time"]),
                "file1d": item["file"],
                "file2d": source["file2d"],
                "wtop1d": item["wtop"],
                "itop1d": item["itop"],
                "jetHeight1d": item["jetHeight"],
                "interfaceX1d": item["interface_x"],
                "wettingFrontX1d": item["wetting_front_x"],
                "airHead1d": item["air_pressure_head_gauge"],
                "vented1d": item["vented"],
                "mode1d": item["mode"],
                "dt_match": abs(float(item["source_time"]) - float(item["time"])),
            }
        )
    return pairs


def main() -> None:
    target_times = [float(v) for v in json.loads(SELECTED_TIMES.read_text(encoding="utf-8"))]
    _, _, vertical = _run_vertical_reference(target_times)
    _, _, horizontal = _run_horizontal(target_times, vertical)
    metadata = _render_frames(target_times, horizontal, vertical)
    pairs = _pair_with_openfoam(metadata)
    FRAME_META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    PAIR_INDEX.write_text(json.dumps(pairs, indent=2), encoding="utf-8")

    # Hard acceptance checks for the failure modes identified by the user.
    second = metadata[1]
    if second["wetting_front_x"] <= 0.30:
        raise RuntimeError(
            "The second frame wets the closed left wall; the dry-bed front is still nonphysical."
        )
    sealed = [row for row in metadata if not row["vented"]]
    invariant = [
        row["air_pressure_abs"] * row["air_volume"] ** 1.4
        for row in sealed
        if row["air_volume"] > 0.0
    ]
    if invariant:
        drift = (max(invariant) - min(invariant)) / max(abs(invariant[0]), 1.0e-30)
        if drift > 5.0e-3:
            raise RuntimeError(f"Pre-vent polytropic invariant drift is too large: {drift:.3e}")

    print(f"Wrote {len(metadata)} Tosan-based 1D frames to {FRAME_DIR}")
    print(f"Wrote synchronized manifest to {PAIR_INDEX}")


if __name__ == "__main__":
    main()

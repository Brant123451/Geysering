"""Re-digitize the three V&W (2011) Case-B repetitions by run identity.

This is a guided vector trace of the native 300-dpi images embedded in the
paper PDF.  It replaces the earlier dark-pixel envelope, which could not
preserve the published line/marker semantics.

Published semantics used here
-----------------------------
Fig. 6 pressure: run 1 = solid; run 2 = dense short-stroke trace;
run 3 = dotted trace.

Fig. 8 levels: Yfs run 1/2/3 = triangle/x/filled circle;
Yint run 1/2/3 = open diamond/square/circle.

The pressure traces are guided by manually inspected control points and then
snapped locally to dark ridges in the native bitmap.  The Fig. 8 coordinates
are marker-centre readings.  Audit overlays are generated so that every
digitized series can be checked directly against the source panel.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter


CASE_ROOT = Path(__file__).resolve().parents[1]
SCANS = CASE_ROOT / "reference" / "paper_scans"
DIG = CASE_ROOT / "data" / "digitized"
OUT = CASE_ROOT / "outputs" / "redigitization_audit"

FIG6 = SCANS / "raw_p5_x101_2000x1457.png"
FIG8 = SCANS / "raw_p7_x121_2145x1534.png"

# Native-image plot boxes from the 3x3 panel detector.  Coordinates include
# the border pixels; data are mapped linearly to the inner raster.
FIG6_BOX = (725, 1335, 481, 884)  # x0, x1, y0, y1; T*=0..5, H*=0..1.5
FIG8_BOX = (800, 1446, 506, 912)  # x0, x1, y0, y1; T*=3..5, Y*=0..1

PRESSURE_CSV = DIG / "fig6_caseB_pressure_runs_v2.csv"
PRESSURE_SUMMARY_CSV = DIG / "fig6_caseB_pressure_mean_range_v3.csv"
LEVEL_CSV = DIG / "fig8_caseB_levels_runs_v2.csv"

RUN_COLORS = {1: "#d62728", 2: "#1f77b4", 3: "#2ca02c"}


# Control points read after magnifying the native Fig. 6 centre panel.  The
# two patterned repetitions overlap closely for most of the plateau, so their
# traces legitimately coincide where the raster does not resolve separation.
PRESSURE_ANCHORS = {
    1: np.array(
        [
            [0.00, 0.40], [0.04, 0.40], [0.055, 0.90], [0.10, 0.76],
            [0.20, 0.70], [0.35, 0.68], [0.50, 0.74], [0.65, 0.80],
            [0.80, 0.84], [1.00, 0.78], [1.20, 0.74], [1.40, 0.78],
            [1.60, 0.77], [1.80, 0.72], [2.00, 0.75], [2.20, 0.77],
            [2.40, 0.74], [2.60, 0.78], [2.80, 0.81], [3.00, 0.80],
            [3.20, 0.80], [3.40, 0.80], [3.55, 0.79], [3.70, 0.76],
            [3.85, 0.72], [3.95, 0.69], [4.02, 0.65], [4.08, 0.59],
            [4.13, 0.52], [4.16, 0.43], [4.17, 0.05], [4.20, 0.00],
            [5.00, 0.00],
        ],
        dtype=float,
    ),
    2: np.array(
        [
            [0.00, 0.82], [0.08, 0.76], [0.20, 0.69], [0.35, 0.65],
            [0.50, 0.70], [0.70, 0.80], [0.85, 0.79], [1.05, 0.72],
            [1.20, 0.68], [1.40, 0.74], [1.60, 0.70], [1.80, 0.68],
            [2.00, 0.75], [2.20, 0.74], [2.40, 0.70], [2.60, 0.76],
            [2.80, 0.79], [3.00, 0.77], [3.20, 0.74], [3.40, 0.76],
            [3.55, 0.74], [3.70, 0.70], [3.82, 0.66], [3.92, 0.58],
            [3.98, 0.48], [4.02, 0.10], [4.03, 0.00], [5.00, 0.00],
        ],
        dtype=float,
    ),
    3: np.array(
        [
            [0.00, 0.79], [0.08, 0.75], [0.20, 0.71], [0.35, 0.67],
            [0.50, 0.71], [0.70, 0.79], [0.85, 0.78], [1.05, 0.73],
            [1.20, 0.70], [1.40, 0.75], [1.60, 0.71], [1.80, 0.70],
            [2.00, 0.75], [2.20, 0.75], [2.40, 0.72], [2.60, 0.77],
            [2.80, 0.78], [3.00, 0.76], [3.20, 0.75], [3.40, 0.77],
            [3.55, 0.75], [3.70, 0.71], [3.82, 0.67], [3.92, 0.60],
            [4.00, 0.52], [4.04, 0.10], [4.05, 0.00], [5.00, 0.00],
        ],
        dtype=float,
    ),
}


# Fig. 8 marker-centre readings.  Baseline markers printed at Y*=0 after a
# quantity left the camera window are excluded from the rising trajectories;
# the paper's actual run-specific marker shapes are retained in the CSV.
LEVEL_POINTS = {
    ("fs", 1): [
        (3.359, 0.840), (3.630, 0.812), (3.670, 0.820), (3.709, 0.832),
        (3.748, 0.855), (3.787, 0.875), (3.827, 0.895), (3.864, 0.920),
        (3.907, 0.951), (3.941, 0.982), (3.981, 1.000),
    ],
    ("fs", 2): [
        (3.022, 0.799), (3.261, 0.839), (3.585, 0.803), (3.635, 0.816),
        (3.675, 0.824), (3.715, 0.836), (3.755, 0.855), (3.795, 0.880),
        (3.835, 0.910), (3.875, 0.946), (3.915, 0.982), (3.940, 1.000),
    ],
    ("fs", 3): [
        (3.279, 0.833), (3.588, 0.803), (3.641, 0.818), (3.669, 0.820),
        (3.706, 0.828), (3.743, 0.842), (3.783, 0.877), (3.820, 0.924),
        (3.861, 0.946), (3.901, 0.975), (3.910, 1.000),
    ],
    ("int", 1): [
        (3.742, 0.064), (3.783, 0.093), (3.820, 0.126), (3.860, 0.162),
        (3.900, 0.213), (3.935, 0.262), (3.976, 0.322), (4.016, 0.394),
        (4.053, 0.469), (4.094, 0.559), (4.133, 0.687), (4.169, 0.892),
    ],
    ("int", 2): [
        (3.648, 0.060), (3.701, 0.092), (3.742, 0.126), (3.788, 0.164),
        (3.817, 0.228), (3.857, 0.288), (3.900, 0.362), (3.933, 0.436),
        (3.974, 0.530), (4.013, 0.634), (4.050, 0.788), (4.093, 0.958),
    ],
    ("int", 3): [
        (3.678, 0.060), (3.718, 0.083), (3.758, 0.113), (3.798, 0.155),
        (3.817, 0.208), (3.857, 0.266), (3.900, 0.333), (3.933, 0.406),
        (3.975, 0.490), (4.015, 0.581), (4.050, 0.722), (4.091, 0.868),
    ],
}


# Isolated upper-left free-surface markers are valid pre-rise observations,
# not the beginning of the monotonic rise toward the tower rim.  Keep them in
# the reproduction, but tag them separately so event-time calculations start
# from the compact trajectories near T*=3.59--3.63.
PRELEVEL_COUNTS = {("fs", 1): 1, ("fs", 2): 2, ("fs", 3): 1}


# Zero-line markers visible in the published panel.  These are retained for
# faithful graphical reproduction but explicitly tagged as sentinels so they
# cannot be mistaken for a falling physical trajectory in later metrics.
BASELINE_POINTS = {
    ("fs", 1): [(3.930, 0.0), (4.020, 0.0), (4.100, 0.0), (4.160, 0.0)],
    ("fs", 2): [(3.900, 0.0), (3.940, 0.0), (3.980, 0.0), (4.020, 0.0), (4.060, 0.0), (4.100, 0.0)],
    ("fs", 3): [
        (3.820, 0.0), (3.850, 0.0), (3.880, 0.0), (3.910, 0.0),
        (3.940, 0.0), (3.970, 0.0), (4.000, 0.0), (4.030, 0.0),
        (4.060, 0.0), (4.090, 0.0), (4.120, 0.0),
    ],
    ("int", 1): [(3.000, 0.0), (3.350, 0.0), (3.610, 0.0), (4.200, 0.0)],
    ("int", 2): [(3.010, 0.0), (3.220, 0.0), (3.600, 0.0), (4.130, 0.0)],
    ("int", 3): [(3.230, 0.0), (3.560, 0.0), (4.080, 0.0)],
}


def _gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=float)


def _snap_pressure(gray: np.ndarray, run: int) -> tuple[np.ndarray, np.ndarray]:
    x0, x1, y0, y1 = FIG6_BOX
    panel = gray[y0 : y1 + 1, x0 : x1 + 1]
    height, width = panel.shape
    anchors = PRESSURE_ANCHORS[run]
    t = np.linspace(0.0, 5.0, width)
    guide = np.interp(t, anchors[:, 0], anchors[:, 1])

    # Local ridge score.  Grid lines and the two in-panel condition boxes are
    # masked before tracing.  Their coordinates are known from the panel scan.
    ink = gaussian_filter(255.0 - panel, sigma=(0.75, 0.65))
    ink[:87, :160] = 0.0
    for tx in (0, 1, 2, 3, 4, 5):
        px = int(round((tx / 5.0) * (width - 1)))
        ink[:, max(0, px - 2) : min(width, px + 3)] = 0.0
    for hh in (0.0, 0.5, 1.0, 1.5):
        py = int(round((1.5 - hh) / 1.5 * (height - 1)))
        ink[max(0, py - 2) : min(height, py + 3), :] = 0.0

    traced = guide.copy()
    previous_row = None
    for ix, (tt, expected_h) in enumerate(zip(t, guide)):
        # The steep/vertical release and the final zero line are encoded by
        # inspected control points, rather than by a single-valued ridge.
        if tt < 0.22 or tt > anchors[-3, 0] or expected_h < 0.12:
            previous_row = None
            continue
        expected_row = (1.5 - expected_h) / 1.5 * (height - 1)
        radius = 5 if run == 1 else 6
        lo = max(0, int(round(expected_row)) - radius)
        hi = min(height - 1, int(round(expected_row)) + radius)
        candidates = np.arange(lo, hi + 1)
        local = ink[candidates, ix]
        penalty = 8.0 * ((candidates - expected_row) / max(radius, 1)) ** 2
        if previous_row is not None:
            penalty += 3.0 * ((candidates - previous_row) / max(radius, 1)) ** 2
        best = int(candidates[np.argmax(local - penalty)])
        if local.max() > 18.0:
            traced[ix] = 1.5 - 1.5 * best / (height - 1)
            previous_row = best

    # Remove one-pixel JPEG jitter while retaining the measured undulation.
    traced = median_filter(traced, size=3, mode="nearest")
    traced = np.clip(traced, 0.0, 1.5)
    return t, traced


def _write_csvs(pressure: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
    DIG.mkdir(parents=True, exist_ok=True)
    with PRESSURE_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "Tstar", "Hstar", "source_style", "method"])
        styles = {1: "solid", 2: "dense_short_stroke", 3: "dotted"}
        for run in (1, 2, 3):
            t, h = pressure[run]
            for tt, hh in zip(t, h):
                writer.writerow([run, f"{tt:.6f}", f"{hh:.6f}", styles[run], "guided_native_raster_trace"])

    # All three traces share the native raster's time grid.  The main-paper
    # curve uses their pointwise arithmetic mean; the min--max span is kept
    # as a descriptive repeatability band (n=3), not a confidence interval.
    t = pressure[1][0]
    stack = np.vstack([pressure[run][1] for run in (1, 2, 3)])
    with PRESSURE_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Tstar", "Hstar_mean", "Hstar_min", "Hstar_max", "n"])
        for tt, mean, low, high in zip(
            t, np.mean(stack, axis=0), np.min(stack, axis=0), np.max(stack, axis=0)
        ):
            writer.writerow([f"{tt:.6f}", f"{mean:.6f}", f"{low:.6f}", f"{high:.6f}", 3])

    with LEVEL_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["kind", "run", "Tstar", "Ystar", "source_marker", "role", "method"])
        markers = {
            ("fs", 1): "triangle", ("fs", 2): "x", ("fs", 3): "filled_circle",
            ("int", 1): "open_diamond", ("int", 2): "open_square", ("int", 3): "open_circle",
        }
        for key in (("fs", 1), ("fs", 2), ("fs", 3), ("int", 1), ("int", 2), ("int", 3)):
            for index, (tt, yy) in enumerate(LEVEL_POINTS[key]):
                role = "initial_level_reading" if index < PRELEVEL_COUNTS.get(key, 0) else "rising_track"
                writer.writerow([key[0], key[1], f"{tt:.6f}", f"{yy:.6f}", markers[key], role, "native_marker_centre"])
            for tt, yy in BASELINE_POINTS[key]:
                writer.writerow([key[0], key[1], f"{tt:.6f}", f"{yy:.6f}", markers[key], "baseline_sentinel", "native_marker_centre"])


def _audit_pressure(gray: np.ndarray, pressure: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
    x0, x1, y0, y1 = FIG6_BOX
    panel = gray[y0 : y1 + 1, x0 : x1 + 1]
    fig, ax = plt.subplots(figsize=(12.0, 6.1))
    ax.imshow(panel, cmap="gray", vmin=0, vmax=255, extent=(0, 5, 0, 1.5), origin="upper", aspect="auto")
    for run, linestyle in ((1, "-"), (2, "--"), (3, ":")):
        t, h = pressure[run]
        ax.plot(t, h, color=RUN_COLORS[run], lw=1.25, ls=linestyle, label=f"digitized run {run}")
    ax.set(xlim=(0, 5), ylim=(0, 1.5), xlabel=r"$T^*_{rel}$", ylabel=r"$H^*$")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.set_title("Fig. 6 centre panel: run-specific trace audit")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_caseB_pressure_runs_overlay_audit.png", dpi=260)
    plt.close(fig)


def _audit_levels(gray: np.ndarray) -> None:
    x0, x1, y0, y1 = FIG8_BOX
    panel = gray[y0 : y1 + 1, x0 : x1 + 1]
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.imshow(panel, cmap="gray", vmin=0, vmax=255, extent=(3, 5, 0, 1), origin="upper", aspect="auto")
    marker_style = {
        ("fs", 1): ("^", True), ("fs", 2): ("x", True), ("fs", 3): ("o", True),
        ("int", 1): ("D", False), ("int", 2): ("s", False), ("int", 3): ("o", False),
    }
    for kind, run in marker_style:
        pts = np.asarray(LEVEL_POINTS[(kind, run)], dtype=float)
        baseline = np.asarray(BASELINE_POINTS[(kind, run)], dtype=float)
        marker, filled = marker_style[(kind, run)]
        ax.plot(
            pts[:, 0], pts[:, 1], ls="none", marker=marker, ms=6.5,
            mfc=(RUN_COLORS[run] if filled else "none"), mec=RUN_COLORS[run], mew=1.25,
            label=f"{kind} run {run}",
        )
        ax.plot(
            baseline[:, 0], baseline[:, 1], ls="none", marker=marker, ms=4.5,
            mfc=(RUN_COLORS[run] if filled else "none"), mec=RUN_COLORS[run], mew=1.0,
        )
    ax.set(xlim=(3, 5), ylim=(0, 1), xlabel=r"$T^*_{rel}$", ylabel=r"$Y^*$")
    ax.legend(ncol=2, loc="lower right", frameon=True, fontsize=8.5)
    ax.set_title("Fig. 8 centre panel: six-series marker audit")
    fig.tight_layout()
    fig.savefig(OUT / "fig8_caseB_level_runs_overlay_audit.png", dpi=260)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gray6 = _gray(FIG6)
    gray8 = _gray(FIG8)
    pressure = {run: _snap_pressure(gray6, run) for run in (1, 2, 3)}
    _write_csvs(pressure)
    _audit_pressure(gray6, pressure)
    _audit_levels(gray8)

    manifest = {
        "case": "VW2011 Test B: Ha0=0.610 m, WLinit=0.356 m, Dt*=0.135",
        "source": "native 300-dpi images embedded in vasconcelos2011.pdf pages 6 and 8",
        "supersedes": ["fig6_caseB_Hstar_band.csv", "fig8_caseB_levels.csv"],
        "pressure_semantics": {"1": "solid", "2": "dense short-stroke", "3": "dotted"},
        "level_semantics": {
            "Yfs": {"1": "triangle", "2": "x", "3": "filled circle"},
            "Yint": {"1": "open diamond", "2": "open square", "3": "open circle"},
        },
        "pressure_csv": str(PRESSURE_CSV.relative_to(CASE_ROOT)),
        "pressure_summary_csv": str(PRESSURE_SUMMARY_CSV.relative_to(CASE_ROOT)),
        "levels_csv": str(LEVEL_CSV.relative_to(CASE_ROOT)),
        "audit_outputs": [
            "outputs/redigitization_audit/fig6_caseB_pressure_runs_overlay_audit.png",
            "outputs/redigitization_audit/fig8_caseB_level_runs_overlay_audit.png",
        ],
        "limitations": [
            "Initial pressure traces overlap densely for T*<0.25.",
            "The low initial value of pressure run 1 is visible in the published source and is retained in the run-level CSV; it contributes to the n=3 mean and min--max band.",
            "Pressure runs 2 and 3 are locally coincident where the raster cannot resolve separation.",
            "Isolated upper-left Yfs markers are retained with role=initial_level_reading and are not used as rise-onset events.",
            "Baseline Y*=0 markers are retained with role=baseline_sentinel and are excluded from rising-track metrics.",
        ],
    }
    (OUT / "caseB_redigitization_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

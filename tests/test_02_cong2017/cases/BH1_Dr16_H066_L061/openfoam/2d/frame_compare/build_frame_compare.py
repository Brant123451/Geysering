#!/usr/bin/env python3
"""Build a gated offline H1 1D--OpenFOAM 2D frame viewer.

Run this script under WSL after the 13 s OpenFOAM calculation.  It refuses to
publish partial/smoke data and checks the paper geometry before rendering.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OF2D = HERE.parent
CASE = OF2D.parents[1]
RESULTS = OF2D / "results"
RUN = Path(os.environ.get("BH1_2D_RUN", "/tmp/bh1-2d-study/paper_tau0p2_areaeq"))
HTML = HERE / "bh1_1d2d_frame_compare_13s.html"
ONE_FRAMES = HERE / "one_d_frames"
TWO_FRAMES = HERE / "two_d_frames"
MODEL_DIR = CASE / "model"
ONE_OUTPUT = CASE / "outputs" / "paper_layout_1d"

T_END = 13.0
FRAME_DT = 0.10
FRAME_TIMES = np.round(np.arange(0.0, T_END + 0.5 * FRAME_DT, FRAME_DT), 2)

D = 0.05
DR = 0.016
W2D = DR * DR / D
H0 = 0.66
RIM = 1.8
X_TEE = 3.47
X_VALVE = 5.98
X_END = 6.59
Z_CROWN = 0.025
Z_RIM = 1.825
Z_TOP = 3.025


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def completion_gate() -> tuple[dict, dict, dict]:
    metrics_path = RESULTS / "openfoam_2d_metrics.json"
    audit_path = RESULTS / "run_record" / "paper_audit.json"
    log_path = RESULTS / "run_record" / "log.solve"
    one_metrics_path = ONE_OUTPUT / "caseA_comparison_metrics.json"
    missing = [path for path in (metrics_path, audit_path, log_path, one_metrics_path) if not path.exists()]
    if missing:
        raise RuntimeError("Formal comparison inputs are missing: " + ", ".join(map(str, missing)))

    metrics = load_json(metrics_path)
    audit = load_json(audit_path)
    one_metrics = load_json(one_metrics_path)
    status = metrics.get("status", {})
    if status.get("fatal_error") or not status.get("ended_normally"):
        raise RuntimeError(f"OpenFOAM run is not normally complete: {status}")
    if float(status.get("last_log_time_s") or -1) < 12.99:
        raise RuntimeError(f"OpenFOAM run stopped before 13 s: {status}")
    log_text = "\n".join(
        path.read_text(errors="ignore")
        for path in sorted((RESULTS / "run_record").glob("log.solve*"))
    )
    if "FOAM FATAL" in log_text or not re.search(r"^End\s*$", log_text, re.M):
        raise RuntimeError("Final solver log lacks a clean End marker")

    cfg = metrics["paper_contract"]
    geom = cfg["physical_geometry_m"]
    initial = cfg["initial_conditions"]
    expected = {
        "pipe_length": X_END,
        "tee_axis_x": X_TEE,
        "release_valve_x": X_VALVE,
        "closed_cap_x": X_END,
        "riser_inner_diameter": DR,
        "riser_height_above_pipe_crown": RIM,
    }
    for key, value in expected.items():
        if not close(geom[key], value):
            raise RuntimeError(f"Paper geometry mismatch: {key}={geom[key]} expected {value}")
    if not close(initial["H0_m_above_pipe_invert"], H0):
        raise RuntimeError("Initial H0 is not the B-H1 paper value")
    if not close(initial["pocket_length_m"], 0.61):
        raise RuntimeError("Initial pocket length is not the B-H1 paper value")
    if not close(initial["pocket_pressure_Pa_abs"], 101325.0):
        raise RuntimeError("Initial pocket is not atmospheric")
    if any(abs(float(value)) > 1e-12 for value in initial["velocity_m_s"]):
        raise RuntimeError("Initial velocity is not quiescent")

    one_case = one_metrics["case"]
    if not close(one_case["tee_x"], X_TEE) or not close(one_case["valve_x"], X_VALVE):
        raise RuntimeError("Selected 1D output is not the Fig. 1(b) geometry-matched run")
    if one_metrics.get("variant") != "paper_Fig1b_axial_layout":
        raise RuntimeError("Selected 1D output lacks the paper-layout variant marker")
    if audit.get("status") != "PASS" or not all(audit.get("checks", {}).values()):
        raise RuntimeError("Paper-contract audit did not pass")
    if not (RUN / "13").exists():
        raise RuntimeError(f"The 13 s OpenFOAM field directory is missing under {RUN}")
    return metrics, audit, one_metrics


def read_series(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names or ()}


def rel(path: Path) -> str:
    return path.relative_to(HERE).as_posix()


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def setup_plotting() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})


def run_one_d_model() -> tuple[object, dict]:
    sys.path.insert(0, str(MODEL_DIR))
    from cong2017_network_twofluid import NetworkCase, run_network

    case = NetworkCase(
        D=D,
        Dr=DR,
        riser_height=RIM,
        L_up=X_TEE,
        L_mid=X_VALVE - X_TEE,
        L_down=X_END - X_VALVE,
        x_riser_at=X_TEE,
        pocket_downstream=True,
        reservoir_head=H0,
        air_head=0.0,
        init_water_level=H0,
        Hop_cap=10.0,
        x_transducer_at=6.44,
        t_end=T_END,
    )
    return case, run_network(case, verbose=False)


def render_one_d_frames(reuse: bool) -> list[dict]:
    setup_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    series = read_series(ONE_OUTPUT / "caseA_model_series.csv")
    case, rec = run_one_d_model()
    ONE_FRAMES.mkdir(parents=True, exist_ok=True)
    frames_t = np.asarray(rec["frames_t"], dtype=float)
    xt = np.asarray(rec["xt"], dtype=float)
    zr = np.asarray(rec["zr"], dtype=float)
    dx = float(rec["dx"])
    dz = float(rec["dz"])
    manifest: list[dict] = []
    blue, air = "#2b7fff", "#f2f4f8"
    handles = [Patch(facecolor=blue, label="water"), Patch(facecolor=air, edgecolor="0.5", label="air")]

    for index, target in enumerate(FRAME_TIMES):
        k = int(np.argmin(abs(frames_t - target)))
        time_s = float(frames_t[k])
        alt = np.asarray(rec["frames_alt"][k])
        alr = np.clip(np.asarray(rec["frames_alr"][k]), 0.0, 1.0)
        agr = np.clip(np.asarray(rec["frames_agr"][k]), 0.0, 1.0)
        yfs = float(np.interp(time_s, series["t_s"], series["Yfs_m"]))
        yint = float(np.interp(time_s, series["t_s"], series["Yint_m"]))
        head = float(np.interp(time_s, series["t_s"], series["pocket_head_m"]))
        full = ONE_FRAMES / f"full_{index:04d}.png"
        zoom = ONE_FRAMES / f"zoom_{index:04d}.png"

        if not (reuse and full.exists()):
            fig, ax = plt.subplots(figsize=(14.0, 5.0), dpi=115)
            ax.add_patch(Rectangle((0, -D), X_END, D, facecolor=air, edgecolor="0.45", lw=0.7))
            for x, fraction in zip(xt, alt):
                fraction = float(np.clip(fraction, 0.0, 1.0))
                if fraction > 0.01:
                    ax.add_patch(Rectangle((x - 0.5 * dx, -D), dx, fraction * D, facecolor=blue, edgecolor="none"))
            ax.add_patch(Rectangle((X_TEE - 0.5 * DR, 0), DR, RIM, facecolor=air, edgecolor="0.4", lw=0.8))
            for z, liquid, gas in zip(zr, alr, agr):
                if liquid > 0.01:
                    side = 0.5 * float(liquid) * DR
                    ax.add_patch(Rectangle((X_TEE - 0.5 * DR, z - 0.5 * dz), side, dz, facecolor=blue, edgecolor="none"))
                    ax.add_patch(Rectangle((X_TEE + 0.5 * DR - side, z - 0.5 * dz), side, dz, facecolor=blue, edgecolor="none"))
                if gas > 0.01 and z <= yfs:
                    width = float(gas) * DR
                    ax.add_patch(Rectangle((X_TEE - 0.5 * width, z - 0.5 * dz), width, dz, facecolor="white", edgecolor="none"))
            ax.add_patch(Rectangle((-0.22, -D - 0.05), 0.18, H0 + D + 0.05, facecolor="#dbeafe", edgecolor="0.5"))
            ax.axvline(X_VALVE, ymin=0.018, ymax=0.075, color="#111827", ls=":", lw=1.0)
            ax.axhline(RIM, xmin=(X_TEE - 0.04) / 6.9, xmax=(X_TEE + 0.04) / 6.9, color="#ef4444", ls="--")
            ax.set(xlim=(-0.25, 6.72), ylim=(-0.11, 3.08), xlabel="horizontal distance x (m)", ylabel="height above pipe crown (m)")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Geometry-matched frozen 1D model — B-H1, t={time_s:.2f} s")
            ax.text(0.01, 0.96, f"Yfs={yfs:.3f} m   Yint={yint:.3f} m   pocket head={head/H0:.3f} H0", transform=ax.transAxes, va="top")
            ax.legend(handles=handles, loc="upper right", frameon=False)
            fig.tight_layout()
            fig.savefig(full, facecolor="white")
            plt.close(fig)

        if not (reuse and zoom.exists()):
            fig, ax = plt.subplots(figsize=(3.0, 6.4), dpi=115)
            ax.add_patch(Rectangle((0, 0), 1, RIM, facecolor=air, edgecolor="0.4"))
            for z, liquid, gas in zip(zr, alr, agr):
                if liquid > 0.01:
                    side = 0.5 * float(liquid)
                    ax.add_patch(Rectangle((0, z - 0.5 * dz), side, dz, facecolor=blue, edgecolor="none"))
                    ax.add_patch(Rectangle((1 - side, z - 0.5 * dz), side, dz, facecolor=blue, edgecolor="none"))
                if gas > 0.01 and z <= yfs:
                    ax.add_patch(Rectangle((0.5 * (1 - gas), z - 0.5 * dz), gas, dz, facecolor="white", edgecolor="none"))
            ax.axhline(yfs, color="#1d4ed8", lw=1.2, label="free surface")
            if yint > 0:
                ax.axhline(yint, color="#dc2626", lw=1.1, ls="--", label="gas front")
            ax.set(xlim=(0, 1), ylim=(0, RIM), xticks=[], ylabel="height above pipe crown (m)")
            ax.set_title(f"1D riser (normalized width)\nt={time_s:.2f} s")
            ax.legend(loc="lower right", frameon=False, fontsize=7)
            fig.tight_layout()
            fig.savefig(zoom, facecolor="white")
            plt.close(fig)

        manifest.append({
            "file": rel(full),
            "riserFile": rel(zoom),
            "time": round(time_s, 6),
            "Yfs": finite_or_none(yfs),
            "Yint": finite_or_none(yint),
            "headOverH0": finite_or_none(head / H0),
        })
        print(f"1D frame {index + 1}/{len(FRAME_TIMES)} t={time_s:.2f} s", flush=True)
    return manifest


def foam_environment() -> str:
    root = "/usr/lib/openfoam/openfoam2512"
    platform = f"{root}/platforms/linux64GccDPInt32Opt"
    return (
        f"export WM_PROJECT_DIR={root}; "
        f"export PATH={platform}/bin:{root}/bin:/usr/bin:/bin; "
        f"export LD_LIBRARY_PATH={platform}/lib:{platform}/lib/sys-openmpi:/usr/lib/x86_64-linux-gnu/openmpi/lib; "
    )


def export_vtu(time_s: float) -> Path:
    command = (
        foam_environment()
        + f"cd {RUN}; nice -n 10 foamToVTK -ascii -fields '(alpha.water)' "
        + f"-no-boundary -no-point-data -time {time_s:.2f} -name VTK_FRAME_WORK -overwrite "
        + "> log.foamToVTK.frame_compare 2>&1"
    )
    completed = subprocess.run(["bash", "-lc", command], text=True)
    if completed.returncode != 0:
        log = RUN / "log.foamToVTK.frame_compare"
        detail = log.read_text(errors="ignore")[-3000:] if log.exists() else "no converter log"
        raise RuntimeError(f"foamToVTK failed at {time_s:.2f} s: {detail}")
    candidates = list((RUN / "VTK_FRAME_WORK").glob("*/internal.vtu"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one VTU at {time_s:.2f} s, found {len(candidates)}")
    return candidates[0]


def read_vtu(path: Path, need_geometry: bool) -> tuple[float, np.ndarray, np.ndarray | None]:
    root = ET.parse(path).getroot()
    time_node = root.find(".//FieldData/DataArray[@Name='TimeValue']")
    alpha_node = root.find(".//CellData/DataArray[@Name='alpha.water']")
    if time_node is None or alpha_node is None:
        raise RuntimeError(f"Missing TimeValue or alpha.water in {path}")
    time_s = float((time_node.text or "0").strip())
    alpha = np.fromstring(alpha_node.text or "", sep=" ")
    centres = None
    if need_geometry:
        points_node = root.find(".//Points/DataArray")
        arrays = {node.attrib.get("Name"): node for node in root.findall(".//Cells/DataArray")}
        if points_node is None or "connectivity" not in arrays or "offsets" not in arrays:
            raise RuntimeError(f"Missing mesh geometry in {path}")
        points = np.fromstring(points_node.text or "", sep=" ").reshape(-1, 3)
        connectivity = np.fromstring(arrays["connectivity"].text or "", sep=" ", dtype=int)
        offsets = np.fromstring(arrays["offsets"].text or "", sep=" ", dtype=int)
        starts = np.r_[0, offsets[:-1]]
        centres = np.empty((len(offsets), 2))
        for i, (start, end) in enumerate(zip(starts, offsets)):
            cell_points = points[connectivity[start:end]]
            centres[i] = (cell_points[:, 0].mean(), cell_points[:, 2].mean())
    return time_s, alpha, centres


def make_raster_map(
    centres: np.ndarray,
    xlim: tuple[float, float],
    zlim: tuple[float, float],
    nx: int,
    nz: int,
    full_domain: bool,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    xs = np.linspace(xlim[0], xlim[1], nx)
    zs = np.linspace(zlim[0], zlim[1], nz)
    xx, zz = np.meshgrid(xs, zs)
    if full_domain:
        pipe = (xx >= 0) & (xx <= X_END) & (zz >= -D / 2) & (zz <= D / 2)
        riser = (abs(xx - X_TEE) <= W2D / 2) & (zz >= D / 2) & (zz <= Z_RIM)
        external = (abs(xx - X_TEE) <= 0.15) & (zz >= Z_RIM) & (zz <= Z_TOP)
        mask = pipe | riser | external
    else:
        mask = np.ones(xx.shape, dtype=bool)
    index = np.full(xx.shape, -1, dtype=np.int32)
    _, nearest = cKDTree(centres).query(np.column_stack((xx[mask], zz[mask])), k=1)
    index[mask] = nearest.astype(np.int32)
    return index, mask


def raster_alpha(alpha: np.ndarray, index: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = np.full(index.shape, np.nan, dtype=float)
    image[mask] = np.clip(alpha[index[mask]], 0.0, 1.0)
    return image


def render_two_d_image(data: np.ndarray, time_s: float, out: Path, zoom: bool, yfs: float, yint: float, head: float) -> None:
    setup_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    cmap = LinearSegmentedColormap.from_list("air_water", [(0.0, "#f4f6f8"), (0.08, "#dbeafe"), (1.0, "#2563eb")])
    cmap.set_bad("white", alpha=0.0)
    if zoom:
        fig, ax = plt.subplots(figsize=(3.0, 6.4), dpi=115)
        ax.imshow(data, origin="lower", extent=(0, 1, 0, RIM), aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.add_patch(Rectangle((0, 0), 1, RIM, fill=False, edgecolor="0.35", lw=0.8))
        if np.isfinite(yfs):
            ax.axhline(yfs, color="#1d4ed8", lw=1.2, label="free surface")
        if np.isfinite(yint):
            ax.axhline(yint, color="#dc2626", lw=1.1, ls="--", label="gas front")
        ax.set(xlim=(0, 1), ylim=(0, RIM), xticks=[], ylabel="height above pipe crown (m)")
        ax.set_title(f"2D area-equivalent riser\nt={time_s:.2f} s")
        ax.legend(loc="lower right", frameon=False, fontsize=7)
    else:
        fig, ax = plt.subplots(figsize=(14.0, 5.0), dpi=115)
        ax.imshow(data, origin="lower", extent=(-0.25, 6.72, -0.11, 3.08), aspect="equal", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.plot([0, X_END], [-D / 2, -D / 2], color="0.35", lw=0.6)
        ax.plot([0, X_END], [D / 2, D / 2], color="0.35", lw=0.6)
        ax.plot([X_TEE - W2D / 2] * 2, [D / 2, Z_RIM], color="0.35", lw=0.7)
        ax.plot([X_TEE + W2D / 2] * 2, [D / 2, Z_RIM], color="0.35", lw=0.7)
        ax.axvline(X_VALVE, ymin=0.018, ymax=0.075, color="#111827", ls=":", lw=1.0)
        ax.set(xlim=(-0.25, 6.72), ylim=(-0.11, 3.08), xlabel="horizontal distance x (m)", ylabel="global z (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"OpenFOAM 2D paper-layout field — B-H1, t={time_s:.2f} s")
        ax.text(0.01, 0.96, f"Yfs={yfs:.3f} m   Yint={yint:.3f} m   PT1={head/H0:.3f} H0", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def render_two_d_frames(reuse: bool) -> list[dict]:
    series = read_series(RESULTS / "openfoam_2d_riser_series.csv")
    pt1 = read_series(RESULTS / "openfoam_2d_pt1_series.csv")
    TWO_FRAMES.mkdir(parents=True, exist_ok=True)
    full_map = full_mask = zoom_map = zoom_mask = None
    manifest: list[dict] = []

    for index, target in enumerate(FRAME_TIMES):
        full = TWO_FRAMES / f"full_{index:04d}.png"
        zoom = TWO_FRAMES / f"zoom_{index:04d}.png"
        time_s = float(target)
        yfs = float(np.interp(time_s, series["t_s"], series["Yfs_m_above_crown"]))
        yint = float(np.interp(time_s, series["t_s"], series["Yint_m_above_crown"]))
        head = float(np.interp(time_s, pt1["t_s"], pt1["head_m_water"]))
        if not (reuse and full.exists() and zoom.exists()):
            vtu = export_vtu(time_s)
            actual, alpha, centres = read_vtu(vtu, need_geometry=full_map is None)
            time_s = actual
            if full_map is None:
                if centres is None or len(centres) != len(alpha):
                    raise RuntimeError("VTK geometry and alpha field sizes differ")
                full_map, full_mask = make_raster_map(
                    centres, (-0.25, 6.72), (-0.11, 3.08), 1394, 638, full_domain=True
                )
                riser_centres = centres.copy()
                riser_centres[:, 0] = (riser_centres[:, 0] - (X_TEE - W2D / 2)) / W2D
                riser_centres[:, 1] -= Z_CROWN
                zoom_map, zoom_mask = make_raster_map(
                    riser_centres, (0, 1), (0, RIM), 180, 900, full_domain=False
                )
            assert full_map is not None and full_mask is not None and zoom_map is not None and zoom_mask is not None
            render_two_d_image(raster_alpha(alpha, full_map, full_mask), time_s, full, False, yfs, yint, head)
            render_two_d_image(raster_alpha(alpha, zoom_map, zoom_mask), time_s, zoom, True, yfs, yint, head)
        manifest.append({
            "file": rel(full),
            "riserFile": rel(zoom),
            "time": round(time_s, 6),
            "Yfs": finite_or_none(yfs),
            "Yint": finite_or_none(yint),
            "headOverH0": finite_or_none(head / H0),
        })
        print(f"2D frame {index + 1}/{len(FRAME_TIMES)} t={time_s:.2f} s", flush=True)
    return manifest


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cong 2017 B-H1：1D 与 OpenFOAM 2D 逐帧对比</title>
<style>
:root{--ink:#17212b;--muted:#667085;--line:#d7dde6;--one:#d55e00;--two:#0072b2;--bg:#f4f6f9;--card:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1540px;margin:auto;padding:18px}h1{font-size:23px;margin:0 0 4px}.note{color:var(--muted);margin:0 0 14px}.toolbar,.card{background:var(--card);border:1px solid var(--line);border-radius:10px;box-shadow:0 2px 8px #26384c0d}.toolbar{padding:12px 14px;margin-bottom:14px;display:grid;grid-template-columns:auto auto auto minmax(260px,1fr);gap:10px;align-items:center}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.card{padding:12px;min-width:0}.card h2{font-size:17px;margin:0 0 8px}.one h2{color:var(--one)}.two h2{color:var(--two)}.viewport{height:335px;border:1px solid #e4e8ed;border-radius:7px;background:#fff;display:flex;align-items:center;justify-content:center;overflow:hidden}.viewport.zoom{height:610px}.viewport img{width:100%;height:100%;object-fit:contain;display:block}.controls{display:grid;grid-template-columns:auto auto 1fr auto auto;gap:7px;align-items:center;margin-top:10px}button{border:1px solid #b9c2ce;background:#fff;border-radius:6px;padding:6px 10px;cursor:pointer}button:hover{background:#eef3f7}input[type=range]{width:100%}.meta{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:9px;color:var(--muted)}.meta b{color:var(--ink)}@media(max-width:900px){.grid{grid-template-columns:1fr}.toolbar{grid-template-columns:1fr}.viewport{height:250px}.viewport.zoom{height:520px}.meta{grid-template-columns:1fr 1fr}}
</style></head><body><main>
<h1>Cong 2017 B-H1：几何匹配的 1D 与 OpenFOAM 2D 逐帧对比</h1>
<p class="note">D=50 mm，Dr=16 mm，H0=0.66 m，L0=0.61 m；三通 x=3.47 m，释放阀 x=5.98 m，封闭端 x=6.59 m。原始物理时间同步，无时间平移、无结果拟合。2D 竖管采用面积等效宽度 5.12 mm。</p>
<section class="toolbar"><button id="view">切换为竖管放大</button><button id="sync">按当前1D时间同步2D</button><button id="jump">跳到实验到达时刻 8.07 s</button><label>共同物理时间 <input id="master" type="range" min="0" max="13" step="0.01" value="0"></label></section>
<section class="grid">
<article class="card one"><h2>冻结1D模型（论文 Fig. 1(b) 布置复算）</h2><div class="viewport"><img id="img1" alt="1D frame"></div><div class="controls"><button id="prev1">◀</button><button id="play1">播放</button><input id="range1" type="range"><button id="next1">▶</button><span id="count1"></span></div><div id="meta1" class="meta"></div></article>
<article class="card two"><h2>OpenFOAM 2D（论文布置，面积等效竖管）</h2><div class="viewport"><img id="img2" alt="2D frame"></div><div class="controls"><button id="prev2">◀</button><button id="play2">播放</button><input id="range2" type="range"><button id="next2">▶</button><span id="count2"></span></div><div id="meta2" class="meta"></div></article>
</section>
<script>
const data1=__DATA1__,data2=__DATA2__;let i1=0,i2=0,zoom=false,timer=null;const $=id=>document.getElementById(id);const nearest=(d,t)=>d.reduce((b,x,i)=>Math.abs(x.time-t)<Math.abs(d[b].time-t)?i:b,0);const fmt=(x,n=3)=>x==null||!Number.isFinite(Number(x))?'—':Number(x).toFixed(n);
function render(side){const d=side===1?data1:data2,i=side===1?i1:i2,f=d[i];$('img'+side).src=zoom?f.riserFile:f.file;$('range'+side).max=d.length-1;$('range'+side).value=i;$('count'+side).textContent=`${i+1}/${d.length}`;$('meta'+side).innerHTML=`<span>Time <b>${fmt(f.time,2)} s</b></span><span>Yfs <b>${fmt(f.Yfs)} m</b></span><span>Yint <b>${fmt(f.Yint)} m</b></span><span>head/H0 <b>${fmt(f.headOverH0)}</b></span>`;[-1,1].forEach(k=>{const j=Math.max(0,Math.min(d.length-1,i+k));new Image().src=zoom?d[j].riserFile:d[j].file})}
function setIndex(side,v){if(side===1)i1=Math.max(0,Math.min(data1.length-1,v));else i2=Math.max(0,Math.min(data2.length-1,v));render(side)}function sync(t){i1=nearest(data1,t);i2=nearest(data2,t);render(1);render(2);$('master').value=t}
function play(){if(timer){clearInterval(timer);timer=null;$('play1').textContent='播放';$('play2').textContent='播放'}else{timer=setInterval(()=>sync(+$('master').value>=13?0:+$('master').value+0.1),180);$('play1').textContent='暂停';$('play2').textContent='暂停'}}
$('range1').oninput=e=>setIndex(1,+e.target.value);$('range2').oninput=e=>setIndex(2,+e.target.value);$('prev1').onclick=()=>setIndex(1,i1-1);$('next1').onclick=()=>setIndex(1,i1+1);$('prev2').onclick=()=>setIndex(2,i2-1);$('next2').onclick=()=>setIndex(2,i2+1);$('play1').onclick=play;$('play2').onclick=play;$('master').oninput=e=>sync(+e.target.value);$('sync').onclick=()=>{i2=nearest(data2,data1[i1].time);render(2)};$('jump').onclick=()=>sync(8.07);$('view').onclick=()=>{zoom=!zoom;document.querySelectorAll('.viewport').forEach(x=>x.classList.toggle('zoom',zoom));$('view').textContent=zoom?'切换为完整管道':'切换为竖管放大';render(1);render(2)};document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')sync(Math.max(0,+$('master').value-.1));if(e.key==='ArrowRight')sync(Math.min(13,+$('master').value+.1));if(e.code==='Space'){e.preventDefault();play()}});sync(0);
</script></main></body></html>'''


def write_outputs(one: list[dict], two: list[dict], metrics: dict, audit: dict, one_metrics: dict) -> None:
    html = HTML_TEMPLATE.replace("__DATA1__", json.dumps(one, ensure_ascii=False, allow_nan=False))
    html = html.replace("__DATA2__", json.dumps(two, ensure_ascii=False, allow_nan=False))
    HTML.write_text(html, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "case": "B-H1",
        "formal_openfoam_complete": True,
        "openfoam_status": metrics["status"],
        "paper_audit_passed": audit.get("status") == "PASS" and all(audit.get("checks", {}).values()),
        "one_d_variant": one_metrics.get("variant"),
        "native_time_no_shift": True,
        "frame_interval_s": FRAME_DT,
        "n_frames_1d": len(one),
        "n_frames_2d": len(two),
        "geometry_m": {"D": D, "Dr": DR, "H0": H0, "L0": 0.61, "tee_x": X_TEE, "valve_x": X_VALVE, "cap_x": X_END},
        "planar_riser_width_m": W2D,
        "html": HTML.name,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def validate_outputs(one: list[dict], two: list[dict]) -> None:
    errors: list[str] = []
    expected_count = len(FRAME_TIMES)
    if len(one) != expected_count or len(two) != expected_count:
        errors.append(f"frame counts are {len(one)}/{len(two)}, expected {expected_count}/{expected_count}")
    if one and (abs(one[0]["time"]) > 0.02 or abs(one[-1]["time"] - T_END) > 0.06):
        errors.append("1D frame time coverage is not 0--13 s")
    if two and (abs(two[0]["time"]) > 0.02 or abs(two[-1]["time"] - T_END) > 0.02):
        errors.append("2D frame time coverage is not 0--13 s")
    for side, frames in (("1D", one), ("2D", two)):
        for frame in frames:
            for key in ("file", "riserFile"):
                path = HERE / frame[key]
                if not path.exists() or path.stat().st_size < 1000:
                    errors.append(f"{side} frame asset missing or empty: {path}")
                    if len(errors) > 20:
                        break
            if len(errors) > 20:
                break

    html = HTML.read_text(encoding="utf-8") if HTML.exists() else ""
    for marker in ("几何匹配的 1D", "OpenFOAM 2D", "共同物理时间", "const data1=", "data2="):
        if marker not in html:
            errors.append(f"HTML marker missing: {marker}")
    if "NaN" in html or "锛" in html or "�" in html:
        errors.append("HTML contains a non-JSON number or mojibake marker")

    payload = {
        "status": "PASS" if not errors else "FAIL",
        "checks": {
            "formal_gate_already_passed": True,
            "expected_frame_count": expected_count,
            "one_d_assets": len(one) * 2,
            "two_d_assets": len(two) * 2,
            "html_utf8_markers": not any("HTML" in error or "mojibake" in error for error in errors),
        },
        "errors": errors,
    }
    (HERE / "validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError("Frame comparison validation failed: " + "; ".join(errors[:8]))


def read_embedded_frames() -> tuple[list[dict], list[dict]]:
    """Reuse the already validated frame metadata when only the HTML layout changes."""
    html = HTML.read_text(encoding="utf-8")
    one_text = html.split("const data1=", 1)[1].split(",data2=", 1)[0]
    two_text = html.split(",data2=", 1)[1].split(";let i1=", 1)[0]
    return json.loads(one_text), json.loads(two_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--reuse-frames", action="store_true")
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()
    metrics, audit, one_metrics = completion_gate()
    if args.check_only:
        print("FORMAL_RUN_GATE_PASS")
        return
    if args.html_only:
        one, two = read_embedded_frames()
    else:
        one = render_one_d_frames(args.reuse_frames)
        two = render_two_d_frames(args.reuse_frames)
    write_outputs(one, two, metrics, audit, one_metrics)
    validate_outputs(one, two)
    print(f"WROTE {HTML}")


if __name__ == "__main__":
    main()

"""Render the fine-grid 1-D sensitivity as water/air occupancy frames.

The output preserves the original Case B comparison style: a physical pipe
view of the exploratory fine-grid 1-D state beside the established 2-D
OpenFOAM frame.  It does not alter either solver result.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from PIL import Image, ImageDraw


CASE_ROOT = Path(__file__).resolve().parents[1]
COMPARE_ROOT = CASE_ROOT / "openfoam" / "2d" / "outputs_1d2d_compare"
SENSITIVITY = CASE_ROOT / "outputs" / "sensitivity_wave_1d" / "wave_resolution_metrics.json"
PAIR_INDEX = COMPARE_ROOT / "frames_index_tosan2021.json"
BASELINE_BUILDER = Path(__file__).with_name("caseB_rebuild_1d_tosan2021.py")
FRAME_DIR = CASE_ROOT / "outputs" / "sensitivity_wave_1d" / "frames_fine_occupancy"
ANNOTATED_2D_DIR = CASE_ROOT / "outputs" / "sensitivity_wave_1d" / "frames_2d_annotated"
TWO_D_METADATA = COMPARE_ROOT / "frames_2d_caseB_tosan2021_aligned_meta.json"
TWO_D_PARSER = CASE_ROOT / "openfoam" / "2d" / "_local_render_2d_frames.py"
TRACKING_JSON = CASE_ROOT / "outputs" / "sensitivity_wave_1d" / "wave_tracking_manifest.json"
HTML = CASE_ROOT / "caseB_1d2d_wave_sensitivity.html"

D = 0.094
DT = 0.0127
PIPE_BOTTOM = -0.5 * D
PIPE_CROWN = 0.5 * D
X_TOWER = 3.516
RIM_Y = PIPE_CROWN + 0.610
WATER = "#2f7ff7"
AIR = "#f1f3f6"


def load_builder():
    spec = importlib.util.spec_from_file_location("caseb_wave_occupancy_builder", BASELINE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASELINE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_2d_parser():
    spec = importlib.util.spec_from_file_location("caseb_wave_2d_parser", TWO_D_PARSER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {TWO_D_PARSER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.read_vtu_cell_centers_alpha


def track_spatial_crest(
    x: np.ndarray,
    surface: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    smoothing_length: float,
    minimum_amplitude: float,
    maximum_local_jump: float | None = None,
) -> dict | None:
    """Track a local crest relative to a moving-average background surface."""
    mask = np.isfinite(x) & np.isfinite(surface) & (x >= x_min) & (x <= x_max)
    xx = np.asarray(x[mask], dtype=float)
    yy = np.asarray(surface[mask], dtype=float)
    if xx.size < 9:
        return None
    order = np.argsort(xx)
    xx, yy = xx[order], yy[order]
    dx = float(np.median(np.diff(xx)))
    window = max(5, int(round(smoothing_length / max(dx, 1.0e-9))))
    if window % 2 == 0:
        window += 1
    window = min(window, xx.size - (1 - xx.size % 2))
    if window < 5:
        return None
    kernel = np.ones(window) / window
    smooth = np.convolve(yy, kernel, mode="same")
    half = window // 2
    valid = np.arange(half, xx.size - half)
    if maximum_local_jump is not None:
        # A gas-pocket nose is a moving discontinuity, not a free-surface wave.
        # Remove every candidate whose smoothing stencil crosses such a jump.
        jump_edges = np.flatnonzero(np.abs(np.diff(yy)) > maximum_local_jump)
        for edge in jump_edges:
            valid = valid[np.abs(valid - edge) > half]
        if valid.size < 3:
            return None
    residual = yy - smooth
    crest_index = int(valid[np.argmax(residual[valid])])
    trough_index = int(valid[np.argmin(residual[valid])])
    amplitude = float(residual[crest_index] - residual[trough_index])
    if amplitude < minimum_amplitude:
        return None
    return {
        "crest_x": float(xx[crest_index]),
        "crest_y": float(yy[crest_index]),
        "trough_x": float(xx[trough_index]),
        "trough_y": float(yy[trough_index]),
        "residual_peak_to_peak_m": amplitude,
    }


def extract_2d_surface(vtu: Path, read_vtu) -> tuple[np.ndarray, np.ndarray]:
    """Extract the bottom-connected alpha.water=0.5 surface in the pipe."""
    centers, alpha = read_vtu(vtu)
    x = np.round(centers[:, 0], 9)
    y = centers[:, 1]
    mask = (x >= 2.35) & (x <= 3.99) & (y >= PIPE_BOTTOM) & (y <= PIPE_CROWN)
    x, y, alpha = x[mask], y[mask], np.asarray(alpha)[mask]
    columns = []
    surfaces = []
    for value in np.unique(x):
        cmask = x == value
        cy = y[cmask]
        ca = alpha[cmask]
        order = np.argsort(cy)
        cy, ca = cy[order], ca[order]
        dry = np.flatnonzero(ca < 0.5)
        if dry.size == 0:
            level = PIPE_CROWN
        elif dry[0] == 0:
            level = PIPE_BOTTOM
        else:
            level = 0.5 * (cy[dry[0] - 1] + cy[dry[0]])
        columns.append(float(value))
        surfaces.append(float(level))
    return np.asarray(columns), np.asarray(surfaces)


def png_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def render_frame(index: int, state: dict, tower: dict, builder) -> tuple[Path, dict | None]:
    x = np.asarray(state["x"], dtype=float)
    depth = np.asarray(state["h"], dtype=float)
    output = FRAME_DIR / f"frame_{index:04d}.png"
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman"]}):
        fig, ax = plt.subplots(figsize=(12.4, 3.2))
        ax.add_patch(Rectangle((0.0, PIPE_BOTTOM), 4.006, D, facecolor=AIR,
                               edgecolor="#333333", linewidth=0.8))
        ax.fill_between(x, PIPE_BOTTOM, PIPE_BOTTOM + depth, step="mid",
                        color=WATER, linewidth=0.0)
        # Only partially filled cells can form a horizontal free surface.  Full
        # cells and the sharp gas-pocket nose are excluded from wave tracking.
        wave_surface = np.where((depth > 0.002) & (depth < D - 0.002), depth, np.nan)
        tracking = track_spatial_crest(
            x,
            wave_surface,
            x_min=3.56,
            x_max=3.98,
            smoothing_length=0.10,
            minimum_amplitude=2.0e-4,
            maximum_local_jump=2.0e-3,
        )
        if tracking is not None:
            crest_y = PIPE_BOTTOM + tracking["crest_y"]
            ax.scatter([tracking["crest_x"]], [crest_y], marker="*", s=70,
                       facecolor="#ffe000", edgecolor="#d7191c", linewidth=1.0, zorder=8)
            ax.annotate(
                f"tracked 1D crest\nA_pp={tracking['residual_peak_to_peak_m']*1000:.2f} mm",
                xy=(tracking["crest_x"], crest_y),
                xytext=(tracking["crest_x"] - 0.18, crest_y + 0.10),
                color="#d7191c", ha="center", fontsize=8,
                arrowprops={"arrowstyle": "->", "color": "#d7191c", "lw": 1.0},
                bbox={"facecolor": "white", "edgecolor": "#d7191c", "alpha": 0.9, "pad": 2},
            )
        else:
            ax.text(3.76, PIPE_CROWN + 0.085, "1D: no resolvable moving crest",
                    color="#d7191c", ha="center", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "#d7191c", "alpha": 0.88, "pad": 2})
        builder._draw_tower(ax, tower, X_TOWER, PIPE_CROWN, DT, 0.610)
        ax.axvline(0.546, ymin=0.035, ymax=0.14, color="#202020",
                   linestyle=":", linewidth=0.9)
        ax.plot([3.470, 3.562], [RIM_Y, RIM_Y], color="#ef4444",
                linestyle="--", linewidth=1.0)
        ax.text(0.015, 0.95, f"Time = {state['t']:.2f} s", transform=ax.transAxes,
                ha="left", va="top", fontsize=12)
        ax.text(0.015, 0.86,
                "Exploratory fine-grid 1D occupancy | dx = 0.005 m | equations unchanged",
                transform=ax.transAxes, ha="left", va="top", fontsize=9)
        ax.text(0.546, PIPE_CROWN + 0.025, "valve", ha="center", va="bottom", fontsize=8)
        ax.set_xlim(-0.04, 4.046)
        ax.set_ylim(PIPE_BOTTOM - 0.035, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("horizontal distance [m]")
        ax.set_ylabel("vertical coordinate [m]")
        ax.legend(handles=[Patch(facecolor=WATER, label="water"),
                           Patch(facecolor=AIR, edgecolor="#555555", label="air")],
                  loc="upper right", frameon=False, fontsize=9)
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.94)
        fig.savefig(output, dpi=140)
        plt.close(fig)
    return output, tracking


def annotate_2d(source: Path, output: Path, tracking: dict | None) -> Path:
    """Mark the visible 2-D spatial-wave region without altering field pixels."""
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    # Matplotlib axes used by caseB_render_2d_vertical_aligned.py.
    box_left, box_right = 0.075 * width, 0.985 * width
    top, bottom = (1.0 - 0.94) * height, (1.0 - 0.18) * height
    x_min, x_max = -0.04, 4.046
    y_min, y_max = PIPE_BOTTOM - 0.035, 1.0
    # set_aspect("equal", adjustable="box") narrows the axes inside the
    # subplot rectangle.  Reproduce that geometry before mapping data to PNG.
    data_aspect = (x_max - x_min) / (y_max - y_min)
    axes_width = min(box_right - box_left, (bottom - top) * data_aspect)
    left = 0.5 * (box_left + box_right - axes_width)
    right = left + axes_width
    def px(x_value: float) -> float:
        return left + (x_value - x_min) / (x_max - x_min) * (right - left)
    def py(y_value: float) -> float:
        return bottom - (y_value - y_min) / (y_max - y_min) * (bottom - top)
    if tracking is not None:
        cx, cy = px(tracking["crest_x"]), py(tracking["crest_y"])
        radius = max(5, width // 250)
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius),
                     fill="#ffe000", outline="#d7191c", width=max(2, width // 700))
        label = (
            f"tracked 2D crest  x={tracking['crest_x']:.2f} m  "
            f"A_pp={tracking['residual_peak_to_peak_m']*1000:.1f} mm"
        )
        label_x, label_y = int(max(left + 8, cx - 180)), int(max(top + 8, cy - 45))
        bbox = draw.textbbox((label_x, label_y), label)
        draw.rectangle((bbox[0]-4, bbox[1]-3, bbox[2]+4, bbox[3]+3),
                       fill="white", outline="#d7191c")
        draw.text((label_x, label_y), label, fill="#d7191c")
        draw.line((label_x + 25, bbox[3] + 3, cx, cy), fill="#d7191c", width=max(2, width // 700))
    else:
        label = "2D right branch: no resolvable moving crest"
        label_x, label_y = int(px(3.05)), int(py(0.14))
        bbox = draw.textbbox((label_x, label_y), label)
        draw.rectangle((bbox[0]-4, bbox[1]-3, bbox[2]+4, bbox[3]+3),
                       fill="white", outline="#d7191c")
        draw.text((label_x, label_y), label, fill="#d7191c")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def build_html(frames: list[dict]) -> str:
    payload = json.dumps(frames, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Case B fine-grid 1D vs 2D</title>
<style>body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:0;background:#f4f6f8;color:#18202a}}main{{max-width:1800px;margin:auto;padding:20px}}h1{{font-size:24px;margin:0 0 8px}}.note{{line-height:1.55;color:#46515e}}.controls,.panel{{background:#fff;border:1px solid #cbd2da;padding:12px}}.controls{{margin:14px 0}}.buttons{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}button,select{{font:inherit;padding:6px 10px}}input[type=range]{{width:100%;margin-top:12px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}img{{display:block;width:100%;object-fit:contain}}#info{{font-variant-numeric:tabular-nums}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>Case B — 右端横管动态波峰追踪</h1><p class="note">黄色星标只在竖管右侧至封闭端（x=3.56–3.98 m）逐帧追踪局部自由水面波峰，箭头和幅值随波峰移动；气团前缘的陡直跳变被排除，检测不到可分辨空间波峰时会明确标注。该算法分别作用于1D水深场和2D原始 αw=0.5 自由水面，标注不改变模拟结果。</p>
<section class="controls"><div class="buttons"><button id="prev">上一帧</button><button id="play">播放</button><button id="next">下一帧</button><select id="speed"><option value="360">0.5×</option><option value="180" selected>1×</option><option value="90">2×</option></select><span id="info"></span></div><input id="range" type="range" min="0" max="{len(frames)-1}" value="0" step="1"></section>
<div class="grid"><section class="panel"><h2>1D：右端横管波峰</h2><img id="one" alt="Tracked fine-grid 1D crest"></section><section class="panel"><h2>2D：右端横管波峰</h2><img id="two" alt="Tracked 2D OpenFOAM crest"></section></div>
<script>const f={payload};let i=0,timer=null;const range=document.getElementById('range'),play=document.getElementById('play');function label(t){{return t?`x=${{t.crest_x.toFixed(3)}} m, A_pp=${{(t.residual_peak_to_peak_m*1000).toFixed(2)}} mm`:'未检测到空间波峰'}}function show(n){{i=Math.max(0,Math.min(f.length-1,n));range.value=i;document.getElementById('one').src=f[i].one;document.getElementById('two').src=f[i].two;document.getElementById('info').textContent=`帧 ${{i+1}}/${{f.length}} · t=${{f[i].time.toFixed(2)}} s · 1D: ${{label(f[i].track1d)}} · 2D: ${{label(f[i].track2d)}}`}}function stop(){{if(timer)clearInterval(timer);timer=null;play.textContent='播放'}}function toggle(){{if(timer){{stop();return}}timer=setInterval(()=>show((i+1)%f.length),Number(document.getElementById('speed').value));play.textContent='暂停'}}document.getElementById('prev').onclick=()=>{{stop();show(i-1)}};document.getElementById('next').onclick=()=>{{stop();show(i+1)}};play.onclick=toggle;range.oninput=e=>{{stop();show(Number(e.target.value))}};document.getElementById('speed').onchange=()=>{{if(timer){{stop();toggle()}}}};show(0);</script></main></body></html>"""


def main() -> None:
    sensitivity = json.loads(SENSITIVITY.read_text(encoding="utf-8"))
    fine = min(sensitivity["variants"], key=lambda item: abs(float(item["dx_m"]) - 0.005))
    fine_frames = fine["frames"]
    pairs = json.loads(PAIR_INDEX.read_text(encoding="utf-8"))
    two_d_metadata = json.loads(TWO_D_METADATA.read_text(encoding="utf-8"))["frames"]
    read_vtu = load_2d_parser()
    times = [float(pair["time"]) for pair in pairs]
    builder = load_builder()
    _, _, tower_frames = builder._run_vertical_reference(times)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_2D_DIR.mkdir(parents=True, exist_ok=True)
    rendered = []
    for index, (pair, tower) in enumerate(zip(pairs, tower_frames)):
        state = min(fine_frames, key=lambda row: abs(float(row["t"]) - float(pair["time"])))
        frame, track1d = render_frame(index, state, tower, builder)
        two_d = COMPARE_ROOT / pair["file2d"]
        source_vtu = CASE_ROOT / "openfoam" / "2d" / two_d_metadata[index]["source_vtu"]
        sx, sy = extract_2d_surface(source_vtu, read_vtu)
        track2d = track_spatial_crest(
            sx,
            sy,
            x_min=3.56,
            x_max=3.98,
            smoothing_length=0.10,
            minimum_amplitude=2.0e-4,
        )
        annotated_2d = annotate_2d(
            two_d,
            ANNOTATED_2D_DIR / f"frame_{index:04d}.png",
            track2d,
        )
        rendered.append({"time": float(pair["time"]), "interface_x": float(state["interface_x"]),
                         "track1d": track1d, "track2d": track2d,
                         "one": png_uri(frame), "two": png_uri(annotated_2d)})
        print(f"rendered {index+1}/{len(pairs)}", flush=True)
    HTML.write_text(build_html(rendered), encoding="utf-8")
    tracking_manifest = [
        {
            "time": item["time"],
            "interface_x": item["interface_x"],
            "track1d": item["track1d"],
            "track2d": item["track2d"],
        }
        for item in rendered
    ]
    TRACKING_JSON.write_text(
        json.dumps(tracking_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"HTML -> {HTML}")


if __name__ == "__main__":
    main()

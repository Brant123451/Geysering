"""Render the exploratory 1-D post-impact wave beside the original 2-D field."""
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


CASE_ROOT = Path(__file__).resolve().parents[1]
COMPARE_ROOT = CASE_ROOT / "openfoam" / "2d" / "outputs_1d2d_compare"
METRICS = (
    CASE_ROOT
    / "outputs"
    / "sensitivity_postimpact_wave_1d"
    / "postimpact_topology_metrics.json"
)
TRACK_2D = (
    CASE_ROOT / "outputs" / "sensitivity_wave_1d" / "wave_tracking_manifest.json"
)
PAIR_INDEX = COMPARE_ROOT / "frames_index_tosan2021.json"
BUILDER = Path(__file__).with_name("caseB_rebuild_1d_tosan2021.py")
FRAME_DIR = (
    CASE_ROOT / "outputs" / "sensitivity_postimpact_wave_1d" / "frames_occupancy"
)
HTML = CASE_ROOT / "caseB_1d2d_postimpact_wave.html"

D = 0.094
DT = 0.0127
PIPE_BOTTOM = -0.5 * D
PIPE_CROWN = 0.5 * D
X_TOWER = 3.516
RIM_Y = PIPE_CROWN + 0.610
WATER = "#2f7ff7"
AIR = "#f1f3f6"


def load_builder():
    spec = importlib.util.spec_from_file_location("caseb_postimpact_renderer", BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def png_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def render_1d(index: int, frame: dict, tracking: dict | None, tower: dict, builder) -> Path:
    x = np.asarray(frame["x"], dtype=float)
    depth = np.asarray(frame["h"], dtype=float)
    output = FRAME_DIR / f"frame_{index:04d}.png"
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman"]}):
        fig, ax = plt.subplots(figsize=(12.4, 3.2))
        ax.add_patch(
            Rectangle(
                (0.0, PIPE_BOTTOM), 4.006, D,
                facecolor=AIR, edgecolor="#333333", linewidth=0.8,
            )
        )
        ax.fill_between(
            x, PIPE_BOTTOM, PIPE_BOTTOM + depth,
            step="mid", color=WATER, linewidth=0.0,
        )
        if tracking is not None:
            crest_y = PIPE_BOTTOM + float(tracking["crest_h"])
            crest_x = float(tracking["crest_x"])
            amplitude = float(tracking["residual_peak_to_peak_m"]) * 1000.0
            ax.scatter(
                [crest_x], [crest_y], marker="*", s=75,
                facecolor="#ffe000", edgecolor="#d7191c", linewidth=1.0, zorder=8,
            )
            ax.annotate(
                f"post-impact 1D crest\nA_pp={amplitude:.2f} mm",
                xy=(crest_x, crest_y), xytext=(crest_x - 0.22, crest_y + 0.10),
                color="#d7191c", ha="center", fontsize=8,
                arrowprops={"arrowstyle": "->", "color": "#d7191c", "lw": 1.0},
                bbox={"facecolor": "white", "edgecolor": "#d7191c", "alpha": 0.9, "pad": 2},
            )
        else:
            ax.text(
                3.76, PIPE_CROWN + 0.085, "1D: no resolvable right-branch crest",
                color="#d7191c", ha="center", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#d7191c", "alpha": 0.88, "pad": 2},
            )
        builder._draw_tower(ax, tower, X_TOWER, PIPE_CROWN, DT, 0.610)
        ax.axvline(0.546, ymin=0.035, ymax=0.14, color="#202020", linestyle=":", linewidth=0.9)
        ax.plot([3.470, 3.562], [RIM_Y, RIM_Y], color="#ef4444", linestyle="--", linewidth=1.0)
        ax.text(0.015, 0.95, f"Time = {float(frame['t']):.2f} s", transform=ax.transAxes,
                ha="left", va="top", fontsize=12)
        ax.text(
            0.015, 0.86,
            "Exploratory 1D | post-impact whole-pipe FV | dx = 0.005 m",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
        )
        ax.text(0.546, PIPE_CROWN + 0.025, "valve", ha="center", va="bottom", fontsize=8)
        ax.set_xlim(-0.04, 4.046)
        ax.set_ylim(PIPE_BOTTOM - 0.035, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("horizontal distance [m]")
        ax.set_ylabel("vertical coordinate [m]")
        ax.legend(
            handles=[Patch(facecolor=WATER, label="water"),
                     Patch(facecolor=AIR, edgecolor="#555555", label="air")],
            loc="upper right", frameon=False, fontsize=9,
        )
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.94)
        fig.savefig(output, dpi=140)
        plt.close(fig)
    return output


def build_html(frames: list[dict], impact_time: float, max_wave_mm: float) -> str:
    payload = json.dumps(frames, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Case B post-impact 1D vs 2D</title>
<style>body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:0;background:#f4f6f8;color:#18202a}}main{{max-width:1800px;margin:auto;padding:20px}}h1{{font-size:24px;margin:0 0 8px}}.note{{line-height:1.55;color:#46515e}}.controls,.panel{{background:#fff;border:1px solid #cbd2da;padding:12px}}.controls{{margin:14px 0}}.buttons{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}button,select{{font:inherit;padding:6px 10px}}input[type=range]{{width:100%;margin-top:12px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}img{{display:block;width:100%;object-fit:contain}}#info{{font-variant-numeric:tabular-nums}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>Case B — 撞壁后右端波动：探索性1D vs 2D</h1>
<p class="note">1D在气团前缘于 t={impact_time:.2f} s 到达右壁后，撤销已经不存在的右侧有压分支，并让整条横管继续用原 Saint-Venant 有限体积方程及封闭壁面边界推进。黄色标记追踪 x=3.56–3.98 m 的移动波峰。1D最大峰谷差 {max_wave_mm:.2f} mm；这是数值拓扑敏感性，不替代冻结论文结果。</p>
<section class="controls"><div class="buttons"><button id="prev">上一帧</button><button id="play">播放</button><button id="next">下一帧</button><select id="speed"><option value="600">0.5×</option><option value="300" selected>1×</option><option value="150">2×</option></select><span id="info"></span></div><input id="range" type="range" min="0" max="{len(frames)-1}" value="0" step="1"></section>
<div class="grid"><section class="panel"><h2>1D：撞壁后拓扑切换候选</h2><img id="one"></section><section class="panel"><h2>2D：原始OpenFOAM结果</h2><img id="two"></section></div>
<script>const f={payload};let i=0,timer=null;const range=document.getElementById('range'),play=document.getElementById('play');function label(t){{return t?`x=${{t.crest_x.toFixed(3)}} m, A_pp=${{(t.residual_peak_to_peak_m*1000).toFixed(2)}} mm`:'未检测到'}}function show(n){{i=Math.max(0,Math.min(f.length-1,n));range.value=i;document.getElementById('one').src=f[i].one;document.getElementById('two').src=f[i].two;document.getElementById('info').textContent=`帧 ${{i+1}}/${{f.length}} · 对齐时间 ${{f[i].time.toFixed(2)}} s · 1D: ${{label(f[i].track1d)}} · 2D: ${{label(f[i].track2d)}}`}}function stop(){{if(timer)clearInterval(timer);timer=null;play.textContent='播放'}}function toggle(){{if(timer){{stop();return}}timer=setInterval(()=>show((i+1)%f.length),Number(document.getElementById('speed').value));play.textContent='暂停'}}document.getElementById('prev').onclick=()=>{{stop();show(i-1)}};document.getElementById('next').onclick=()=>{{stop();show(i+1)}};play.onclick=toggle;range.oninput=e=>{{stop();show(Number(e.target.value))}};document.getElementById('speed').onchange=()=>{{if(timer){{stop();toggle()}}}};show(0);</script></main></body></html>"""


def main() -> None:
    result = json.loads(METRICS.read_text(encoding="utf-8"))
    pairs_all = json.loads(PAIR_INDEX.read_text(encoding="utf-8"))
    track2d_all = json.loads(TRACK_2D.read_text(encoding="utf-8"))
    selected = [(index, pair) for index, pair in enumerate(pairs_all) if float(pair["time"]) >= 7.55]
    builder = load_builder()
    times = [float(pair["time"]) for _, pair in selected]
    _, _, towers = builder._run_vertical_reference(times)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    rendered = []
    frame_times = np.asarray([float(row["t"]) for row in result["frames"]])
    for local_index, ((source_index, pair), tower) in enumerate(zip(selected, towers)):
        candidate_index = int(np.argmin(np.abs(frame_times - float(pair["time"]))))
        frame = result["frames"][candidate_index]
        track1d = result["tracking"][candidate_index]
        one = render_1d(local_index, frame, track1d, tower, builder)
        two = (
            CASE_ROOT
            / "outputs"
            / "sensitivity_wave_1d"
            / "frames_2d_annotated"
            / f"frame_{source_index:04d}.png"
        )
        if not two.exists():
            raise FileNotFoundError(two)
        rendered.append(
            {
                "time": float(pair["time"]),
                "track1d": track1d,
                "track2d": track2d_all[source_index]["track2d"],
                "one": png_uri(one),
                "two": png_uri(two),
            }
        )
        print(f"rendered {local_index + 1}/{len(selected)}", flush=True)
    max_wave_mm = float(result["metrics"]["maximum_residual_peak_to_peak_m"]) * 1000.0
    HTML.write_text(
        build_html(rendered, float(result["impact_time_s"]), max_wave_mm),
        encoding="utf-8",
    )
    print(f"HTML -> {HTML}")


if __name__ == "__main__":
    main()

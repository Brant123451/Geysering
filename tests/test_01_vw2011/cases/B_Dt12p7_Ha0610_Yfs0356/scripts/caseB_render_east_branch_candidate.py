"""Render the two-fluid east-branch junction sensitivity beside 2-D Case B."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle


CASE_ROOT = Path(__file__).resolve().parents[1]
COMPARE_ROOT = CASE_ROOT / "openfoam" / "2d" / "outputs_1d2d_compare"
METRICS = (
    CASE_ROOT / "outputs" / "sensitivity_tjunction_east_branch" / "east_branch_metrics.json"
)
PAIR_INDEX = COMPARE_ROOT / "frames_index_tosan2021.json"
FRAME_DIR = (
    CASE_ROOT / "outputs" / "sensitivity_tjunction_east_branch" / "frames_occupancy"
)
HTML = CASE_ROOT / "caseB_1d2d_east_branch_candidate.html"

D = 0.094
BOTTOM = -0.5 * D
CROWN = 0.5 * D
X_T = 3.516
WATER = "#2f7ff7"
AIR = "#f1f3f6"


def png_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def render_1d(index: int, frame: dict) -> Path:
    x = np.asarray(frame["x"], dtype=float)
    h = np.asarray(frame["h"], dtype=float)
    output = FRAME_DIR / f"frame_{index:04d}.png"
    fig, ax = plt.subplots(figsize=(12.4, 2.6))
    ax.add_patch(Rectangle((0.0, BOTTOM), 4.006, D, facecolor=AIR,
                           edgecolor="#333333", linewidth=0.9))
    ax.fill_between(x, BOTTOM, BOTTOM + h, step="mid", color=WATER, linewidth=0.0)
    # Draw the T location only; the panel is intentionally a horizontal-field audit.
    ax.plot([X_T, X_T], [CROWN, 0.20], color="#333333", linewidth=1.0)
    ax.plot([X_T - 0.00635, X_T + 0.00635], [0.20, 0.20],
            color="#ef4444", linestyle="--", linewidth=1.0)
    track = frame.get("track")
    if track is not None and float(frame["t"]) >= 7.0:
        cx = float(track["crest_x"])
        cy = BOTTOM + float(track["crest_h"])
        ax.scatter([cx], [cy], marker="*", s=72, facecolor="#ffe000",
                   edgecolor="#d7191c", linewidth=1.0, zorder=8)
    ax.text(0.015, 0.93, f"Time = {float(frame['t']):.2f} s", transform=ax.transAxes,
            ha="left", va="top", fontsize=12)
    ax.text(0.015, 0.80,
            "Two-fluid sensitivity | east-leg gas + moving-front RH liquid trace",
            transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.text(0.985, 0.80,
            f"right-wall alpha_g = {100.0*float(frame['right_wall_alpha_g']):.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#b91c1c")
    ax.set_xlim(-0.04, 4.046)
    ax.set_ylim(BOTTOM - 0.025, 0.24)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("horizontal distance [m]")
    ax.set_ylabel("y [m]")
    ax.legend(handles=[Patch(facecolor=WATER, label="water"),
                       Patch(facecolor=AIR, edgecolor="#555555", label="air")],
              loc="upper right", frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.94)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def build_html(frames: list[dict], result: dict) -> str:
    payload = json.dumps(frames, ensure_ascii=False, separators=(",", ":"))
    wall_time = float(result["wall_arrival_time_s"])
    wall_alpha = 100.0 * float(result["metrics"]["maximum_wall_alpha_g"])
    wave_mm = 1000.0 * float(result["metrics"]["maximum_postwall_wave_m"])
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Case B east-branch gas candidate</title>
<style>body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:0;background:#f4f6f8;color:#18202a}}main{{max-width:1800px;margin:auto;padding:20px}}h1{{font-size:24px;margin:0 0 8px}}.warn{{background:#fff4e5;border:1px solid #ef9f27;padding:10px;line-height:1.55}}.controls,.panel{{background:#fff;border:1px solid #cbd2da;padding:12px}}.controls{{margin:14px 0}}.buttons{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}button,select{{font:inherit;padding:6px 10px}}input[type=range]{{width:100%;margin-top:12px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}img{{display:block;width:100%;object-fit:contain}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>Case B — FAIL：守恒前缘增强了撞壁响应，但波幅仍不足</h1><p class="warn"><b>失败候选，不是论文结果：</b>本候选没有指定正弦曲线或波幅；它使气体约在 {wall_time:.2f} s 到达右壁，右端最大气相率 {wall_alpha:.1f}%，并保留竖管约 0.60 m 的喷发。加入移动前缘 Rankine–Hugoniot 液相流量条件后，撞壁后最大波幅由上一轮 2.48 mm 增至 {wave_mm:.2f} mm，但 8.30 s 时1D最高水位仅约 76.35 mm（距管顶约 17.65 mm），仍不能复现2D约 87.29 mm、几乎触顶的局部爬升。因此不能作为有效1D复现。</p>
<section class="controls"><div class="buttons"><button id="prev">上一帧</button><button id="play">播放</button><button id="next">下一帧</button><select id="speed"><option value="600">0.5×</option><option value="300" selected>1×</option><option value="150">2×</option></select><span id="info"></span></div><input id="range" type="range" min="0" max="{len(frames)-1}" value="0"></section>
<div class="grid"><section class="panel"><h2>1D两流体 + 移动前缘守恒条件</h2><img id="one"></section><section class="panel"><h2>2D OpenFOAM</h2><img id="two"></section></div>
<script>const f={payload};let i=0,timer=null;const range=document.getElementById('range'),play=document.getElementById('play');function show(n){{i=Math.max(0,Math.min(f.length-1,n));range.value=i;one.src=f[i].one;two.src=f[i].two;info.textContent=`帧 ${{i+1}}/${{f.length}} · t=${{f[i].time.toFixed(2)}} s · 1D右壁气相率=${{(100*f[i].wallAlpha).toFixed(1)}}%`}}function stop(){{if(timer)clearInterval(timer);timer=null;play.textContent='播放'}}function toggle(){{if(timer){{stop();return}}timer=setInterval(()=>show((i+1)%f.length),Number(speed.value));play.textContent='暂停'}}prev.onclick=()=>{{stop();show(i-1)}};next.onclick=()=>{{stop();show(i+1)}};play.onclick=toggle;range.oninput=e=>{{stop();show(Number(e.target.value))}};speed.onchange=()=>{{if(timer){{stop();toggle()}}}};show(0);</script></main></body></html>"""


def main() -> None:
    result = json.loads(METRICS.read_text(encoding="utf-8"))
    pairs = json.loads(PAIR_INDEX.read_text(encoding="utf-8"))
    selected = [(i, row) for i, row in enumerate(pairs) if float(row["time"]) >= 6.35]
    times = np.asarray([float(row["t"]) for row in result["frames"]])
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    rendered = []
    for local, (source_index, pair) in enumerate(selected):
        k = int(np.argmin(np.abs(times - float(pair["time"]))))
        frame = result["frames"][k]
        one = render_1d(local, frame)
        two = (CASE_ROOT / "outputs" / "sensitivity_wave_1d" /
               "frames_2d_annotated" / f"frame_{source_index:04d}.png")
        rendered.append({"time": float(pair["time"]),
                         "wallAlpha": float(frame["right_wall_alpha_g"]),
                         "one": png_uri(one), "two": png_uri(two)})
        print(f"rendered {local+1}/{len(selected)}", flush=True)
    HTML.write_text(build_html(rendered, result), encoding="utf-8")
    print(f"HTML -> {HTML}")


if __name__ == "__main__":
    main()

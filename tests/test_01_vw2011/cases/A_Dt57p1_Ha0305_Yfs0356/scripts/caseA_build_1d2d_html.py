"""Build an offline, manually adjustable 1D--2D Case A frame viewer.

The 1D frames must first be regenerated with ``caseA_make_frame_viewer.py``.
The 2D fields are read from ASCII VTU files exported to ``VTK_CASEA_HTML``.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize


HERE = Path(__file__).resolve().parent
CASE = HERE.parent
VTK_ROOT = CASE / "openfoam/2d/VTK_CASEA_HTML"
FRAME_ROOT = CASE / "openfoam/2d/outputs/html_frames"
HTML_PATH = CASE / "caseA_1d2d_frame_compare.html"

D = 0.094
DT = 0.0571
L = 4.006
TOWER_X = 3.516
TOWER_H = 0.610
TSTAR_PER_SECOND = 1.2269382978252207


def read_vtu(path: Path) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, float]:
    root = ET.parse(path).getroot()
    points_node = root.find(".//Points/DataArray")
    if points_node is None or points_node.text is None:
        raise ValueError(f"No points in {path}")
    points = np.fromstring(points_node.text, sep=" ").reshape(-1, 3)

    arrays = {node.attrib.get("Name"): node for node in root.findall(".//Cells/DataArray")}
    connectivity = np.fromstring(arrays["connectivity"].text or "", sep=" ", dtype=int)
    offsets = np.fromstring(arrays["offsets"].text or "", sep=" ", dtype=int)
    cells = [connectivity[i:j] for i, j in zip(np.r_[0, offsets[:-1]], offsets)]

    alpha_node = root.find(".//CellData/DataArray[@Name='alpha.water']")
    time_node = root.find(".//FieldData/DataArray[@Name='TimeValue']")
    if alpha_node is None or alpha_node.text is None or time_node is None:
        raise ValueError(f"Missing alpha.water or TimeValue in {path}")
    alpha = np.fromstring(alpha_node.text, sep=" ")
    return points, cells, alpha, float((time_node.text or "0").strip())


def read_levels() -> dict[str, np.ndarray]:
    path = CASE / "openfoam/2d/outputs/openfoam_2d_levels.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}


def interp(levels: dict[str, np.ndarray], field: str, time_s: float) -> float:
    return float(np.interp(time_s, levels["time_s"], levels[field]))


def draw_outline(ax) -> None:
    left = TOWER_X - DT / 2
    right = TOWER_X + DT / 2
    wall = dict(color="#343a40", linewidth=1.0, zorder=10)
    ax.plot([0, L], [-D, -D], **wall)
    ax.plot([0, 0], [-D, 0], **wall)
    ax.plot([L, L], [-D, 0], **wall)
    ax.plot([0, left], [0, 0], **wall)
    ax.plot([right, L], [0, 0], **wall)
    ax.plot([left, left], [0, TOWER_H], **wall)
    ax.plot([right, right], [0, TOWER_H], **wall)


def project_cell_faces(points: np.ndarray, cells: list[np.ndarray], zoom: bool) -> list[np.ndarray]:
    translated = points.copy()
    translated[:, 1] -= D / 2  # pipe crown is y=0 in both models
    if zoom:
        translated[:, 0] = (translated[:, 0] - (TOWER_X - DT / 2)) / DT
    polygons = []
    for cell in cells:
        # The mesh is one cell thick in z.  Projecting all eight hexahedron
        # vertices directly creates a self-crossing polygon; selecting vertices
        # by index is also unsafe because the VTK ordering alternates z faces.
        # Deduplicate in x-y and order the four corners around their centroid.
        xy = np.unique(translated[cell, :2], axis=0)
        centre = xy.mean(axis=0)
        order = np.argsort(np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0]))
        polygons.append(xy[order])
    return polygons


def render_frame(polygons: list[np.ndarray], alpha: np.ndarray,
                 time_s: float, out_path: Path, zoom: bool) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "water", [(0.0, "#f7f9fc"), (0.12, "#d8ecff"), (1.0, "#2b7fff")]
    )

    if zoom:
        fig, ax = plt.subplots(figsize=(2.6, 6.2), dpi=110)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, TOWER_H)
        ax.set_aspect("auto")
        ax.set_xticks([])
        ax.set_yticks([0, 0.15, 0.30, 0.45, 0.60])
        ax.set_ylabel("height above pipe crown [m]", fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(14.0, 3.6), dpi=130)
        ax.set_xlim(-0.05, L + 0.05)
        ax.set_ylim(-D - 0.04, TOWER_H + 0.10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_yticks([-0.1, 0, 0.3, 0.6])
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("y (m)", fontsize=9)

    collection = PolyCollection(
        polygons, array=np.clip(alpha, 0, 1), cmap=cmap,
        norm=Normalize(0, 1), edgecolors="none", antialiaseds=False,
        rasterized=True, zorder=1,
    )
    ax.add_collection(collection)
    if zoom:
        ax.plot([0, 0], [0, TOWER_H], color="#343a40", linewidth=1.0, zorder=10)
        ax.plot([1, 1], [0, TOWER_H], color="#343a40", linewidth=1.0, zorder=10)
        ax.set_title(f"tower zoom\n2D OpenFOAM, t={time_s:.2f} s", fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    else:
        draw_outline(ax)
        ax.set_title(f"2D OpenFOAM   Time = {time_s:.2f} s", fontsize=10, pad=5)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=8, direction="out", length=2.5)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def rel(path: Path) -> str:
    return path.relative_to(CASE).as_posix()


def load_1d_frames(index_path: Path) -> list[dict]:
    frames = json.loads(index_path.read_text(encoding="utf-8"))
    selected = frames
    for frame in selected:
        t = float(frame["time"])
        surface_height = float(
            frame.get(
                "visibleWaterTop",
                frame.get("materialHeight", frame["wtop"]),
            )
        )
        frame["Tstar"] = t * TSTAR_PER_SECOND
        frame["Yint"] = float(frame["itop"])
        frame["Yfs"] = surface_height
        frame["YintStar"] = float(frame["itop"]) / TOWER_H
        frame["YfsStar"] = surface_height / TOWER_H
    return selected


def build_2d_frames(reuse_existing: bool = False) -> list[dict]:
    FRAME_ROOT.mkdir(parents=True, exist_ok=True)
    levels = read_levels()
    datasets = []
    for path in VTK_ROOT.glob("*/internal.vtu"):
        data = read_vtu(path)
        if -1e-8 <= data[3] <= 13.0 + 1e-8:
            datasets.append(data)
    datasets.sort(key=lambda item: item[3])
    if len(datasets) != 261:
        raise RuntimeError(f"Expected 261 OpenFOAM frames from 0 to 13 s, found {len(datasets)}")

    full_polygons = project_cell_faces(datasets[0][0], datasets[0][1], zoom=False)
    zoom_polygons = project_cell_faces(datasets[0][0], datasets[0][1], zoom=True)
    manifest = []
    for index, (points, cells, alpha, time_s) in enumerate(datasets):
        full = FRAME_ROOT / f"full_{index:04d}.png"
        zoom = FRAME_ROOT / f"zoom_{index:04d}.png"
        reused = reuse_existing and full.is_file() and zoom.is_file()
        if not reused:
            render_frame(full_polygons, alpha, time_s, full, zoom=False)
            render_frame(zoom_polygons, alpha, time_s, zoom, zoom=True)
        yint_star = interp(levels, "Yint_star", time_s)
        yfs_star = interp(levels, "Yfs_star", time_s)
        manifest.append({
            "file": rel(full), "riserFile": rel(zoom), "time": time_s,
            "Tstar": interp(levels, "Tstar", time_s),
            "Yint": yint_star * TOWER_H, "Yfs": yfs_star * TOWER_H,
            "YintStar": yint_star, "YfsStar": yfs_star,
        })
        action = "Reused" if reused else "Rendered"
        print(f"{action} 2D frame {index + 1:02d}/{len(datasets)} at {time_s:.2f} s")
    return manifest


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Case A mixed-flow 1D vs 2D frame comparison</title>
<style>
:root{--ink:#17212b;--muted:#68717d;--line:#d8dee6;--one:#d55e00;--two:#0072b2;--bg:#f3f6f9;--card:#fff}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1500px;margin:auto;padding:18px} h1{font-size:22px;margin:0 0 5px}.note{color:var(--muted);margin-bottom:14px}
.toolbar,.card{background:var(--card);border:1px solid var(--line);border-radius:10px;box-shadow:0 2px 8px #26384c0d}
.toolbar{padding:12px 14px;margin-bottom:14px;display:grid;grid-template-columns:auto auto auto minmax(240px,1fr) auto;gap:10px;align-items:center}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.card{padding:12px;min-width:0}.card h2{font-size:16px;margin:0 0 8px}.one h2{color:var(--one)}.two h2{color:var(--two)}
.viewport{height:310px;border:1px solid #e4e8ed;border-radius:7px;background:#fff;display:flex;align-items:center;justify-content:center;overflow:hidden}
.viewport.zoom{height:570px}.viewport img{width:100%;height:100%;object-fit:contain;display:block}
.controls{display:grid;grid-template-columns:auto auto 1fr auto auto;gap:7px;align-items:center;margin-top:10px}button{border:1px solid #b9c2ce;background:#fff;border-radius:6px;padding:6px 10px;cursor:pointer}button:hover{background:#eef3f7}input[type=range]{width:100%}
.meta{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:9px;color:var(--muted)}.meta b{color:var(--ink)}
.diff{margin-top:14px;padding:10px 12px;background:#eef4f8;border-radius:7px;display:flex;gap:22px;flex-wrap:wrap}
.pairs{margin-top:14px}.pairs table{width:100%;border-collapse:collapse;background:white}.pairs th,.pairs td{border:1px solid var(--line);padding:6px 8px;text-align:center}.pairs th{background:#eef2f6}
@media(max-width:900px){.grid{grid-template-columns:1fr}.toolbar{grid-template-columns:1fr}.viewport{height:240px}.viewport.zoom{height:480px}}
</style>
</head>
<body><main>
<h1>Case A：__MODEL_LABEL__ 与 2D OpenFOAM 逐帧对照</h1>
<div class="note">__MODEL_NOTE__</div>
<section class="toolbar">
  <button id="view">切换为竖管放大</button><button id="sync">按当前 1D 时间同步 2D</button><button id="jump">跳到入塔窗口</button>
  <label>共同物理时间 <input id="master" type="range" min="0" max="13" step="0.01" value="0"></label>
  <button id="record">记录当前配对</button>
</section>
<section class="grid">
 <article class="card one"><h2>__MODEL_LABEL__</h2><div class="viewport"><img id="img1" alt="Present 1D model frame"></div>
  <div class="controls"><button id="prev1">◀</button><button id="play1">播放</button><input id="range1" type="range"><button id="next1">▶</button><span id="count1"></span></div><div id="meta1" class="meta"></div></article>
 <article class="card two"><h2>2D OpenFOAM</h2><div class="viewport"><img id="img2" alt="2D OpenFOAM frame"></div>
  <div class="controls"><button id="prev2">◀</button><button id="play2">播放</button><input id="range2" type="range"><button id="next2">▶</button><span id="count2"></span></div><div id="meta2" class="meta"></div></article>
</section>
<div class="diff"><span id="dt"></span><span id="dy"></span><span>快捷键：A/D 调 1D；←/→ 调 2D；空格同步播放</span></div>
<section class="pairs"><table><thead><tr><th>#</th><th>1D Time (s)</th><th>2D Time (s)</th><th>ΔTime (s)</th><th>1D Yint (m)</th><th>2D Yint (m)</th><th>ΔYint (m)</th><th></th></tr></thead><tbody id="pairRows"></tbody></table></section>
</main>
<script>
const data1=__DATA1__, data2=__DATA2__; let i1=0,i2=0,zoom=false,masterTimer=null,pairs=[];const timers={1:null,2:null};
const $=id=>document.getElementById(id), nearest=(data,t)=>data.reduce((best,x,i)=>Math.abs(x.time-t)<Math.abs(data[best].time-t)?i:best,0);
function fmt(x,n=3){return Number(x).toFixed(n)}
function render(side){const data=side===1?data1:data2, i=side===1?i1:i2, f=data[i];
  $('img'+side).src=zoom?f.riserFile:f.file;$('range'+side).max=data.length-1;$('range'+side).value=i;$('count'+side).textContent=`${i+1}/${data.length}`;
  $('meta'+side).innerHTML=`<span>Time <b>${fmt(f.time,2)} s</b></span><span>T* <b>${fmt(f.Tstar,2)}</b></span><span>Yint <b>${fmt(f.Yint,3)} m</b></span><span>Yfs <b>${fmt(f.Yfs,3)} m</b></span><span>Yint* <b>${fmt(f.YintStar,3)}</b></span><span>Yfs* <b>${fmt(f.YfsStar,3)}</b></span>`;diff();preload(side)}
function preload(side){const data=side===1?data1:data2,i=side===1?i1:i2;[-1,1].forEach(d=>{let j=Math.max(0,Math.min(data.length-1,i+d));new Image().src=zoom?data[j].riserFile:data[j].file})}
function diff(){if(!data1[i1]||!data2[i2])return;$('dt').innerHTML=`ΔTime = <b>${fmt(data1[i1].time-data2[i2].time,2)} s</b>`;$('dy').innerHTML=`ΔYint = <b>${fmt(data1[i1].Yint-data2[i2].Yint,3)} m</b>`}
function setIndex(side,value){if(side===1)i1=Math.max(0,Math.min(data1.length-1,value));else i2=Math.max(0,Math.min(data2.length-1,value));render(side)}
function syncTime(t){i1=nearest(data1,t);i2=nearest(data2,t);render(1);render(2);$('master').value=t}
function togglePlay(side){let btn=$('play'+side);if(timers[side]){clearInterval(timers[side]);timers[side]=null;btn.textContent='播放'}else{timers[side]=setInterval(()=>setIndex(side,(side===1?i1:i2)+1>= (side===1?data1.length:data2.length)?0:(side===1?i1:i2)+1),180);btn.textContent='暂停'}}
$('range1').oninput=e=>setIndex(1,+e.target.value);$('range2').oninput=e=>setIndex(2,+e.target.value);$('prev1').onclick=()=>setIndex(1,i1-1);$('next1').onclick=()=>setIndex(1,i1+1);$('prev2').onclick=()=>setIndex(2,i2-1);$('next2').onclick=()=>setIndex(2,i2+1);$('play1').onclick=()=>togglePlay(1);$('play2').onclick=()=>togglePlay(2);
$('master').oninput=e=>syncTime(+e.target.value);$('sync').onclick=()=>{i2=nearest(data2,data1[i1].time);render(2);$('master').value=data1[i1].time};
$('jump').onclick=()=>syncTime(6.55);
$('view').onclick=()=>{zoom=!zoom;document.querySelectorAll('.viewport').forEach(x=>x.classList.toggle('zoom',zoom));$('view').textContent=zoom?'切换为完整管道':'切换为竖管放大';render(1);render(2)};
$('record').onclick=()=>{pairs.push([data1[i1],data2[i2]]);drawPairs()};function drawPairs(){$('pairRows').innerHTML=pairs.map((p,k)=>`<tr><td>${k+1}</td><td>${fmt(p[0].time,2)}</td><td>${fmt(p[1].time,2)}</td><td>${fmt(p[0].time-p[1].time,2)}</td><td>${fmt(p[0].Yint)}</td><td>${fmt(p[1].Yint)}</td><td>${fmt(p[0].Yint-p[1].Yint)}</td><td><button onclick="pairs.splice(${k},1);drawPairs()">删除</button></td></tr>`).join('')}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key==='a'||e.key==='A')setIndex(1,i1-1);if(e.key==='d'||e.key==='D')setIndex(1,i1+1);if(e.key==='ArrowLeft')setIndex(2,i2-1);if(e.key==='ArrowRight')setIndex(2,i2+1);if(e.code==='Space'){e.preventDefault();if(masterTimer){clearInterval(masterTimer);masterTimer=null}else masterTimer=setInterval(()=>syncTime(+$('master').value>=13?0:+$('master').value+.05),180)}});
syncTime(0);
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-2d-frames", action="store_true",
        help="Reuse existing 2D PNGs while rebuilding the embedded manifests.",
    )
    parser.add_argument(
        "--one-d-index",
        type=Path,
        default=Path("outputs/frames_index.json"),
        help="1D frame manifest, relative to the Case-A folder unless absolute.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HTML_PATH,
        help="Output HTML path, relative to the Case-A folder unless absolute.",
    )
    parser.add_argument(
        "--model-label",
        default="Present 1D model",
    )
    parser.add_argument(
        "--model-note",
        default=(
            "1D 结果采用守恒材料前缘、面中心对齐的 T 结点及局部 Taylor 液膜回流闭合；"
            "竖管按各网格的真实液相体积分数绘制，不使用人为界面线或整段水柱投影。"
            "2D OpenFOAM 原始结果未修改。"
        ),
    )
    args = parser.parse_args()
    one_d_index = args.one_d_index
    if not one_d_index.is_absolute():
        one_d_index = CASE / one_d_index
    output_path = args.output
    if not output_path.is_absolute():
        output_path = CASE / output_path
    one_d = load_1d_frames(one_d_index)
    two_d = build_2d_frames(reuse_existing=args.reuse_2d_frames)
    html = HTML_TEMPLATE.replace("__DATA1__", json.dumps(one_d, ensure_ascii=False))
    html = html.replace("__DATA2__", json.dumps(two_d, ensure_ascii=False))
    html = html.replace("__MODEL_LABEL__", args.model_label)
    html = html.replace("__MODEL_NOTE__", args.model_note)
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

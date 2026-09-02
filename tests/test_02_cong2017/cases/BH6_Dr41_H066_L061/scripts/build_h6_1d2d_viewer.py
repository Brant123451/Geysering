#!/usr/bin/env python3
"""Render the completed B-H6 1D and OpenFOAM 2D runs as a two-pane viewer."""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree


CASE = Path(__file__).resolve().parents[1]
MODEL_DIR = CASE / "model"
VIEWER_DIR = CASE / "outputs" / "1d2d_viewer"
HTML_PATH = CASE / "BH6_1d2d_simulation_render_13s.html"
VTK_ROOT = Path("/tmp/bh6-2d-study/paper_tau0p2_areaeq/VTK_H6_VIEWER")

sys.path.insert(0, str(MODEL_DIR))
from cong2017_network_twofluid import NetworkCase, run_network  # noqa: E402


T_END = 13.0
FRAME_DT = 0.1
TARGET_TIMES = np.round(np.arange(0.0, T_END + 0.5 * FRAME_DT, FRAME_DT), 2)
PIPE_D = 0.05
PIPE_Z0 = -0.025
PIPE_Z1 = 0.025
PIPE_L = 6.59
TEE_X = 3.47
VALVE_X = 5.98
RISER_W_2D = 0.03362
RISER_W_1D = 0.041
RISER_Z0 = 0.025
RISER_Z1 = 1.825
RISER_H = 1.8

WATER_CMAP = LinearSegmentedColormap.from_list(
    "h6_water", ["#f7fafc", "#cfe8ff", "#5aa9ff", "#1268d3"]
)


def nearest_indices(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.asarray([int(np.argmin(np.abs(values - target))) for target in targets])


def rgba(values: np.ndarray) -> np.ndarray:
    return WATER_CMAP(np.clip(values, 0.0, 1.0))


def style_axis(ax: plt.Axes, time_s: float, title: str, zoom: bool = False) -> None:
    ax.set_facecolor("white")
    ax.tick_params(labelsize=8, direction="out", length=2.5, colors="#52606d")
    for spine in ax.spines.values():
        spine.set_color("#9aa5b1")
        spine.set_linewidth(0.7)
    ax.set_title(f"{title}   t = {time_s:0.2f} s", fontsize=10, pad=7, color="#17212b")
    if zoom:
        ax.set_xlabel("enlarged riser width", fontsize=8)
        ax.set_ylabel("height above pipe crown (m)", fontsize=8)
    else:
        ax.set_xlabel("horizontal distance (m)", fontsize=8)
        ax.set_ylabel("height (m)", fontsize=8)


def draw_full_outline(ax: plt.Axes, riser_width: float) -> None:
    left = TEE_X - 0.5 * riser_width
    right = TEE_X + 0.5 * riser_width
    wall = dict(color="#263238", linewidth=0.9, zorder=5)
    ax.plot([0, PIPE_L], [PIPE_Z0, PIPE_Z0], **wall)
    ax.plot([0, 0], [PIPE_Z0, PIPE_Z1], **wall)
    ax.plot([PIPE_L, PIPE_L], [PIPE_Z0, PIPE_Z1], **wall)
    ax.plot([0, left], [PIPE_Z1, PIPE_Z1], **wall)
    ax.plot([right, PIPE_L], [PIPE_Z1, PIPE_Z1], **wall)
    ax.plot([left, left], [PIPE_Z1, RISER_Z1], **wall)
    ax.plot([right, right], [PIPE_Z1, RISER_Z1], **wall)
    ax.axvline(VALVE_X, color="#e45756", linewidth=0.9, linestyle="--", zorder=6)
    ax.text(VALVE_X, 0.085, "valve 5.98 m", ha="center", fontsize=7, color="#a33a39")
    ax.text(TEE_X, RISER_Z1 + 0.035, "riser", ha="center", fontsize=7, color="#52606d")


def save_full(alpha: np.ndarray, path: Path, time_s: float, title: str,
              riser_width: float) -> None:
    fig, ax = plt.subplots(figsize=(14.0, 4.2), dpi=120)
    ax.imshow(
        rgba(alpha), origin="lower", interpolation="nearest",
        extent=(-0.05, PIPE_L + 0.05, -0.07, 1.91), aspect="auto", zorder=1,
    )
    draw_full_outline(ax, riser_width)
    ax.set_xlim(-0.05, PIPE_L + 0.05)
    ax.set_ylim(-0.07, 1.91)
    ax.set_aspect("equal", adjustable="box")
    style_axis(ax, time_s, title)
    fig.tight_layout(pad=0.8)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def save_zoom(alpha: np.ndarray, path: Path, time_s: float, title: str,
              width_m: float) -> None:
    fig, ax = plt.subplots(figsize=(3.0, 6.4), dpi=120)
    ax.imshow(
        rgba(alpha), origin="lower", interpolation="nearest",
        extent=(-0.5 * width_m, 0.5 * width_m, 0.0, RISER_H), aspect="auto",
    )
    ax.plot([-0.5 * width_m, -0.5 * width_m], [0, RISER_H], color="#263238", lw=1)
    ax.plot([0.5 * width_m, 0.5 * width_m], [0, RISER_H], color="#263238", lw=1)
    ax.set_xlim(-0.55 * width_m, 0.55 * width_m)
    ax.set_ylim(0, RISER_H)
    ax.set_xticks([])
    style_axis(ax, time_s, title, zoom=True)
    fig.tight_layout(pad=0.8)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def render_1d() -> list[dict[str, object]]:
    output = VIEWER_DIR / "frames_1d"
    output.mkdir(parents=True, exist_ok=True)
    case = NetworkCase(
        D=PIPE_D, Dr=RISER_W_1D, riser_height=RISER_H,
        L_up=TEE_X, L_mid=VALVE_X - TEE_X, L_down=PIPE_L - VALVE_X,
        x_riser_at=TEE_X, pocket_downstream=True,
        reservoir_head=0.66, air_head=0.0, init_water_level=0.66,
        valve_open_time=0.20, Hop_cap=10.0,
        x_transducer_at=6.44, t_end=T_END,
    )
    rec = run_network(case, verbose=False)
    source_times = np.asarray(rec["frames_t"], dtype=float)
    selected = nearest_indices(source_times, TARGET_TIMES)
    xt = np.asarray(rec["xt"], dtype=float)
    zr = np.asarray(rec["zr"], dtype=float)

    full_w, full_h = 1600, 475
    xs = np.linspace(-0.05, PIPE_L + 0.05, full_w)
    zs = np.linspace(-0.07, 1.91, full_h)
    pipe_x_index = np.clip(np.searchsorted(xt, xs), 0, len(xt) - 1)
    riser_z_index = np.clip(np.searchsorted(zr, zs - RISER_Z0), 0, len(zr) - 1)
    pipe_x_mask = (xs >= 0.0) & (xs <= PIPE_L)
    riser_x_mask = np.abs(xs - TEE_X) <= 0.5 * RISER_W_1D
    pipe_z_mask = (zs >= PIPE_Z0) & (zs <= PIPE_Z1)
    riser_z_mask = (zs >= RISER_Z0) & (zs <= RISER_Z1)

    zoom_w, zoom_h = 230, 900
    x_norm = np.linspace(-0.5, 0.5, zoom_w)
    z_zoom = np.linspace(0.0, RISER_H, zoom_h)
    zoom_z_index = np.clip(np.searchsorted(zr, z_zoom), 0, len(zr) - 1)

    manifest: list[dict[str, object]] = []
    for frame_no, source_index in enumerate(selected):
        time_s = float(source_times[source_index])
        alt = np.clip(np.asarray(rec["frames_alt"][source_index]), 0.0, 1.0)
        agr = np.clip(np.asarray(rec["frames_agr"][source_index]), 0.0, 1.0)
        wtop = float(rec["wtop"][source_index])

        full = np.zeros((full_h, full_w), dtype=np.float32)
        pipe_fill = alt[pipe_x_index]
        for row in np.flatnonzero(pipe_z_mask):
            fraction = (zs[row] - PIPE_Z0) / PIPE_D
            full[row, pipe_x_mask] = (fraction <= pipe_fill[pipe_x_mask]).astype(np.float32)
        for row in np.flatnonzero(riser_z_mask & (zs <= RISER_Z0 + wtop)):
            gas_fraction = float(agr[riser_z_index[row]])
            column = np.flatnonzero(riser_x_mask)
            if column.size:
                normalized = (xs[column] - TEE_X) / RISER_W_1D
                full[row, column] = (np.abs(normalized) >= 0.5 * gas_fraction).astype(np.float32)

        zoom = np.zeros((zoom_h, zoom_w), dtype=np.float32)
        for row in np.flatnonzero(z_zoom <= wtop):
            gas_fraction = float(agr[zoom_z_index[row]])
            zoom[row, :] = (np.abs(x_norm) >= 0.5 * gas_fraction).astype(np.float32)

        full_path = output / f"full_{frame_no:04d}.png"
        zoom_path = output / f"zoom_{frame_no:04d}.png"
        save_full(full, full_path, time_s, "Present 1D model", RISER_W_1D)
        save_zoom(zoom, zoom_path, time_s, "Present 1D model", RISER_W_1D)
        manifest.append({
            "time": round(float(TARGET_TIMES[frame_no]), 2),
            "sourceTime": round(time_s, 4),
            "file": full_path.relative_to(CASE).as_posix(),
            "zoomFile": zoom_path.relative_to(CASE).as_posix(),
        })
        if frame_no % 20 == 0 or frame_no + 1 == len(selected):
            print(f"1D frames: {frame_no + 1}/{len(selected)}")
    return manifest


def vtk_series() -> list[tuple[float, Path]]:
    series_path = next(VTK_ROOT.glob("*.vtm.series"))
    payload = json.loads(series_path.read_text())
    return sorted(
        (float(item["time"]), VTK_ROOT / Path(item["name"]).with_suffix("") / "internal.vtu")
        for item in payload["files"]
    )


def data_array(root: ET.Element, name: str) -> np.ndarray:
    node = root.find(f".//CellData/DataArray[@Name='{name}']")
    if node is None or node.text is None:
        raise ValueError(f"Missing {name}")
    return np.fromstring(node.text, sep=" ")


def mesh_and_alpha(path: Path, need_mesh: bool) -> tuple[np.ndarray | None, np.ndarray]:
    root = ET.parse(path).getroot()
    alpha = data_array(root, "alpha.water")
    if not need_mesh:
        return None, alpha
    points_node = root.find(".//Points/DataArray")
    if points_node is None or points_node.text is None:
        raise ValueError(f"Missing points in {path}")
    points = np.fromstring(points_node.text, sep=" ").reshape(-1, 3)
    arrays = {node.attrib.get("Name"): node for node in root.findall(".//Cells/DataArray")}
    connectivity = np.fromstring(arrays["connectivity"].text or "", sep=" ", dtype=int)
    offsets = np.fromstring(arrays["offsets"].text or "", sep=" ", dtype=int)
    starts = np.r_[0, offsets[:-1]]
    centres = np.asarray([points[connectivity[a:b]][:, (0, 2)].mean(axis=0)
                          for a, b in zip(starts, offsets)])
    return centres, alpha


def render_2d() -> list[dict[str, object]]:
    output = VIEWER_DIR / "frames_2d"
    output.mkdir(parents=True, exist_ok=True)
    series = vtk_series()
    times = np.asarray([entry[0] for entry in series])
    selected = nearest_indices(times, TARGET_TIMES)

    centres, first_alpha = mesh_and_alpha(series[selected[0]][1], need_mesh=True)
    assert centres is not None
    tree = cKDTree(centres)

    full_w, full_h = 1600, 475
    xs = np.linspace(-0.05, PIPE_L + 0.05, full_w)
    zs = np.linspace(-0.07, 1.91, full_h)
    xx, zz = np.meshgrid(xs, zs)
    full_mask = (((xx >= 0) & (xx <= PIPE_L) & (zz >= PIPE_Z0) & (zz <= PIPE_Z1)) |
                 ((np.abs(xx - TEE_X) <= 0.5 * RISER_W_2D) &
                  (zz >= RISER_Z0) & (zz <= RISER_Z1)))
    full_map = np.full(xx.shape, -1, dtype=np.int32)
    full_map[full_mask] = tree.query(np.c_[xx[full_mask], zz[full_mask]], workers=-1)[1]

    zoom_w, zoom_h = 230, 900
    x_zoom = np.linspace(TEE_X - 0.5 * RISER_W_2D, TEE_X + 0.5 * RISER_W_2D, zoom_w)
    z_zoom = np.linspace(RISER_Z0, RISER_Z1, zoom_h)
    zx, zz2 = np.meshgrid(x_zoom, z_zoom)
    zoom_map = tree.query(np.c_[zx.ravel(), zz2.ravel()], workers=-1)[1].reshape(zx.shape)

    manifest: list[dict[str, object]] = []
    for frame_no, source_index in enumerate(selected):
        time_s, path = series[source_index]
        if frame_no == 0:
            alpha = first_alpha
        else:
            _, alpha = mesh_and_alpha(path, need_mesh=False)
        alpha = np.clip(alpha, 0.0, 1.0)
        full = np.zeros(full_map.shape, dtype=np.float32)
        full[full_mask] = alpha[full_map[full_mask]]
        zoom = alpha[zoom_map].astype(np.float32)

        full_path = output / f"full_{frame_no:04d}.png"
        zoom_path = output / f"zoom_{frame_no:04d}.png"
        save_full(full, full_path, time_s, "2D OpenFOAM", RISER_W_2D)
        save_zoom(zoom, zoom_path, time_s, "2D OpenFOAM", RISER_W_2D)
        manifest.append({
            "time": round(float(TARGET_TIMES[frame_no]), 2),
            "sourceTime": round(float(time_s), 4),
            "file": full_path.relative_to(CASE).as_posix(),
            "zoomFile": zoom_path.relative_to(CASE).as_posix(),
        })
        if frame_no % 20 == 0 or frame_no + 1 == len(selected):
            print(f"2D frames: {frame_no + 1}/{len(selected)}")
    return manifest


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>B-H6 1D–2D simulation rendering</title>
<style>
:root{--ink:#17212b;--muted:#66717d;--line:#d8e0e8;--bg:#f4f7fa;--card:#fff;--one:#d85b24;--two:#1769aa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1580px;margin:auto;padding:18px}header{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:13px}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted)}.status{background:#e7f6ec;color:#24723a;border:1px solid #bfe3ca;border-radius:999px;padding:5px 10px;white-space:nowrap}
.toolbar,.card{background:var(--card);border:1px solid var(--line);border-radius:11px;box-shadow:0 2px 9px #26384c0d}.toolbar{padding:11px 13px;margin-bottom:14px;display:grid;grid-template-columns:auto auto auto minmax(260px,1fr) auto auto;gap:9px;align-items:center}
button{border:1px solid #b8c3cf;background:#fff;border-radius:7px;padding:7px 11px;cursor:pointer;color:var(--ink)}button:hover{background:#edf3f7}input[type=range]{width:100%}.time{font-variant-numeric:tabular-nums;font-weight:650;min-width:62px;text-align:right}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.card{padding:12px;min-width:0}.card-head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}.card h2{font-size:16px;margin:0}.one h2{color:var(--one)}.two h2{color:var(--two)}.source{color:var(--muted);font-size:12px}
.viewport{height:440px;border:1px solid #e3e8ee;border-radius:8px;background:#fff;display:flex;align-items:center;justify-content:center;overflow:hidden}.viewport.zoom{height:650px}.viewport img{width:100%;height:100%;object-fit:contain;display:block}
.frame-time{margin-top:8px;color:var(--muted);text-align:center;font-variant-numeric:tabular-nums}
@media(max-width:920px){header{display:block}.status{display:inline-block;margin-top:8px}.toolbar{grid-template-columns:auto auto auto 1fr}.grid{grid-template-columns:1fr}.viewport{height:300px}.viewport.zoom{height:560px}}
</style></head><body><main>
<header><div><h1>Cong 2017 B-H6：1D 与 OpenFOAM 2D 同步渲染</h1><div class="sub">论文 Fig. 1(b) 布置 · D=50 mm · Dr=41 mm · H0=0.66 m · L0=0.61 m · 0–13 s</div></div><div class="status">模拟已完成</div></header>
<section class="toolbar"><button id="play">播放</button><button id="prev">◀</button><button id="next">▶</button><input id="slider" type="range" min="0" max="130" step="1" value="0"><span id="time" class="time">0.00 s</span><button id="view">立管放大</button></section>
<section class="grid">
 <article class="card one"><div class="card-head"><h2>Present 1D model</h2><span class="source">geometry-matched 1D</span></div><div class="viewport"><img id="img1" alt="H6 1D simulation frame"></div><div id="meta1" class="frame-time"></div></article>
 <article class="card two"><div class="card-head"><h2>2D OpenFOAM</h2><span class="source">completed planar run</span></div><div class="viewport"><img id="img2" alt="H6 OpenFOAM 2D simulation frame"></div><div id="meta2" class="frame-time"></div></article>
</section></main><script>
const one=__ONE__,two=__TWO__;let index=0,playing=null,zoom=false;const $=id=>document.getElementById(id);
function draw(){const a=one[index],b=two[index];$('img1').src=zoom?a.zoomFile:a.file;$('img2').src=zoom?b.zoomFile:b.file;$('slider').value=index;$('time').textContent=a.time.toFixed(2)+' s';$('meta1').textContent=`frame ${index+1}/${one.length} · source time ${a.sourceTime.toFixed(2)} s`;$('meta2').textContent=`frame ${index+1}/${two.length} · source time ${b.sourceTime.toFixed(2)} s`;[-1,1].forEach(d=>{const j=Math.max(0,Math.min(one.length-1,index+d));new Image().src=zoom?one[j].zoomFile:one[j].file;new Image().src=zoom?two[j].zoomFile:two[j].file})}
function setIndex(i){index=(i+one.length)%one.length;draw()}function toggle(){if(playing){clearInterval(playing);playing=null;$('play').textContent='播放'}else{playing=setInterval(()=>setIndex(index+1),140);$('play').textContent='暂停'}}
$('play').onclick=toggle;$('prev').onclick=()=>setIndex(index-1);$('next').onclick=()=>setIndex(index+1);$('slider').oninput=e=>setIndex(+e.target.value);$('view').onclick=()=>{zoom=!zoom;document.querySelectorAll('.viewport').forEach(v=>v.classList.toggle('zoom',zoom));$('view').textContent=zoom?'完整管道':'立管放大';draw()};document.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();toggle()}if(e.key==='ArrowLeft')setIndex(index-1);if(e.key==='ArrowRight')setIndex(index+1)});draw();
</script></body></html>'''


def write_html(one: list[dict[str, object]], two: list[dict[str, object]]) -> None:
    if len(one) != len(two):
        raise ValueError(f"Frame-count mismatch: 1D={len(one)}, 2D={len(two)}")
    html = HTML.replace("__ONE__", json.dumps(one, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("__TWO__", json.dumps(two, ensure_ascii=False, separators=(",", ":")))
    HTML_PATH.write_text(html, encoding="utf-8")
    (VIEWER_DIR / "frames_1d.json").write_text(json.dumps(one, indent=2) + "\n")
    (VIEWER_DIR / "frames_2d.json").write_text(json.dumps(two, indent=2) + "\n")
    print(f"HTML: {HTML_PATH}")


def load_manifest(name: str) -> list[dict[str, object]]:
    return json.loads((VIEWER_DIR / name).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("1d", "2d", "html", "all"), default="all")
    args = parser.parse_args()
    VIEWER_DIR.mkdir(parents=True, exist_ok=True)
    one = render_1d() if args.only in ("1d", "all") else load_manifest("frames_1d.json")
    two = render_2d() if args.only in ("2d", "all") else (
        load_manifest("frames_2d.json") if (VIEWER_DIR / "frames_2d.json").exists() else []
    )
    if args.only == "1d":
        (VIEWER_DIR / "frames_1d.json").write_text(json.dumps(one, indent=2) + "\n")
    elif args.only == "2d":
        (VIEWER_DIR / "frames_2d.json").write_text(json.dumps(two, indent=2) + "\n")
        one = load_manifest("frames_1d.json")
        write_html(one, two)
    else:
        write_html(one, two)


if __name__ == "__main__":
    main()

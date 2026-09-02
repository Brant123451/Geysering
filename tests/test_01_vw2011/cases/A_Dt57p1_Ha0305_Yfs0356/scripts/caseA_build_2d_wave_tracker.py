"""Build an offline diagnostic viewer for the Case-A 2-D interface wave.

The tracker reads ``alpha.water`` from the archived OpenFOAM VTU fields.  It
does not infer the interface from PNG colours.  For each axial mesh column,
the equivalent liquid depth is the area-weighted liquid volume divided by the
planar column area.  A short Savitzky--Golay window removes cell-scale VOF
stair-stepping only for *diagnostic display*; a longer window defines the
slowly varying background used to quantify the undular residual.

The generated HTML keeps the original 2-D frame and the extracted profile in
lock-step so that the physical feature can be checked before changing the 1-D
model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter


HERE = Path(__file__).resolve().parent
CASE = HERE.parent
VTK_ROOT = CASE / "openfoam/2d/VTK_CASEA_HTML"
FRAME_ROOT = CASE / "openfoam/2d/outputs/html_frames"
OUTPUT = CASE / "caseA_2d_horizontal_wave_tracking.html"
METRICS = CASE / "outputs/caseA_openfoam2d_wave_tracking.json"

D = 0.094
L_TUNNEL = 4.006
TOWER_X = 3.516
TOWER_LEFT = TOWER_X - 0.0571 / 2.0
ROI = (2.45, 3.45)
RESOLVE_LENGTH = 0.025
TREND_LENGTH = 0.30
PEAK_PROMINENCE = 0.0015
PEAK_SEPARATION = 0.08
ACTIVE_PTP = 0.004


def _array(node: ET.Element, ncomp: int = 1) -> np.ndarray:
    if node.attrib.get("format") != "ascii":
        raise ValueError("Expected ASCII foamToVTK output")
    values = np.fromstring(node.text or "", sep=" ")
    return values.reshape(-1, ncomp) if ncomp > 1 else values


def _read_field(path: Path, *, geometry: bool) -> dict[str, object]:
    root = ET.parse(path).getroot()
    time_node = root.find(".//FieldData/DataArray[@Name='TimeValue']")
    alpha_node = root.find(".//CellData/DataArray[@Name='alpha.water']")
    if time_node is None or alpha_node is None:
        raise ValueError(f"Missing TimeValue or alpha.water in {path}")
    result: dict[str, object] = {
        "time": float((time_node.text or "0").strip()),
        "alpha": np.clip(_array(alpha_node), 0.0, 1.0),
    }
    if not geometry:
        return result

    points_node = root.find(".//Points/DataArray")
    if points_node is None:
        raise ValueError(f"Missing points in {path}")
    points = _array(points_node, 3)
    arrays = {
        node.attrib.get("Name"): node
        for node in root.findall(".//Cells/DataArray")
    }
    connectivity = _array(arrays["connectivity"]).astype(int)
    offsets = _array(arrays["offsets"]).astype(int)
    cells = [
        connectivity[i:j]
        for i, j in zip(np.r_[0, offsets[:-1]], offsets)
    ]
    centres = np.empty((len(cells), 2), dtype=float)
    areas = np.empty(len(cells), dtype=float)
    for index, cell in enumerate(cells):
        xy = np.unique(points[cell, :2], axis=0)
        centre = xy.mean(axis=0)
        order = np.argsort(
            np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0])
        )
        polygon = xy[order]
        x = polygon[:, 0]
        y = polygon[:, 1]
        centres[index] = centre
        areas[index] = 0.5 * abs(
            float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        )
    result.update(centres=centres, areas=areas)
    return result


def _all_vtu_paths() -> list[tuple[float, Path]]:
    found: list[tuple[float, Path]] = []
    for path in VTK_ROOT.glob("*/internal.vtu"):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            header = handle.read(512)
        match = re.search(r"\btime='([^']+)'", header)
        if match is None:
            raise ValueError(f"Cannot read time from VTU header: {path}")
        time = float(match.group(1))
        if -1.0e-8 <= time <= 13.0 + 1.0e-8:
            found.append((time, path))
    found.sort(key=lambda item: item[0])
    if len(found) != 261:
        raise RuntimeError(
            f"Expected 261 OpenFOAM frames from 0 to 13 s, found {len(found)}"
        )
    return found


def _odd_window(length: float, dx: float, n: int, minimum: int) -> int:
    value = max(minimum, int(round(length / dx)))
    if value % 2 == 0:
        value += 1
    maximum = n if n % 2 else n - 1
    return min(value, maximum)


def _dominant_mode(
    x: np.ndarray,
    residual: np.ndarray,
) -> tuple[float | None, float]:
    centred = residual - float(np.mean(residual))
    if float(np.max(np.abs(centred))) < 1.0e-10:
        return None, 0.0
    dx = float(np.median(np.diff(x)))
    spectrum = np.fft.rfft(centred * np.hanning(len(centred)))
    frequency = np.fft.rfftfreq(len(centred), d=dx)
    power = np.abs(spectrum) ** 2
    use = (frequency > 0.0) & (frequency >= 1.0 / 0.8) & (
        frequency <= 1.0 / 0.08
    )
    if not np.any(use) or float(np.sum(power[use])) <= 0.0:
        return None, 0.0
    candidates = np.flatnonzero(use)
    best = int(candidates[np.argmax(power[use])])
    return float(1.0 / frequency[best]), float(power[best] / np.sum(power[use]))


def _compact(values: np.ndarray) -> list[float]:
    return np.round(np.asarray(values, dtype=float), 6).tolist()


def build_payload() -> tuple[dict[str, object], dict[str, object]]:
    paths = _all_vtu_paths()
    first = _read_field(paths[0][1], geometry=True)
    centres = np.asarray(first["centres"], dtype=float)
    areas = np.asarray(first["areas"], dtype=float)
    pipe = (
        (centres[:, 0] >= -1.0e-8)
        & (centres[:, 0] <= L_TUNNEL + 1.0e-8)
        & (centres[:, 1] >= -0.5 * D - 1.0e-8)
        & (centres[:, 1] <= 0.5 * D + 1.0e-8)
    )
    x, column_index = np.unique(
        np.round(centres[pipe, 0], 8), return_inverse=True
    )
    column_area = np.bincount(column_index, weights=areas[pipe])
    dx = float(np.median(np.diff(x)))
    resolve_window = _odd_window(RESOLVE_LENGTH, dx, len(x), minimum=5)
    trend_window = _odd_window(TREND_LENGTH, dx, len(x), minimum=9)
    peak_distance = max(1, int(round(PEAK_SEPARATION / dx)))
    roi = (x >= ROI[0]) & (x <= ROI[1])

    frames: list[dict[str, object]] = []
    metrics_frames: list[dict[str, object]] = []
    for index, (time_hint, path) in enumerate(paths):
        field = first if index == 0 else _read_field(path, geometry=False)
        alpha = np.asarray(field["alpha"], dtype=float)
        liquid_fraction = np.bincount(
            column_index, weights=alpha[pipe] * areas[pipe]
        ) / column_area
        depth = D * np.clip(liquid_fraction, 0.0, 1.0)
        resolved = savgol_filter(
            depth, resolve_window, 2, mode="interp"
        )
        trend = savgol_filter(
            resolved, trend_window, 2, mode="interp"
        )
        residual = resolved - trend
        residual_roi = residual[roi]
        x_roi = x[roi]
        peaks, peak_data = find_peaks(
            residual_roi,
            prominence=PEAK_PROMINENCE,
            distance=peak_distance,
        )
        troughs, trough_data = find_peaks(
            -residual_roi,
            prominence=PEAK_PROMINENCE,
            distance=peak_distance,
        )
        wavelength, mode_fraction = _dominant_mode(x_roi, residual_roi)
        p2p = float(np.ptp(residual_roi))
        rms = float(np.sqrt(np.mean(residual_roi * residual_roi)))
        time = float(field["time"])
        if abs(time - time_hint) > 1.0e-5:
            raise ValueError(f"VTU header/data time mismatch in {path}")
        image = FRAME_ROOT / f"full_{index:04d}.png"
        if not image.is_file():
            raise FileNotFoundError(f"Missing rendered 2-D frame: {image}")
        frame_metrics: dict[str, object] = {
            "index": index,
            "time_s": time,
            "residual_peak_to_peak_m": p2p,
            "residual_rms_m": rms,
            "dominant_wavelength_m": wavelength,
            "dominant_mode_energy_fraction": mode_fraction,
            "crest_count": int(len(peaks)),
            "trough_count": int(len(troughs)),
            "crest_x_m": _compact(x_roi[peaks]),
            "trough_x_m": _compact(x_roi[troughs]),
            "crest_prominence_m": _compact(peak_data["prominences"]),
            "trough_prominence_m": _compact(trough_data["prominences"]),
            "wave_detected": bool(p2p >= ACTIVE_PTP),
        }
        metrics_frames.append(frame_metrics)
        frames.append(
            {
                **frame_metrics,
                "image": image.relative_to(CASE).as_posix(),
                "depth_m": _compact(depth),
                "resolved_m": _compact(resolved),
                "trend_m": _compact(trend),
            }
        )
        if index % 20 == 0 or index == len(paths) - 1:
            print(
                f"Extracted {index + 1:03d}/{len(paths)}: "
                f"Time={time:.2f} s, wave p-p={1e3 * p2p:.2f} mm"
            )

    method = {
        "source": "OpenFOAM VTK_CASEA_HTML alpha.water cell field",
        "interface_definition": (
            "planar-pipe equivalent liquid depth h=D*column-area-weighted(alpha.water)"
        ),
        "diagnostic_resolve_window_m": RESOLVE_LENGTH,
        "diagnostic_resolve_window_cells": resolve_window,
        "background_trend_window_m": TREND_LENGTH,
        "background_trend_window_cells": trend_window,
        "wave_region_m": list(ROI),
        "crest_prominence_m": PEAK_PROMINENCE,
        "minimum_crest_separation_m": PEAK_SEPARATION,
        "wave_detection_peak_to_peak_threshold_m": ACTIVE_PTP,
        "note": (
            "Smoothing is used only to identify the archived 2-D interface; "
            "it is not applied to either numerical solver."
        ),
    }
    payload: dict[str, object] = {
        "method": method,
        "diameter_m": D,
        "tunnel_length_m": L_TUNNEL,
        "tower_x_m": TOWER_X,
        "tower_left_m": TOWER_LEFT,
        "x_m": _compact(x),
        "frames": frames,
    }
    metrics: dict[str, object] = {
        "method": method,
        "frame_count": len(frames),
        "time_interval_s": [float(paths[0][0]), float(paths[-1][0])],
        "native_axial_spacing_m": dx,
        "frames": metrics_frames,
    }
    return payload, metrics


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Case A — 2D横管界面波动态识别</title>
<style>
:root{--ink:#18212b;--muted:#687381;--line:#d9e0e8;--blue:#2478d4;--orange:#e46f1a;--trend:#6d7480;--roi:#fff1c7;--card:#fff;--bg:#f3f6f9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Times New Roman","Microsoft YaHei",serif}
main{max-width:1500px;margin:auto;padding:18px}h1{font-size:23px;margin:0 0 5px}.subtitle{color:var(--muted);margin:0 0 14px}
.toolbar,.card{background:var(--card);border:1px solid var(--line);border-radius:10px;box-shadow:0 2px 9px #26384c0d}.toolbar{padding:12px 14px;display:grid;grid-template-columns:auto auto auto minmax(300px,1fr) auto;gap:9px;align-items:center;margin-bottom:14px}
button{border:1px solid #b8c3cf;background:#fff;border-radius:6px;padding:6px 11px;cursor:pointer}button:hover{background:#edf3f8}input[type=range]{width:100%}
.grid{display:grid;grid-template-columns:1.08fr .92fr;gap:14px}.card{padding:12px;min-width:0}.card h2{font-size:16px;margin:0 0 8px}.raw-wrap{height:360px;border:1px solid #e3e8ee;border-radius:7px;display:flex;align-items:center;background:#fff;overflow:hidden}.raw-wrap img{width:100%;height:100%;object-fit:contain}
.canvas-wrap{height:360px;border:1px solid #e3e8ee;border-radius:7px;background:#fff}.canvas-wrap canvas,.timeline canvas{width:100%;height:100%;display:block}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px}.metric{background:#f7f9fb;border:1px solid #e8edf2;border-radius:6px;padding:7px 9px;color:var(--muted)}.metric b{display:block;color:var(--ink);font-size:16px}.metric.status.detected b{color:#b54b00}.metric.status.quiet b{color:#53606d}
.legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);margin-top:9px}.sw{display:inline-block;width:24px;height:0;border-top:3px solid;margin-right:6px;vertical-align:middle}.sw.raw{border-color:var(--blue);border-top-width:1px}.sw.resolved{border-color:var(--orange)}.sw.trend{border-color:var(--trend);border-top-style:dashed;border-top-width:2px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#c2185b;margin-right:6px}
.timeline{height:180px;margin-top:14px;padding:10px;background:#fff;border:1px solid var(--line);border-radius:10px}.method{margin-top:12px;padding:10px 12px;background:#eef4f8;border-radius:7px;color:#45515e}.method b{color:var(--ink)}
@media(max-width:900px){.toolbar{grid-template-columns:1fr}.grid{grid-template-columns:1fr}.raw-wrap,.canvas-wrap{height:270px}.metrics{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body><main>
<h1>Case A：2D横管气–水界面波动态识别</h1>
<p class="subtitle">左侧为未经修改的 OpenFOAM 相分数帧；右侧曲线由同一帧的 <i>alpha.water</i> 单元场直接提取。黄色区域是竖管左侧的波动核对区。</p>
<section class="toolbar">
  <button id="prev">◀ 上一帧</button><button id="play">播放</button><button id="next">下一帧 ▶</button>
  <label>Time = <b id="timeLabel">0.00 s</b> <input id="slider" type="range" min="0" step="1" value="0"></label>
  <button id="jump">跳到 Time = 9.35 s</button>
</section>
<section class="grid">
  <article class="card"><h2>原始2D OpenFOAM场</h2><div class="raw-wrap"><img id="raw" alt="OpenFOAM 2D alpha.water frame"></div></article>
  <article class="card"><h2>自动提取的横管界面</h2><div class="canvas-wrap"><canvas id="profile"></canvas></div>
    <div class="legend"><span><i class="sw raw"></i>守恒等效液深（原始）</span><span><i class="sw resolved"></i>25 mm尺度识别线</span><span><i class="sw trend"></i>300 mm背景趋势</span><span><i class="dot"></i>识别的波峰/波谷</span></div>
  </article>
</section>
<section class="metrics">
  <div id="status" class="metric status"><span>识别状态</span><b>—</b></div>
  <div class="metric"><span>去趋势峰–峰值</span><b id="p2p">—</b></div>
  <div class="metric"><span>去趋势 RMS</span><b id="rms">—</b></div>
  <div class="metric"><span>主导波长</span><b id="waveLength">—</b></div>
  <div class="metric"><span>波峰 / 波谷数</span><b id="counts">—</b></div>
  <div class="metric"><span>单一主模态能量占比</span><b id="mode">—</b></div>
  <div class="metric"><span>分析区间</span><b>2.45–3.45 m</b></div>
  <div class="metric"><span>竖管左壁</span><b>3.487 m</b></div>
</section>
<div class="timeline"><canvas id="history"></canvas></div>
<div class="method"><b>识别方法：</b>每个轴向网格列对 <i>alpha.water</i> 做面积加权，得到守恒等效液深；25 mm窗口只去除VOF网格台阶，300 mm窗口只用于分离背景趋势。两种处理均仅用于诊断显示，没有回写1D或2D求解结果。</div>
</main>
<script>
const payload=__PAYLOAD__;const frames=payload.frames,x=payload.x_m,D=payload.diameter_m,roi=payload.method.wave_region_m;let index=0,timer=null;
const $=id=>document.getElementById(id);$('slider').max=frames.length-1;
function sizeCanvas(canvas){const r=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(1,Math.round(r.width*dpr)),h=Math.max(1,Math.round(r.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);return {c,w:r.width,h:r.height}}
function path(c,values,X,Y,color,width,dash=[]){c.save();c.beginPath();for(let j=0;j<x.length;j++){const px=X(x[j]),py=Y(values[j]);if(j===0)c.moveTo(px,py);else c.lineTo(px,py)}c.strokeStyle=color;c.lineWidth=width;c.setLineDash(dash);c.stroke();c.restore()}
function drawProfile(){const f=frames[index],{c,w,h}=sizeCanvas($('profile')),p={l:48,r:16,t:18,b:38};c.clearRect(0,0,w,h);const X=v=>p.l+(v/payload.tunnel_length_m)*(w-p.l-p.r),Y=v=>h-p.b-(v/D)*(h-p.t-p.b);
 c.fillStyle='#fff1c7';c.fillRect(X(roi[0]),p.t,X(roi[1])-X(roi[0]),h-p.t-p.b);c.strokeStyle='#dce3ea';c.lineWidth=1;for(let k=0;k<=4;k++){const yy=D*k/4;c.beginPath();c.moveTo(p.l,Y(yy));c.lineTo(w-p.r,Y(yy));c.stroke();c.fillStyle='#56616e';c.font='12px Times New Roman';c.textAlign='right';c.fillText((yy*1000).toFixed(0),p.l-7,Y(yy)+4)}for(let k=0;k<=4;k++){const xx=k;c.beginPath();c.moveTo(X(xx),p.t);c.lineTo(X(xx),h-p.b);c.stroke();c.textAlign='center';c.fillText(xx.toFixed(0),X(xx),h-p.b+18)}
 c.strokeStyle='#202833';c.lineWidth=1.2;c.strokeRect(p.l,p.t,w-p.l-p.r,h-p.t-p.b);path(c,f.depth_m,X,Y,'#2478d4',1);path(c,f.trend_m,X,Y,'#6d7480',1.7,[6,4]);path(c,f.resolved_m,X,Y,'#e46f1a',2.4);
 const mark=(xs,color)=>{c.fillStyle=color;for(const xx of xs){let j=0,b=Infinity;for(let q=0;q<x.length;q++){const d=Math.abs(x[q]-xx);if(d<b){b=d;j=q}}c.beginPath();c.arc(X(x[j]),Y(f.resolved_m[j]),3.2,0,2*Math.PI);c.fill()}};mark(f.crest_x_m,'#c2185b');mark(f.trough_x_m,'#c2185b');
 c.fillStyle='#27313c';c.textAlign='center';c.font='13px Times New Roman';c.fillText('Horizontal distance, x (m)',(p.l+w-p.r)/2,h-8);c.save();c.translate(14,(p.t+h-p.b)/2);c.rotate(-Math.PI/2);c.fillText('Equivalent liquid depth, h (mm)',0,0);c.restore();c.textAlign='left';c.fillStyle='#8b6a00';c.fillText('wave-identification region',X(roi[0])+7,p.t+15)}
function drawHistory(){const {c,w,h}=sizeCanvas($('history')),p={l:48,r:16,t:18,b:34},vals=frames.map(f=>1000*f.residual_peak_to_peak_m),vmax=Math.max(8,Math.ceil(Math.max(...vals)/2)*2),X=v=>p.l+(v/13)*(w-p.l-p.r),Y=v=>h-p.b-(v/vmax)*(h-p.t-p.b);c.clearRect(0,0,w,h);c.fillStyle='#fff';c.fillRect(0,0,w,h);c.strokeStyle='#dce3ea';for(let k=0;k<=4;k++){const yy=vmax*k/4;c.beginPath();c.moveTo(p.l,Y(yy));c.lineTo(w-p.r,Y(yy));c.stroke();c.fillStyle='#56616e';c.textAlign='right';c.font='11px Times New Roman';c.fillText(yy.toFixed(0),p.l-7,Y(yy)+4)}c.beginPath();frames.forEach((f,j)=>{const px=X(f.time_s),py=Y(vals[j]);if(j===0)c.moveTo(px,py);else c.lineTo(px,py)});c.strokeStyle='#e46f1a';c.lineWidth=2;c.stroke();c.setLineDash([5,4]);c.beginPath();c.moveTo(p.l,Y(1000*payload.method.wave_detection_peak_to_peak_threshold_m));c.lineTo(w-p.r,Y(1000*payload.method.wave_detection_peak_to_peak_threshold_m));c.strokeStyle='#8f98a3';c.stroke();c.setLineDash([]);c.beginPath();c.moveTo(X(frames[index].time_s),p.t);c.lineTo(X(frames[index].time_s),h-p.b);c.strokeStyle='#c2185b';c.lineWidth=1.5;c.stroke();c.fillStyle='#27313c';c.textAlign='center';c.font='12px Times New Roman';c.fillText('Time (s)',(p.l+w-p.r)/2,h-7);c.save();c.translate(13,(p.t+h-p.b)/2);c.rotate(-Math.PI/2);c.fillText('wave p-p (mm)',0,0);c.restore()}
function render(){const f=frames[index];$('raw').src=f.image;$('slider').value=index;$('timeLabel').textContent=f.time_s.toFixed(2)+' s';$('p2p').textContent=(1000*f.residual_peak_to_peak_m).toFixed(2)+' mm';$('rms').textContent=(1000*f.residual_rms_m).toFixed(2)+' mm';$('waveLength').textContent=f.dominant_wavelength_m==null?'—':f.dominant_wavelength_m.toFixed(3)+' m';$('counts').textContent=f.crest_count+' / '+f.trough_count;$('mode').textContent=(100*f.dominant_mode_energy_fraction).toFixed(1)+'%';const s=$('status');s.className='metric status '+(f.wave_detected?'detected':'quiet');s.querySelector('b').textContent=f.wave_detected?'检测到界面波':'低于识别阈值';drawProfile();drawHistory();preload()}
function preload(){[-1,1].forEach(d=>{const j=Math.max(0,Math.min(frames.length-1,index+d));new Image().src=frames[j].image})}function setIndex(v){index=Math.max(0,Math.min(frames.length-1,Math.round(v)));render()}function toggle(){if(timer){clearInterval(timer);timer=null;$('play').textContent='播放'}else{timer=setInterval(()=>setIndex(index+1>=frames.length?0:index+1),170);$('play').textContent='暂停'}}
$('prev').onclick=()=>setIndex(index-1);$('next').onclick=()=>setIndex(index+1);$('play').onclick=toggle;$('slider').oninput=e=>setIndex(+e.target.value);$('jump').onclick=()=>setIndex(frames.reduce((b,f,j)=>Math.abs(f.time_s-9.35)<Math.abs(frames[b].time_s-9.35)?j:b,0));window.addEventListener('resize',render);document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')setIndex(index-1);if(e.key==='ArrowRight')setIndex(index+1);if(e.code==='Space'){e.preventDefault();toggle()}});render();
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else CASE / args.output
    metrics_path = args.metrics if args.metrics.is_absolute() else CASE / args.metrics
    payload, metrics = build_payload()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    output.write_text(
        HTML_TEMPLATE.replace(
            "__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()

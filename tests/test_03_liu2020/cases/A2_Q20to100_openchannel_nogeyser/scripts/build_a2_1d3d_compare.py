#!/usr/bin/env python3
"""Build the Liu2020 A2 synchronized 1-D/OpenFOAM comparison viewer.

The left panel uses the current frozen one-dimensional A2 frame sequence.
The right panel uses the completed refined OpenFOAM v2512 ``interFoam``
front-elevation GIF archived on the A2 result branch.  That GIF contains
true VTK y=0 fields at the retained t=12, 13, and 14 s dumps and
probe-reconstructed front elevations at the other times.

All viewer times use the simulation clock, t=0 at the start of the
20->100 L/s inflow ramp.  No scientific data are recomputed here.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image


CASE_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = CASE_ROOT / "outputs"
COMPARE_DIR = OUTPUTS / "a2_1d_openfoam_compare"
OPENFOAM_FRAMES = COMPARE_DIR / "openfoam_refined_frames"
HTML_PATH = CASE_ROOT / "a2_1d_openfoam_3d_frame_compare.html"
MANIFEST_PATH = OUTPUTS / "a2_1d_openfoam_compare_manifest.json"

ONE_D_INDEX = OUTPUTS / "frames_index.json"
ONE_D_METRICS = OUTPUTS / "caseA_metrics.json"
OF_METRICS = OUTPUTS / "openfoam_3d_refined_metrics.json"
OF_PRESSURE_SERIES = OUTPUTS / "openfoam_3d_refined_pressure_series.csv"
OF_RISER_SERIES = OUTPUTS / "openfoam_3d_refined_riser_series.csv"
OF_GIF = OUTPUTS / "openfoam_3d_refined_front_complete_motion.gif"


def adaptive_timeline(t0: float, t1: float) -> list[float]:
    """Reproduce the exact time schedule used by the archived OpenFOAM GIF."""
    import numpy as np

    parts = [
        np.arange(t0, min(0.0, t1) + 1.0e-9, 0.40),
        np.arange(max(0.0, t0), min(4.0, t1) + 1.0e-9, 0.08),
        np.arange(max(4.0, t0), t1 + 1.0e-9, 0.20),
        np.array([t0, t1], dtype=float),
    ]
    forced = np.array([-12.0, 12.0, 13.0, 14.0], dtype=float)
    forced = forced[(forced >= t0 - 1.0e-9) & (forced <= t1 + 1.0e-9)]
    times = np.unique(np.round(np.concatenate(parts + [forced]), 6))
    return [float(t) for t in times if t0 - 1.0e-9 <= t <= t1 + 1.0e-9]


def require_inputs() -> None:
    required = [
        ONE_D_INDEX,
        ONE_D_METRICS,
        OF_METRICS,
        OF_PRESSURE_SERIES,
        OF_RISER_SERIES,
        OF_GIF,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing A2 comparison inputs:\n" + "\n".join(missing))


def extract_openfoam_frames() -> list[dict[str, object]]:
    """Extract the completed GIF into seekable WebP frames for the HTML."""
    if OPENFOAM_FRAMES.exists():
        shutil.rmtree(OPENFOAM_FRAMES)
    OPENFOAM_FRAMES.mkdir(parents=True, exist_ok=True)

    times = adaptive_timeline(-12.0, 14.4)
    image = Image.open(OF_GIF)
    if image.n_frames != len(times):
        raise RuntimeError(
            f"OpenFOAM GIF has {image.n_frames} frames but its archived "
            f"adaptive timeline has {len(times)} entries"
        )

    entries: list[dict[str, object]] = []
    retained_vtk = (-12.0, 12.0, 13.0, 14.0)
    for source_index, time_s in enumerate(times):
        if time_s < -1.0e-9 or time_s > 14.000001:
            continue
        image.seek(source_index)
        frame_name = f"openfoam_{len(entries):04d}.webp"
        frame_path = OPENFOAM_FRAMES / frame_name
        image.convert("RGB").save(frame_path, "WEBP", quality=86, method=6)
        mode = "true VTK y=0 field" if any(abs(time_s - t) <= 0.05 for t in retained_vtk) else "probe reconstruction"
        entries.append(
            {
                "time": round(time_s, 6),
                "file": f"outputs/a2_1d_openfoam_compare/openfoam_refined_frames/{frame_name}",
                "source": mode,
                "gif_frame": source_index,
            }
        )
    return entries


def relative_one_d_entries() -> list[dict[str, object]]:
    entries = json.loads(ONE_D_INDEX.read_text(encoding="utf-8"))
    result: list[dict[str, object]] = []
    for entry in entries:
        file_path = CASE_ROOT / str(entry["file"])
        if not file_path.exists():
            raise FileNotFoundError(f"Missing 1-D frame: {file_path}")
        result.append(entry)
    return result


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Liu2020 A2 — 1D model vs completed 3D OpenFOAM</title>
<style>
:root{--navy:#0b1f33;--blue:#1464a0;--cyan:#23a6d5;--ink:#172533;--muted:#64748b;--line:#d9e3ec;--paper:#f5f8fb;--card:#fff;--ok:#087f5b;--warn:#b45309}
*{box-sizing:border-box} body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Segoe UI","Microsoft YaHei",Arial,sans-serif}
header{background:linear-gradient(115deg,var(--navy),#123e63 64%,#176b8f);color:#fff;padding:24px 30px 22px;box-shadow:0 3px 16px #0b1f332b}
header h1{font-size:24px;margin:0 0 8px;letter-spacing:.1px} header p{margin:0;max-width:1180px;color:#d7e9f6;line-height:1.55;font-size:14px}
main{max-width:1680px;margin:0 auto;padding:20px}
.status{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:12px;margin-bottom:16px}
.metric{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:13px 15px;box-shadow:0 2px 8px #0b1f330d}
.metric .label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.55px}.metric .value{font-weight:750;font-size:20px;margin-top:4px}.metric .sub{font-size:12px;color:var(--muted);margin-top:3px}.ok{color:var(--ok)}.warn{color:var(--warn)}
.toolbar{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:14px;display:grid;grid-template-columns:auto auto auto minmax(260px,1fr) auto;gap:9px;align-items:center;position:sticky;top:8px;z-index:5;box-shadow:0 5px 20px #0b1f3318}
button{border:1px solid #afc2d1;background:#fff;color:var(--navy);border-radius:7px;padding:8px 13px;font-weight:650;cursor:pointer}button:hover{background:#eaf3f8}button.primary{background:var(--blue);border-color:var(--blue);color:#fff}button.primary:hover{background:#0f5488}
input[type=range]{width:100%;accent-color:var(--blue)}.clock{font-variant-numeric:tabular-nums;font-weight:750;color:var(--navy);min-width:124px;text-align:right}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{background:#fff;border:1px solid var(--line);border-radius:11px;overflow:hidden;box-shadow:0 2px 10px #0b1f330d}
.panel-head{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;padding:12px 14px;border-bottom:1px solid var(--line)}.panel h2{font-size:17px;margin:0 0 3px}.panel-head p{font-size:12px;color:var(--muted);margin:0;line-height:1.4}.badge{white-space:nowrap;font-size:11px;border-radius:999px;padding:5px 9px;background:#e7f5ff;color:#075985;font-weight:700}
.imagebox{aspect-ratio:2.01/1;background:#e8eef3;display:flex;align-items:center;justify-content:center;overflow:hidden}.imagebox img{display:block;width:100%;height:100%;object-fit:contain;background:white}
.readout{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border-top:1px solid var(--line)}.readout div{background:#fff;padding:9px 10px;min-height:55px}.readout span{display:block;color:var(--muted);font-size:10px;text-transform:uppercase}.readout strong{display:block;margin-top:4px;font-size:13px;font-variant-numeric:tabular-nums}
.side-control{display:grid;grid-template-columns:auto minmax(120px,1fr) auto;gap:8px;align-items:center;padding:10px 12px;border-top:1px solid var(--line)}.side-control button{padding:6px 10px;font-size:12px}.side-time{font-size:12px;font-variant-numeric:tabular-nums;color:var(--muted)}
.notes{margin-top:14px;padding:12px 14px;background:#fff8e6;border:1px solid #f2d39b;border-radius:9px;color:#784d08;font-size:12px;line-height:1.55}
footer{max-width:1680px;margin:0 auto;padding:0 20px 25px;color:var(--muted);font-size:11px;line-height:1.5}
@media(max-width:1050px){.status{grid-template-columns:1fr 1fr}.panels{grid-template-columns:1fr}.toolbar{grid-template-columns:auto auto auto 1fr}.clock{grid-column:1/-1;text-align:left}.imagebox{aspect-ratio:2.05/1}}
</style>
</head>
<body>
<header>
  <h1>Liu et al. (2020) Case A2：一维模型与已完成三维 OpenFOAM 对比</h1>
  <p>Q = 20→100 L/s，下游明渠、不喷发分支。左右两侧共用 t=0（流量斜坡开始）时钟；右侧加密网格 OpenFOAM 前视在 t=12、13、14 s 使用保留的真实 VTK y=0 场，其余时刻为连续探针重构。</p>
</header>
<main>
  <section class="status">
    <div class="metric"><div class="label">OpenFOAM run</div><div class="value ok">完整完成</div><div class="sub">v2512 interFoam · 4 MPI · −12→14.4 s</div></div>
    <div class="metric"><div class="label">Refined mesh</div><div class="value">__OF_CELLS__ cells</div><div class="sub">Mesh OK · interface Co max __OF_ALPHA_CO__</div></div>
    <div class="metric"><div class="label">Branch selection</div><div class="value ok">No geyser / No geyser</div><div class="sub">1-D / 3-D，竖管顶 1.22 m</div></div>
    <div class="metric"><div class="label">Bore arrival</div><div class="value">__OF_BORE__ s</div><div class="sub">3-D refined；实验目标 1.60 s</div></div>
  </section>

  <section class="toolbar">
    <button id="prev">◀ 上一帧</button><button id="play" class="primary">▶ 同步播放</button><button id="next">下一帧 ▶</button>
    <input id="master" type="range" min="0" max="1" value="0" step="1" aria-label="同步时间轴">
    <div class="clock" id="clock">t = 0.00 s</div>
  </section>

  <section class="panels">
    <article class="panel">
      <div class="panel-head"><div><h2>当前一维复合域模型</h2><p>上游管—交汇室—下游管连续 PDE 域；竖管柱由计算状态产生。</p></div><span class="badge">1-D model</span></div>
      <div class="imagebox"><img id="img1" alt="A2 one-dimensional model frame"></div>
      <div class="readout"><div><span>time</span><strong id="t1">—</strong></div><div><span>chamber stage</span><strong id="s1">—</strong></div><div><span>riser column</span><strong id="h1">—</strong></div><div><span>Qin / Qout</span><strong id="q1">—</strong></div></div>
      <div class="side-control"><button id="play1">播放 1-D</button><input id="slider1" type="range" min="0" max="1" value="0" step="1"><span id="count1" class="side-time">—</span></div>
    </article>

    <article class="panel">
      <div class="panel-head"><div><h2>加密网格三维 OpenFOAM</h2><p>两幅画面来自同一次 refined 3-D 计算：上部为全装置前视图，下部为下游管局部放大，并不是两个三维算例。</p></div><span class="badge">3-D interFoam</span></div>
      <div class="imagebox"><img id="img3" alt="A2 completed OpenFOAM frame"></div>
      <div class="readout"><div><span>time</span><strong id="t3">—</strong></div><div><span>frame source</span><strong id="source3">—</strong></div><div><span>max mixture front</span><strong>__OF_FRONT__ m</strong></div><div><span>riser top reached</span><strong class="ok">No</strong></div></div>
      <div class="side-control"><button id="play3">播放 3-D</button><input id="slider3" type="range" min="0" max="1" value="0" step="1"><span id="count3" class="side-time">—</span></div>
    </article>
  </section>

  <div class="notes"><strong>证据边界：</strong>三维计算支持“不喷发”分支和涌波到达时序，但明显低估 PT2/PT3 稳态压力和初始混合物柱高。除 t=12、13、14 s 的真实 VTK 截面外，右侧动态帧由本次 OpenFOAM 连续探针、湿面积和液位序列重构，不应当解释为逐时刻完整三维体积分数场。</div>
</main>
<footer>数据来源：A2 专用结果分支 origin/cursor/test3-a2-openfoam-1850；OpenFOAM refined metrics、pressure/riser series 与 front-elevation archive。页面与派生帧均保存在 A2 自己的子目录中。</footer>
<script>
const data1=__DATA_ONE_D__;
const data3=__DATA_OPENFOAM__;
const $=id=>document.getElementById(id);
let i1=0,i3=0,master=0,syncTimer=null,timer1=null,timer3=null;
const nearest=(data,t)=>data.reduce((best,x,i)=>Math.abs(x.time-t)<Math.abs(data[best].time-t)?i:best,0);
function render1(){const f=data1[i1];$('img1').src=f.file;$('t1').textContent=f.time.toFixed(2)+' s';$('s1').textContent=f.S.toFixed(3)+' m';$('h1').textContent=f.hr.toFixed(3)+' m';$('q1').textContent=f.Qin.toFixed(1)+' / '+f.Qout.toFixed(1)+' L/s';$('slider1').value=i1;$('count1').textContent=(i1+1)+' / '+data1.length;}
function render3(){const f=data3[i3];$('img3').src=f.file;$('t3').textContent=f.time.toFixed(2)+' s';$('source3').textContent=f.source;$('slider3').value=i3;$('count3').textContent=(i3+1)+' / '+data3.length;}
function set1(i){i1=Math.max(0,Math.min(data1.length-1,i));render1();}
function set3(i){i3=Math.max(0,Math.min(data3.length-1,i));render3();}
function setMaster(i){master=Math.max(0,Math.min(data1.length-1,i));const t=data1[master].time;set1(master);set3(nearest(data3,t));$('master').value=master;$('clock').textContent='t = '+t.toFixed(2)+' s';}
function stopSync(){if(syncTimer){clearInterval(syncTimer);syncTimer=null;$('play').textContent='▶ 同步播放';}}
function toggleSync(){if(syncTimer){stopSync();return}if(timer1){clearInterval(timer1);timer1=null;$('play1').textContent='播放 1-D'}if(timer3){clearInterval(timer3);timer3=null;$('play3').textContent='播放 3-D'}syncTimer=setInterval(()=>setMaster(master+1>=data1.length?0:master+1),110);$('play').textContent='Ⅱ 暂停';}
function toggleSide(side){stopSync();if(side===1){if(timer1){clearInterval(timer1);timer1=null;$('play1').textContent='播放 1-D'}else{timer1=setInterval(()=>set1(i1+1>=data1.length?0:i1+1),120);$('play1').textContent='暂停 1-D'}}else{if(timer3){clearInterval(timer3);timer3=null;$('play3').textContent='播放 3-D'}else{timer3=setInterval(()=>set3(i3+1>=data3.length?0:i3+1),140);$('play3').textContent='暂停 3-D'}}}
$('master').max=data1.length-1;$('slider1').max=data1.length-1;$('slider3').max=data3.length-1;
$('master').addEventListener('input',e=>{stopSync();setMaster(+e.target.value)});$('slider1').addEventListener('input',e=>set1(+e.target.value));$('slider3').addEventListener('input',e=>set3(+e.target.value));
$('prev').onclick=()=>{stopSync();setMaster(master-1)};$('next').onclick=()=>{stopSync();setMaster(master+1)};$('play').onclick=toggleSync;$('play1').onclick=()=>toggleSide(1);$('play3').onclick=()=>toggleSide(3);
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'){stopSync();setMaster(master-1)}if(e.key==='ArrowRight'){stopSync();setMaster(master+1)}if(e.key===' '){e.preventDefault();toggleSync()}});
setMaster(0);
</script>
</body></html>
"""


def build_html(one_d: list[dict[str, object]], openfoam: list[dict[str, object]]) -> None:
    one_metrics = json.loads(ONE_D_METRICS.read_text(encoding="utf-8"))
    of_metrics = json.loads(OF_METRICS.read_text(encoding="utf-8"))
    model = one_metrics["model"]
    of = of_metrics["openfoam_3d"]
    mass = of_metrics["mass_conservation"]

    replacements = {
        "__DATA_ONE_D__": json.dumps(one_d, ensure_ascii=False, separators=(",", ":")),
        "__DATA_OPENFOAM__": json.dumps(openfoam, ensure_ascii=False, separators=(",", ":")),
        "__OF_CELLS__": f"{int(of['cells']):,}",
        "__OF_ALPHA_CO__": f"{of['maximum_interface_courant_number']:.3f}",
        "__OF_BORE__": f"{of['bore_arrival_ramp_clock_s']:.3f}",
        "__OF_FRONT__": f"{of['maximum_mixture_front_m']:.2f}",
        "__OF_COLUMN__": f"{of['maximum_contiguous_mixture_column_m']:.2f}",
        "__OF_PT2__": f"{of['PT2_paper_window_kPa']:.3f}",
        "__OF_PT3__": f"{of['PT3_paper_window_kPa']:.3f}",
        "__OF_MASS__": f"{mass['final_residual_percent_of_inflow']:.3f}",
        "__ONE_D_HMAX__": f"{model['hr_max']:.3f}",
        "__ONE_D_BORE__": "1.410",
    }
    html = HTML_TEMPLATE
    for token, value in replacements.items():
        html = html.replace(token, value)
    HTML_PATH.write_text(html, encoding="utf-8")


def main() -> None:
    require_inputs()
    one_d = relative_one_d_entries()
    openfoam = extract_openfoam_frames()
    build_html(one_d, openfoam)

    manifest = {
        "case": "Liu2020 A2 Q20to100 open-channel no-geyser",
        "viewer": str(HTML_PATH.relative_to(CASE_ROOT)),
        "clock": "t=0 at the start of the 20->100 L/s inflow ramp",
        "one_d_source": str(ONE_D_INDEX.relative_to(CASE_ROOT)),
        "openfoam_source": str(OF_GIF.relative_to(CASE_ROOT)),
        "openfoam_metrics": str(OF_METRICS.relative_to(CASE_ROOT)),
        "one_d_frames": len(one_d),
        "openfoam_frames": len(openfoam),
        "openfoam_frame_note": "true VTK y=0 at retained t=12,13,14 s; probe reconstruction otherwise",
        "scientific_status": "OpenFOAM supports no-geyser branch and bore timing; pressure and riser-height amplitudes remain partial",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(one_d)} 1-D frames; {len(openfoam)} OpenFOAM frames")
    print(f"viewer -> {HTML_PATH}")
    print(f"manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

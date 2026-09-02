"""Exploratory Case B 1-D wave-resolution sensitivity.

This diagnostic keeps the governing equations, physical parameters, boundary
conditions, and tower-pressure coupling unchanged.  It compares the frozen
``dx=0.02 m`` horizontal discretisation with a ``dx=0.005 m`` refinement and
writes a self-contained HTML viewer.  Outputs are sensitivity evidence only;
they do not replace the manuscript baseline.
"""
from __future__ import annotations

import importlib.util
import argparse
import json
import sys
from pathlib import Path

import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
MODEL = CASE_ROOT / "model"
OUTPUT = CASE_ROOT / "outputs" / "sensitivity_wave_1d"
HTML = CASE_ROOT / "caseB_1d_wave_sensitivity.html"
BASELINE_BUILDER = Path(__file__).with_name("caseB_rebuild_1d_tosan2021.py")
SOLVER_PATH = MODEL / "tosan2021_horizontal_shockfit.py"
T_END = 8.95
OUTPUT_DT = 0.02
DX_VALUES = (0.02, 0.005)
PIPE_D = 0.094


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_variant(dx: float, vertical_frames: list[dict]) -> list[dict]:
    solver_module = load_module(f"caseb_wave_solver_{str(dx).replace('.', 'p')}", SOLVER_PATH)
    builder = load_module(f"caseb_wave_builder_{str(dx).replace('.', 'p')}", BASELINE_BUILDER)
    config = solver_module.HorizontalConfig(
        length=4.006,
        diameter=PIPE_D,
        valve_x=0.546,
        vent_x=3.516,
        dx=dx,
        wave_speed=100.0,
        gamma=1.4,
        initial_air_head=0.610,
        initial_water_head=0.356,
        wetting_front_report_fraction=builder.VISIBLE_WATER_AREA_FRACTION,
    )
    solver = solver_module.Tosan2021HorizontalShockFit(
        config,
        vent_pressure_hook=builder._tower_pressure_hook(vertical_frames),
    )
    raw = solver.run(solver.case_b_initial_state(), t_end=T_END, output_dt=OUTPUT_DT)
    return [solver.snapshot(item) if not isinstance(item, dict) else item for item in raw]


def depth_profiles(snapshots: list[dict], dx: float) -> dict:
    solver_module = load_module(f"caseb_wave_geometry_{str(dx).replace('.', 'p')}", SOLVER_PATH)
    section = solver_module.CircularSection(PIPE_D)
    rows = []
    gas_mass = []
    water_volume = []
    probe_x = (3.60, 3.75, 3.90)
    probes = {f"x{value:.2f}": [] for value in probe_x}
    for snapshot in snapshots:
        x = np.asarray(snapshot["x"], dtype=float)
        area = np.asarray(snapshot["area"], dtype=float)
        depth = np.asarray(section.depth_from_area(np.clip(area, 0.0, section.full_area)))
        rows.append(
            {
                "t": round(float(snapshot["time"]), 6),
                "x": np.round(x, 6).tolist(),
                "h": np.round(depth, 7).tolist(),
                "interface_x": round(float(snapshot["interface_x"]), 6),
            }
        )
        gas_mass.append(float(snapshot["air_mass"]))
        water_volume.append(float(snapshot["water_volume"]))
        for value in probe_x:
            probes[f"x{value:.2f}"].append(float(np.interp(value, x, depth)))

    time = np.asarray([row["t"] for row in rows])
    mask = time >= 4.5
    probe_metrics = {}
    for key, values in probes.items():
        array = np.asarray(values)
        segment = array[mask]
        probe_metrics[key] = {
            "minimum_m": float(np.min(segment)),
            "maximum_m": float(np.max(segment)),
            "peak_to_peak_m": float(np.ptp(segment)),
            "standard_deviation_m": float(np.std(segment)),
        }
    gm = np.asarray(gas_mass)
    wv = np.asarray(water_volume)
    return {
        "dx_m": dx,
        "output_dt_s": OUTPUT_DT,
        "frames": rows,
        "probes": {key: np.round(values, 7).tolist() for key, values in probes.items()},
        "metrics_after_4p5s": probe_metrics,
        "gas_mass_relative_drift": float((np.max(gm) - np.min(gm)) / gm[0]),
        "water_volume_relative_range": float((np.max(wv) - np.min(wv)) / wv[0]),
    }


def build_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Case B 1D wave-resolution sensitivity</title>
<style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:0;background:#f4f6f8;color:#18202a}}main{{max-width:1500px;margin:auto;padding:20px}}
h1{{font-size:24px;margin:0 0 8px}}.note{{line-height:1.55;color:#46515e}}.controls,.card{{background:#fff;border:1px solid #cbd2da;padding:12px;margin:12px 0}}
.controls div{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}button,select{{font:inherit;padding:6px 10px}}input[type=range]{{width:100%;margin-top:12px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}canvas{{width:100%;height:300px;border:1px solid #dde2e8;background:#fff}}.metric{{font-variant-numeric:tabular-nums}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Case B — 1D 波动分辨率敏感性</h1>
<p class="note">控制方程、物理参数和边界条件不变；仅比较论文基线 Δx=0.020 m 与探索性细网格 Δx=0.005 m。图中显示横管水深，右端局部视图将竖向波动放大。该页面是 sensitivity 诊断，不替代论文冻结结果。</p>
<section class="controls"><div><button id="prev">上一帧</button><button id="play">播放</button><button id="next">下一帧</button><select id="speed"><option value="160">1×</option><option value="80" selected>2×</option><option value="40">4×</option></select><span id="info" class="metric"></span></div><input id="range" type="range" min="0" value="0" step="1"></section>
<div class="grid"><section class="card"><h2>基线 Δx=0.020 m</h2><canvas id="coarse"></canvas><div id="coarseMetric" class="metric"></div></section><section class="card"><h2>细网格 Δx=0.005 m</h2><canvas id="fine"></canvas><div id="fineMetric" class="metric"></div></section></div>
<section class="card"><h2>右端局部放大（x=3.50–4.006 m）</h2><canvas id="zoom"></canvas><p class="note">蓝色：基线；红色：细网格。纵轴为水深并随当前帧自动放大，单位为 mm；请结合上方全管视图判断实际波幅。</p></section>
<script>
const data={data}; const coarse=data.variants[0], fine=data.variants[1]; let i=0,timer=null;
const range=document.getElementById('range'); range.max=coarse.frames.length-1;
function plot(canvas,series,x0=0,x1=4.006,y0=0,y1=0.094){{const dpr=devicePixelRatio||1,r=canvas.getBoundingClientRect();canvas.width=r.width*dpr;canvas.height=r.height*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);const w=r.width,h=r.height,p=42;c.clearRect(0,0,w,h);c.strokeStyle='#333';c.strokeRect(p,12,w-p-10,h-p-10);for(const s of series){{c.beginPath();c.strokeStyle=s.color;c.lineWidth=1.6;let started=false;for(let k=0;k<s.x.length;k++){{if(s.x[k]<x0||s.x[k]>x1)continue;const xx=p+(s.x[k]-x0)/(x1-x0)*(w-p-10);const yy=12+(y1-s.h[k])/(y1-y0)*(h-p-10);started?c.lineTo(xx,yy):c.moveTo(xx,yy);started=true}}c.stroke()}}c.fillStyle='#333';c.font='12px Segoe UI';c.fillText(x0.toFixed(2)+' m',p,h-6);c.fillText(x1.toFixed(3)+' m',w-58,h-6);c.fillText((y1*1000).toFixed(1)+' mm',2,20);c.fillText((y0*1000).toFixed(1)+' mm',2,h-p+14)}}
function zoomBounds(series){{let values=[];for(const s of series)for(let k=0;k<s.x.length;k++)if(s.x[k]>=3.50&&s.x[k]<=4.006)values.push(s.h[k]);let lo=Math.min(...values),hi=Math.max(...values);const span=Math.max(hi-lo,0.002);return [Math.max(0,lo-0.15*span),Math.min(0.094,hi+0.15*span)]}}
function show(n){{i=Math.max(0,Math.min(coarse.frames.length-1,n));range.value=i;const a=coarse.frames[i],b=fine.frames[i];plot(document.getElementById('coarse'),[{{x:a.x,h:a.h,color:'#2563eb'}}]);plot(document.getElementById('fine'),[{{x:b.x,h:b.h,color:'#dc2626'}}]);const zs=[{{x:a.x,h:a.h,color:'#2563eb'}},{{x:b.x,h:b.h,color:'#dc2626'}}],yb=zoomBounds(zs);plot(document.getElementById('zoom'),zs,3.50,4.006,yb[0],yb[1]);document.getElementById('info').textContent=`帧 ${{i+1}}/${{coarse.frames.length}} · t=${{a.t.toFixed(2)}} s · 局部纵轴 ${{(yb[0]*1000).toFixed(1)}}–${{(yb[1]*1000).toFixed(1)}} mm`;document.getElementById('coarseMetric').textContent=`界面 x=${{a.interface_x.toFixed(3)}} m`;document.getElementById('fineMetric').textContent=`界面 x=${{b.interface_x.toFixed(3)}} m`}}
function stop(){{if(timer)clearInterval(timer);timer=null;document.getElementById('play').textContent='播放'}}function toggle(){{if(timer){{stop();return}}timer=setInterval(()=>show((i+1)%coarse.frames.length),Number(document.getElementById('speed').value));document.getElementById('play').textContent='暂停'}}
document.getElementById('prev').onclick=()=>{{stop();show(i-1)}};document.getElementById('next').onclick=()=>{{stop();show(i+1)}};document.getElementById('play').onclick=toggle;range.oninput=e=>{{stop();show(Number(e.target.value))}};document.getElementById('speed').onchange=()=>{{if(timer){{stop();toggle()}}}};addEventListener('resize',()=>show(i));show(0);
</script></main></body></html>"""


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    builder = load_module("caseb_wave_vertical_builder", BASELINE_BUILDER)
    times = np.arange(0.0, T_END + 0.5 * OUTPUT_DT, OUTPUT_DT).tolist()
    _, _, vertical = builder._run_vertical_reference(times)
    variants = []
    for dx in DX_VALUES:
        print(f"Running dx={dx:.3f} m", flush=True)
        variants.append(depth_profiles(run_variant(dx, vertical), dx))
    payload = {
        "status": "exploratory_sensitivity_only",
        "governing_equations_modified": False,
        "physical_parameters_modified": False,
        "variants": variants,
    }
    (OUTPUT / "wave_resolution_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    HTML.write_text(build_html(payload), encoding="utf-8")
    print(f"Metrics -> {OUTPUT / 'wave_resolution_metrics.json'}")
    print(f"HTML -> {HTML}")


def rebuild_html_only() -> None:
    metrics = OUTPUT / "wave_resolution_metrics.json"
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    HTML.write_text(build_html(payload), encoding="utf-8")
    print(f"HTML -> {HTML}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()
    rebuild_html_only() if args.html_only else main()

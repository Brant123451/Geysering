#!/usr/bin/env python3
"""Build the offline B-H3 project-1D/OpenFOAM-2D frame comparison."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "BH3_1d_openfoam2d_frame_compare_13s.html"


def show(value: object, digits: int = 3) -> str:
    if value is None or isinstance(value, bool):
        return "—" if value is None else ("是" if value else "否")
    number = float(value)
    return "—" if not math.isfinite(number) else f"{number:.{digits}f}"


def metric_row(label: str, unit: str, experiment: object, one_d: object, two_d: object) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(label)}</td><td>{html.escape(unit)}</td>"
        f"<td>{show(experiment)}</td><td>{show(one_d)}</td><td>{show(two_d)}</td>"
        "</tr>"
    )


def main() -> None:
    one_frames = json.loads((HERE / "model_1d" / "frames.json").read_text(encoding="utf-8"))
    two_frames = json.loads((HERE / "openfoam_2d" / "frames.json").read_text(encoding="utf-8"))

    template = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cong 2017 B-H3：项目 1D 与 OpenFOAM 2D 逐帧对比</title>
<style>
:root{--ink:#17212b;--muted:#68717d;--line:#d8dee6;--one:#d55e00;--two:#0072b2;--bg:#f3f6f9;--card:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:18px}
h1{font-size:23px;margin:0 0 5px}.note{color:var(--muted);margin:0 0 14px}.toolbar,.card,.metrics{background:var(--card);border:1px solid var(--line);border-radius:10px;box-shadow:0 2px 8px #26384c0d}
.toolbar{padding:12px 14px;margin-bottom:14px;display:grid;grid-template-columns:auto auto auto minmax(260px,1fr);gap:10px;align-items:center}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.card{padding:12px;min-width:0}.card h2{font-size:16px;margin:0 0 8px}.one h2{color:var(--one)}.two h2{color:var(--two)}
.viewport{height:360px;border:1px solid #e4e8ed;border-radius:7px;background:#fff;display:flex;align-items:center;justify-content:center;overflow:hidden}.viewport.zoom{height:620px}.viewport img{width:100%;height:100%;object-fit:contain;display:block}
.controls{display:grid;grid-template-columns:auto auto 1fr auto auto;gap:7px;align-items:center;margin-top:10px}button{border:1px solid #b9c2ce;background:#fff;border-radius:6px;padding:6px 10px;cursor:pointer}button:hover{background:#eef3f7}input[type=range]{width:100%}
.meta{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:9px;color:var(--muted)}.meta b{color:var(--ink)}.diff{margin:14px 0;padding:10px 12px;background:#eef4f8;border-radius:7px;display:flex;gap:22px;flex-wrap:wrap}
.metrics{padding:12px;margin-bottom:14px;overflow:auto}.metrics h2{font-size:16px;margin:0 0 8px}.metrics table,.pairs table{width:100%;border-collapse:collapse;background:white}.metrics th,.metrics td,.pairs th,.pairs td{border:1px solid var(--line);padding:6px 8px;text-align:center}.metrics th,.pairs th{background:#eef2f6}.audit{font-size:12px;color:var(--muted);margin-top:8px}.pairs{margin-top:14px;overflow:auto}
@media(max-width:900px){.grid{grid-template-columns:1fr}.toolbar{grid-template-columns:1fr}.viewport{height:270px}.viewport.zoom{height:520px}.meta{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main>
<h1>Cong et al. (2017) Campaign 2，B-H3：项目 1D 模型与 OpenFOAM 2D 逐帧对比</h1>
<p class="note">两侧采用相同论文布置：D=50 mm，Dr=26 mm，水箱—立管 3.47 m，立管—阀门 2.51 m，L0=0.61 m，总长 6.59 m；H0=0.66 m（自管底），对应初始水面距管顶 0.61 m。时间轴未平移，参数未按 B-H3 结果反演。2D 为保持圆管面积比的平面等效宽度 13.52 mm，不能代表三维环状液膜。</p>
<section class="toolbar"><button id="view">切换为立管放大</button><button id="sync">按当前 1D 时间同步 2D</button><button id="jump">跳到实验 Ta=8.18 s</button><label>共同物理时间 <input id="master" type="range" min="0" max="13" step="0.01" value="0"></label></section>
<section class="grid"><article class="card one"><h2>项目 1D 两流体网络模型</h2><div class="viewport"><img id="img1" alt="1D frame"></div><div class="controls"><button id="prev1">◀</button><button id="play1">播放</button><input id="range1" type="range"><button id="next1">▶</button><span id="count1"></span></div><div id="meta1" class="meta"></div></article>
<article class="card two"><h2>OpenFOAM 2D（原始 alpha.water）</h2><div class="viewport"><img id="img2" alt="OpenFOAM 2D frame"></div><div class="controls"><button id="prev2">◀</button><button id="play2">播放</button><input id="range2" type="range"><button id="next2">▶</button><span id="count2"></span></div><div id="meta2" class="meta"></div></article></section>
</main><script>
const data1=__DATA1__,data2=__DATA2__;let i1=0,i2=0,zoom=false,masterTimer=null;const timers={1:null,2:null};const $=id=>document.getElementById(id);const nearest=(data,t)=>data.reduce((b,x,i)=>Math.abs(x.time-t)<Math.abs(data[b].time-t)?i:b,0);const fmt=(x,n=3)=>Number.isFinite(Number(x))?Number(x).toFixed(n):'—';
function render(side){const data=side===1?data1:data2,i=side===1?i1:i2,f=data[i];$('img'+side).src=zoom?f.riserFile:f.file;$('range'+side).max=data.length-1;$('range'+side).value=i;$('count'+side).textContent=`${i+1}/${data.length}`;$('meta'+side).innerHTML=`<span>Time <b>${fmt(f.time,2)} s</b></span><span>Yfs <b>${fmt(f.Yfs)} m</b></span><span>Yint <b>${fmt(f.Yint)} m</b></span><span>压力头 <b>${fmt(f.head)} m</b></span>`;diff();preload(side)}
function preload(side){const data=side===1?data1:data2,i=side===1?i1:i2;[-1,1].forEach(d=>{const j=Math.max(0,Math.min(data.length-1,i+d));new Image().src=zoom?data[j].riserFile:data[j].file})}
function diff(){}
function setIndex(side,value){if(side===1)i1=Math.max(0,Math.min(data1.length-1,value));else i2=Math.max(0,Math.min(data2.length-1,value));render(side)}function syncTime(t){i1=nearest(data1,t);i2=nearest(data2,t);render(1);render(2);$('master').value=t}
function togglePlay(side){const btn=$('play'+side);if(timers[side]){clearInterval(timers[side]);timers[side]=null;btn.textContent='播放'}else{timers[side]=setInterval(()=>setIndex(side,((side===1?i1:i2)+1)%(side===1?data1.length:data2.length)),180);btn.textContent='暂停'}}
$('range1').oninput=e=>setIndex(1,+e.target.value);$('range2').oninput=e=>setIndex(2,+e.target.value);$('prev1').onclick=()=>setIndex(1,i1-1);$('next1').onclick=()=>setIndex(1,i1+1);$('prev2').onclick=()=>setIndex(2,i2-1);$('next2').onclick=()=>setIndex(2,i2+1);$('play1').onclick=()=>togglePlay(1);$('play2').onclick=()=>togglePlay(2);$('master').oninput=e=>syncTime(+e.target.value);$('sync').onclick=()=>{i2=nearest(data2,data1[i1].time);render(2);$('master').value=data1[i1].time};$('jump').onclick=()=>syncTime(8.18);
$('view').onclick=()=>{zoom=!zoom;document.querySelectorAll('.viewport').forEach(x=>x.classList.toggle('zoom',zoom));$('view').textContent=zoom?'切换为完整管道':'切换为立管放大';render(1);render(2)};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key==='a'||e.key==='A')setIndex(1,i1-1);if(e.key==='d'||e.key==='D')setIndex(1,i1+1);if(e.key==='ArrowLeft')setIndex(2,i2-1);if(e.key==='ArrowRight')setIndex(2,i2+1);if(e.code==='Space'){e.preventDefault();if(masterTimer){clearInterval(masterTimer);masterTimer=null}else masterTimer=setInterval(()=>syncTime(+$('master').value>=13?0:+$('master').value+.05),180)}});syncTime(0);
</script></body></html>'''
    page = template.replace("__DATA1__", json.dumps(one_frames, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    page = page.replace("__DATA2__", json.dumps(two_frames, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

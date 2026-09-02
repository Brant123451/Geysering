#!/usr/bin/env python3
"""Build the dated, non-final 1D/OpenFOAM 2D progress viewer."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
TEST_ROOT = STUDY.parents[1]
CASES_ROOT = TEST_ROOT / "cases"
MODEL_ROOT = STUDY / "case1_frozen_complete" / "model_1d"
EXTENSION_INDEX = HERE / "frames_2d_extension.json"
OUTPUT_HTML = HERE / "bh1_bh3_bh6_1d2d_progress_20260810.html"
OUTPUT_MANIFEST = HERE / "progress_viewer_manifest.json"
FRAME_RENDERER = (
    CASES_ROOT
    / "BH3_Dr26_H066_L061"
    / "openfoam"
    / "2d"
    / "comparison"
    / "build_1d_frames.py"
)
ASSET_VERSION = "progress-20260810-v1"

CASE_SPECS = {
    "BH1": {
        "label": "B-H1",
        "dr": 0.016,
        "model_end": 20.0,
        "result": "1D：喷发（与试验一致） · 2D：已越口喷发 · 试验：喷发",
        "status": "2D 喷发已确认；有效计算至 14.85 s",
        "status_class": "yes",
    },
    "BH3": {
        "label": "B-H3",
        "dr": 0.026,
        "model_end": 23.0,
        "result": "1D：未喷发（模型漏报） · 2D：仍在续算 · 试验：喷发",
        "status": "2D 正在续算；本页安全检查点至 16.80 s",
        "status_class": "pending",
    },
    "BH6": {
        "label": "B-H6",
        "dr": 0.041,
        "model_end": 20.0,
        "result": "1D：未喷发（与试验一致） · 2D：仍在续算 · 试验：未喷发",
        "status": "2D 正在续算；本页安全检查点至 17.40 s",
        "status_class": "pending",
    },
}


def rel(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), HERE)).as_posix()


def load_renderer():
    spec = importlib.util.spec_from_file_location("campaign2_frame_renderer", FRAME_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frame renderer: {FRAME_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DISPLAY_TOP = 1.90
    module.PIPE_D = 0.20
    return module


def render_1d_frames(case_id: str, case_spec: dict[str, object], renderer) -> list[dict[str, object]]:
    source = MODEL_ROOT / case_id / "case1_full_network_frames.npz"
    data = np.load(source, allow_pickle=False)
    source_times = np.asarray(data["frames_t"], dtype=float)
    end_time = float(case_spec["model_end"])
    targets = np.arange(0.0, end_time + 0.0001, 0.1)
    indices = [int(np.argmin(np.abs(source_times - target))) for target in targets]
    out_dir = HERE / "frames_1d" / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer.RISER_D = float(case_spec["dr"])
    rows: list[dict[str, object]] = []
    for frame_no, source_index in enumerate(indices):
        time_s = float(source_times[source_index])
        output = out_dir / f"full_{frame_no:04d}.svg"
        svg = renderer.full_svg(
            time_s,
            data["xt"],
            data["frames_alt"][source_index],
            float(data["dx"][0]),
            data["zr"],
            data["frames_alr"][source_index],
            data["frames_agr"][source_index],
            float(data["wtop"][source_index]),
            float(data["itop"][source_index]),
            float(data["pocket_head"][source_index]),
        ).replace("B-H3", str(case_spec["label"]))
        svg = svg.replace("Pipe and riser use the paper dimensions", "")
        output.write_text(svg, encoding="utf-8")
        rows.append({"time": round(time_s, 8), "src": f"{rel(output)}?v={ASSET_VERSION}"})
    return rows


def base_2d_frames(case_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if case_id == "BH1":
        directory = CASES_ROOT / "BH1_Dr16_H066_L061" / "openfoam" / "2d" / "frame_compare" / "two_d_frames"
        for index, path in enumerate(sorted(directory.glob("full_*.png"))):
            rows.append({"time": round(index * 0.1, 8), "src": rel(path)})
    elif case_id == "BH3":
        comparison = CASES_ROOT / "BH3_Dr26_H066_L061" / "openfoam" / "2d" / "comparison"
        index_path = comparison / "openfoam_2d" / "frames.json"
        for item in json.loads(index_path.read_text(encoding="utf-8")):
            path = comparison / str(item["file"])
            rows.append({"time": float(item["time"]), "src": rel(path)})
    elif case_id == "BH6":
        case_root = CASES_ROOT / "BH6_Dr41_H066_L061"
        index_path = case_root / "outputs" / "1d2d_viewer" / "frames_2d.json"
        for item in json.loads(index_path.read_text(encoding="utf-8")):
            path = case_root / str(item["file"])
            rows.append({"time": float(item["time"]), "src": rel(path)})
    else:
        raise KeyError(case_id)
    if not rows or rows[0]["time"] != 0.0 or rows[-1]["time"] < 13.0:
        raise RuntimeError(f"Incomplete baseline 2D frames for {case_id}")
    return rows


def load_2d_frames(case_id: str, extensions: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    rows = base_2d_frames(case_id)
    for item in extensions[case_id]:
        path = HERE / str(item["file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"time": float(item["time"]), "src": rel(path)})
    unique = {round(float(item["time"]), 8): item for item in rows}
    result = [unique[key] for key in sorted(unique)]
    if any(float(b["time"]) <= float(a["time"]) for a, b in zip(result, result[1:])):
        raise RuntimeError(f"Non-increasing 2D times for {case_id}")
    return result


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Cong 2017 · B-H1 / B-H3 / B-H6 · 最新1D/2D进度</title>
  <style>
    :root{--ink:#17212b;--muted:#667085;--line:#d9e0e8;--bg:#edf2f7;--card:#fff;--one:#b45309;--two:#075985;--yes:#166534;--pending:#9a3412}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}
    main{max-width:1580px;margin:auto;padding:18px}h1{margin:0;font-size:24px}.subtitle{color:var(--muted);margin:3px 0 13px}
    .warning{padding:11px 14px;border:1px solid #f59e0b;border-radius:9px;background:#fff7ed;color:#9a3412;font-weight:700;margin-bottom:13px}
    .toolbar,.case{background:var(--card);border:1px solid var(--line);border-radius:11px;box-shadow:0 2px 9px #1f29370b}
    .toolbar{display:grid;grid-template-columns:auto auto minmax(300px,1fr) auto;gap:10px;align-items:center;padding:11px 13px;position:sticky;top:0;z-index:2}
    button{border:1px solid #bac5d1;border-radius:7px;background:white;padding:7px 12px;cursor:pointer}button:hover{background:#f1f5f9}input[type=range]{width:100%}
    .time{font-variant-numeric:tabular-nums;font-weight:800;color:#0f4c81;min-width:92px;text-align:right}
    .case{margin-top:14px;padding:13px}.case-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:9px}.case h2{margin:0;font-size:19px}.result{color:var(--muted);margin-top:2px}
    .badge{border-radius:999px;padding:5px 10px;font-weight:700;white-space:nowrap}.badge.yes{background:#dcfce7;color:var(--yes)}.badge.pending{background:#ffedd5;color:var(--pending)}
    .panes{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pane{border:1px solid #e2e8f0;border-radius:9px;overflow:hidden;background:#fff}.pane-head{display:flex;justify-content:space-between;padding:7px 10px;background:#f8fafc;font-weight:750}.one .pane-head{color:var(--one)}.two .pane-head{color:var(--two)}
    .frame{height:330px;display:flex;align-items:center;justify-content:center;background:#fff;overflow:hidden}.frame img{width:100%;height:100%;display:block;object-fit:contain}
    .case-foot{display:flex;justify-content:space-between;gap:10px;color:var(--muted);font-size:12px;margin-top:8px}.hold{color:#9a3412;font-weight:650}
    .note{color:var(--muted);font-size:12px;margin:13px 2px 0}
    @media(max-width:900px){.toolbar{grid-template-columns:auto auto 1fr}.time{grid-column:1/-1;text-align:left}.panes{grid-template-columns:1fr}.frame{height:250px}.case-head,.case-foot{display:block}.badge{display:inline-block;margin-top:7px}}
  </style>
</head>
<body><main>
  <h1>Cong et al. (2017) Series B · 三工况1D / OpenFOAM 2D</h1>
  <div class="subtitle">B-H1、B-H3、B-H6 · 完整横管—竖管域 · 原始物理时间</div>
  <div class="warning">进度版（2026-08-10）：H3、H6 的2D仍在续算。本页可查看最新已写入检查点，但不可作为最终论文证据。</div>
  <section class="toolbar">
    <button id="play" type="button">▶ 播放</button>
    <button id="reset" type="button">回到 0 s</button>
    <input id="clock" type="range" min="0" max="__MAX_TICK__" step="1" value="0" aria-label="公共物理时间">
    <div class="time" id="time">t = 0.00 s</div>
  </section>
  <div id="cases"></div>
  <p class="note">1D采用哈希锁定的冻结全网络核心；数值近干薄膜在显示时按干区处理。1D冻结配置的阀门开启时间为0.25 s；论文记载约0.2 s，2D采用0.20 s。横管仅在1D绘图厚度上放大4倍，轴向几何与模拟数据未改变。</p>
</main>
<script>
const cases=__CASE_DATA__,DT=0.1,END=__END_TIME__;
const root=document.getElementById('cases');
function nearest(frames,t){let lo=0,hi=frames.length-1;while(lo<hi){const mid=Math.floor((lo+hi)/2);if(frames[mid].time<t)lo=mid+1;else hi=mid}if(lo===0)return frames[0];const a=frames[lo-1],b=frames[lo];return Math.abs(a.time-t)<=Math.abs(b.time-t)?a:b}
root.innerHTML=cases.map(c=>`<section class="case"><div class="case-head"><div><h2>${c.label} · Dr/D=${c.ratio}</h2><div class="result">${c.result}</div></div><span class="badge ${c.statusClass}">${c.status}</span></div><div class="panes"><article class="pane one"><div class="pane-head"><span>冻结1D模型</span><span id="${c.id}-t1">0.00 s</span></div><div class="frame"><img id="${c.id}-im1" alt="${c.label} 1D完整管道帧"></div></article><article class="pane two"><div class="pane-head"><span>OpenFOAM 2D</span><span id="${c.id}-t2">0.00 s</span></div><div class="frame"><img id="${c.id}-im2" alt="${c.label} OpenFOAM 2D完整管道帧"></div></article></div><div class="case-foot"><span>1D ${c.frames1d.length}帧，至 ${c.end1d.toFixed(2)} s；2D ${c.frames2d.length}帧，至 ${c.end2d.toFixed(2)} s</span><span class="hold" id="${c.id}-hold"></span></div></section>`).join('');
const slider=document.getElementById('clock'),label=document.getElementById('time'),play=document.getElementById('play');let tick=0,playing=false,last=0;
function draw(){tick=Math.max(0,Math.min(Math.round(END/DT),tick));slider.value=String(tick);const t=tick*DT;label.textContent=`t = ${t.toFixed(2)} s`;for(const c of cases){const a=nearest(c.frames1d,t),b=nearest(c.frames2d,t),im1=document.getElementById(`${c.id}-im1`),im2=document.getElementById(`${c.id}-im2`);if(im1.getAttribute('src')!==a.src)im1.src=a.src;if(im2.getAttribute('src')!==b.src)im2.src=b.src;document.getElementById(`${c.id}-t1`).textContent=`${a.time.toFixed(2)} s`;document.getElementById(`${c.id}-t2`).textContent=`${b.time.toFixed(2)} s`;document.getElementById(`${c.id}-hold`).textContent=t>c.end2d+0.05?`2D已保持在最新检查点 ${c.end2d.toFixed(2)} s`:''}}
function animate(now){if(!playing)return;if(now-last>90){tick++;last=now;if(tick>Math.round(END/DT)){tick=0}draw()}requestAnimationFrame(animate)}
play.onclick=()=>{playing=!playing;play.textContent=playing?'❚❚ 暂停':'▶ 播放';if(playing){last=performance.now();requestAnimationFrame(animate)}};document.getElementById('reset').onclick=()=>{tick=0;draw()};slider.oninput=()=>{tick=Number(slider.value);draw()};document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'){tick--;draw()}if(e.key==='ArrowRight'){tick++;draw()}if(e.code==='Space'){e.preventDefault();play.click()}});draw();
</script></body></html>'''


def main() -> None:
    extensions = json.loads(EXTENSION_INDEX.read_text(encoding="utf-8"))
    renderer = load_renderer()
    cases: list[dict[str, object]] = []
    for case_id, spec in CASE_SPECS.items():
        frames_1d = render_1d_frames(case_id, spec, renderer)
        frames_2d = load_2d_frames(case_id, extensions)
        cases.append({
            "id": case_id,
            "label": spec["label"],
            "ratio": f"{float(spec['dr']) / 0.05:.2f}",
            "result": spec["result"],
            "status": spec["status"],
            "statusClass": spec["status_class"],
            "frames1d": frames_1d,
            "frames2d": frames_2d,
            "end1d": float(frames_1d[-1]["time"]),
            "end2d": float(frames_2d[-1]["time"]),
        })

    end_time = max(float(case["end2d"]) for case in cases)
    payload = json.dumps(cases, ensure_ascii=False, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__CASE_DATA__", payload)
    html = html.replace("__END_TIME__", f"{end_time:.2f}")
    html = html.replace("__MAX_TICK__", str(round(end_time / 0.1)))
    OUTPUT_HTML.write_text(html, encoding="utf-8")

    manifest = {
        "artifact": OUTPUT_HTML.name,
        "status": "interim_progress_not_for_manuscript",
        "created_local_date": "2026-08-10",
        "scientific_change": "1D replaced by accepted full-event archives; sparse 2D extension checkpoints appended without modifying solvers",
        "display_only_change": "1D horizontal pipe drawing thickness is enlarged 4x",
        "cases": [
            {
                "id": case["id"],
                "end_1d_s": case["end1d"],
                "end_2d_checkpoint_s": case["end2d"],
                "frames_1d": len(case["frames1d"]),
                "frames_2d": len(case["frames2d"]),
                "status": case["status"],
            }
            for case in cases
        ],
        "provenance_note": "BH1 baseline frames use the formal 13 s run and extension frames use the refined Co=0.15 run; BH3/BH6 extensions continue their original 13 s runs.",
        "not_final_reason": "BH3 and BH6 OpenFOAM extensions have not reached their event-completion gates.",
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT_HTML)
    for case in cases:
        print(f"{case['id']}: 1D->{case['end1d']:.2f}s; 2D->{case['end2d']:.2f}s")


if __name__ == "__main__":
    main()

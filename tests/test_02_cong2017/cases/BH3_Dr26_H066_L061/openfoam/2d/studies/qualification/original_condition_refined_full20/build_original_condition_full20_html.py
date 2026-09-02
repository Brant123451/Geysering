#!/usr/bin/env python3
"""Build the original-condition B-H3 refined 0--20 s front-view HTML.

The page uses native OpenFOAM time.  No frame is shifted, stretched, blended,
or re-labelled.  The final filename is written only after the complete real
0--20 s coverage gate passes.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import tempfile
from bisect import bisect_left
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CASE = HERE.parents[4]
QUALIFICATION = CASE / "openfoam" / "2d" / "qualification" / "h3_refined_iso_riser20"
METRICS = QUALIFICATION / "results" / "openfoam_2d_metrics.json"
RISER_CSV = QUALIFICATION / "results" / "openfoam_2d_riser_series.csv"
AUDIT = (
    CASE.parents[1]
    / "studies"
    / "BH1_BH3_BH6_1d_animation"
    / "2d_physical_outlet_audit"
    / "h3_refined_iso_end20.json"
)
SHORT_BUILDER = (
    HERE.parent
    / "sensitivity"
    / "outcome_forcing_H180_p401325"
    / "build_front_view_animation.py"
)
HTML_PATH = HERE / "H3_original_condition_refined_full20_front_view_animation.html"
RESULTS = HERE / "results"
FRAMES_DIR = RESULTS / "front_view_frames"
MANIFEST_PATH = RESULTS / "front_view_animation_manifest.json"
VERIFY_PATH = RESULTS / "front_view_animation_verification.json"
PREVIEW_PATH = RESULTS / "front_view_preview_t10p1.png"

PAPER_TA = 8.18
MODEL_TA = 8.04
EXPERIMENT_OVERRIM_INFERRED = 8.18 + (1.80 - 0.61) / 0.657
MODEL_PROXY = 10.084942263797627


def load_render_helpers():
    if not SHORT_BUILDER.exists():
        raise FileNotFoundError(f"Missing shared renderer: {SHORT_BUILDER}")
    spec = importlib.util.spec_from_file_location("h3_short_front_builder", SHORT_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SHORT_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load_render_helpers()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_series(vtk_root: Path) -> list[tuple[float, Path]]:
    series_files = list(vtk_root.glob("*.vtm.series"))
    found: list[tuple[float, Path]] = []
    if len(series_files) == 1:
        payload = json.loads(series_files[0].read_text(encoding="utf-8"))
        for item in payload["files"]:
            stem = Path(item["name"]).with_suffix("")
            found.append((float(item["time"]), vtk_root / stem / "internal.vtu"))
    else:
        for path in vtk_root.glob("*/internal.vtu"):
            header = path.read_bytes()[:4096].decode("ascii", errors="ignore")
            match = re.search(r"\btime=['\"]([^'\"]+)", header)
            if not match:
                raise ValueError(f"Cannot discover time for {path}")
            found.append((float(match.group(1)), path))
    found.sort(key=lambda item: item[0])
    if not found:
        raise RuntimeError(f"No internal.vtu series found below {vtk_root}")
    missing = [str(path) for _, path in found if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} VTK pieces; first: {missing[0]}")
    return found


def validate_coverage(series: list[tuple[float, Path]], frame_dt: float = 0.1) -> None:
    expected = np.round(np.arange(0.0, 20.0 + 0.5 * frame_dt, frame_dt), 10)
    actual = np.asarray([time_s for time_s, _ in series], dtype=float)
    if len(actual) != len(expected) or not np.allclose(actual, expected, atol=2e-7):
        raise RuntimeError(
            "Refusing to create the final HTML: expected native times "
            f"0:{frame_dt}:20 ({len(expected)} frames), got {len(actual)} frames "
            f"from {actual[0] if len(actual) else None} to {actual[-1] if len(actual) else None}."
        )


def fast_alpha(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    positions = [raw.find(b'Name="alpha.water"'), raw.find(b"Name='alpha.water'")]
    position = max(positions)
    if position < 0:
        raise ValueError(f"Missing alpha.water in {path}")
    begin = raw.find(b">", position)
    end = raw.find(b"</DataArray>", begin)
    if begin < 0 or end < 0:
        raise ValueError(f"Malformed alpha.water DataArray in {path}")
    alpha = np.fromstring(raw[begin + 1 : end].decode("ascii"), sep=" ")
    return np.clip(alpha, 0.0, 1.0)


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in rows[0]
    }


def interpolate(series: dict[str, np.ndarray], key: str, time_s: float) -> float:
    return float(np.interp(time_s, series["t_s"], series[key]))


def nearest_audit(frames: list[dict[str, object]], time_s: float) -> dict[str, object] | None:
    if not frames:
        return None
    times = [float(item["time_s"]) for item in frames]
    position = bisect_left(times, time_s)
    candidates = [i for i in (position - 1, position) if 0 <= i < len(frames)]
    best = min(candidates, key=lambda i: abs(times[i] - time_s))
    if abs(times[best] - time_s) > 0.026:
        return None
    return frames[best]


def png_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>B-H3 论文原条件加密二维 0–20 s 正视图</title>
<style>
:root{--ink:#17212b;--muted:#61707d;--line:#d6e0e8;--bg:#eef3f7;--card:#fff;--blue:#1268d3;--red:#c92f32;--amber:#b87900;--green:#19734b}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(180deg,#eaf1f7,#f8fafb 42%);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei","PingFang SC",sans-serif}
main{max-width:1580px;margin:auto;padding:18px}header{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:13px}h1{font-size:23px;line-height:1.25;margin:0 0 5px}.subtitle{color:var(--muted)}
.badge{white-space:nowrap;color:#fff;background:var(--red);border-radius:999px;padding:7px 12px;font-weight:750;box-shadow:0 2px 8px #c92f3233}.warning{background:#fff3f2;border:1px solid #efc6c4;border-left:5px solid var(--red);border-radius:10px;padding:11px 13px;margin-bottom:12px;color:#692426}.warning strong{color:#a51f23}
.toolbar,.viewer,.facts,.evidence{background:var(--card);border:1px solid var(--line);border-radius:12px;box-shadow:0 3px 14px #24384c12}.toolbar{display:grid;grid-template-columns:auto auto auto minmax(280px,1fr) auto auto;gap:8px;align-items:center;padding:10px 12px;margin-bottom:12px}button,select{font:inherit;color:var(--ink);background:#fff;border:1px solid #b8c5d1;border-radius:8px;padding:7px 11px;cursor:pointer}button:hover{background:#edf4fa}input[type=range]{width:100%;accent-color:var(--blue)}.time{font-weight:760;font-variant-numeric:tabular-nums;min-width:86px;text-align:right}
.viewer{padding:11px}.viewport{position:relative;background:#fff;border:1px solid #e0e7ed;border-radius:9px;overflow:hidden;min-height:350px;display:flex;align-items:center;justify-content:center}.viewport img{width:100%;height:auto;display:block}.overlay{position:absolute;left:12px;top:10px;background:#fffffff2;border:1px solid #dce5ec;border-radius:8px;padding:7px 10px;box-shadow:0 2px 8px #20364a17;font-variant-numeric:tabular-nums}.overlay strong{display:block;color:var(--blue)}
.timeline{position:relative;height:39px;margin:6px 5px 0}.axis{position:absolute;left:0;right:0;top:10px;height:4px;border-radius:4px;background:#dce5ed}.mark{position:absolute;top:1px;transform:translateX(-50%);width:2px;height:22px;background:var(--ink)}.mark::after{content:attr(data-label);position:absolute;top:23px;left:50%;transform:translateX(-50%);font-size:10px;white-space:nowrap;color:var(--muted)}.mark.paper{background:var(--green)}.mark.infer{background:var(--amber)}.mark.proxy{background:var(--red)}
.events{display:flex;gap:7px;flex-wrap:wrap;margin-top:11px}.event{font-size:12px;padding:5px 9px}.paper{border-color:#90c8ab;color:#176f48}.infer{border-color:#dfc57b;color:#865f00}.proxy{border-color:#e4a1a3;color:#a52327}.legend{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:10px 2px 0;color:var(--muted);font-size:12px}.swatch{display:inline-block;width:36px;height:10px;border-radius:6px;margin-right:5px;vertical-align:middle}.water{background:linear-gradient(90deg,#f7fafc,#74b9f5,#1268d3)}.rim{border-top:2px dashed var(--red);height:1px}
.facts{display:grid;grid-template-columns:repeat(5,1fr);gap:0;margin-top:12px;overflow:hidden}.fact{padding:10px 12px;border-right:1px solid var(--line)}.fact:last-child{border-right:0}.fact span{display:block;color:var(--muted);font-size:12px}.fact b{font-variant-numeric:tabular-nums}.evidence{margin-top:12px;padding:12px 14px}.evidence h2{font-size:15px;margin:0 0 7px}.evidence ul{margin:0;padding-left:20px}.source{margin-top:9px;color:var(--muted);font-size:12px}
@media(max-width:900px){header{display:block}.badge{display:inline-block;margin-top:9px}.toolbar{grid-template-columns:auto auto auto 1fr}.time{grid-column:1/-1;text-align:left}.facts{grid-template-columns:1fr 1fr}.fact{border-bottom:1px solid var(--line)}.viewport{min-height:220px}.mark::after{display:none}}
</style></head><body><main>
<header><div><h1>Cong2017 Campaign 2 · B-H3 论文原条件加密二维正视图</h1><div class="subtitle">refined isoAdvector · 原生绝对时钟 0–20 s · 无时间平移/拉伸 · 主视口完整 x-z 计算域</div></div><div class="badge">原条件 2D 漏报</div></header>
<div class="warning"><strong>必须保留的结论：</strong>Cong et al. 实验将 B-H3 分类为 <b>GEYSER</b>；但这套论文原条件 2D 加密算例没有通过统一的真实 z=1.825 m 物理出口界面支撑门槛。旧“液位达到竖管高度 98%”代理在 10.08494 s 给出喷发，只能作为近井口代理，不能替代真实出口审计。</div>
<section class="toolbar"><button id="play">播放</button><button id="prev">◀</button><button id="next">▶</button><input id="slider" type="range" min="0" max="__MAX_INDEX__" step="1" value="0"><span id="time" class="time">t = 0.00 s</span><select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></section>
<section class="viewer"><div class="viewport"><img id="frame" alt="B-H3 original-condition refined OpenFOAM front-view frame"><div class="overlay"><strong id="phase">阀门开启起点</strong><span id="frameNo">frame 1 / __FRAME_COUNT__</span></div></div>
<div class="timeline"><div class="axis"></div><i class="mark" style="left:40.2%" data-label="模拟 Ta 8.04"></i><i class="mark paper" style="left:40.9%" data-label="论文 Ta 8.18"></i><i class="mark infer" style="left:49.955%" data-label="推算越顶 ≈9.99"></i><i class="mark proxy" style="left:50.4247%" data-label="旧代理 10.08494"></i></div>
<div class="events"><button class="event" data-time="0">0.00 s 阀门开启起点</button><button class="event" data-time="8.04">8.04 s 模拟 Ta</button><button class="event paper" data-time="8.18">8.18 s 论文 Ta</button><button class="event infer" data-time="9.991">≈9.99 s 实验越顶推算</button><button class="event proxy" data-time="10.0849422638">10.08494 s 模拟旧代理</button><button class="event" data-time="20">20.00 s 正常 End</button></div>
<div class="legend"><span><i class="swatch water"></i>固定 alpha.water 色标：气相 0 → 液相 1</span><span><i class="swatch rim"></i>真实物理竖管出口 z=1.825 m</span><span>1×：每 0.1 s 物理时间显示 0.1 s</span></div></section>
<section class="facts"><div class="fact"><span>当前自由液位（冠顶以上）</span><b id="yfs">—</b></div><div class="fact"><span>当前出口 max α</span><b id="rimAlpha">—</b></div><div class="fact"><span>累计正液量</span><b id="cumVol">—</b></div><div class="fact"><span>模拟/论文 Ta</span><b>8.04 / 8.18 s</b></div><div class="fact"><span>求解范围</span><b>0–20 s · normal End</b></div></section>
<section class="evidence"><h2>时间标记与证据边界</h2><ul><li><b>8.18 s</b> 是论文 B-H3 的 Ta；<b>8.04 s</b> 是本算例的同定义模拟 Ta。</li><li><b>≈9.99 s</b> 不是论文直接报告的喷出口时刻，而是由 8.18 + (1.80−0.61)/0.657 = 9.991 s 得到的实验速度外推。</li><li><b>10.08494 s</b> 是旧 98% 液位代理到达时刻；统一真实出口审计的最终 evidence gate 为 no，分类为 INDETERMINATE_EVIDENCE_GAP（400/401 时刻，缺 t=0 出口面样本）。</li></ul></section>
<div class="source">各帧直接取自论文原条件加密算例的 cell-centred alpha.water；仅做固定空间栅格化和固定色标着色。无时间插值、无界面重绘、无阈值修改。键盘：空格播放/暂停，←/→逐帧。</div>
</main><script>
const frames=__FRAMES__;let index=0,timer=null;const $=id=>document.getElementById(id);
function sci(v){return v==null?'—':(v===0?'0':Number(v).toExponential(3))}
function phase(t){if(t<8.04)return '气囊到达竖管前';if(t<8.18)return '模拟 Ta 已到 / 论文 Ta 前';if(t<9.991)return '论文 Ta 后 · 上升段';if(t<10.0849422638)return '推算实验越顶后 · 旧代理前';return '旧近井口代理后 · 真实出口仍未通过'}
function draw(){const f=frames[index];$('frame').src=f.uri;$('slider').value=index;$('time').textContent=`t = ${f.time.toFixed(2)} s`;$('frameNo').textContent=`frame ${index+1} / ${frames.length}`;$('phase').textContent=phase(f.time);$('yfs').textContent=f.yfs==null?'—':f.yfs.toFixed(4)+' m';$('rimAlpha').textContent=f.rimAlpha==null?'—':Number(f.rimAlpha).toPrecision(4);$('cumVol').textContent=sci(f.cumulativeVolume)+(f.cumulativeVolume==null?'':' m³');for(const j of [Math.max(0,index-1),Math.min(frames.length-1,index+1)]){const img=new Image();img.src=frames[j].uri}}
function setIndex(i){index=Math.max(0,Math.min(frames.length-1,i));draw()}function nearest(t){let best=0;for(let i=1;i<frames.length;i++)if(Math.abs(frames[i].time-t)<Math.abs(frames[best].time-t))best=i;return best}function stop(){if(timer){clearInterval(timer);timer=null;$('play').textContent='播放'}}function toggle(){if(timer){stop();return}const speed=Number($('speed').value);timer=setInterval(()=>{if(index===frames.length-1){stop();return}setIndex(index+1)},100/speed);$('play').textContent='暂停'}
$('play').onclick=toggle;$('prev').onclick=()=>{stop();setIndex(index-1)};$('next').onclick=()=>{stop();setIndex(index+1)};$('slider').oninput=e=>{stop();setIndex(Number(e.target.value))};$('speed').onchange=()=>{if(timer){stop();toggle()}};document.querySelectorAll('[data-time]').forEach(b=>b.onclick=()=>{stop();setIndex(nearest(Number(b.dataset.time)))});document.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();toggle()}else if(e.key==='ArrowLeft'){stop();setIndex(index-1)}else if(e.key==='ArrowRight'){stop();setIndex(index+1)}});draw();
</script></body></html>'''


def build(vtk_root: Path) -> None:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    riser = read_csv(RISER_CSV)
    if metrics["status"]["last_log_time_s"] != 20.0 or not metrics["status"]["ended_normally"]:
        raise RuntimeError("Refusing to build from an incomplete original-condition run")
    if metrics["experiment"]["classification"] != "GEYSER":
        raise RuntimeError("Unexpected experimental classification")
    if audit["decision"]["final_evidence_gate"]:
        raise RuntimeError("Archived strict audit unexpectedly passes; review evidence before building")

    series = discover_series(vtk_root)
    validate_coverage(series)
    first_time, first_alpha, centres = R.read_vtu(series[0][1], geometry=True)
    if abs(first_time) > 2e-7:
        raise RuntimeError(f"First VTK time is not native t=0: {first_time}")
    assert centres is not None

    cfg = metrics["paper_contract"]
    geometry = cfg["physical_geometry_m"]
    mapping = cfg["planar_mapping"]
    full_bounds = (
        -0.08,
        float(geometry["pipe_length"]) + 0.08,
        float(geometry["pipe_invert_z"]) - 0.055,
        float(geometry["riser_rim_z"]) + float(mapping["external_height_above_rim_m"]) + 0.055,
    )
    full_shape = (650, 1400)
    raster = R.raster_map(centres, full_shape[1], full_shape[0], full_bounds)
    audit_frames = sorted(audit["frames"], key=lambda item: float(item["time_s"]))
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    html_records: list[dict[str, object]] = []
    manifest_records: list[dict[str, object]] = []
    for frame_no, (time_s, path) in enumerate(series):
        alpha = first_alpha if frame_no == 0 else fast_alpha(path)
        if len(alpha) != len(centres):
            raise ValueError(f"Cell-count mismatch at t={time_s}: {len(alpha)} != {len(centres)}")
        rgb = R.raster_alpha(alpha, *raster, full_shape)
        R.annotate_full(rgb, cfg)
        png_path = FRAMES_DIR / f"full_{frame_no:04d}.png"
        R.write_png(png_path, rgb)
        evidence = nearest_audit(audit_frames, time_s)
        yfs = interpolate(riser, "Yfs_m_above_crown", time_s)
        record = {
            "time": round(float(time_s), 10),
            "uri": png_uri(png_path),
            "yfs": yfs,
            "rimAlpha": None if evidence is None else float(evidence["max_alpha"]),
            "cumulativeVolume": None
            if evidence is None
            else float(evidence["cumulative_positive_liquid_volume_m3"]),
        }
        html_records.append(record)
        manifest_records.append(
            {
                "frame": frame_no,
                "native_time_s": float(time_s),
                "source_vtu": str(path),
                "png": str(png_path.relative_to(HERE)),
                "png_sha256": sha256(png_path),
                "Yfs_m_above_crown": yfs,
                "physical_rim_max_alpha": record["rimAlpha"],
                "physical_rim_cumulative_positive_liquid_volume_m3": record["cumulativeVolume"],
            }
        )
        if frame_no % 20 == 0 or frame_no + 1 == len(series):
            print(f"Rendered {frame_no + 1}/{len(series)} at native t={time_s:.2f} s", flush=True)

    preview_index = min(range(len(series)), key=lambda i: abs(series[i][0] - 10.1))
    shutil.copyfile(FRAMES_DIR / f"full_{preview_index:04d}.png", PREVIEW_PATH)
    html = HTML_TEMPLATE.replace("__MAX_INDEX__", str(len(series) - 1))
    html = html.replace("__FRAME_COUNT__", str(len(series)))
    html = html.replace(
        "__FRAMES__", json.dumps(html_records, ensure_ascii=False, separators=(",", ":"))
    )

    # Atomic write prevents a partial final-named artifact.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=HERE, suffix=".html.tmp") as stream:
        stream.write(html)
        temporary = Path(stream.name)
    temporary.replace(HTML_PATH)

    manifest = {
        "schema_version": 1,
        "artifact": str(HTML_PATH),
        "artifact_sha256": sha256(HTML_PATH),
        "scope": "Cong2017 B-H3 original-condition refined isoAdvector; full 0--20 s",
        "source_case": audit["coverage"]["source_case"],
        "source_field": "cell-centred alpha.water",
        "native_clock": {"start_s": 0.0, "end_s": 20.0, "frame_dt_s": 0.1, "shift_s": 0.0},
        "view": {"orientation": "front along y, x-z shown", "bounds_m": list(full_bounds), "pixels": [1400, 650], "fixed_alpha_range": [0.0, 1.0]},
        "events": {
            "paper_Ta_s": PAPER_TA,
            "model_Ta_s": MODEL_TA,
            "experiment_over_rim_inferred_s": EXPERIMENT_OVERRIM_INFERRED,
            "experiment_over_rim_inference": "8.18 + (1.80 - 0.61) / 0.657",
            "model_legacy_98pct_proxy_s": MODEL_PROXY,
        },
        "experimental_classification": "GEYSER",
        "strict_physical_outlet_audit": audit["decision"],
        "presentation_conclusion": "original-condition 2D misses the experimental geyser outcome",
        "evidence_sha256": {"metrics": sha256(METRICS), "riser_csv": sha256(RISER_CSV), "physical_rim_audit": sha256(AUDIT)},
        "frames": manifest_records,
        "preview": {"path": str(PREVIEW_PATH.relative_to(HERE)), "native_time_s": series[preview_index][0], "sha256": sha256(PREVIEW_PATH)},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = verify()
    print(f"HTML: {HTML_PATH}")
    print(f"Frames: {report['frame_count']}; size: {report['html_bytes']} bytes")


def verify() -> dict[str, object]:
    if not HTML_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError("Final HTML/manifest does not exist")
    html = HTML_PATH.read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    checks = {
        "artifact_sha256_matches": sha256(HTML_PATH) == manifest["artifact_sha256"],
        "frame_count_201": len(manifest["frames"]) == 201,
        "embedded_png_count_201": html.count("data:image/png;base64,") == 201,
        "native_start_zero": manifest["frames"][0]["native_time_s"] == 0.0,
        "native_end_twenty": manifest["frames"][-1]["native_time_s"] == 20.0,
        "native_shift_zero": manifest["native_clock"]["shift_s"] == 0.0,
        "paper_Ta_present": "8.18 s 论文 Ta" in html,
        "model_Ta_present": "8.04 s 模拟 Ta" in html,
        "inference_disclosed": "不是论文直接报告" in html and "9.991" in html,
        "legacy_proxy_present": "10.08494" in html,
        "miss_disclosed": "原条件 2D 漏报" in html,
        "strict_gate_disclosed": "INDETERMINATE_EVIDENCE_GAP" in html,
        "no_external_frame_links": not re.search(r'<img[^>]+src=["\'](?!data:)', html, flags=re.I),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"HTML verification failed: {failed}")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "artifact": str(HTML_PATH),
        "html_bytes": HTML_PATH.stat().st_size,
        "html_sha256": sha256(HTML_PATH),
        "frame_count": len(manifest["frames"]),
        "native_time_start_s": manifest["frames"][0]["native_time_s"],
        "native_time_end_s": manifest["frames"][-1]["native_time_s"],
        "checks": checks,
    }
    VERIFY_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def self_test() -> None:
    required = [METRICS, RISER_CSV, AUDIT, SHORT_BUILDER]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Self-test missing required evidence: {missing}")
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert metrics["experiment"]["classification"] == "GEYSER"
    assert abs(float(metrics["experiment"]["Ta_s"]) - PAPER_TA) < 1e-12
    assert abs(float(metrics["model"]["Ta_s"]) - MODEL_TA) < 1e-12
    assert abs(float(metrics["model"]["t_free_surface_at_98pct_rim_s"]) - MODEL_PROXY) < 1e-12
    assert audit["decision"]["classification"] == "INDETERMINATE_EVIDENCE_GAP"
    assert audit["decision"]["final_evidence_gate"] is False
    assert abs(EXPERIMENT_OVERRIM_INFERRED - 9.991111111111111) < 1e-12
    assert "__FRAMES__" in HTML_TEMPLATE and "完整 x-z 计算域" in HTML_TEMPLATE
    result = {
        "schema_version": 1,
        "status": "SCRIPT_READY",
        "real_vtk_export_run": False,
        "reason": "live OpenFOAM/reconstruction work detected; renderer prepared without competing for resources",
        "evidence_checks": {
            "experiment_GEYSER": True,
            "paper_Ta_8p18": True,
            "model_Ta_8p04": True,
            "inferred_over_rim_9p991_disclosed": True,
            "legacy_proxy_10p084942": True,
            "strict_physical_gate_false": True,
        },
        "expected_final_artifact": str(HTML_PATH),
        "expected_frame_count": 201,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "renderer_static_self_test.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vtk-root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    selected = sum(bool(value) for value in (args.vtk_root, args.verify_only, args.self_test))
    if selected != 1:
        parser.error("choose exactly one of --vtk-root, --verify-only, or --self-test")
    if args.verify_only:
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
    elif args.self_test:
        self_test()
    else:
        build(args.vtk_root)


if __name__ == "__main__":
    main()

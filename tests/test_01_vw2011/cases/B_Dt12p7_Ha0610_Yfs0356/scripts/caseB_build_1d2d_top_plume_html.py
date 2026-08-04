#!/usr/bin/env python3
"""Build the latest Case B synchronized viewer with a visible outlet air space."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parents[1]
BASE_ASSET_ROOT = CASE_ROOT / "openfoam" / "2d" / "outputs_1d2d_compare"
BASE_MANIFEST = BASE_ASSET_ROOT / "frames_index_tosan2021.json"
TOP_ROOT = CASE_ROOT / "openfoam" / "2d_top_plume" / "outputs_viewer"
TOP_MANIFEST = TOP_ROOT / "frames_index_top_plume.json"
OUTPUT = CASE_ROOT / "caseB_1d2d_frame_compare_top_plume.html"
MAX_SOURCE_TIME_OFFSET = 0.031


def _case_relative(path: Path) -> str:
    return str(path.relative_to(CASE_ROOT)).replace("\\", "/")


def _asset_url(relative_path: str, version: int) -> str:
    normalized = str(relative_path).replace("\\", "/").lstrip("/")
    return f"{normalized}?v={version}"


def _base_asset(relative_path: str) -> str:
    return _case_relative(BASE_ASSET_ROOT / str(relative_path))


def build_frames(base_frames: list[dict], top_frames: list[dict]) -> list[dict]:
    combined = []
    for base in base_frames:
        source_time = float(base["time"])
        item = dict(base)
        item["file1d"] = _base_asset(str(base["file1d"]))
        item["file2dOld"] = _base_asset(str(base["file2d"]))
        item.pop("file2d", None)
        nearest = min(
            top_frames,
            key=lambda frame: abs(float(frame["source_time_s"]) - source_time),
        )
        offset = float(nearest["source_time_s"]) - source_time
        if abs(offset) <= MAX_SOURCE_TIME_OFFSET:
            item.update(
                {
                    "hasTopPlume": True,
                    "fileTopOverlay": str(nearest["file_overlay"]),
                    "fileTopZoom": str(nearest["file_zoom"]),
                    "fileTopSpace": str(nearest["file_space"]),
                    "topLocalTime": float(nearest["local_time_s"]),
                    "topSourceTime": float(nearest["source_time_s"]),
                    "topSourceTimeOffset": offset,
                    "topVisibleWaterCellTopY": nearest.get("visible_water_cell_top_y_m"),
                }
            )
        else:
            item.update(
                {
                    "hasTopPlume": False,
                    "fileTopOverlay": None,
                    "fileTopZoom": None,
                    "fileTopSpace": None,
                    "topLocalTime": None,
                    "topSourceTime": None,
                    "topSourceTimeOffset": None,
                    "topVisibleWaterCellTopY": None,
                }
            )
        combined.append(item)
    return combined


def main(output: Path = OUTPUT) -> None:
    base_frames = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    top_frames = json.loads(TOP_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(base_frames, list) or not base_frames:
        raise ValueError(f"Expected a non-empty frame list in {BASE_MANIFEST}")
    if not isinstance(top_frames, list) or not top_frames:
        raise ValueError(f"Expected a non-empty frame list in {TOP_MANIFEST}")

    asset_version = max(BASE_MANIFEST.stat().st_mtime_ns, TOP_MANIFEST.stat().st_mtime_ns)
    # The latest viewer is intentionally limited to the interval covered by
    # the completed top-only calculation, so every displayed time has a real
    # synchronized exterior-air-domain result.
    viewer_frames = [
        frame for frame in build_frames(base_frames, top_frames)
        if frame["hasTopPlume"]
    ]
    for frame in viewer_frames:
        frame["file1d"] = _asset_url(frame["file1d"], asset_version)
        frame["file2dOld"] = _asset_url(frame["file2dOld"], asset_version)
        if frame["hasTopPlume"]:
            frame["fileTopOverlay"] = _asset_url(frame["fileTopOverlay"], asset_version)
            frame["fileTopZoom"] = _asset_url(frame["fileTopZoom"], asset_version)
            frame["fileTopSpace"] = _asset_url(frame["fileTopSpace"], asset_version)

    frames_json = json.dumps(viewer_frames, ensure_ascii=False, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Case B：Present 1D model 与顶部自由喷流 2D 对比</title>
  <style>
    :root {{ color-scheme: light; font-family: "Times New Roman", "Microsoft YaHei", serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #f4f6f8; color: #151515; }}
    main {{ max-width: 2200px; margin: 0 auto; padding: 22px; }}
    h1 {{ margin: 0 0 7px; font-size: 25px; font-weight: 600; }}
    .note {{ margin: 0 0 10px; color: #414141; font-family: "Microsoft YaHei", sans-serif; font-size: 14px; line-height: 1.55; }}
    .caution {{ padding-left: 11px; border-left: 3px solid #b45309; }}
    .controls {{ padding: 14px 16px; margin: 16px 0; background: #fff; border: 1px solid #c9cdd2; }}
    .button-row {{ display: flex; align-items: center; gap: 9px; flex-wrap: wrap; margin-bottom: 9px; }}
    .button-row:last-child {{ margin-bottom: 0; }}
    button {{ min-width: 88px; padding: 7px 13px; border: 1px solid #60656b; background: #fff; font-family: "Microsoft YaHei", sans-serif; font-size: 14px; cursor: pointer; }}
    button:hover {{ background: #edf2f7; }}
    button.active {{ color: #fff; background: #1d4f91; border-color: #1d4f91; }}
    button:disabled {{ color: #8b9198; background: #f2f3f5; cursor: not-allowed; }}
    button:focus-visible, input:focus-visible {{ outline: 2px solid #1d4f91; outline-offset: 2px; }}
    #frame-info {{ margin-left: 8px; font-size: 16px; font-variant-numeric: tabular-nums; }}
    #state-info {{ margin-top: 10px; color: #30343a; font-size: 15px; font-variant-numeric: tabular-nums; line-height: 1.5; }}
    input[type="range"] {{ width: 100%; margin: 12px 0 1px; accent-color: #1d4f91; }}
    .panels {{ display: grid; grid-template-columns: 1fr 1fr 0.72fr; gap: 16px; align-items: start; }}
    .panel {{ padding: 11px; background: #fff; border: 1px solid #aeb4bc; }}
    h2 {{ margin: 0 0 8px; font-size: 18px; font-weight: 600; }}
    .single-image, .full-stack {{ display: block; width: 100%; border: 1px solid #d4d8dd; background: #fff; }}
    .single-image {{ min-height: 250px; object-fit: contain; }}
    .full-stack {{ position: relative; }}
    .full-stack img {{ display: block; width: 100%; height: auto; }}
    #top-overlay {{ position: absolute; inset: 0; pointer-events: none; }}
    .space-image {{ display: block; width: 100%; height: auto; border: 1px solid #b9c7d1; background: #f2f8fc; }}
    .panel-subnote {{ margin: 8px 0 0; color: #4a5560; font-family: "Microsoft YaHei", sans-serif; font-size: 13px; line-height: 1.45; }}
    .hint {{ margin: 12px 0 0; color: #565656; font-family: "Microsoft YaHei", sans-serif; font-size: 13px; }}
    kbd {{ padding: 1px 5px; border: 1px solid #aeb4bc; background: #fff; font-family: "Times New Roman", serif; }}
    @media (max-width: 1200px) {{ .panels {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .space-panel {{ grid-column: 1 / -1; max-width: 560px; justify-self: center; }} }}
    @media (max-width: 760px) {{ main {{ padding: 12px; }} .panels {{ grid-template-columns: 1fr; }} .space-panel {{ grid-column: auto; }} }}
  </style>
</head>
<body>
<main>
  <h1>Case B：Present 1D model 与 2D OpenFOAM</h1>
  <p class="note">
    三栏画面按同一物理时刻同步：Present 1D model、完整 2D 管网，以及竖管出口上方的 2D 外部空气域。
    完整 2D 视图保留旧 2D 的水平管和竖管结果，并在物理管口
    <i>y</i> = 0.657 m 上方替换为 top-only 外部自由喷流。顶部局部模型入口位于管口下方
    1.4375 mm；其局部时钟与原 2D 时钟满足
    <i>t</i><sub>source</sub> = <i>t</i><sub>local</sub> + 6.5 s。
  </p>
  <p class="note">
    第三栏明确显示管口以上 0.08 m 的空气空间，仅显示 α<sub>water</sub> ≥ 0.01 的水相；
    实体管壁在真实开口处终止，开口上方没有假延伸管、顶盖或封闭边界。
  </p>
  <p class="note caution">
    完整 2D 栏是证据可追溯的展示合成：下部来自旧全域 2D，出口上方来自独立的单向耦合
    top-only 计算，并非同一个双向耦合 CFD 场。第三栏直接显示该 top-only 数值场，不放大液相高度。
  </p>
  <section class="controls" aria-label="帧与视图控制">
    <div class="button-row">
      <button id="previous" type="button">上一帧</button>
      <button id="play" type="button">播放</button>
      <button id="next" type="button">下一帧</button>
      <span id="frame-info" aria-live="polite"></span>
    </div>
    <input id="scrubber" type="range" min="0" max="{len(viewer_frames) - 1}" value="0" step="1" aria-label="帧序号">
    <div id="state-info"></div>
  </section>
  <section class="panels">
    <article class="panel">
      <h2>Present 1D model</h2>
      <img id="present-image" class="single-image" alt="Present 1D model frame">
    </article>
    <article class="panel">
      <h2>2D OpenFOAM — whole system</h2>
      <div id="full-wrap" class="full-stack">
        <img id="old-2d-image" alt="Archived full-domain 2D OpenFOAM frame">
        <img id="top-overlay" alt="One-way top-plume replacement overlay">
      </div>
    </article>
    <article class="panel space-panel">
      <h2>2D OpenFOAM — outlet air space</h2>
      <img id="top-space" class="space-image" alt="Open air space above the 2D outlet">
      <p class="panel-subnote">显示窗口：管口以上 0.08 m；水相几何尺寸未人为放大。</p>
    </article>
  </section>
  <p class="hint">快捷键：<kbd>←</kbd>/<kbd>→</kbd> 逐帧，<kbd>Space</kbd> 播放或暂停。</p>
</main>
<script>
"use strict";
const frames = {frames_json};
let current = 0;
let timer = null;
const scrubber = document.getElementById("scrubber");
const presentImage = document.getElementById("present-image");
const old2dImage = document.getElementById("old-2d-image");
const topOverlay = document.getElementById("top-overlay");
const topSpace = document.getElementById("top-space");
const frameInfo = document.getElementById("frame-info");
const stateInfo = document.getElementById("state-info");
const playButton = document.getElementById("play");

function formatNumber(value, digits = 3) {{
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}}
function preload(index) {{
  if (index < 0 || index >= frames.length) return;
  const frame = frames[index];
  for (const source of [frame.file1d, frame.file2dOld, frame.fileTopOverlay, frame.fileTopSpace]) {{
    if (source) {{ const image = new Image(); image.src = source; }}
  }}
}}
function showFrame(index) {{
  current = Math.max(0, Math.min(frames.length - 1, index));
  const frame = frames[current];
  scrubber.value = String(current);
  presentImage.src = frame.file1d;
  old2dImage.src = frame.file2dOld;
  topOverlay.src = frame.fileTopOverlay;
  topOverlay.style.display = "block";
  topSpace.src = frame.fileTopSpace;

  frameInfo.textContent = `Frame ${{current + 1}}/${{frames.length}}  |  Source time = ${{formatNumber(frame.time, 2)}} s`;
  const topStatus = frame.hasTopPlume
    ? `top local time = ${{formatNumber(frame.topLocalTime, 2)}} s  |  ` +
      `top source time = ${{formatNumber(frame.topSourceTime, 2)}} s  |  ` +
      `Δt = ${{formatNumber(frame.topSourceTimeOffset, 3)}} s  |  ` +
      `visible-water cell top y = ${{formatNumber(frame.topVisibleWaterCellTopY, 3)}} m`
    : "top-only result unavailable before source time 6.5 s";
  stateInfo.textContent = `${{topStatus}}  |  display threshold: α_water ≥ 0.01`;
  preload(current + 1);
}}
function stop() {{
  if (timer !== null) {{ clearInterval(timer); timer = null; }}
  playButton.textContent = "播放";
}}
function togglePlay() {{
  if (timer !== null) {{ stop(); return; }}
  timer = setInterval(() => showFrame((current + 1) % frames.length), 190);
  playButton.textContent = "暂停";
}}
document.getElementById("previous").addEventListener("click", () => {{ stop(); showFrame(current - 1); }});
document.getElementById("next").addEventListener("click", () => {{ stop(); showFrame(current + 1); }});
playButton.addEventListener("click", togglePlay);
scrubber.addEventListener("input", event => {{ stop(); showFrame(Number(event.target.value)); }});
document.addEventListener("keydown", event => {{
  if (event.key === "ArrowLeft") {{ stop(); showFrame(current - 1); }}
  else if (event.key === "ArrowRight") {{ stop(); showFrame(current + 1); }}
  else if (event.key === " ") {{ event.preventDefault(); togglePlay(); }}
}});
showFrame(0);
</script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output} with {len(viewer_frames)} base frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    main(parser.parse_args().output)

#!/usr/bin/env python3
"""Build the Case B Present-model/2D external-plume frame viewer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    CASE_ROOT
    / "openfoam"
    / "2d_external_plume"
    / "outputs_1d2d_compare"
    / "frames_index_external_plume.json"
)
OUTPUT = CASE_ROOT / "caseB_1d2d_frame_compare_external_plume.html"


def _asset_url(relative_path: str, version: int) -> str:
    normalized = str(relative_path).replace("\\", "/").lstrip("/")
    return f"{normalized}?v={version}"


def main(output: Path = OUTPUT) -> None:
    frames = json.loads(SOURCE.read_text(encoding="utf-8"))
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Expected a non-empty frame list in {SOURCE}")

    asset_version = SOURCE.stat().st_mtime_ns
    viewer_frames = []
    for index, frame in enumerate(frames):
        required = {"file1d", "file2d", "time", "sourceTime2d"}
        missing = required.difference(frame)
        if missing:
            raise KeyError(f"Frame {index} is missing {sorted(missing)}")
        item = dict(frame)
        item["file1d"] = _asset_url(item["file1d"], asset_version)
        item["file2d"] = _asset_url(item["file2d"], asset_version)
        viewer_frames.append(item)

    frames_json = json.dumps(
        viewer_frames,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Case B：Present model 与 2D OpenFOAM 对比</title>
  <style>
    :root {{ color-scheme: light; font-family: "Times New Roman", "Microsoft YaHei", serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #f4f6f8; color: #151515; }}
    main {{ max-width: 1800px; margin: 0 auto; padding: 22px; }}
    h1 {{ margin: 0 0 7px; font-size: 25px; font-weight: 600; }}
    .note {{
      margin: 0 0 12px; color: #414141;
      font-family: "Microsoft YaHei", sans-serif; font-size: 14px; line-height: 1.55;
    }}
    .controls {{ padding: 14px 16px; margin: 16px 0; background: #fff; border: 1px solid #c9cdd2; }}
    .buttons {{ display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }}
    button {{
      min-width: 88px; padding: 7px 13px; border: 1px solid #60656b; background: #fff;
      font-family: "Microsoft YaHei", sans-serif; font-size: 14px; cursor: pointer;
    }}
    button:hover {{ background: #edf2f7; }}
    button:focus-visible, input:focus-visible {{ outline: 2px solid #1d4f91; outline-offset: 2px; }}
    #frame-info {{ margin-left: 8px; font-size: 16px; font-variant-numeric: tabular-nums; }}
    #state-info {{ margin-top: 10px; color: #30343a; font-size: 15px; font-variant-numeric: tabular-nums; line-height: 1.5; }}
    input[type="range"] {{ width: 100%; margin: 14px 0 1px; accent-color: #1d4f91; }}
    .panels {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .panel {{ padding: 11px; background: #fff; border: 1px solid #aeb4bc; }}
    h2 {{ margin: 0 0 8px; font-size: 18px; font-weight: 600; }}
    img {{ display: block; width: 100%; min-height: 250px; object-fit: contain; border: 1px solid #d4d8dd; background: #fff; }}
    .hint {{ margin: 12px 0 0; color: #565656; font-family: "Microsoft YaHei", sans-serif; font-size: 13px; }}
    kbd {{ padding: 1px 5px; border: 1px solid #aeb4bc; background: #fff; font-family: "Times New Roman", serif; }}
    @media (max-width: 900px) {{ main {{ padding: 12px; }} .panels {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>Case B：Present model 与 2D OpenFOAM</h1>
  <p class="note">
    两个面板按同一目标时刻配对。右侧二维模型的实体竖管在
    <i>y</i> = 0.657 m 的真实开口处终止，开口上方为外部空气域，因此喷出液体可以横向扩散，
    不再被数值延伸管约束。
  </p>
  <p class="note">
    显示约定：二维计算仍采用面积等效狭缝
    <i>W</i><sub>eq</sub> = <i>D</i><sub>t</sub><sup>2</sup>/<i>D</i> = 1.716 mm；
    绘图仅将管口映射为物理直径 <i>D</i><sub>t</sub> = 12.7 mm，且保持外部空气域边界不变。
    外部空气单元不着色，只显示 α<sub>water</sub> ≥ 0.01 的液体。该映射不修改求解结果。
  </p>
  <section class="controls" aria-label="帧控制">
    <div class="buttons">
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
      <h2>Present model</h2>
      <img id="present-image" alt="Present model frame">
    </article>
    <article class="panel">
      <h2>2D OpenFOAM</h2>
      <img id="openfoam-image" alt="2D OpenFOAM external-plume frame">
    </article>
  </section>
  <p class="hint">
    快捷键：<kbd>←</kbd>/<kbd>→</kbd> 逐帧，<kbd>Space</kbd> 播放或暂停，
    <kbd>Home</kbd>/<kbd>End</kbd> 跳至首尾帧。
  </p>
</main>
<script>
"use strict";
const frames = {frames_json};
let current = 0;
let timer = null;
const scrubber = document.getElementById("scrubber");
const presentImage = document.getElementById("present-image");
const openfoamImage = document.getElementById("openfoam-image");
const frameInfo = document.getElementById("frame-info");
const stateInfo = document.getElementById("state-info");
const playButton = document.getElementById("play");

function firstDefined(frame, names) {{
  for (const name of names) {{
    if (frame[name] !== undefined && frame[name] !== null) return frame[name];
  }}
  return null;
}}
function formatNumber(value, digits = 3) {{
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}}
function preload(index) {{
  if (index < 0 || index >= frames.length) return;
  const first = new Image();
  const second = new Image();
  first.src = frames[index].file1d;
  second.src = frames[index].file2d;
}}
function showFrame(index) {{
  current = Math.max(0, Math.min(frames.length - 1, index));
  const frame = frames[current];
  const interfaceX = firstDefined(frame, ["interface_x", "interfaceX1d"]);
  const airHead = firstDefined(frame, ["air_pressure_head_gauge", "airHead1d"]);
  const mode = firstDefined(frame, ["mode", "mode1d"]) ?? "—";
  scrubber.value = String(current);
  presentImage.src = frame.file1d;
  openfoamImage.src = frame.file2d;
  frameInfo.textContent =
    `Frame ${{current + 1}}/${{frames.length}}  |  Time = ${{formatNumber(frame.time, 2)}} s`;
  stateInfo.textContent =
    `2D source time = ${{formatNumber(frame.sourceTime2d, 2)}} s  |  ` +
    `Δt = ${{formatNumber(frame.dtMatch2d, 3)}} s  |  ` +
    `interface_x = ${{formatNumber(interfaceX)}} m  |  ` +
    `H_air = ${{formatNumber(airHead)}} m  |  mode = ${{String(mode)}}`;
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
  else if (event.key === "Home") {{ stop(); showFrame(0); }}
  else if (event.key === "End") {{ stop(); showFrame(frames.length - 1); }}
  else if (event.key === " ") {{ event.preventDefault(); togglePlay(); }}
}});
showFrame(0);
</script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output} with {len(viewer_frames)} paired frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    main(parser.parse_args().output)

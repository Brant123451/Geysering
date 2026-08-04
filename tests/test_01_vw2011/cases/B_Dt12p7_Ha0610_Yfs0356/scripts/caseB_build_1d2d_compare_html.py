"""Build a self-contained Case B 1D--2D frame comparison viewer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = CASE_ROOT / "openfoam" / "2d" / "outputs_1d2d_compare" / "frames_index.json"
OUTPUT = CASE_ROOT / "caseB_1d2d_frame_compare.html"
ASSET_ROOT = "openfoam/2d/outputs_1d2d_compare/"


def main(output: Path = OUTPUT) -> None:
    frames = json.loads(SOURCE.read_text(encoding="utf-8"))
    asset_version = SOURCE.stat().st_mtime_ns
    for frame in frames:
        frame["file1d"] = (
            ASSET_ROOT
            + frame["file1d"]
            + f"?v={asset_version}"
        )
        frame["file2d"] = ASSET_ROOT + frame["file2d"]
    frames_json = json.dumps(frames, ensure_ascii=False, separators=(",", ":"))

    html = f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Case B: Present model vs 2D OpenFOAM</title>
  <style>
    :root {{ color-scheme: light; font-family: "Times New Roman", "Microsoft YaHei", serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #f4f6f8; color: #151515; }}
    main {{ max-width: 1800px; margin: 0 auto; padding: 22px; }}
    h1 {{ margin: 0 0 7px; font-size: 25px; font-weight: 600; }}
    .note {{ margin: 0 0 16px; color: #414141; font-size: 15px; line-height: 1.45; }}
    .controls {{ padding: 14px 16px; margin-bottom: 16px; background: #fff; border: 1px solid #c9cdd2; }}
    .buttons {{ display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }}
    button {{ min-width: 90px; padding: 7px 13px; border: 1px solid #60656b; background: #fff; font: inherit; cursor: pointer; }}
    button:hover {{ background: #edf2f7; }}
    #info {{ margin-left: 8px; font-size: 16px; font-variant-numeric: tabular-nums; }}
    input[type="range"] {{ width: 100%; margin: 14px 0 1px; accent-color: #1d4f91; }}
    .panels {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .panel {{ padding: 11px; background: #fff; border: 1px solid #aeb4bc; }}
    h2 {{ margin: 0 0 8px; font-size: 18px; font-weight: 600; }}
    img {{ display: block; width: 100%; min-height: 250px; object-fit: contain; border: 1px solid #d4d8dd; background: #fff; }}
    .hint {{ margin: 12px 0 0; color: #565656; font-size: 13px; }}
    kbd {{ padding: 1px 5px; border: 1px solid #aeb4bc; background: #fff; font-family: inherit; }}
    @media (max-width: 900px) {{ main {{ padding: 12px; }} .panels {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>Case B — Present model and 2D OpenFOAM</h1>
  <p class="note">左右两图按同一目标物理时刻配对。拖动滑块可手动调帧；显示的 <i>Time</i> 取 2D 输出时刻。喷发阶段同时显示 Present model 的塔内水面和塔口以上喷柱顶高。</p>
  <section class="controls" aria-label="Frame controls">
    <div class="buttons">
      <button id="previous" type="button">上一帧</button>
      <button id="play" type="button">播放</button>
      <button id="next" type="button">下一帧</button>
      <span id="info"></span>
    </div>
    <input id="scrubber" type="range" min="0" max="{len(frames) - 1}" value="0" step="1" aria-label="Frame number">
  </section>
  <section class="panels">
    <article class="panel"><h2 id="present-label">Present model</h2><img id="present-image" alt="Present model frame"></article>
    <article class="panel"><h2 id="openfoam-label">2D OpenFOAM</h2><img id="openfoam-image" alt="2D OpenFOAM frame"></article>
  </section>
  <p class="hint">快捷键：<kbd>←</kbd> / <kbd>→</kbd> 逐帧；<kbd>Space</kbd> 播放或暂停。</p>
</main>
<script>
const frames = {frames_json};
let current = 0;
let timer = null;
const scrubber = document.getElementById("scrubber");
const presentImage = document.getElementById("present-image");
const openfoamImage = document.getElementById("openfoam-image");
const info = document.getElementById("info");
const playButton = document.getElementById("play");

function showFrame(index) {{
  current = Math.max(0, Math.min(frames.length - 1, index));
  const frame = frames[current];
  scrubber.value = current;
  presentImage.src = frame.file1d;
  openfoamImage.src = frame.file2d;
  document.getElementById("present-label").textContent = "Present model";
  document.getElementById("openfoam-label").textContent = "2D OpenFOAM";
  info.textContent = `Frame ${{current + 1}}/${{frames.length}}  |  Time = ${{frame.time.toFixed(2)}} s  |  Yfs(1D) = ${{frame.wtop1d.toFixed(3)}} m  |  jet top(1D) = ${{frame.jetHeight1d.toFixed(3)}} m  |  matching error = ${{frame.dt_match.toFixed(3)}} s`;
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
  if (event.key === "ArrowRight") {{ stop(); showFrame(current + 1); }}
  if (event.key === " ") {{ event.preventDefault(); togglePlay(); }}
}});
showFrame(0);
</script>
</body>
</html>
'''
    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output} with {len(frames)} paired frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    main(parser.parse_args().output)

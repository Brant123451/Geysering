"""Build a self-contained Case B 1D--2D full-domain comparison viewer.

The full-domain panels retain the physical 12.7 mm tower diameter.  A separate
shaft-detail row is deliberately enlarged laterally so that the reconstructed
1D wall films and the 2D VOF structure remain visible without falsifying the
geometry in the complete-domain view.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = CASE_ROOT / "openfoam" / "2d" / "outputs_1d2d_compare" / "frames_index_tosan2021.json"
OUTPUT = CASE_ROOT / "caseB_1d2d_frame_compare.html"
ASSET_ROOT = CASE_ROOT / "openfoam" / "2d" / "outputs_1d2d_compare"
TSTAR_SCALE_S = 1.7281978911310492


def data_uri(path: Path) -> str:
    """Embed one frame so the viewer is portable as a single HTML file."""
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def main(output: Path = OUTPUT) -> None:
    frames = json.loads(SOURCE.read_text(encoding="utf-8"))
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Expected a non-empty frame list in {SOURCE}")

    for frame in frames:
        path_1d = ASSET_ROOT / frame["file1d"]
        path_2d = ASSET_ROOT / frame["file2d"]
        if not path_1d.is_file() or not path_2d.is_file():
            raise FileNotFoundError(f"Missing paired frame: {path_1d} / {path_2d}")
        frame["file1d"] = data_uri(path_1d)
        frame["file2d"] = data_uri(path_2d)
        frame["tstar"] = float(frame["time"]) / TSTAR_SCALE_S

    frames_json = json.dumps(frames, ensure_ascii=False, separators=(",", ":"))
    html = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Case B：1D模型与2D OpenFOAM完整域对比</title>
  <style>
    :root { color-scheme: light; font-family: "Times New Roman", "Microsoft YaHei", serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f4f6f8; color: #151515; }
    main { max-width: 1800px; margin: 0 auto; padding: 22px; }
    h1 { margin: 0 0 7px; font-size: 25px; font-weight: 600; }
    .note { margin: 0 0 12px; color: #414141; font: 14px/1.55 "Microsoft YaHei", sans-serif; }
    .emphasis { color: #8d2b1f; }
    .controls { padding: 14px 16px; margin: 14px 0 16px; background: #fff; border: 1px solid #c9cdd2; }
    .buttons { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }
    button { min-width: 90px; padding: 7px 13px; border: 1px solid #60656b; background: #fff; font: 14px "Microsoft YaHei", sans-serif; cursor: pointer; }
    button:hover { background: #edf2f7; }
    #info { margin-left: 8px; font-size: 16px; font-variant-numeric: tabular-nums; }
    input[type="range"] { width: 100%; margin: 14px 0 1px; accent-color: #1d4f91; }
    .panels, .details { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .panel, .detail { padding: 11px; background: #fff; border: 1px solid #aeb4bc; }
    h2 { margin: 0 0 8px; font-size: 18px; font-weight: 600; }
    h3 { margin: 0 0 5px; font: 600 16px "Microsoft YaHei", sans-serif; }
    img { display: block; width: 100%; min-height: 250px; object-fit: contain; border: 1px solid #d4d8dd; background: #fff; }
    .detail-heading { margin: 18px 0 8px; font: 600 18px "Microsoft YaHei", sans-serif; }
    .detail p { margin: 0 0 7px; color: #555; font: 13px/1.45 "Microsoft YaHei", sans-serif; }
    canvas { display: block; width: 100%; height: 430px; border: 1px solid #d4d8dd; background: #fff; }
    .hint { margin: 12px 0 0; color: #565656; font: 13px "Microsoft YaHei", sans-serif; }
    kbd { padding: 1px 5px; border: 1px solid #aeb4bc; background: #fff; font-family: inherit; }
    @media (max-width: 900px) {
      main { padding: 12px; }
      .panels, .details { grid-template-columns: 1fr; }
      canvas { height: 390px; }
    }
  </style>
</head>
<body>
<main>
  <h1>Case B — 1D模型与2D OpenFOAM</h1>
  <p class="note">上排为完整计算域，并按相同物理时刻配对。完整域中的竖管使用真实物理直径 <i>D</i><sub>t</sub> = 12.7 mm，没有额外加粗。</p>
  <p class="note emphasis">下排是竖管局部横向放大，仅用于辨认水膜、气核和液塞；不能从放大宽度量取真实管径。1D局部图由截面平均相含率重构，并不表示模型解析了径向界面。</p>
  <section class="controls" aria-label="帧控制">
    <div class="buttons">
      <button id="previous" type="button">上一帧</button>
      <button id="play" type="button">播放</button>
      <button id="next" type="button">下一帧</button>
      <span id="info"></span>
    </div>
    <input id="scrubber" type="range" min="0" max="__MAX_FRAME__" value="0" step="1" aria-label="帧序号">
  </section>
  <section class="panels">
    <article class="panel"><h2>1D present model — 完整域（真实管径）</h2><img id="present-image" alt="一维模型完整计算域"></article>
    <article class="panel"><h2>2D OpenFOAM — 完整域（真实管径映射）</h2><img id="openfoam-image" alt="二维OpenFOAM完整计算域"></article>
  </section>

  <div class="detail-heading">竖管局部放大</div>
  <section class="details">
    <article class="detail">
      <h3>1D截面平均相含率重构</h3>
      <p>气核单元采用 α<sub>g</sub>=0.88、α<sub>l</sub>=0.12；蓝色水膜分置于两侧，每侧物理等效厚度约0.39 mm。</p>
      <canvas id="present-detail" width="520" height="430" aria-label="一维竖管水膜和气核局部放大"></canvas>
    </article>
    <article class="detail">
      <h3>2D VOF原始场局部裁剪</h3>
      <p>从同帧完整域直接裁剪并横向放大；蓝色为水，白色为气体，可观察非对称壁面水体。</p>
      <canvas id="openfoam-detail" width="520" height="430" aria-label="二维竖管VOF局部放大"></canvas>
    </article>
  </section>
  <p class="hint">快捷键：<kbd>←</kbd>/<kbd>→</kbd>逐帧，<kbd>Space</kbd>播放或暂停。</p>
</main>
<script>
"use strict";
const frames = __FRAMES_JSON__;
let current = 0;
let timer = null;
const scrubber = document.getElementById("scrubber");
const presentImage = document.getElementById("present-image");
const openfoamImage = document.getElementById("openfoam-image");
const presentDetail = document.getElementById("present-detail");
const openfoamDetail = document.getElementById("openfoam-detail");
const info = document.getElementById("info");
const playButton = document.getElementById("play");

function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
function finiteOr(value, fallback = 0) { const n = Number(value); return Number.isFinite(n) ? n : fallback; }

function drawOneDDetail(frame) {
  const canvas = presentDetail;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const water = "#2f7ff7", air = "#f7f8fa", wall = "#3f454b";
  const shaftTop = 42, shaftBottom = 370, shaftWidth = 128;
  const xLeft = 0.5 * (W - shaftWidth), xRight = xLeft + shaftWidth;
  const shaftHeight = 0.610;
  const wtop = clamp(finiteOr(frame.wtop1d, 0), 0, shaftHeight);
  const itop = clamp(finiteOr(frame.itop1d, 0), 0, wtop);
  const jetTop = Math.max(finiteOr(frame.jetHeight1d, 0), 0);
  const yOf = z => shaftBottom - clamp(z / shaftHeight, 0, 1) * (shaftBottom - shaftTop);
  const yWater = yOf(wtop), yInterface = yOf(itop);
  const alphaG = 0.88;
  const filmPhysicalFraction = 0.5 * (1 - Math.sqrt(alphaG));
  const filmPx = Math.max(shaftWidth * filmPhysicalFraction, 4);

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = air; ctx.fillRect(xLeft, shaftTop, shaftWidth, shaftBottom - shaftTop);

  if (itop <= 1e-9) {
    ctx.fillStyle = water; ctx.fillRect(xLeft, yWater, shaftWidth, shaftBottom - yWater);
  } else {
    ctx.fillStyle = water;
    if (yInterface > yWater) ctx.fillRect(xLeft, yWater, shaftWidth, yInterface - yWater);
    ctx.fillRect(xLeft, yInterface, filmPx, shaftBottom - yInterface);
    ctx.fillRect(xRight - filmPx, yInterface, filmPx, shaftBottom - yInterface);
  }

  if (jetTop > shaftHeight) {
    const jetWidth = 0.42 * shaftWidth;
    const jetHeightPx = Math.min(26 + 58 * (jetTop - shaftHeight) / shaftHeight, 68);
    ctx.fillStyle = water;
    ctx.fillRect(0.5 * W - 0.5 * jetWidth, shaftTop - jetHeightPx, jetWidth, jetHeightPx);
  }

  ctx.strokeStyle = wall; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(xLeft, shaftBottom); ctx.lineTo(xLeft, shaftTop); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(xRight, shaftBottom); ctx.lineTo(xRight, shaftTop); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(xLeft - 58, shaftTop); ctx.lineTo(xLeft, shaftTop); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(xRight, shaftTop); ctx.lineTo(xRight + 58, shaftTop); ctx.stroke();

  ctx.setLineDash([6, 5]); ctx.lineWidth = 1.4;
  ctx.strokeStyle = "#1d4f91"; ctx.beginPath(); ctx.moveTo(xLeft - 18, yWater); ctx.lineTo(xRight + 18, yWater); ctx.stroke();
  if (itop > 1e-9) {
    ctx.strokeStyle = "#c44933"; ctx.beginPath(); ctx.moveTo(xLeft - 12, yInterface); ctx.lineTo(xRight + 12, yInterface); ctx.stroke();
  }
  ctx.setLineDash([]);

  ctx.font = '15px "Microsoft YaHei", sans-serif'; ctx.fillStyle = "#202428";
  ctx.fillText(`Yfs = ${wtop.toFixed(3)} m`, xRight + 28, clamp(yWater + 5, 58, H - 40));
  ctx.fillText(`Yint = ${itop.toFixed(3)} m`, 18, clamp(yInterface + 5, 75, H - 22));
  ctx.font = '14px "Microsoft YaHei", sans-serif';
  ctx.fillStyle = water; ctx.fillRect(18, 394, 20, 12); ctx.fillStyle = "#202428"; ctx.fillText("水/壁面水膜", 45, 405);
  ctx.strokeStyle = "#9aa0a6"; ctx.strokeRect(195, 394, 20, 12); ctx.fillText("气体", 222, 405);
  ctx.fillStyle = "#6b2020"; ctx.fillText("横向放大示意（非几何比例）", 330, 405);
}

function drawTwoDDetail(image) {
  const canvas = openfoamDetail;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
  if (!image.naturalWidth || !image.naturalHeight) return;
  const sw = image.naturalWidth * 0.055;
  const sx = image.naturalWidth * 0.804 - 0.5 * sw;
  const sy = image.naturalHeight * 0.015;
  const sh = image.naturalHeight * 0.82;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(image, sx, sy, sw, sh, 44, 18, W - 88, H - 60);
  ctx.strokeStyle = "#7a7f85"; ctx.lineWidth = 1.5; ctx.strokeRect(44, 18, W - 88, H - 60);
  ctx.font = '14px "Microsoft YaHei", sans-serif'; ctx.fillStyle = "#6b2020";
  ctx.fillText("同帧原始场裁剪；横向放大（非几何比例）", 112, H - 17);
}

function preload(index) {
  if (index < 0 || index >= frames.length) return;
  const a = new Image(), b = new Image(); a.src = frames[index].file1d; b.src = frames[index].file2d;
}

function showFrame(index) {
  current = Math.max(0, Math.min(frames.length - 1, index));
  const frame = frames[current];
  scrubber.value = String(current);
  presentImage.src = frame.file1d;
  openfoamImage.onload = () => drawTwoDDetail(openfoamImage);
  openfoamImage.src = frame.file2d;
  drawOneDDetail(frame);
  info.textContent = `帧 ${current + 1}/${frames.length} | t=${finiteOr(frame.time).toFixed(2)} s | T*=${finiteOr(frame.tstar).toFixed(2)} | Yfs=${finiteOr(frame.wtop1d).toFixed(3)} m | Yint=${finiteOr(frame.itop1d).toFixed(3)} m`;
  preload(current + 1);
}
function stop() { if (timer !== null) { clearInterval(timer); timer = null; } playButton.textContent = "播放"; }
function togglePlay() { if (timer !== null) { stop(); return; } timer = setInterval(() => showFrame((current + 1) % frames.length), 190); playButton.textContent = "暂停"; }
document.getElementById("previous").addEventListener("click", () => { stop(); showFrame(current - 1); });
document.getElementById("next").addEventListener("click", () => { stop(); showFrame(current + 1); });
playButton.addEventListener("click", togglePlay);
scrubber.addEventListener("input", event => { stop(); showFrame(Number(event.target.value)); });
document.addEventListener("keydown", event => {
  if (event.key === "ArrowLeft") { stop(); showFrame(current - 1); }
  if (event.key === "ArrowRight") { stop(); showFrame(current + 1); }
  if (event.key === " ") { event.preventDefault(); togglePlay(); }
});
showFrame(0);
</script>
</body>
</html>
'''
    html = html.replace("__MAX_FRAME__", str(len(frames) - 1)).replace("__FRAMES_JSON__", frames_json)
    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output} with {len(frames)} paired frames and shaft-detail views.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    main(parser.parse_args().output)

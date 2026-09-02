"""Build a Case-B 1D/2D viewer using the raw 61-cell riser solution.

The archived comparison manifest stores only three scalar riser trajectories
(`wtop`, `itop`, and `jetHeight`).  Reconstructing a shaft from those scalars
necessarily looks like a lumped ODE model.  This candidate reruns the frozen
network solver and embeds its actual axial liquid/gas area-fraction fields in
the detail animation.  No sinusoidal perturbation or cosmetic wave is added.

This is a diagnostic/candidate artifact.  It does not replace the frozen paper
result until the raw-field presentation and its conservation checks are
accepted.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import mimetypes
import sys
from pathlib import Path

import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = CASE_ROOT / "model" / "vw2011_network_twofluid.py"
ASSET_ROOT = CASE_ROOT / "openfoam" / "2d" / "outputs_1d2d_compare"
SOURCE = ASSET_ROOT / "frames_index_tosan2021.json"
OUTPUT = CASE_ROOT / "caseB_1d2d_raw_riser_compare.html"
DIAGNOSTICS = CASE_ROOT / "outputs" / "riser_raw_field_candidate" / "diagnostics.json"
TSTAR_SCALE_S = 1.7281978911310492


def _load_model():
    spec = importlib.util.spec_from_file_location("vw2011_network_twofluid_raw", MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _extrema_count(profile: np.ndarray, threshold: float = 2.0e-3) -> int:
    slope = np.diff(np.asarray(profile, dtype=float))
    slope[np.abs(slope) < threshold] = 0.0
    nz = slope[slope != 0.0]
    if nz.size < 2:
        return 0
    return int(np.count_nonzero(np.sign(nz[1:]) != np.sign(nz[:-1])))


def _run_and_pair(source_frames: list[dict], vertical_candidate: bool = False) -> tuple[list[dict], dict]:
    model = _load_model()
    t_end = max(float(frame["time"]) for frame in source_frames)
    case = model.selected_case(t_end=t_end)
    if vertical_candidate:
        case.enable_vertical_interphase_reaction = True
        case.riser_viscosity_factor = 0.25
    rec = model.run_network(case, verbose=True)

    raw_t = np.asarray(rec["frames_t"], dtype=float)
    raw_al = np.asarray(rec["frames_alr"], dtype=float)
    raw_ag = np.asarray(rec["frames_agr"], dtype=float)
    z = np.asarray(rec["zr"], dtype=float)
    dz = float(rec["dz"])

    paired: list[dict] = []
    phase_volume = []
    for frame in source_frames:
        time = float(frame["time"])
        k = int(np.argmin(np.abs(raw_t - time)))
        al = np.clip(raw_al[k], 0.0, 1.0)
        ag = np.clip(raw_ag[k], 0.0, 1.0)
        total_variation = float(np.sum(np.abs(np.diff(ag))))
        extrema = _extrema_count(ag)
        local_sum = al + ag
        phase_volume.append(
            {
                "time": time,
                "liquid_volume_m3": float(case.Ar * dz * np.sum(al)),
                "resolved_gas_volume_m3": float(case.Ar * dz * np.sum(ag)),
                "max_alpha_sum": float(np.max(local_sum)),
            }
        )
        paired.append(
            {
                "time": time,
                "tstar": time / TSTAR_SCALE_S,
                "file1d": _data_uri(ASSET_ROOT / frame["file1d"]),
                "file2d": _data_uri(ASSET_ROOT / frame["file2d"]),
                "alpha_l": np.round(al, 6).tolist(),
                "alpha_g": np.round(ag, 6).tolist(),
                "z": np.round(z, 6).tolist(),
                "dz": dz,
                "raw_time": float(raw_t[k]),
                "wtop_raw": float(rec["wtop"][k]),
                "itop_raw": float(rec["itop"][k]),
                "gas_mass_raw": float(rec["frames_core_mass"][k]),
                "total_variation": total_variation,
                "extrema": extrema,
            }
        )

    maxima = max(paired, key=lambda item: item["total_variation"])
    diagnostics = {
        "status": "candidate_not_for_paper",
        "model": str(MODEL_PATH),
        "vertical_candidate": vertical_candidate,
        "vertical_interphase_reaction": bool(case.enable_vertical_interphase_reaction),
        "riser_repack": "instantaneous_saturation_preserving",
        "riser_viscosity_factor": float(case.riser_viscosity_factor),
        "riser_cells": int(raw_al.shape[1]),
        "dz_m": dz,
        "frames": len(paired),
        "max_total_variation": maxima["total_variation"],
        "max_total_variation_time_s": maxima["time"],
        "max_extrema_count": max(item["extrema"] for item in paired),
        "phase_volume": phase_volume,
        "note": (
            ("Vertical candidate with equal-and-opposite interphase drag reaction and "
             "25% of the frozen riser artificial momentum viscosity. "
             if vertical_candidate else "Raw area-fraction output from the frozen network solver. ")
            + "No imposed sinusoid. The full-domain 1D PNG still uses the archived "
              "horizontal renderer; the enlarged shaft panel is the raw 61-cell field."
        ),
    }
    return paired, diagnostics


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Case B：1D 原始竖管场与 2D OpenFOAM 对比</title>
  <style>
    :root { color-scheme: light; font-family: "Times New Roman", "Microsoft YaHei", serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f4f6f8; color: #171717; }
    main { max-width: 1800px; margin: 0 auto; padding: 20px; }
    h1 { margin: 0 0 7px; font-size: 25px; font-weight: 600; }
    .note { margin: 4px 0; font: 14px/1.55 "Microsoft YaHei", sans-serif; color: #3f4347; }
    .warning { color: #8c2f23; }
    .controls { margin: 14px 0 16px; padding: 13px 15px; background: #fff; border: 1px solid #b7bdc4; }
    .buttons { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }
    button { min-width: 82px; padding: 7px 12px; background: #fff; border: 1px solid #5e646a; cursor: pointer; font: 14px "Microsoft YaHei", sans-serif; }
    button:hover { background: #edf2f7; }
    #info { margin-left: 7px; font-size: 15px; font-variant-numeric: tabular-nums; }
    input[type="range"] { width: 100%; margin: 13px 0 0; accent-color: #1d4f91; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 15px; }
    .panel { padding: 10px; background: #fff; border: 1px solid #aeb4bc; }
    h2 { margin: 0 0 7px; font-size: 18px; font-weight: 600; }
    .panel p { margin: 0 0 6px; color: #555; font: 13px/1.45 "Microsoft YaHei", sans-serif; }
    img { display: block; width: 100%; min-height: 245px; object-fit: contain; border: 1px solid #d4d8dd; background: #fff; }
    .detail-title { margin: 17px 0 8px; font: 600 18px "Microsoft YaHei", sans-serif; }
    canvas { display: block; width: 100%; height: 490px; border: 1px solid #d4d8dd; background: #fff; }
    .legend { display: flex; gap: 17px; margin-top: 7px; font: 13px "Microsoft YaHei", sans-serif; }
    .swatch { display: inline-block; width: 18px; height: 11px; margin-right: 5px; vertical-align: -1px; border: 1px solid #8d939a; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } main { padding: 11px; } }
  </style>
</head>
<body>
<main>
  <h1>Case B：1D 原始竖管双流体场 / 2D OpenFOAM</h1>
  <p class="note">下方 1D 竖管逐格读取求解器的 61 个轴向网格：液相面积分数 α<sub>l</sub>(z,t) 与解析气相面积分数 α<sub>g</sub>(z,t)。没有使用固定 α<sub>g</sub>=0.88，也没有叠加正弦曲线。</p>
  <p class="note warning">__MODEL_NOTE__ 横向放大只用于看清相含率波动；上方完整域保持真实几何比例。确认后再决定是否替换论文冻结结果。</p>
  <section class="controls">
    <div class="buttons">
      <button id="prev" type="button">上一帧</button>
      <button id="play" type="button">播放</button>
      <button id="next" type="button">下一帧</button>
      <span id="info"></span>
    </div>
    <input id="scrub" type="range" min="0" max="__MAX__" value="0" step="1">
  </section>

  <section class="grid">
    <article class="panel"><h2>1D 完整域（真实管径）</h2><img id="one-d" alt="一维模型完整域"></article>
    <article class="panel"><h2>2D OpenFOAM 完整域</h2><img id="two-d" alt="二维OpenFOAM完整域"></article>
  </section>

  <div class="detail-title">竖管局部横向放大（同一物理时刻）</div>
  <section class="grid">
    <article class="panel">
      <h2>1D 原始 61 网格场</h2>
      <p>每个高度的气核宽度由该网格的 α<sub>g</sub> 映射，剩余截面为液相壁膜/液塞；边界随原始网格值移动。</p>
      <canvas id="raw-riser" width="650" height="490"></canvas>
      <div class="legend"><span><i class="swatch" style="background:#2f7ff7"></i>液相</span><span><i class="swatch" style="background:#f7f8fa"></i>气相</span><span><i class="swatch" style="background:#c94935"></i>局部峰谷标记</span></div>
    </article>
    <article class="panel">
      <h2>2D VOF 原始场裁剪</h2>
      <p>从同帧完整域直接裁剪；蓝色为水，白色为空气。</p>
      <canvas id="vof-riser" width="650" height="490"></canvas>
    </article>
  </section>
</main>
<script>
"use strict";
const frames = __FRAMES__;
let current = 0, timer = null;
const oneD = document.getElementById("one-d"), twoD = document.getElementById("two-d");
const rawCanvas = document.getElementById("raw-riser"), vofCanvas = document.getElementById("vof-riser");
const scrub = document.getElementById("scrub"), info = document.getElementById("info"), play = document.getElementById("play");

function localExtrema(a) {
  const result = [];
  for (let i = 1; i < a.length - 1; i++) {
    const d0 = a[i] - a[i - 1], d1 = a[i + 1] - a[i];
    if (Math.abs(d0) > 0.002 && Math.abs(d1) > 0.002 && d0 * d1 < 0) result.push(i);
  }
  return result;
}

function drawRaw(frame) {
  const c = rawCanvas, x = c.getContext("2d"), W = c.width, H = c.height;
  const top = 28, bottom = 445, left = 205, width = 190, right = left + width;
  const zmax = 0.610, al = frame.alpha_l, ag = frame.alpha_g, dz = frame.dz;
  const y = z => bottom - Math.max(0, Math.min(1, z / zmax)) * (bottom - top);
  x.clearRect(0, 0, W, H); x.fillStyle = "#fff"; x.fillRect(0, 0, W, H);
  x.fillStyle = "#f7f8fa"; x.fillRect(left, top, width, bottom - top);

  for (let i = 0; i < al.length; i++) {
    const z0 = Math.max(0, frame.z[i] - 0.5 * dz), z1 = Math.min(zmax, frame.z[i] + 0.5 * dz);
    const yt = y(z1), yb = y(z0), liquid = Math.max(0, Math.min(1, al[i]));
    if (liquid <= 0.001) continue;
    x.fillStyle = "#2f7ff7"; x.fillRect(left, yt, width, Math.max(1, yb - yt + 0.4));
    const gas = Math.max(0, Math.min(1, ag[i]));
    if (gas > 0.001) {
      const coreWidth = width * Math.sqrt(gas);
      x.fillStyle = "#f7f8fa";
      x.fillRect(0.5 * (left + right - coreWidth), yt, coreWidth, Math.max(1, yb - yt + 0.4));
    }
  }

  x.strokeStyle = "#34393e"; x.lineWidth = 2;
  x.beginPath(); x.moveTo(left, bottom); x.lineTo(left, top); x.stroke();
  x.beginPath(); x.moveTo(right, bottom); x.lineTo(right, top); x.stroke();
  x.beginPath(); x.moveTo(left - 58, top); x.lineTo(left, top); x.stroke();
  x.beginPath(); x.moveTo(right, top); x.lineTo(right + 58, top); x.stroke();

  x.strokeStyle = "rgba(45,55,65,0.16)"; x.lineWidth = 0.7;
  for (let i = 0; i <= al.length; i++) {
    const yy = y(i * dz); x.beginPath(); x.moveTo(left, yy); x.lineTo(right, yy); x.stroke();
  }

  const extrema = localExtrema(ag);
  x.fillStyle = "#c94935";
  for (const i of extrema) {
    const gas = Math.max(0, Math.min(1, ag[i]));
    if (gas < 0.01) continue;
    const coreWidth = width * Math.sqrt(gas);
    x.beginPath(); x.arc(0.5 * (left + right + coreWidth), y(frame.z[i]), 3.2, 0, 2 * Math.PI); x.fill();
  }

  x.fillStyle = "#24282c"; x.font = '14px "Microsoft YaHei", sans-serif';
  for (const ztick of [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) {
    const yy = y(ztick); x.fillText(ztick.toFixed(1), 155, yy + 4);
    x.strokeStyle = "#6c7278"; x.beginPath(); x.moveTo(left - 6, yy); x.lineTo(left, yy); x.stroke();
  }
  x.save(); x.translate(112, 260); x.rotate(-Math.PI / 2); x.fillText("z [m]", 0, 0); x.restore();
  x.fillText(`原始场时刻 ${frame.raw_time.toFixed(3)} s`, 430, 66);
  x.fillText(`TV(αg) = ${frame.total_variation.toFixed(3)}`, 430, 92);
  x.fillText(`内部峰谷 = ${frame.extrema}`, 430, 118);
  x.fillText(`m_g = ${(1000 * frame.gas_mass_raw).toFixed(3)} g`, 430, 144);
  x.fillStyle = "#8c2f23"; x.fillText("红点仅标记原始局部峰谷", 430, 176);
}

function drawVof(image) {
  const c = vofCanvas, x = c.getContext("2d"), W = c.width, H = c.height;
  x.clearRect(0, 0, W, H); x.fillStyle = "#fff"; x.fillRect(0, 0, W, H);
  if (!image.naturalWidth || !image.naturalHeight) return;
  const sw = image.naturalWidth * 0.058;
  const sx = image.naturalWidth * 0.804 - 0.5 * sw;
  const sy = image.naturalHeight * 0.012;
  const sh = image.naturalHeight * 0.84;
  x.imageSmoothingEnabled = false;
  x.drawImage(image, sx, sy, sw, sh, 92, 18, W - 184, H - 56);
  x.strokeStyle = "#7a7f85"; x.lineWidth = 1.5; x.strokeRect(92, 18, W - 184, H - 56);
}

function show(index) {
  current = Math.max(0, Math.min(frames.length - 1, index));
  const f = frames[current]; scrub.value = String(current);
  oneD.src = f.file1d;
  twoD.onload = () => drawVof(twoD); twoD.src = f.file2d;
  drawRaw(f);
  info.textContent = `帧 ${current + 1}/${frames.length} | t=${f.time.toFixed(2)} s | T*=${f.tstar.toFixed(2)} | TV(αg)=${f.total_variation.toFixed(3)} | 峰谷=${f.extrema}`;
}
function stop() { if (timer !== null) { clearInterval(timer); timer = null; } play.textContent = "播放"; }
function toggle() { if (timer !== null) { stop(); return; } timer = setInterval(() => show((current + 1) % frames.length), 190); play.textContent = "暂停"; }
document.getElementById("prev").onclick = () => { stop(); show(current - 1); };
document.getElementById("next").onclick = () => { stop(); show(current + 1); };
play.onclick = toggle; scrub.oninput = event => { stop(); show(Number(event.target.value)); };
document.addEventListener("keydown", event => {
  if (event.key === "ArrowLeft") { stop(); show(current - 1); }
  if (event.key === "ArrowRight") { stop(); show(current + 1); }
  if (event.key === " ") { event.preventDefault(); toggle(); }
});
show(0);
</script>
</body>
</html>
'''


def main(output: Path = OUTPUT, vertical_candidate: bool = False) -> None:
    source_frames = json.loads(SOURCE.read_text(encoding="utf-8"))
    if not isinstance(source_frames, list) or not source_frames:
        raise ValueError(f"Expected a non-empty frame list in {SOURCE}")
    paired, diagnostics = _run_and_pair(source_frames, vertical_candidate=vertical_candidate)
    diagnostics_path = (
        CASE_ROOT / "outputs" / "riser_vertical_coupling_candidate" / "diagnostics.json"
        if vertical_candidate else DIAGNOSTICS
    )
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = json.dumps(paired, ensure_ascii=False, separators=(",", ":"))
    model_note = (
        "竖管闭合候选：气液阻力采用等量反作用，人工动量黏性为冻结值的 25%；瞬时饱和重排保留。"
        if vertical_candidate else "冻结方程原始场显示版：未改变控制方程或闭合。"
    )
    html = (HTML_TEMPLATE.replace("__MAX__", str(len(paired) - 1))
            .replace("__FRAMES__", payload).replace("__MODEL_NOTE__", model_note))
    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output} with {len(paired)} paired frames")
    print(f"Wrote {diagnostics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--vertical-candidate", action="store_true")
    args = parser.parse_args()
    main(args.output, vertical_candidate=args.vertical_candidate)

#!/usr/bin/env python3
"""Build a synchronized B-H1/B-H3/B-H6 1D–OpenFOAM 2D viewer."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE.parents[1]
CASES = TEST_ROOT / "cases"
HTML_PATH = HERE / "bh1_bh3_bh6_1d_animation.html"
MANIFEST_PATH = HERE / "viewer_manifest.json"
REPAIRED_ROOT = HERE / "case1_model_rerun"
REPAIRED_MODEL_ROOT = REPAIRED_ROOT / "model_1d"
REPAIRED_FRAME_ROOT = REPAIRED_ROOT / "frames"
FRAME_RENDERER = (
    CASES
    / "BH3_Dr26_H066_L061"
    / "openfoam"
    / "2d"
    / "comparison"
    / "build_1d_frames.py"
)
ASSET_VERSION = "case1-horizontal-v3-frozen-20260809"


CASE_SOURCES = [
    {
        "id": "BH1",
        "label": "B-H1",
        "detail": "Dr = 16 mm · Dr/D = 0.32 · 1D未喷发 / 2D未喷发 / 实验喷发",
        "valve": "0.20 s",
        "Dr": 0.016,
        "npz": REPAIRED_MODEL_ROOT / "BH1" / "case1_horizontal_1d_frames.npz",
        "index_1d": REPAIRED_FRAME_ROOT / "BH1" / "frames.json",
        "two_d": {
            "mode": "pattern",
            "directory": CASES
            / "BH1_Dr16_H066_L061"
            / "openfoam"
            / "2d"
            / "frame_compare"
            / "two_d_frames",
            "pattern": "full_*.png",
            "step": 0.1,
            "source": "OpenFOAM formal 13 s run (paper_tau0p2_areaeq)",
        },
    },
    {
        "id": "BH3",
        "label": "B-H3",
        "detail": "Dr = 26 mm · Dr/D = 0.52 · 1D未喷发 / 2D未喷发 / 实验喷发",
        "valve": "0.20 s",
        "Dr": 0.026,
        "npz": REPAIRED_MODEL_ROOT / "BH3" / "case1_horizontal_1d_frames.npz",
        "index_1d": REPAIRED_FRAME_ROOT / "BH3" / "frames.json",
        "two_d": {
            "mode": "index",
            "index": CASES
            / "BH3_Dr26_H066_L061"
            / "openfoam"
            / "2d"
            / "comparison"
            / "openfoam_2d"
            / "frames.json",
            "base": CASES
            / "BH3_Dr26_H066_L061"
            / "openfoam"
            / "2d"
            / "comparison",
            "file_key": "file",
            "source": "OpenFOAM completed 13 s comparison run",
        },
    },
    {
        "id": "BH6",
        "label": "B-H6",
        "detail": "Dr = 41 mm · Dr/D = 0.82 · 1D未喷发 / 2D未喷发 / 实验未喷发",
        "valve": "0.20 s",
        "Dr": 0.041,
        "npz": REPAIRED_MODEL_ROOT / "BH6" / "case1_horizontal_1d_frames.npz",
        "index_1d": REPAIRED_FRAME_ROOT / "BH6" / "frames.json",
        "two_d": {
            "mode": "index",
            "index": CASES
            / "BH6_Dr41_H066_L061"
            / "outputs"
            / "1d2d_viewer"
            / "frames_2d.json",
            "base": CASES / "BH6_Dr41_H066_L061",
            "file_key": "file",
            "source": "OpenFOAM completed 13 s comparison run",
        },
    },
]


def local_href(path: Path) -> str:
    """Return a slash-normalized path relative to the generated HTML."""
    return Path(os.path.relpath(path, HERE)).as_posix()


def render_repaired_viewer_frames() -> None:
    """Render all repaired 1D archives with one geometry-preserving style."""
    module_spec = importlib.util.spec_from_file_location(
        "campaign2_frame_renderer", FRAME_RENDERER
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load frame renderer: {FRAME_RENDERER}")
    renderer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(renderer)
    renderer.DISPLAY_TOP = 1.90
    # Display-only vertical exaggeration. Model fields and axial coordinates
    # remain unchanged; it keeps the 50 mm tunnel visible beside the riser.
    renderer.PIPE_D = 0.20
    targets = np.arange(0.0, 13.0001, 0.05)
    for spec in CASE_SOURCES:
        data = np.load(Path(spec["npz"]), allow_pickle=False)
        renderer.RISER_D = float(spec["Dr"])
        output_dir = Path(spec["index_1d"]).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        source_times = np.asarray(data["frames_t"], dtype=float)
        indices = np.asarray(
            [int(np.argmin(np.abs(source_times - target))) for target in targets]
        )
        rows: list[dict[str, float | str]] = []
        for frame_no, source_index in enumerate(indices):
            time_s = float(source_times[source_index])
            output = output_dir / f"pipe_visible_{frame_no:04d}.svg"
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
            ).replace("B-H3", str(spec["label"]))
            svg = svg.replace(
                "Pipe and riser use the paper dimensions",
                "",
            )
            output.write_text(svg, encoding="utf-8")
            rows.append({"time": time_s, "file": local_href(output)})
        Path(spec["index_1d"]).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def validate_frames(frames: list[dict[str, object]], label: str) -> None:
    if not frames:
        raise ValueError(f"No frames for {label}")
    frames.sort(key=lambda item: float(item["time"]))
    times = [float(item["time"]) for item in frames]
    if abs(times[0]) > 1.0e-6 or times[-1] < 13.0 - 1.0e-6:
        raise ValueError(f"{label} does not cover 0–13 s: {times[0]}–{times[-1]}")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f"{label} frame times are not strictly increasing")


def load_1d_frames(spec: dict[str, object]) -> list[dict[str, object]]:
    index = Path(spec["index_1d"])
    rows = json.loads(index.read_text(encoding="utf-8"))
    frames: list[dict[str, object]] = []
    for row in rows:
        source = (HERE / str(row["file"])).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        frames.append(
            {
                "time": float(row["time"]),
                "src": f"{local_href(source)}?v={ASSET_VERSION}",
            }
        )
    validate_frames(frames, f"{spec['id']} 1D")
    return frames


def load_2d_frames(spec: dict[str, object]) -> tuple[list[dict[str, object]], str]:
    source_spec = dict(spec["two_d"])
    frames: list[dict[str, object]] = []
    if source_spec["mode"] == "pattern":
        directory = Path(source_spec["directory"])
        files = sorted(directory.glob(str(source_spec["pattern"])))
        step = float(source_spec["step"])
        for number, source in enumerate(files):
            frames.append({"time": number * step, "src": local_href(source.resolve())})
        index_label = local_href(directory)
    elif source_spec["mode"] == "index":
        index = Path(source_spec["index"])
        rows = json.loads(index.read_text(encoding="utf-8"))
        base = Path(source_spec["base"])
        file_key = str(source_spec["file_key"])
        for row in rows:
            source = (base / str(row[file_key])).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            frames.append({"time": float(row["time"]), "src": local_href(source)})
        index_label = local_href(index)
    else:
        raise ValueError(f"Unknown 2D source mode: {source_spec['mode']}")
    validate_frames(frames, f"{spec['id']} OpenFOAM 2D")
    return frames, index_label


def load_case(spec: dict[str, object]) -> dict[str, object]:
    frames_1d = load_1d_frames(spec)
    frames_2d, index_2d = load_2d_frames(spec)
    return {
        "id": spec["id"],
        "label": spec["label"],
        "detail": spec["detail"],
        "valve": spec["valve"],
        "index1d": local_href(Path(spec["index_1d"])),
        "index2d": index_2d,
        "source2d": dict(spec["two_d"])["source"],
        "count1d": len(frames_1d),
        "count2d": len(frames_2d),
        "start1d": frames_1d[0]["time"],
        "end1d": frames_1d[-1]["time"],
        "start2d": frames_2d[0]["time"],
        "end2d": frames_2d[-1]["time"],
        "frames1d": frames_1d,
        "frames2d": frames_2d,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cong 2017 · B-H1 / B-H3 / B-H6 · 1D与OpenFOAM 2D同步对比</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17212b;
      --muted: #66717e;
      --line: #d8e0e8;
      --panel: #ffffff;
      --page: #eef3f7;
      --blue: #1268d3;
      --blue-soft: #e8f2ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--page);
      color: var(--ink);
      font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
    }
    .page { width: min(1880px, 100%); margin: 0 auto; padding: 18px; }
    header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 12px;
    }
    h1 { margin: 0; font-size: clamp(20px, 2.2vw, 31px); font-weight: 730; }
    .subtitle { margin-top: 5px; color: var(--muted); font-size: 14px; }
    .time-badge {
      min-width: 142px;
      padding: 10px 15px;
      border-radius: 12px;
      background: var(--blue);
      color: white;
      font-size: 22px;
      font-variant-numeric: tabular-nums;
      text-align: center;
      box-shadow: 0 5px 18px rgba(18,104,211,.18);
    }
    .controls {
      position: sticky;
      top: 8px;
      z-index: 20;
      display: grid;
      grid-template-columns: auto auto auto minmax(180px, 1fr) auto;
      gap: 9px;
      align-items: center;
      padding: 11px;
      margin-bottom: 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,.96);
      box-shadow: 0 7px 25px rgba(31,50,73,.09);
      backdrop-filter: blur(8px);
    }
    button, select {
      min-height: 38px;
      border: 1px solid #bac7d4;
      border-radius: 9px;
      background: white;
      color: var(--ink);
      font: inherit;
      padding: 0 12px;
      cursor: pointer;
    }
    button:hover { border-color: var(--blue); background: var(--blue-soft); }
    button.primary {
      min-width: 90px;
      color: white;
      border-color: var(--blue);
      background: var(--blue);
    }
    input[type="range"] { width: 100%; accent-color: var(--blue); cursor: pointer; }
    .viewer { display: grid; grid-template-columns: 1fr; gap: 14px; }
    .case {
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 13px;
      background: var(--panel);
      box-shadow: 0 4px 15px rgba(31,50,73,.06);
    }
    .case-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 13px;
      border-bottom: 1px solid var(--line);
      background: #f8fafc;
    }
    .case-name { font-size: 18px; font-weight: 720; }
    .case-detail { margin-left: 8px; color: var(--muted); font-size: 13px; font-weight: 450; }
    .case-time { color: var(--blue); font-variant-numeric: tabular-nums; font-weight: 650; }
    .compare-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      padding: 10px;
      background: #f3f6f9;
    }
    .result-pane {
      min-width: 0;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
    }
    .pane-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
      font-weight: 700;
      background: #fbfcfe;
    }
    .pane-time { color: var(--muted); font-weight: 550; font-variant-numeric: tabular-nums; }
    .frame-wrap {
      display: grid;
      place-items: center;
      height: clamp(260px, 28vw, 440px);
      background: white;
    }
    .frame {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: white;
    }
    .case-foot {
      padding: 8px 13px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }
    footer { margin-top: 12px; color: var(--muted); font-size: 12px; text-align: center; }
    @media (max-width: 820px) {
      .page { padding: 10px; }
      header { display: block; }
      .time-badge { margin-top: 10px; width: 100%; }
      .controls { grid-template-columns: repeat(3, auto); }
      .controls input { grid-column: 1 / -1; grid-row: 2; }
      .compare-grid { grid-template-columns: 1fr; }
      .case-head { align-items: flex-start; }
      .case-detail { display: block; margin: 2px 0 0; }
    }
  </style>
</head>
<body>
<main class="page">
  <header>
    <div>
      <h1>Cong et al. (2017) Series B · 三工况1D / OpenFOAM 2D同步对比</h1>
      <div class="subtitle">B-H1、B-H3、B-H6 · 完整横管—竖管域 · 公共物理时钟 0–13 s</div>
    </div>
    <div class="time-badge" id="timeBadge">t = 0.00 s</div>
  </header>

  <div style="margin:0 0 14px;padding:12px 16px;border:1px solid #f59e0b;border-radius:10px;background:#fff7ed;color:#9a3412;font-weight:700;line-height:1.55">
    ⚠ 本页是已废弃的 0–13 s 截断版，不可用于判定是否喷发或写入论文。1D 已重算至完整事件；H3/H6 的 OpenFOAM 2D 正在从 13 s 续算。
  </div>

  <section class="controls" aria-label="动画控制">
    <button class="primary" id="playButton" type="button">▶ 播放</button>
    <button id="backButton" type="button" title="后退0.1秒">−0.1 s</button>
    <button id="forwardButton" type="button" title="前进0.1秒">+0.1 s</button>
    <input id="timeSlider" type="range" min="0" max="130" step="1" value="0" aria-label="时间">
    <select id="speedSelect" aria-label="播放速度">
      <option value="0.5">0.5×</option>
      <option value="1" selected>1×</option>
      <option value="2">2×</option>
      <option value="4">4×</option>
    </select>
  </section>

  <section class="viewer" id="viewer"></section>
  <footer>蓝色为水相，白色或浅灰为气相；1D与2D均按最接近公共时刻的原始帧显示，未作时间平移。2D采用截面积等效平面模型。</footer>
</main>

<script>
  const CASES = __CASE_DATA__;
  const START = 0;
  const END = 13;
  const DT = 0.1;
  let time = 0;
  let playing = false;
  let lastStamp = null;

  const viewer = document.getElementById('viewer');
  const slider = document.getElementById('timeSlider');
  const timeBadge = document.getElementById('timeBadge');
  const playButton = document.getElementById('playButton');
  const speedSelect = document.getElementById('speedSelect');

  function nearestFrame(frames, target) {
    let lo = 0, hi = frames.length - 1;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (frames[mid].time < target) lo = mid + 1;
      else hi = mid;
    }
    if (lo === 0) return frames[0];
    const before = frames[lo - 1];
    const after = frames[lo];
    return Math.abs(before.time - target) <= Math.abs(after.time - target) ? before : after;
  }

  for (const item of CASES) {
    const card = document.createElement('article');
    card.className = 'case';
    card.innerHTML = `
      <div class="case-head">
        <div><span class="case-name">${item.label}</span><span class="case-detail">${item.detail}</span></div>
        <span class="case-time" id="${item.id}-time">t = 0.00 s</span>
      </div>
      <div class="compare-grid">
        <section class="result-pane">
          <div class="pane-head"><span>1D 模型</span><span class="pane-time" id="${item.id}-1d-time">0.00 s</span></div>
          <div class="frame-wrap"><img class="frame" id="${item.id}-1d-image" alt="${item.label} 1D模拟帧"></div>
        </section>
        <section class="result-pane">
          <div class="pane-head"><span>OpenFOAM 2D</span><span class="pane-time" id="${item.id}-2d-time">0.00 s</span></div>
          <div class="frame-wrap"><img class="frame" id="${item.id}-2d-image" alt="${item.label} OpenFOAM 2D模拟帧"></div>
        </section>
      </div>
      <div class="case-foot">1D：${item.count1d}帧 · 2D：${item.count2d}帧 · 阀门开启时间：${item.valve} · 原始物理时钟，无时间平移</div>`;
    viewer.appendChild(card);
  }

  function preloadFrames(frames, target) {
    for (const delta of [-0.2, 0.2, 0.4]) {
      const frame = nearestFrame(frames, Math.max(START, Math.min(END, target + delta)));
      const image = new Image();
      image.src = frame.src;
    }
  }

  function render() {
    time = Math.max(START, Math.min(END, time));
    slider.value = String(Math.round(time / DT));
    timeBadge.textContent = `t = ${time.toFixed(2)} s`;
    for (const item of CASES) {
      const frame1d = nearestFrame(item.frames1d, time);
      const frame2d = nearestFrame(item.frames2d, time);
      const image1d = document.getElementById(`${item.id}-1d-image`);
      const image2d = document.getElementById(`${item.id}-2d-image`);
      if (image1d.getAttribute('src') !== frame1d.src) image1d.setAttribute('src', frame1d.src);
      if (image2d.getAttribute('src') !== frame2d.src) image2d.setAttribute('src', frame2d.src);
      document.getElementById(`${item.id}-1d-time`).textContent = `${frame1d.time.toFixed(2)} s`;
      document.getElementById(`${item.id}-2d-time`).textContent = `${frame2d.time.toFixed(2)} s`;
      document.getElementById(`${item.id}-time`).textContent = `公共时刻 ${time.toFixed(2)} s`;
      preloadFrames(item.frames1d, time);
      preloadFrames(item.frames2d, time);
    }
  }

  function setPlaying(value) {
    playing = value;
    playButton.textContent = playing ? 'Ⅱ 暂停' : '▶ 播放';
    playButton.setAttribute('aria-pressed', String(playing));
    lastStamp = null;
    if (playing) requestAnimationFrame(tick);
  }

  function tick(stamp) {
    if (!playing) return;
    if (lastStamp === null) lastStamp = stamp;
    const elapsed = (stamp - lastStamp) / 1000;
    if (elapsed >= 0.095) {
      const speed = Number(speedSelect.value);
      const steps = Math.max(1, Math.floor(elapsed / 0.1));
      time += steps * DT * speed;
      lastStamp = stamp;
      if (time > END + 1e-9) time = START;
      time = Math.round(time * 10) / 10;
      render();
    }
    requestAnimationFrame(tick);
  }

  playButton.addEventListener('click', () => setPlaying(!playing));
  document.getElementById('backButton').addEventListener('click', () => { time -= DT; render(); });
  document.getElementById('forwardButton').addEventListener('click', () => { time += DT; render(); });
  slider.addEventListener('input', () => { time = Number(slider.value) * DT; render(); });
  document.addEventListener('keydown', (event) => {
    if (event.code === 'Space') { event.preventDefault(); setPlaying(!playing); }
    if (event.code === 'ArrowLeft') { time -= DT; render(); }
    if (event.code === 'ArrowRight') { time += DT; render(); }
  });

  render();
</script>
</body>
</html>
'''


def main() -> None:
    render_repaired_viewer_frames()
    cases = [load_case(spec) for spec in CASE_SOURCES]
    payload = json.dumps(cases, ensure_ascii=False, separators=(",", ":"))
    HTML_PATH.write_text(HTML_TEMPLATE.replace("__CASE_DATA__", payload), encoding="utf-8")

    manifest_cases = []
    for case in cases:
        manifest_cases.append(
            {
                key: value
                for key, value in case.items()
                if key
                in {
                    "id",
                    "label",
                    "detail",
                    "valve",
                    "index1d",
                    "index2d",
                    "source2d",
                    "count1d",
                    "count2d",
                    "start1d",
                    "end1d",
                    "start2d",
                    "end2d",
                }
            }
        )
    manifest = {
        "artifact": HTML_PATH.name,
        "purpose": "synchronized comparison of Case-1-core horizontal 1D reruns and completed OpenFOAM 2D results",
        "common_display_clock_s": {"start": 0.0, "end": 13.0, "step": 0.1},
        "cases": manifest_cases,
        "common_conditions": "paper H0=0.66 m from tunnel invert; model crown datum=0.61 m; valve_open_time=0.20 s",
        "outcomes_at_13s": "BH1: 1D/2D no geyser, experiment geyser; BH3: 1D/2D no geyser, experiment geyser; BH6: 1D/2D/experiment no geyser",
        "time_alignment": "native physical time, nearest frame, no time shift",
        "viewer_only_adjustment": "1D horizontal-pipe drawing uses 4x vertical exaggeration for readability; axial geometry and simulation states are unchanged",
        "two_d_geometry": "cross-sectional-area-equivalent planar OpenFOAM model",
        "frame_asset_version": ASSET_VERSION,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(HTML_PATH)
    for case in cases:
        print(
            f"{case['id']}: 1D {case['count1d']} frames "
            f"({case['start1d']:.2f}-{case['end1d']:.2f} s); "
            f"2D {case['count2d']} frames "
            f"({case['start2d']:.2f}-{case['end2d']:.2f} s)"
        )


if __name__ == "__main__":
    main()

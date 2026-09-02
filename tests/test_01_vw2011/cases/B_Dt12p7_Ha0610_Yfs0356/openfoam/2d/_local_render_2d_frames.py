#!/usr/bin/env python3
"""Render OpenFOAM 2D alpha frames from existing VTK/*.vtu and build HTML."""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs_1d2d_compare"
FRAMES_1D = OUT / "frames_1d"
FRAMES_2D = OUT / "frames_2d"
VTK_DIR = HERE / "VTK"
SERIES = VTK_DIR / "2d.vtm.series"


def parse_series() -> list[tuple[float, Path]]:
    data = json.loads(SERIES.read_text(encoding="utf-8"))
    out = []
    for item in data["files"]:
        t = float(item["time"])
        stem = Path(item["name"]).stem  # 2d_12262
        vtu = VTK_DIR / stem / "internal.vtu"
        if vtu.exists():
            out.append((t, vtu))
    return out


def _floats_from_text(text: str) -> np.ndarray:
    return np.fromstring(text, sep=" ", dtype=np.float64)


def _ints_from_text(text: str) -> np.ndarray:
    return np.fromstring(text, sep=" ", dtype=np.int64)


def read_vtu_cell_centers_alpha(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # Fast-ish ascii VTU reader for OF foamToVTK output.
    raw = path.read_text(encoding="utf-8", errors="ignore")
    # Points
    mpts = re.search(
        r"<DataArray[^>]*Name='Points'[^>]*>(.*?)</DataArray>",
        raw,
        flags=re.S,
    )
    if not mpts:
        raise RuntimeError(f"no Points in {path}")
    pts = _floats_from_text(mpts.group(1)).reshape(-1, 3)

    mconn = re.search(
        r"<DataArray[^>]*Name='connectivity'[^>]*>(.*?)</DataArray>",
        raw,
        flags=re.S,
    )
    moff = re.search(
        r"<DataArray[^>]*Name='offsets'[^>]*>(.*?)</DataArray>",
        raw,
        flags=re.S,
    )
    if not mconn or not moff:
        raise RuntimeError(f"no connectivity in {path}")
    conn = _ints_from_text(mconn.group(1))
    offs = _ints_from_text(moff.group(1))

    # alpha.water cell data
    malpha = re.search(
        r"<DataArray[^>]*Name='alpha\.water'[^>]*>(.*?)</DataArray>",
        raw,
        flags=re.S,
    )
    if not malpha:
        raise RuntimeError(f"no alpha.water in {path}")
    alpha = _floats_from_text(malpha.group(1))

    centers = np.zeros((len(offs), 3), dtype=np.float64)
    start = 0
    for i, end in enumerate(offs):
        ids = conn[start:end]
        centers[i] = pts[ids].mean(axis=0)
        start = int(end)
    if len(alpha) != len(centers):
        n = min(len(alpha), len(centers))
        return centers[:n], alpha[:n]
    return centers, alpha


def render_frames() -> list[dict]:
    FRAMES_2D.mkdir(parents=True, exist_ok=True)
    for old in FRAMES_2D.glob("*.png"):
        old.unlink()
    series = parse_series()
    cmap = LinearSegmentedColormap.from_list(
        "airwater", ["#eef2f7", "#9ec5fe", "#2b7fff"], N=256
    )
    frames = []
    for i, (t, vtu) in enumerate(series):
        centers, alpha = read_vtu_cell_centers_alpha(vtu)
        x, y, a = centers[:, 0], centers[:, 1], alpha
        mask = (x >= -0.05) & (x <= 4.1) & (y >= -0.06) & (y <= 1.0)
        x, y, a = x[mask], y[mask], a[mask]

        fig, ax = plt.subplots(figsize=(12.0, 3.2))
        sc = ax.scatter(
            x, y, c=a, cmap=cmap, s=10, vmin=0.0, vmax=1.0, marker="s", linewidths=0
        )
        ax.axhline(0.657, color="#ef4444", ls="--", lw=1.0)
        ax.axvline(0.546, color="#111827", ls=":", lw=0.8)
        ax.axvline(3.516, color="#111827", ls=":", lw=0.8)
        ax.set_xlim(-0.05, 4.05)
        ax.set_ylim(-0.05, 0.98)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(
            f"OpenFOAM 2D Case B | W=Dt^2/D, sigma=0 | t={t:.2f} s", fontsize=10
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label("alpha.water")
        fig.tight_layout()
        name = f"frame_{i:04d}.png"
        fig.savefig(FRAMES_2D / name, dpi=120)
        plt.close(fig)
        frames.append(
            {
                "index": i,
                "file": f"frames_2d/{name}",
                "time": float(t),
                "label": f"2D OpenFOAM  t={t:.2f}s",
            }
        )
        print(f"rendered {i+1}/{len(series)} t={t:.2f}", flush=True)
    return frames


def load_1d_frames() -> list[dict]:
    files = sorted(FRAMES_1D.glob("frame_*.png"))
    # Recover approximate times from previous index if present
    idx_path = OUT / "selected_times.json"
    # 1D frames were generated with linspace over ~9s; rebuild labels from PNG order
    # Prefer matching via frames_index if a previous partial exists; else use index*dt
    meta_guess = OUT / "_1d_times_guess.json"
    frames = []
    # Try to read times from 1D by re-parsing filenames only -> use selected 2D times later for pairing
    for i, p in enumerate(files):
        frames.append(
            {
                "index": i,
                "file": f"frames_1d/{p.name}",
                "time": float("nan"),
                "label": f"1D frame {i:04d}",
            }
        )
    # Better: re-extract times by reading companion if we saved them
    saved = OUT / "frames_1d_meta.json"
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf-8"))
    # Estimate uniform over 0..9 if unknown
    if frames:
        for i, f in enumerate(frames):
            f["time"] = 9.0 * i / max(len(frames) - 1, 1)
            f["label"] = f"1D  t={f['time']:.2f}s"
    return frames


def write_html(pairs: list[dict]) -> Path:
    html_path = OUT / "compare_1d_vs_2d.html"
    payload = json.dumps(pairs)
    html_path.write_text(
        f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>Case B · 1D vs OpenFOAM 2D</title>
<style>
  body {{ font-family: Segoe UI, sans-serif; margin: 16px; background: #f7f8fa; color: #111; }}
  h1 {{ font-size: 20px; margin: 0 0 8px; }}
  .meta {{ color: #555; font-size: 13px; margin-bottom: 12px; line-height: 1.45; }}
  .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .panel {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 8px; }}
  .panel h2 {{ font-size: 14px; margin: 0 0 6px; min-height: 2.4em; }}
  img {{ width: 100%; height: auto; background: #fff; border: 1px solid #eee; }}
  .controls {{ margin: 14px 0; background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
  input[type=range] {{ width: 100%; }}
  .btns button {{ margin-right: 8px; padding: 6px 12px; cursor: pointer; }}
  kbd {{ background:#eef2ff; padding:1px 5px; border-radius:4px; border:1px solid #c7d2fe; }}
</style>
</head>
<body>
<h1>Vasconcelos &amp; Wright (2011) Case B — 自研 1D vs OpenFOAM 2D</h1>
<div class="meta">
  左：自研两流体网络模型（真实几何比例示意）。右：OpenFOAM 2D 水气分布 α.water（面积等效塔宽 W=Dt²/D，σ=0）。<br/>
  拖动滑条手动调帧；也可点按钮或用键盘 <kbd>←</kbd> <kbd>→</kbd>、<kbd>空格</kbd> 播放/暂停。
</div>
<div class="controls">
  <div class="btns">
    <button id="prev" type="button">上一帧</button>
    <button id="play" type="button">播放/暂停</button>
    <button id="next" type="button">下一帧</button>
    <span id="info"></span>
  </div>
  <input id="scrub" type="range" min="0" max="{max(len(pairs)-1,0)}" value="0" step="1"/>
</div>
<div class="row">
  <div class="panel"><h2 id="t1">1D</h2><img id="img1d" alt="1D"/></div>
  <div class="panel"><h2 id="t2">2D</h2><img id="img2d" alt="2D"/></div>
</div>
<script>
const frames = {payload};
let i = 0, timer = null;
const scrub = document.getElementById('scrub');
const img1 = document.getElementById('img1d');
const img2 = document.getElementById('img2d');
const info = document.getElementById('info');
const t1 = document.getElementById('t1');
const t2 = document.getElementById('t2');
function show(k) {{
  if (!frames.length) return;
  i = Math.max(0, Math.min(frames.length-1, k));
  const f = frames[i];
  img1.src = f.file1d;
  img2.src = f.file2d;
  scrub.value = i;
  info.textContent = `帧 ${{i+1}}/${{frames.length}}  ·  t = ${{f.time.toFixed(2)}} s  ·  1D对齐误差 ${{f.dt_match.toFixed(3)}} s`;
  t1.textContent = f.label1d;
  t2.textContent = f.label2d;
}}
scrub.addEventListener('input', e => show(Number(e.target.value)));
document.getElementById('prev').onclick = () => show(i-1);
document.getElementById('next').onclick = () => show(i+1);
document.getElementById('play').onclick = () => {{
  if (timer) {{ clearInterval(timer); timer=null; return; }}
  timer = setInterval(() => show((i+1) % frames.length), 180);
}};
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft') show(i-1);
  if (e.key === 'ArrowRight') show(i+1);
  if (e.key === ' ') {{ e.preventDefault(); document.getElementById('play').click(); }}
}});
show(0);
</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not FRAMES_1D.exists() or not list(FRAMES_1D.glob("frame_*.png")):
        raise SystemExit("missing 1D frames; run full compare builder first")
    # Prefer exact 1D meta if present
    meta1 = OUT / "frames_1d_meta.json"
    if not meta1.exists():
        # Rebuild 1D meta quickly by re-running only make_case_frames times
        import sys

        sys.path.insert(0, str(HERE.parent.parent / "model"))
        from vw2011_network_twofluid import NetworkCase, make_case_frames, run_network

        case = NetworkCase(Dr=0.0127, air_head=0.610, init_water_level=0.356, t_end=9.0)
        rec = run_network(case, verbose=False)
        # Don't rewrite images; just recover times from same sampling
        nF = len(rec["frames_t"])
        sel = np.unique(np.linspace(0, nF - 1, min(60, nF)).astype(int))
        frames_1d = []
        pngs = sorted(FRAMES_1D.glob("frame_*.png"))
        for i, (k, p) in enumerate(zip(sel, pngs)):
            frames_1d.append(
                {
                    "index": i,
                    "file": f"frames_1d/{p.name}",
                    "time": float(rec["frames_t"][k]),
                    "label": f"1D  t={rec['frames_t'][k]:.2f}s  Yfs={rec['wtop'][k]:.3f}m",
                }
            )
        meta1.write_text(json.dumps(frames_1d, indent=2), encoding="utf-8")
    frames_1d = json.loads(meta1.read_text(encoding="utf-8"))

    frames_2d = render_frames()
    pairs = []
    for f2 in frames_2d:
        j = int(np.argmin([abs(f1["time"] - f2["time"]) for f1 in frames_1d]))
        f1 = frames_1d[j]
        pairs.append(
            {
                "time": f2["time"],
                "file1d": f1["file"],
                "file2d": f2["file"],
                "label1d": f1["label"],
                "label2d": f2["label"],
                "dt_match": abs(f1["time"] - f2["time"]),
            }
        )
    html = write_html(pairs)
    (OUT / "frames_index.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print(f"HTML -> {html}")
    print(f"pairs = {len(pairs)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build Case B 1D-vs-2D scrubbable HTML comparison.

1D frames: re-run the frozen two-fluid model and dump PNG frames.
2D frames: reconstruct selected OpenFOAM times, foamToVTK, plot alpha.water.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parent.parent
MODEL = CASE_ROOT / "model"
OUT = HERE / "outputs_1d2d_compare"
FRAMES_1D = OUT / "frames_1d"
FRAMES_2D = OUT / "frames_2d"
VTK_DIR = HERE / "VTK"
MAX_FRAMES = 60
END_T = 8.95

sys.path.insert(0, str(MODEL))


def run_bash(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    # Avoid nested-quote pain: write a tiny temp script.
    script = HERE / "_tmp_of_cmd.sh"
    script.write_text(
        "#!/bin/bash\n"
        "source /usr/share/modules/init/bash 2>/dev/null || true\n"
        "set +e\nset +u\n"
        "source /usr/lib/openfoam/openfoam2512/etc/bashrc\n"
        "set -euo pipefail\n"
        f"cd {HERE.as_posix()}\n"
        f"{cmd}\n",
        encoding="utf-8",
    )
    print("+", cmd)
    return subprocess.run(["bash", str(script)], check=check, text=True)


def export_1d_frames() -> list[dict]:
    from vw2011_network_twofluid import NetworkCase, make_case_frames, run_network

    FRAMES_1D.mkdir(parents=True, exist_ok=True)
    for old in FRAMES_1D.glob("*.png"):
        old.unlink()

    case = NetworkCase(Dr=0.0127, air_head=0.610, init_water_level=0.356, t_end=9.0)
    rec = run_network(case, verbose=False)
    # make_case_frames writes into out_dir/frames; we redirect that path.
    tmp = OUT / "_1d_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    meta = make_case_frames(case, rec, tmp, "B", max_frames=MAX_FRAMES)
    src = tmp / "frames"
    for p in sorted(src.glob("frame_*.png")):
        shutil.copy2(p, FRAMES_1D / p.name)
    frames = []
    for i, item in enumerate(meta):
        frames.append(
            {
                "index": i,
                "file": f"frames_1d/frame_{i:04d}.png",
                "time": float(item["time"]),
                "label": f"1D  t={item['time']:.2f}s  Yfs={item['wtop']:.3f}m",
            }
        )
    shutil.rmtree(tmp, ignore_errors=True)
    return frames


def list_processor_times() -> list[float]:
    times = []
    for p in (HERE / "processor0").iterdir():
        name = p.name
        if re.fullmatch(r"[0-9]+(\.[0-9]+)?", name):
            times.append(float(name))
    return sorted(times)


def select_times(all_times: list[float], n: int = MAX_FRAMES) -> list[float]:
    usable = [t for t in all_times if t <= END_T + 1e-9]
    if not usable:
        raise RuntimeError("no processor time directories found")
    if len(usable) <= n:
        return usable
    idx = np.unique(np.linspace(0, len(usable) - 1, n).astype(int))
    return [usable[i] for i in idx]


def reconstruct_times(times: list[float]) -> None:
    # Batch reconstruct to avoid one giant command line.
    chunk = 15
    for i in range(0, len(times), chunk):
        part = times[i : i + chunk]
        spec = ",".join(f"{t:g}" for t in part)
        run_bash(f"reconstructPar -time '{spec}' -fields '(alpha.water)' > log.reconstruct_compare 2>&1 || true")


def foam_to_vtk(times: list[float]) -> None:
    if VTK_DIR.exists():
        shutil.rmtree(VTK_DIR)
    spec = ",".join(f"{t:g}" for t in times)
    run_bash(
        f"foamToVTK -fields '(alpha.water)' -time '{spec}' -ascii > log.foamToVTK_compare 2>&1"
    )


def _parse_vtk_unstructured(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cell centers (N,3), alpha (N,), and a bbox."""
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    # POINTS
    i = 0
    while i < len(text) and not text[i].startswith("POINTS"):
        i += 1
    if i >= len(text):
        raise RuntimeError(f"no POINTS in {path}")
    n_points = int(text[i].split()[1])
    i += 1
    coords: list[float] = []
    while len(coords) < n_points * 3:
        coords.extend(float(x) for x in text[i].split())
        i += 1
    pts = np.asarray(coords, dtype=float).reshape(n_points, 3)

    while i < len(text) and not text[i].startswith("CELLS"):
        i += 1
    n_cells = int(text[i].split()[1])
    i += 1
    centers = np.zeros((n_cells, 3), dtype=float)
    for c in range(n_cells):
        vals = [int(x) for x in text[i].split()]
        i += 1
        ids = vals[1:]
        centers[c] = pts[ids].mean(axis=0)

    while i < len(text) and "alpha.water" not in text[i]:
        i += 1
    if i >= len(text):
        raise RuntimeError(f"no alpha.water in {path}")
    # next non-empty numeric block after LOOKUP_TABLE
    while i < len(text) and not text[i].strip().replace(".", "").replace("-", "").replace("e", "").replace("E", "").replace("+", "").isdigit() and "LOOKUP" not in text[i] and not text[i][0].isdigit():
        # advance to LOOKUP_TABLE then data
        if text[i].startswith("LOOKUP_TABLE") or text[i].startswith("SCALARS"):
            i += 1
            continue
        i += 1
    # More robust: find LOOKUP_TABLE line then read n_cells floats
    j = 0
    while j < len(text) and not text[j].startswith("LOOKUP_TABLE"):
        j += 1
    j += 1
    alphas: list[float] = []
    while len(alphas) < n_cells and j < len(text):
        if text[j].startswith("FIELD") or text[j].startswith("CELL_DATA") or text[j].startswith("POINT_DATA"):
            break
        alphas.extend(float(x) for x in text[j].split())
        j += 1
    alpha = np.asarray(alphas[:n_cells], dtype=float)
    return centers, alpha, pts


def find_vtk_for_time(t: float) -> Path | None:
    # foamToVTK names vary: case_0.vtk / case_0.100000.vtk etc.
    cands = sorted(VTK_DIR.glob("*.vtk"))
    best = None
    best_err = 1e9
    for p in cands:
        m = re.search(r"_([0-9]+(?:\.[0-9]+)?)\.vtk$", p.name)
        if not m:
            continue
        tt = float(m.group(1))
        err = abs(tt - t)
        if err < best_err:
            best_err = err
            best = p
    if best is not None and best_err < 0.03:
        return best
    return None


def render_2d_frames(times: list[float]) -> list[dict]:
    FRAMES_2D.mkdir(parents=True, exist_ok=True)
    for old in FRAMES_2D.glob("*.png"):
        old.unlink()

    cmap = LinearSegmentedColormap.from_list(
        "airwater", ["#eef2f7", "#9ec5fe", "#2b7fff"], N=256
    )
    frames = []
    for i, t in enumerate(times):
        vtk = find_vtk_for_time(t)
        if vtk is None:
            print(f"WARN missing VTK for t={t}")
            continue
        centers, alpha, _pts = _parse_vtk_unstructured(vtk)
        x = centers[:, 0]
        y = centers[:, 1]
        # Keep only the physical plane region of interest
        mask = (x >= -0.05) & (x <= 4.1) & (y >= -0.06) & (y <= 1.0)
        x, y, a = x[mask], y[mask], alpha[mask]

        fig, ax = plt.subplots(figsize=(12.0, 3.2))
        # Scatter/hexbin-like display using triangulation-free colored points scaled by local dx
        sc = ax.scatter(x, y, c=a, cmap=cmap, s=8, vmin=0.0, vmax=1.0, marker="s", linewidths=0)
        ax.axhline(0.657, color="#ef4444", ls="--", lw=1.0, label="tower rim")
        ax.axvline(0.546, color="#111827", ls=":", lw=0.8)
        ax.axvline(3.516, color="#111827", ls=":", lw=0.8)
        ax.set_xlim(-0.05, 4.05)
        ax.set_ylim(-0.05, 0.98)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(
            f"OpenFOAM 2D Case B  |  W=Dt^2/D, sigma=0  |  t={t:.2f}s",
            fontsize=10,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label("alpha.water")
        fig.tight_layout()
        name = f"frame_{i:04d}.png"
        fig.savefig(FRAMES_2D / name, dpi=130)
        plt.close(fig)
        frames.append(
            {
                "index": i,
                "file": f"frames_2d/{name}",
                "time": float(t),
                "label": f"2D  t={t:.2f}s",
            }
        )
        print(f"2D frame {i}/{len(times)} t={t:.2f}")
    return frames


def align_frames(frames_1d: list[dict], frames_2d: list[dict]) -> list[dict]:
    """Pair by nearest time; master clock = 2D times (actual CFD)."""
    if not frames_2d:
        raise RuntimeError("no 2D frames")
    out = []
    for f2 in frames_2d:
        t = f2["time"]
        j = int(np.argmin([abs(f1["time"] - t) for f1 in frames_1d]))
        f1 = frames_1d[j]
        out.append(
            {
                "time": t,
                "file1d": f1["file"],
                "file2d": f2["file"],
                "label1d": f1["label"],
                "label2d": f2["label"],
                "dt_match": abs(f1["time"] - t),
            }
        )
    return out


def write_html(pairs: list[dict]) -> Path:
    html_path = OUT / "compare_1d_vs_2d.html"
    payload = json.dumps(pairs)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>Case B · 1D vs OpenFOAM 2D</title>
<style>
  body {{ font-family: Segoe UI, sans-serif; margin: 16px; background: #f7f8fa; color: #111; }}
  h1 {{ font-size: 20px; margin: 0 0 8px; }}
  .meta {{ color: #555; font-size: 13px; margin-bottom: 12px; }}
  .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .panel {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 8px; }}
  .panel h2 {{ font-size: 14px; margin: 0 0 6px; }}
  img {{ width: 100%; height: auto; background: #fff; border: 1px solid #eee; }}
  .controls {{ margin: 14px 0; background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
  input[type=range] {{ width: 100%; }}
  .btns button {{ margin-right: 8px; padding: 6px 12px; }}
  code {{ background: #eef2ff; padding: 1px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Vasconcelos &amp; Wright (2011) Case B — 自研 1D vs OpenFOAM 2D</h1>
<div class="meta">
  左：自研两流体网络模型（真实几何比例示意）。右：OpenFOAM 2D（面积等效塔宽 W=Dt²/D，σ=0）。<br/>
  可用滑条手动调帧；左右按最近时刻对齐。
</div>
<div class="controls">
  <div class="btns">
    <button id="prev" type="button">上一帧</button>
    <button id="play" type="button">播放/暂停</button>
    <button id="next" type="button">下一帧</button>
    <span id="info"></span>
  </div>
  <input id="scrub" type="range" min="0" max="{len(pairs)-1}" value="0" step="1"/>
</div>
<div class="row">
  <div class="panel">
    <h2 id="t1">1D</h2>
    <img id="img1d" alt="1D frame"/>
  </div>
  <div class="panel">
    <h2 id="t2">2D</h2>
    <img id="img2d" alt="2D frame"/>
  </div>
</div>
<script>
const frames = {payload};
let i = 0;
let timer = null;
const scrub = document.getElementById('scrub');
const img1 = document.getElementById('img1d');
const img2 = document.getElementById('img2d');
const info = document.getElementById('info');
const t1 = document.getElementById('t1');
const t2 = document.getElementById('t2');
function show(k) {{
  i = Math.max(0, Math.min(frames.length-1, k));
  const f = frames[i];
  img1.src = f.file1d;
  img2.src = f.file2d;
  scrub.value = i;
  info.textContent = `帧 ${{i+1}}/${{frames.length}}  ·  t=${{f.time.toFixed(2)}} s  ·  1D对齐误差 ${{f.dt_match.toFixed(3)}} s`;
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
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== 1D frames ===")
    frames_1d = export_1d_frames()
    print(f"1D frames: {len(frames_1d)}")

    print("=== 2D reconstruct + VTK ===")
    all_times = list_processor_times()
    times = select_times(all_times, MAX_FRAMES)
    (OUT / "selected_times.json").write_text(json.dumps(times, indent=2), encoding="utf-8")
    print(f"selected {len(times)} times from {len(all_times)} dumps")
    reconstruct_times(times)
    foam_to_vtk(times)
    frames_2d = render_2d_frames(times)
    print(f"2D frames: {len(frames_2d)}")

    pairs = align_frames(frames_1d, frames_2d)
    html = write_html(pairs)
    (OUT / "frames_index.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print(f"HTML -> {html}")


if __name__ == "__main__":
    main()

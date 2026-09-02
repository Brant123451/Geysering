"""Build the honest B3 OpenFOAM front-view status/player page.

The page never draws a synthetic OpenFOAM state.  Its poster and eventual MP4
are rendered from sampled alpha.water fields written by OpenFOAM.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
CASE = HERE.parent
OUTPUT = CASE / "caseB_1d3d_frame_compare.html"
MANIFEST = CASE / "outputs" / "caseB_1d3d_viewer_manifest.json"
PROGRESS = CASE / "openfoam" / "2d_front_vof" / "outputs" / "run_progress.json"
MOVIE_REL = "openfoam/2d_front_vof/outputs/B3_openfoam_quasi2d_front_view.mp4"
POSTER_REL = "openfoam/3d_real_vof/outputs/real_front_slice_frames/frame_00000.png"


def load_progress() -> dict:
    if not PROGRESS.exists():
        return {"status": "preparing", "solver_time_s": None, "front_slice_frames": 0}
    return json.loads(PROGRESS.read_text(encoding="utf-8"))


def main() -> None:
    progress = load_progress()
    solver_time = progress.get("solver_time_s")
    solver_text = "尚未开始" if solver_time is None else f"{solver_time:.6f} / 5.300000 s"
    frame_count = int(progress.get("front_slice_frames", 0))
    status = str(progress.get("status", "preparing"))
    generated = datetime.now(timezone.utc).isoformat()

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>Liu2020 B3 — 真实 OpenFOAM 正视剖面</title>
<style>
:root{{--ink:#17212b;--muted:#667085;--line:#d5dce5;--blue:#075aa6;--cyan:#00cfe8;--bg:#f2f5f8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1500px;margin:auto;padding:20px}}h1{{margin:0 0 5px;font-size:24px}}.sub{{color:var(--muted);margin-bottom:14px}}
.status{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px}}.card,.viewer,.note{{background:#fff;border:1px solid var(--line);border-radius:10px;box-shadow:0 2px 8px #23344a0c}}
.card{{padding:11px 13px}}.card small{{display:block;color:var(--muted)}}.card b{{font-size:15px}}.running{{color:#9a5a00}}.ok{{color:#147d3f}}
.viewer{{padding:13px}}video{{display:block;width:100%;max-height:76vh;background:#e9eef3;border:1px solid #cfd7e2;border-radius:8px}}
.legend{{display:flex;gap:20px;flex-wrap:wrap;color:var(--muted);margin:10px 2px 0}}.line{{display:inline-block;width:22px;border-top:3px solid;margin-right:6px;vertical-align:middle}}.wall{{border-color:#111827}}.interface{{border-color:var(--cyan)}}
.note{{margin-top:14px;padding:12px 14px}}.note b{{color:#7a4300}}code{{background:#f1f4f7;padding:1px 4px;border-radius:4px}}.foot{{margin-top:12px;color:var(--muted);font-size:12px;overflow-wrap:anywhere}}
@media(max-width:850px){{.status{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>Liu2020 Case B3：真实 OpenFOAM 水–气界面正视剖面</h1>
<div class="sub">穿过管道轴线的中心剖面；上游管—检查井—下游管—竖井来自同一个连通网格。</div>
<section class="status">
 <div class="card"><small>3D 基准求解</small><b class="ok">16.4 s 已完成</b></div>
 <div class="card"><small>峰值体场</small><b>原运行被 purgeWrite 清理</b></div>
 <div class="card"><small>正视剖面重跑</small><b class="running">{status}：{solver_text}</b></div>
 <div class="card"><small>已写出真实界面帧</small><b>{frame_count}</b></div>
</section>
<section class="viewer">
 <video controls preload="metadata" poster="{POSTER_REL}">
  <source src="{MOVIE_REL}" type="video/mp4">
  浏览器不支持 MP4 播放。
 </video>
 <div class="legend"><span><i class="line wall"></i>OpenFOAM 流体域边界</span><span><i class="line interface"></i><code>alpha.water = 0.5</code> 水气界面</span><span>深蓝：水；浅色：空气</span></div>
</section>
<div class="note"><b>当前状态：</b>上图海报已经是三维 OpenFOAM 网格的真实初始剖面，不再显示探针重建动画。完整 MP4 正在由相同边界条件的连续 OpenFOAM 求解生成；完成后本页刷新即可播放。二维快速结果仅用于完整界面渲染，论文中的 3D 标量结果不会被它替换。</div>
<div class="foot">Generated {generated}. Solver progress: openfoam/2d_front_vof/outputs/run_progress.json. Movie: {MOVIE_REL}</div>
</main></body></html>"""
    OUTPUT.write_text(html, encoding="utf-8")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "case": "Liu2020 B3 Q20to100 full-pipe geyser",
                "output": OUTPUT.name,
                "generator": "scripts/caseB_build_real_vof_html.py",
                "generated_utc": generated,
                "openfoam_view": "real alpha.water centre-plane only; no probe reconstruction",
                "production_3d_complete": True,
                "peak_volume_field_rerun_status": status,
                "quasi2d_solver_time_s": solver_time,
                "quasi2d_front_slice_frames": frame_count,
                "scientific_status": "quasi-2D rendering is diagnostic; production 3-D metrics remain primary",
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {MANIFEST}")


if __name__ == "__main__":
    main()

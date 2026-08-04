# -*- coding: utf-8 -*-
"""Case A of the V&W (2011) reproduction: LARGE tower, no geyser.

    D_t = 57.1 mm  (D_t* = 0.607),  H_a0 = 0.305 m,  Y_fs0 = 0.356 m.

Paper experimental data for this condition (middle-row / left-column panels):
  * Fig. 5  -- normalized pressure head H*(T*),  axes T* = 0..10, H* = 0..1.5
  * Fig. 7  -- normalized levels Y*fs / Y*int,   axes T* = 7..10, Y* = 0..1

This script digitizes both panels, runs the same decoupled two-fluid network
model with the case-A parameters, and produces overlay figures + metrics +
a standalone report.html inside this folder.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "model"))     # frozen per-case copies of the solver + digitizer

from digitize_paper_curves import (load_gray, find_panels, draw_panel_debug,
                                   label_boxes_in_panel, digitize_fig6, digitize_fig8)
from vw2011_network_twofluid import G, NetworkCase, run_network
from PIL import Image

FIG5 = HERE / "paper_scans" / "raw_p4_x92_1992x1446.png"    # pressure heads, Dt*=0.607
FIG7 = HERE / "paper_scans" / "raw_p6_x111_2175x1527.png"   # levels, Dt*=0.607
PANEL = (1, 0)                              # middle row (WL=0.356), left col (Ha=0.305)

DIG = HERE / "digitized"
OUT = HERE / "outputs"
DIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

C_MODEL = "#d62728"
C_MODEL2 = "#f59e0b"
C_PAPER = "#374151"
C_PAPER_BAND = "#9ca3af"
C_FS = "#1f77b4"
C_INT = "#111827"

CASE = dict(Dr=0.0571, air_head=0.305, init_water_level=0.356, L=0.610)


def crop_panel(gray, box, dst: Path, margin: int = 70):
    x0, x1, y0, y1 = box
    H, W = gray.shape
    r0 = max(0, y0 - margin); r1 = min(H, y1 + margin)
    c0 = max(0, x0 - margin); c1 = min(W, x1 + margin)
    Image.fromarray(gray[r0:r1, c0:c1]).save(dst)


def digitize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---------------- Fig. 5 : pressure head (T* 0..10) ----------------
    g5 = load_gray(FIG5)
    panels5, _ = find_panels(g5)
    draw_panel_debug(g5, panels5, DIG / "debug_fig5_panels.png", center=PANEL)
    box5 = panels5[PANEL[0]][PANEL[1]]
    crop_panel(g5, box5, DIG / "fig5_caseA_panel.png")
    masks5 = label_boxes_in_panel(g5, box5)
    T5, med5, lo5, hi5, bin5 = digitize_fig6(g5, box5, xlim=(0.0, 10.0), ylim=(0.0, 1.5),
                                             label_masks=masks5)
    with (DIG / "fig5_caseA_Hstar_band.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Tstar", "Hstar_med", "Hstar_min", "Hstar_max"])
        for row in zip(T5, med5, lo5, hi5):
            w.writerow([f"{v:.5f}" for v in row])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.imshow(bin5, cmap="gray_r")
    a1.set_title("Fig.5 case-A panel: extracted dark pixels (labels masked)")
    a2.fill_between(T5, lo5, hi5, color=C_PAPER_BAND, alpha=0.45, label="digitized min-max band")
    a2.plot(T5, med5, color=C_INT, lw=1.2, label="digitized median")
    a2.set_xlim(0, 10); a2.set_ylim(0, 1.5); a2.grid(alpha=0.3)
    a2.set_xlabel("T*_ref"); a2.set_ylabel("H*"); a2.legend(frameon=False, fontsize=8)
    a2.set_title("digitized H* (paper Fig.5, Ha0=0.305, WL=0.356)")
    fig.tight_layout(); fig.savefig(DIG / "debug_fig5_extract.png", dpi=140); plt.close(fig)

    # ---------------- Fig. 7 : levels (T* 7..10) ----------------
    g7 = load_gray(FIG7)
    panels7, _ = find_panels(g7)
    draw_panel_debug(g7, panels7, DIG / "debug_fig7_panels.png", center=PANEL)
    box7 = panels7[PANEL[0]][PANEL[1]]
    crop_panel(g7, box7, DIG / "fig7_caseA_panel.png")
    masks7 = label_boxes_in_panel(g7, box7)
    pts7, comps7, bin7 = digitize_fig8(g7, box7, xlim=(7.0, 10.0), ylim=(0.0, 1.0),
                                       label_masks=masks7,
                                       max_cluster_h=200, max_cluster_w=700,
                                       reclass_int_below=0.45)
    with (DIG / "fig7_caseA_levels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Tstar", "Ystar", "kind"])
        for tstar, ystar, kind in sorted(pts7):
            w.writerow([f"{tstar:.5f}", f"{ystar:.5f}", kind])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.imshow(bin7, cmap="gray_r")
    for cx, cy, kind, size, hr in comps7:
        a1.plot(cx, cy, "o", ms=4, mfc="none",
                mec=("#ef4444" if kind == "int" else "#2b7fff"), mew=1.0)
    a1.set_title("Fig.7 case-A panel: components (blue=Y*fs, red=Y*int)")
    for kind, color, mk, lbl in (("fs", "#2b7fff", "^", "Y*_fs digitized"),
                                 ("int", "#ef4444", "o", "Y*_int digitized")):
        xs = [p[0] for p in pts7 if p[2] == kind]
        ys = [p[1] for p in pts7 if p[2] == kind]
        a2.plot(xs, ys, mk, ms=4, mfc="none", mec=color, label=lbl)
    a2.set_xlim(7, 10); a2.set_ylim(0, 1.02); a2.grid(alpha=0.3)
    a2.set_xlabel("T*_ref"); a2.set_ylabel("Y*"); a2.legend(frameon=False, fontsize=8)
    a2.set_title("digitized levels (paper Fig.7, Ha0=0.305, WL=0.356)")
    fig.tight_layout(); fig.savefig(DIG / "debug_fig7_extract.png", dpi=140); plt.close(fig)

    print(f"fig5 samples: {len(T5)}   fig7 markers: {len(pts7)} "
          f"(fs={sum(1 for p in pts7 if p[2] == 'fs')}, int={sum(1 for p in pts7 if p[2] == 'int')})")
    fig5 = dict(T=np.array(T5), med=np.array(med5), lo=np.array(lo5), hi=np.array(hi5))
    fig7 = {k: np.array(sorted([(t, y) for t, y, kk in pts7 if kk == k])) for k in ("fs", "int")}
    return fig5, fig7


def run_model(t_end: float = 9.0):
    case = NetworkCase(Dr=CASE["Dr"], air_head=CASE["air_head"],
                       init_water_level=CASE["init_water_level"], t_end=t_end)
    rec = run_network(case, verbose=False)
    L = case.riser_height
    sgd = math.sqrt(G * case.Dr)
    t = np.asarray(rec["t"])
    n = min(len(t), len(rec["tr_head"]) + 1)
    series = dict(
        t=t[:n],
        Tstar=t[:n] * sgd / L,
        Yfs=np.asarray(rec["wtop"])[:n] / L,
        Yint=np.asarray(rec["itop"])[:n] / L,
        pocket=np.asarray(rec["up_head"])[:n] / L,
        tr=np.concatenate([[np.nan], np.asarray(rec["tr_head"])])[:n] / L,
    )
    return case, series


def first_crossing(x, y, thresh, above=True, after=0.0):
    for xi, yi in zip(x, y):
        if xi < after:
            continue
        if (yi >= thresh) if above else (yi <= thresh):
            return float(xi)
    return None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig5, fig7 = digitize()
    case, s = run_model()
    L = case.riser_height
    tsc = L / math.sqrt(G * case.Dr)
    yfs0 = case.init_water_level / L

    # ------------------------------------------------ pressure overlay
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.fill_between(fig5["T"], fig5["lo"], fig5["hi"], color=C_PAPER_BAND, alpha=0.45,
                    label="V&W(2011) experiment, Fig.5 panel (3 repetitions, digitized band)")
    ax.plot(fig5["T"], fig5["med"], color=C_PAPER, lw=1.4,
            label="V&W(2011) experiment, digitized median")
    # Cycle-averaged trace only: the raw trace carries the model's resolved
    # slug-on-pocket-spring oscillation (~0.7 s period, amplitude far above the
    # experiment's smooth traces) -- reviewer asked to drop it from the overlay;
    # it remains available in outputs/caseA_model_series.csv (column tr).
    tr = np.asarray(s["tr"], dtype=float)
    tt = np.asarray(s["t"], dtype=float)
    win = 0.8  # seconds ~ one slosh period
    tr_avg = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        mwin = (tt >= tt[i] - win / 2) & (tt <= tt[i] + win / 2)
        if np.any(np.isfinite(tr[mwin])):
            tr_avg[i] = np.nanmean(tr[mwin])
    ax.plot(s["Tstar"], tr_avg, color=C_MODEL, lw=2.2,
            label="model: transducer head, cycle-averaged (0.8 s window)")
    ax.set_ylim(0, 1.5)
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.5)
    ax.set_xlabel(r"$T^*_{ref} = t\,\sqrt{g D_t}/L$")
    ax.set_ylabel(r"$H^* = H/L$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Case A pressure head at the transducer -- "
                 r"$D_t^*=0.607$, $H_{a0}=0.305$ m, $WL_{init}=0.356$ m"
                 "\nmodel (decoupled two-fluid) vs V&W(2011) JHE Fig.5 (digitized)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "caseA_comparison_pressure.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------ levels overlay
    # Panel order matches the reading priority: LEFT = the paper's own Fig.7
    # window (T*=7..10, the直接对比对象), RIGHT = full model trajectory for
    # context (the paper never plots T*<7 for the levels).
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    # The paper's Fig.7 Yint circles are THREE separate repetitions of the same
    # run (butterfly valve opened by hand -> each run's climb starts at a
    # slightly different instant), i.e. three parallel climbing sequences.
    # Cluster the digitized circles by their climb-line intercept so each run
    # can be drawn as its own connected curve instead of an unreadable cloud.
    int_runs = []
    if fig7["int"].size:
        ip = fig7["int"]
        climb = ip[ip[:, 1] > 0.03]
        if climb.shape[0] >= 6:
            k_slope = 0.55  # approx dY*/dT* of the climb (same for all runs)
            b = climb[:, 0] - climb[:, 1] / k_slope   # liftoff intercept T0
            # 1-D k-means (k=3) on the intercept -- the three repetitions are
            # parallel climb lines, so their intercepts form three tight groups
            cents = np.quantile(b, [1 / 6, 3 / 6, 5 / 6])
            labels = np.zeros(climb.shape[0], dtype=int)
            for _ in range(20):
                labels = np.argmin(np.abs(b[:, None] - cents[None, :]), axis=1)
                new = np.array([b[labels == ci].mean() if (labels == ci).any()
                                else cents[ci] for ci in range(3)])
                if np.allclose(new, cents):
                    break
                cents = new
            bounds = [0, 0, 0, 0]  # only len(bounds)-1 = 3 clusters used below
            # refine: fit a line per run, reassign each point to the nearest
            # line (kills the sawtooth mis-assignments where the runs overlap)
            for _ in range(4):
                lines = []
                for ci in range(len(bounds) - 1):
                    pts = climb[labels == ci]
                    if pts.shape[0] >= 2:
                        lines.append(np.polyfit(pts[:, 0], pts[:, 1], 1))
                    else:
                        lines.append(None)
                for pi in range(climb.shape[0]):
                    dists = [abs(np.polyval(ln, climb[pi, 0]) - climb[pi, 1])
                             if ln is not None else 1e9 for ln in lines]
                    labels[pi] = int(np.argmin(dists))
            for ci in range(len(bounds) - 1):
                pts = climb[labels == ci]
                if pts.shape[0] >= 2:
                    int_runs.append(pts[np.argsort(pts[:, 0])])
    for ax, xlim in ((a1, (7.0, 10.0)), (a2, (0.0, 10.0))):
        ax.plot(s["Tstar"], s["Yfs"], color=C_MODEL, lw=2.0, label=r"model $Y^*_{fs}$ (free surface)")
        ax.plot(s["Tstar"], s["Yint"], color=C_MODEL2, lw=1.8, ls="--", label=r"model $Y^*_{int}$ (gas front)")
        if fig7["fs"].size:
            ax.plot(fig7["fs"][:, 0], fig7["fs"][:, 1], "^", ms=6, mfc="none", mec=C_FS,
                    label=r"experiment $Y^*_{fs}$ (Fig.7, digitized, 3 repetitions)")
        for rk, run in enumerate(int_runs):
            # connect only the unambiguous climb (y<0.55); near the top the three
            # repetitions converge and interleave, so leave those as loose circles
            lo = run[run[:, 1] < 0.55]
            hi = run[run[:, 1] >= 0.55]
            if lo.shape[0] >= 2:
                ax.plot(lo[:, 0], lo[:, 1], "o-", ms=5, lw=0.9, mfc="none", mec=C_INT,
                        color=C_INT, alpha=0.85,
                        label=(r"experiment $Y^*_{int}$ -- 3 repetitions "
                               "(climb connected)" if rk == 0 else None))
            if hi.shape[0]:
                ax.plot(hi[:, 0], hi[:, 1], "o", ms=5, mfc="none", mec=C_INT, alpha=0.7)
        if not int_runs and fig7["int"].size:
            ax.plot(fig7["int"][:, 0], fig7["int"][:, 1], "o", ms=6, mfc="none", mec=C_INT,
                    label=r"experiment $Y^*_{int}$ (Fig.7, digitized)")
        ax.axhline(1.0, color="#16a34a", ls=":", lw=1.2)
        ax.axhline(yfs0, color="0.65", ls=":", lw=1.0)
        ax.set_xlim(*xlim); ax.set_ylim(0, 1.05)
        ax.set_xlabel(r"$T^*_{ref} = t\,\sqrt{g D_t}/L$")
        ax.set_ylabel(r"$Y^* = Y/L$")
        ax.grid(alpha=0.3)
    # slope comparison: replot the model gas front time-shifted so its climb
    # aligns with the MIDDLE experimental repetition (valve opening instant is
    # manual in the experiment, so a rigid time shift is legitimate)
    if len(int_runs) >= 2:
        mid = sorted(int_runs, key=lambda r: r[0, 0])[len(int_runs) // 2]
        y_ref = 0.30
        t_exp = float(np.interp(y_ref, mid[:, 1], mid[:, 0]))
        yint_m = np.asarray(s["Yint"]); ts_m = np.asarray(s["Tstar"])
        rising = yint_m > 0.02
        if rising.any():
            i0 = int(np.argmax(rising))
            seg = slice(i0, int(np.argmax(yint_m)) + 1)
            t_mod = float(np.interp(y_ref, yint_m[seg], ts_m[seg]))
            dshift = t_exp - t_mod
            a1.plot(ts_m + dshift, yint_m, color=C_MODEL2, lw=1.2, ls=":",
                    label=rf"model $Y^*_{{int}}$ shifted +{dshift:.2f} $T^*$ (slope check)")
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    a1.set_title("paper Fig.7 window ($T^*$ = 7..10) -- primary comparison", fontsize=10)
    a2.set_title("full model trajectory (context; paper plots $T^*\\geq 7$ only)", fontsize=10)
    a2.axvspan(7.0, 10.0, color="#f3f4f6", zorder=0)
    fig.suptitle("Case A tower free-surface and air-water interface -- "
                 r"$D_t^*=0.607$, $H_{a0}=0.305$ m, $WL_{init}=0.356$ m", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "caseA_comparison_levels.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------ metrics
    fs_pts, int_pts = fig7["fs"], fig7["int"]
    fs_plateau = float(np.median(fs_pts[:, 1])) if fs_pts.size else None
    mask57 = (fig5["T"] > 1.0) & (fig5["T"] < 7.0)
    m = dict(
        case=dict(**CASE, Dt_over_D=CASE["Dr"] / 0.094),
        Tstar_scale_s=tsc,
        model=dict(
            Yfs_max=float(np.nanmax(s["Yfs"])),
            rise_frac=float((np.nanmax(s["Yfs"]) - yfs0)),
            geyser=bool(np.nanmax(s["Yfs"]) >= 0.98),
            int_liftoff_Tstar=first_crossing(s["Tstar"], s["Yint"], 0.05),
            int_catch_fs_Tstar=first_crossing(
                s["Tstar"], np.where(s["Yint"] > 0.10, s["Yint"] - s["Yfs"], -1.0), -0.02),
            Hstar_plateau_tr=float(np.nanmedian(s["tr"][(s["Tstar"] > 1.0) & (s["Tstar"] < 7.0)])),
            Hstar_max=float(np.nanmax(s["pocket"])),
        ),
        paper=dict(
            Yfs_plateau=fs_plateau,
            Yfs_max=float(fs_pts[:, 1].max()) if fs_pts.size else None,
            geyser=False,
            int_first_Tstar=float(int_pts[:, 0].min()) if int_pts.size else None,
            int_catch_fs_Tstar=(float(int_pts[int_pts[:, 1] >= fs_plateau - 0.05][:, 0].min())
                                if (int_pts.size and fs_plateau is not None
                                    and (int_pts[:, 1] >= fs_plateau - 0.05).any()) else None),
            Hstar_plateau=float(np.median(fig5["med"][mask57])),
            Hstar_drop_Tstar=first_crossing(fig5["T"], fig5["med"], 0.3, above=False, after=4.0),
            Yfs0=yfs0,
        ),
    )
    (OUT / "caseA_comparison_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

    with (OUT / "caseA_model_series.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "Tstar", "Yfs_star", "Yint_star", "pocket_Hstar", "transducer_Hstar"])
        for i in range(len(s["t"])):
            w.writerow([f"{s['t'][i]:.4f}", f"{s['Tstar'][i]:.4f}", f"{s['Yfs'][i]:.4f}",
                        f"{s['Yint'][i]:.4f}", f"{s['pocket'][i]:.4f}", f"{s['tr'][i]:.4f}"])

    build_report(m)
    print(json.dumps(m, indent=2))
    print(f"-> {OUT / 'caseA_comparison_pressure.png'}")
    print(f"-> {OUT / 'caseA_comparison_levels.png'}")
    print(f"-> {HERE / 'report.html'}")


def build_report(m: dict):
    mo, pa = m["model"], m["paper"]
    tsc = m["Tstar_scale_s"]

    def _f(v, nd=2):
        return "—" if v is None else f"{v:.{nd}f}"

    def _sec(v):
        return "—" if v is None else f"{v * tsc:.1f} s"

    rows = [
        ("是否喷发（水面到塔顶）", "否（水面最高 ~" + _f(pa.get("Yfs_max")) + "L）",
         ("是" if mo.get("geyser") else "否") + f"（水面最高 {_f(mo.get('Yfs_max'))}L）", ""),
        ("水面平台 Y*fs", _f(pa.get("Yfs_plateau")), _f(mo.get("Yfs_max")), "实验为 Fig.7 散点中位"),
        ("界面出现在图窗（Y*int 离底）", f"T*={_f(pa.get('int_first_Tstar'))}（{_sec(pa.get('int_first_Tstar'))}）",
         f"T*={_f(mo.get('int_liftoff_Tstar'))}（{_sec(mo.get('int_liftoff_Tstar'))}）", ""),
        ("气水界面追上自由水面", f"T*={_f(pa.get('int_catch_fs_Tstar'))}（{_sec(pa.get('int_catch_fs_Tstar'))}）",
         f"T*={_f(mo.get('int_catch_fs_Tstar'))}（{_sec(mo.get('int_catch_fs_Tstar'))}）", "气体经水柱排出=不喷发分支"),
        ("压头平台 H*（T*≈1–7 中位）", _f(pa.get("Hstar_plateau")), _f(mo.get("Hstar_plateau_tr")), ""),
        ("压头骤降（气囊排空）", f"T*={_f(pa.get('Hstar_drop_Tstar'))}（{_sec(pa.get('Hstar_drop_Tstar'))}）", "见叠加图", ""),
    ]
    trs = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in rows)
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Case A: V&amp;W(2011) Dt=57.1mm 大塔（不喷发）对比</title>
<style>
body{{font-family:-apple-system,Segoe UI,Arial,'Microsoft YaHei',sans-serif;margin:0;background:#f6f8fb;color:#1f2937}}
.wrap{{max-width:1180px;margin:24px auto;padding:0 18px}}
.panel{{background:#fff;border:1px solid #ddd;border-radius:12px;padding:16px;margin-bottom:16px}}
img{{width:100%;border:1px solid #ddd;border-radius:10px;background:#fff}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid #eee}}
th{{background:#f3f4f6}}
p{{line-height:1.55;color:#374151}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start}}
.muted{{font-size:13px;color:#6b7280}}
</style></head><body><div class="wrap">
<h1>Case A — V&amp;W(2011) 大塔工况（不喷发分支）</h1>
<p>工况：<code>Dt=57.1 mm</code>（Dt/D=0.607）、<code>Ha0=0.305 m</code>、<code>WL=0.356 m</code>。
实验数据取自论文 <b>Fig.5</b>（压头，中排左列面板）与 <b>Fig.7</b>（水面/界面，同面板），
数字化后与同一套解耦两流体模型的输出叠加。无量纲化：T*=t·√(gDt)/L，H*=H/L，Y*=Y/L
（L=0.610&nbsp;m；本工况 T*=1 对应 {tsc:.2f}&nbsp;s）。</p>
<div class="panel">
  <h2 style="margin-top:0">论文原图（扫描面板）</h2>
  <div class="grid2">
    <div><h3 style="margin:4px 0">Fig.5 本工况面板（压头）</h3><img src="digitized/fig5_caseA_panel.png"></div>
    <div><h3 style="margin:4px 0">Fig.7 本工况面板（水面/界面）</h3><img src="digitized/fig7_caseA_panel.png"></div>
  </div>
</div>
<div class="panel">
  <h2 style="margin-top:0">叠加对比</h2>
  <h3>压头 H*(T*)</h3><img src="outputs/caseA_comparison_pressure.png">
  <h3>塔内水面与气水界面 Y*(T*)</h3><img src="outputs/caseA_comparison_levels.png">
  <h3>事件时刻与量值对照</h3>
  <table>
    <tr><th>指标</th><th>论文实验（数字化）</th><th>模型</th><th>备注</th></tr>
    {trs}
  </table>
  <p class="muted">数字化中间产物（面板检测、取点叠加）见 <code>digitized/</code> 下的 debug 图；
  模型无量纲序列见 <code>outputs/caseA_model_series.csv</code>。</p>
</div>
<div class="panel">
  <h2 style="margin-top:0">附：连通气囊闭合（pocket_bleed）A/B 对比</h2>
  <p class="muted">为复现论文"爬升期压头渐进衰减"（其 TPA 模型为 0 维气囊 ODE），我们给两流体模型
  加了可选闭合：<b>气囊-塔内气柱连通泄压</b>（气囊压力向"界面上方水柱 + 膜份额"的静压环境松弛）
  + <b>膜支撑静压覆盖</b>（气柱跨段只承担界面以上水重）。A/B 结论：闭合版把水面涨升（swell）
  物理化了（爬升期 Yfs 0.59→0.64，实验 0.55→0.65），且贯通/塌落时刻不变；但压头在贴近贯通时
  先隆起到 ~0.68 再塌落（基线版全程贴实验带更好），塌落尾段也偏慢。故<b>正式版默认关闭</b>该闭合，
  下图为闭合版效果存档（求解器开关 <code>pocket_bleed / film_span_overlay</code>）。</p>
  <div class="grid2">
    <div><h3 style="margin:4px 0">闭合版：压头对比</h3><img src="outputs/caseA_comparison_pressure_bleedON.png"></div>
    <div><h3 style="margin:4px 0">闭合版：水位/界面对比</h3><img src="outputs/caseA_comparison_levels_bleedON.png"></div>
  </div>
  <p class="muted">深层原因：1D 两流体的分散气泡云拓扑里，塔底始终承担整根水柱的重量，
  气囊压力只能保持平台直到贯通；论文的"边爬边泄"需要 Taylor 气弹贴壁薄膜绕流拓扑
  （水膜由壁面剪切支撑、界面以上水柱直接压在气囊上），这正是论文选择 0 维气囊 ODE 而非
  场模型的原因。我们把该拓扑作为闭合注入后能得到趋势，但稳定参数下无法同时保住
  基线已对齐的时序指标，故按"基线为主、闭合存档"交付。</p>
</div>
__EXTRA_SECTIONS__
</div></body></html>"""
    html = html.replace("__EXTRA_SECTIONS__", build_extra_sections())
    (HERE / "report.html").write_text(html, encoding="utf-8")


def build_extra_sections() -> str:
    """Fig.4 快照对比 / Table 2 速度对比 / Fig.10（论文自家模型）对比——
    产物由 caseA_fig4_and_table2.py 与 caseA_fig10_compare.py 生成，存在才嵌入。"""
    parts = []
    # ---- 出版级重绘图（仓库 paper/ 工程实际收录的矢量图源） ----
    pub_imgs = [
        ("caseA_pressure_manual.png",
         "压头 H*：三次重复逐点手工数字化（黑）+ 模型 0.8 s 周期平均（红）"),
        ("caseA_levels_manual.png",
         "水位/界面 Y*：实验散点（三角=水面，圆=界面）+ 模型曲线"
         "（前沿画至追上水面为止；水面画至实验覆盖终点）"),
        ("caseA_tpa_redrawn.png",
         "与论文自家 TPA 模型对比（Fig.10 数字化重绘，非扫描叠加）"),
    ]
    have = [(f, cap) for f, cap in pub_imgs if (OUT / f).exists()]
    if have:
        imgs = "\n".join(
            f'  <h3>{cap}</h3><img src="outputs/{f}">' for f, cap in have)
        parts.append(f"""
<div class="panel">
  <h2 style="margin-top:0">出版级重绘图（论文实际收录版本）</h2>
  <p class="muted">以下为论文（E:\\Geysering\\paper）收录的自绘矢量图的 PNG 预览，
  由 caseA_plot_manual_pressure.py / caseA_plot_manual_levels.py /
  caseA_fig10_redraw.py 生成；数据与上方叠加对比图同源（同一冻结求解器输出）。</p>
{imgs}
</div>""")
    v_json = OUT / "caseA_table2_velocities.json"
    if (OUT / "caseA_fig4_snapshots.png").exists():
        vel_rows = ""
        if v_json.exists():
            v = json.loads(v_json.read_text(encoding="utf-8"))
            mo, pa = v["model"], v["paper_table2_Dt0607"]
            f4 = v["paper_fig4"]
            vel_rows = f"""
  <h3>Table 2 速度对照（V* = V/√(gDt)）</h3>
  <table>
    <tr><th>量</th><th>论文 Table 2（实验平均）</th><th>论文 Fig.4 照片估算</th><th>模型</th></tr>
    <tr><td>气水界面上升速度 V*int</td><td>{pa['Vint_star']:.2f}</td>
        <td>{f4['avg_speed_star']:.2f}（0.05→0.26 m / 0.56 s）</td>
        <td>{mo['Vint_star']:.2f}</td></tr>
    <tr><td>自由水面上升速度 V*fs</td><td>{pa['Vfs_star']:.3f}</td><td>—</td>
        <td>{mo['Vfs_star']:.3f}（近零，同量级）</td></tr>
  </table>
  <p class="muted">说明：Table 2 是纯实验统计（论文用自家模型对速度的对比在 Fig.10 的两个速度子图里，
  不在表里）。模型 V*int 取界面穿越 Fig.4 同一高度带（0.05–0.30 m）的拟合斜率。</p>"""
        parts.append(f"""
<div class="panel">
  <h2 style="margin-top:0">Fig.4 对照 — 塔内气囊上升快照（定性+速度）</h2>
  <div class="grid2">
    <div><h3 style="margin:4px 0">论文 Fig.4（相机快照，间隔 0.14 s）</h3>
      <img src="paper_scans/fig4_photo_strip.png">
      <p class="muted" style="margin:6px 0 0 0">注：照片下方的 22.31–22.85 s 是<b>摄像机录像带时钟</b>
      （录像早于开阀就开始了），不是"开阀后经过的时间"；有物理意义的是快照间隔 0.14 s 与前沿爬升区间
      0.05→0.26 m。模型对应段为开阀后 t≈5.4–5.9 s。</p></div>
    <div><h3 style="margin:4px 0">模型同节奏快照（前沿越过 0.05 m 起，间隔 0.14 s）</h3>
      <img src="outputs/caseA_fig4_snapshots.png"></div>
  </div>{vel_rows}
</div>""")
    frames_json = OUT / "frames_index.json"
    if frames_json.exists():
        frames_data = frames_json.read_text(encoding="utf-8")
        parts.append("""
<div class="panel">
  <h2 style="margin-top:0">两流体模拟逐帧查看器 — 水平管 + 通风塔全场演化</h2>
  <p class="muted">同一冻结求解器的 Case A 全场演化（水=蓝，气=白/浅灰）：
  阀开冲击 → 气囊分层推进 → 抵塔爬升 → 贯通排气 → 压力塌落。
  <b>左图为真实比例 1:1</b>（管径 D=94 mm、塔径 Dt=57.1 mm 均按真实尺寸），
  因此塔是细柱、塔内气泡在全局图上只有细丝——塔内细节请看右侧放大同步视图
  （按当地两流体气含率 α_g 画气核宽度）。左右方向键 / 滑块 / 播放均可调帧。</p>
  <p class="muted"><b>时间跨度说明</b>：t=0 为开阀时刻，模拟到 9 s 结束——论文所有曲线图
  （Fig.5/7/10）的横轴上限 T*=10 换算成真实时间只有 8.15 s（1 T* = 0.815 s），
  9 s 已完整覆盖论文展示的全过程（平台段→气囊抵塔→爬升→贯通排气→压头归零）。
  Fig.4 照片下方的 22 s 是摄像机自己的录像时钟（录像早于开阀十几秒就开始了），
  不是开阀后经过的时间：按 Fig.7，实验气囊在开阀后 T*≈7.3（约 5.9 s）才进塔，
  与模型的 5.4–5.9 s 对应同一事件。</p>
  <div class="meta" style="display:flex;gap:18px;margin:10px 0;font-weight:700;flex-wrap:wrap">
    <span id="vIdx"></span><span id="vTime"></span><span id="vWtop"></span>
    <span id="vItop"></span><span id="vMass"></span><span id="vHead"></span></div>
  <div style="display:grid;grid-template-columns:minmax(0,2.9fr) minmax(220px,1fr);gap:14px;align-items:start">
    <div><h3 style="margin:0 0 8px 0;font-size:15px">全局 1:1 视图</h3>
      <img id="vFrame" style="width:100%">
      <p class="muted" style="margin:6px 0 0 0">水平管按截面含水率显示分层气团推进。</p></div>
    <div><h3 style="margin:0 0 8px 0;font-size:15px">竖管放大同步视图</h3>
      <img id="vRiser" style="width:100%">
      <p class="muted" style="margin:6px 0 0 0">气核宽度 = 当地 α_g（无阈值化）；
      蓝线=可见水面（含气泡涌胀），红虚线=气相前沿。</p></div>
  </div>
  <div style="margin-top:8px">
    <button id="vPrev" style="padding:8px 14px;margin:6px;border:1px solid #bbb;border-radius:8px;background:#fff;cursor:pointer">上一帧</button>
    <button id="vPlay" style="padding:8px 14px;margin:6px;border:1px solid #bbb;border-radius:8px;background:#fff;cursor:pointer">播放</button>
    <input id="vSlider" type="range" style="width:60%;vertical-align:middle">
    <button id="vNext" style="padding:8px 14px;margin:6px;border:1px solid #bbb;border-radius:8px;background:#fff;cursor:pointer">下一帧</button>
  </div>
  <p class="muted">整段 GIF 备份：<a href="outputs/caseA_animation.gif">caseA_animation.gif</a></p>
</div>
<script>
const vFrames=""" + frames_data + """;
let vI=0,vTimer=null;
const vImg=document.getElementById('vFrame'),vRImg=document.getElementById('vRiser'),
      vSld=document.getElementById('vSlider'),vBtn=document.getElementById('vPlay');
vSld.min=0;vSld.max=Math.max(0,vFrames.length-1);vSld.value=0;
function vShow(k){
  vI=Math.max(0,Math.min(vFrames.length-1,k));
  const f=vFrames[vI];
  vImg.src=f.file; vRImg.src=f.riserFile; vSld.value=vI;
  vIdx.textContent=`帧 ${vI+1}/${vFrames.length}`;
  vTime.textContent=`t=${f.time.toFixed(2)} s`;
  vWtop.textContent=`塔内可见水位=${f.wtop.toFixed(3)} m`;
  vItop.textContent=`竖管气相前沿=${f.itop.toFixed(3)} m`;
  vMass.textContent=`竖管解析气体质量=${f.coreMassMg.toFixed(2)} mg`;
  vHead.textContent=`气囊压力头=${f.head.toFixed(3)} m`;
}
function vStop(){if(vTimer)clearInterval(vTimer);vTimer=null;vBtn.textContent='播放';}
vPrev.onclick=()=>{vStop();vShow(vI-1)};
vNext.onclick=()=>{vStop();vShow(vI+1)};
vSld.oninput=e=>{vStop();vShow(Number(e.target.value))};
vBtn.onclick=()=>{if(vTimer){vStop();return};vBtn.textContent='暂停';
  vTimer=setInterval(()=>vShow(vI>=vFrames.length-1?0:vI+1),260)};
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowLeft'){vStop();vShow(vI-1)}
  if(e.key==='ArrowRight'){vStop();vShow(vI+1)}});
vShow(0);
</script>""")
    elif (OUT / "caseA_animation.gif").exists():
        parts.append("""
<div class="panel">
  <h2 style="margin-top:0">两流体模拟动画 — 水平管 + 通风塔全场演化</h2>
  <p class="muted">同一冻结求解器的 Case A 全场动画（水=蓝，气=白；塔径为可视化放宽绘制）：
  阀开冲击 → 气囊分层推进 → 抵塔爬升（Taylor 气泡）→ 贯通排气 → 压力塌落。</p>
  <img src="outputs/caseA_animation.gif" style="max-width:900px">
</div>""")
    if (OUT / "caseA_fig10_overlay.png").exists():
        parts.append("""
<div class="panel">
  <h2 style="margin-top:0">Fig.10 对照 — 模型曲线直接叠画在论文原图上</h2>
  <p class="muted">论文 Fig.10 工况为 <b>WL=0.254 m</b>（非本文件夹的 0.356），我们用同一冻结求解器
  改初始水位复跑，并把结果（<b style="color:#e11d48">红线</b>）按论文坐标系直接画进原图五个面板：
  黑粗线=论文自家 TPA 模型，虚线散点=实验三次重复。时间轴做 <b>+0.63 T* 折中刚体平移</b>：
  Y*<sub>int</sub> 爬升想要 +1.0、H* 衰减想要 +0.25，一个刚体平移无法同时满足——
  <b>结构性差异</b>在于我们的气囊在界面爬完之后才贯通泄压，而实验（薄膜绕流拓扑）是边爬边泄。</p>
  <img src="outputs/caseA_fig10_overlay.png">
  <p class="muted">读图（折中平移下五个面板都是"平行、略有错位"）：
  Y*<sub>int</sub> 斜率与跨度一致、整体早 ~0.4 T*；V*<sub>int</sub> 平台 ~0.37 vs 论文 ~0.4；
  Y*<sub>fs</sub> 爬升幅度一致（终段回落=我们的排气排水偏快）；
  V*<sub>fs</sub> 量级一致但欠振荡（实验有 0.03–0.13 波动）；
  H* 衰减段斜率与论文实验平行、起点晚 ~0.4 T*。若只看某一个面板，把红线整体左/右平移
  0.3–0.4 T* 即可与该面板重合——残余错位反映的是上述结构性差异，不是量级误差。</p>
</div>""")
    elif (OUT / "caseA_fig10_model_panels.png").exists():
        parts.append("""
<div class="panel">
  <h2 style="margin-top:0">Fig.10 对照 — 论文自家数值模型（TPA）同轴对比</h2>
  <p class="muted">注意：论文 Fig.10 的工况是 <b>WL=0.254 m</b>（不是本文件夹的 0.356），
  为对齐我们用同一冻结求解器改初始水位复跑。左=论文原图（实验 3 次 + 他们的模型），右=我们的模型同轴输出。</p>
  <div class="grid2">
    <div><img src="paper_scans/fig10_full.png"></div>
    <div><img src="outputs/caseA_fig10_model_panels.png"></div>
  </div>
</div>""")
    return "".join(parts)


if __name__ == "__main__":
    main()

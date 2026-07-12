# -*- coding: utf-8 -*-
"""Case A of the Cong, Chan & Lee (2017) reproduction: Run B-H1 (geysering).

    D = 0.05 m, Dr = 0.016 m (Dr/D = 0.32), H0 = 0.66 m, L0 = 0.61 m,
    V*air = 9.03  ->  experiment: GEYSER in the high-speed-camera run.

Layout (Series B, Fig. 1b): constant-head tank -- 2.88 m pipe -- tee/riser
(1.8 m tall) -- 2.51 m pipe -- ball valve -- 0.61 m ATMOSPHERIC air pocket --
closed cap.  The tee position follows Table 2's own kinematic pair for B-H1
(Ta = 8.07 s at Uf = 0.444*sqrt(gD) = 0.311 m/s -> valve-to-tee = 2.51 m);
the "~1.8 m" in the text is the camera field of view, not the full distance.

Opening the valve releases the pocket: it migrates along the crown toward the
riser, partially enters, the supply slug arrests and COMPRESSES the pocket
(measured surge ~1.9*H0), and the column is ejected (geyser at t ~ 9.55 s).

This driver runs the frozen per-case solver copy (model/), overlays the
digitized Fig. 9(a) trajectories and Fig. 10(a) PT1 trace, writes metrics +
the model series, and builds report.html.

The tracked 1-D CSV is a frozen legacy comparison. Re-running this
threshold-sensitive solver under a different NumPy/runtime can change its
branch, so replacing those artifacts requires an explicit command-line flag.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(CASE_ROOT / "model"))

from cong2017_network_twofluid import G, NetworkCase, run_network

DIG = CASE_ROOT / "data" / "digitized"
OUT = CASE_ROOT / "outputs"
REPORT = OUT / "report.html"
OUT.mkdir(parents=True, exist_ok=True)

C_MODEL = "#d62728"
C_MODEL2 = "#f59e0b"
C_PAPER_BAND = "#9ca3af"
C_FS = "#2563eb"
C_INT = "#111827"

H0 = 0.66
HR = 1.8
CASE_KW = dict(
    D=0.05, Dr=0.016, riser_height=HR,
    L_up=2.88, L_mid=2.51, L_down=0.61,
    x_riser_at=2.88,
    pocket_downstream=True,
    reservoir_head=H0,
    air_head=0.0,
    init_water_level=H0,
    Hop_cap=10.0,
    x_transducer_at=5.85,
)
T_END = 13.0

EXP = dict(geyser=True, Ta=8.07, v_fs=0.924, v_int=1.231,
           t_rim=9.55, PT1_peak_over_H0=1.9)


def run_model():
    case = NetworkCase(**CASE_KW, t_end=T_END)
    rec = run_network(case, verbose=False)
    t = np.asarray(rec["t"])
    n = min(len(t), len(rec["tr_head"]) + 1)
    series = dict(
        t=t[:n],
        Yfs=np.asarray(rec["wtop"])[:n],
        Yint=np.asarray(rec["itop"])[:n],
        pocket=np.asarray(rec["up_head"])[:n],
        tr=np.concatenate([[np.nan], np.asarray(rec["tr_head"])])[:n],
        pj=np.concatenate([[np.nan], np.asarray(rec["pj_head"])])[:n],
    )
    return case, series


def load_digitized():
    fs, gi = [], []
    with (DIG / "fig9a_levels.csv").open() as f:
        for row in csv.DictReader(f):
            (fs if row["kind"] == "fs" else gi).append((float(row["t_s"]), float(row["Y_m"])))
    p10 = dict(t=[], med=[], lo=[], hi=[])
    with (DIG / "fig10a_pt1.csv").open() as f:
        for row in csv.DictReader(f):
            p10["t"].append(float(row["t_s"]))
            p10["med"].append(float(row["HoverH0_med"]))
            p10["lo"].append(float(row["HoverH0_min"]))
            p10["hi"].append(float(row["HoverH0_max"]))
    return np.array(sorted(fs)), np.array(sorted(gi)), {k: np.array(v) for k, v in p10.items()}


def first_crossing(x, y, thresh, above=True, after=0.0):
    for xi, yi in zip(x, y):
        if xi < after:
            continue
        if (yi >= thresh) if above else (yi <= thresh):
            return float(xi)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite-frozen",
        action="store_true",
        help="explicitly replace the tracked legacy 1-D comparison artifacts",
    )
    args = parser.parse_args()
    if not args.overwrite_frozen:
        parser.error(
            "tracked outputs are frozen; pass --overwrite-frozen only when "
            "intentionally re-baselining the legacy 1-D comparison"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs_pts, int_pts, p10 = load_digitized()
    case, s = run_model()
    t = s["t"]

    # ---------------- headline events ----------------
    Ta_m = first_crossing(t, s["Yint"], 0.02)
    t_rim_m = first_crossing(t, s["Yfs"], 0.98 * HR)
    geyser_m = t_rim_m is not None

    # eruption speeds on the FIRST coherent surge (the segment comparable to
    # the experiment's single climb, Fig. 9a): from the last quiescent minimum
    # of Yfs before the first crossing of H0+0.5 up to that surge's own peak.
    # The model needs a second pulse to reach the rim (documented honestly);
    # averaging across the inter-pulse valley would not measure a climb.
    v_fs_m = v_int_m = None
    t_srg = first_crossing(t, s["Yfs"], H0 + 0.5)
    if t_srg is not None:
        i_c = int(np.argmax(t >= t_srg))
        i0 = i_c
        while i0 > 1 and s["Yfs"][i0 - 1] > s["Yfs"][i0 - 2] - 1e-9:
            i0 -= 1
        i1 = i_c
        while i1 + 1 < len(t) and s["Yfs"][i1 + 1] >= s["Yfs"][i1] - 1e-9:
            i1 += 1
        v_fs_m = float((s["Yfs"][i1] - s["Yfs"][i0]) / max(t[i1] - t[i0], 1e-9))
        seg = (t >= t[i0]) & (t <= t[i1] + 0.3) & (s["Yint"] > 0.05)
        if np.count_nonzero(seg) >= 3:
            v_int_m = float(np.polyfit(t[seg], s["Yint"][seg], 1)[0])

    # compression surge = the pocket peak around the FIRST ejection pulse (the
    # geysering drive the paper reports as ~1.9*H0); the violent post-eruption
    # ringing after the column re-falls is excluded (documented separately)
    t_surge1 = first_crossing(t, s["Yfs"], H0 + 0.5)
    if t_surge1 is not None:
        w1 = (t >= t_surge1 - 1.2) & (t <= t_surge1 + 0.3)
        pk = float(np.nanmax(s["pocket"][w1]))
        i_pk = int(np.nanargmax(np.where(w1, s["pocket"], -np.inf)))
    else:
        pk = float(np.nanmax(s["pocket"]))
        i_pk = int(np.nanargmax(s["pocket"]))

    # ---------------- levels overlay (Fig. 9a) ----------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.0))
    for ax, xlim in ((a1, (0.0, T_END)), (a2, (7.5, 12.5))):
        ax.plot(t, s["Yfs"], color=C_MODEL, lw=2.0, label="model $Y_{fs}$ (free surface)")
        ax.plot(t, s["Yint"], color=C_MODEL2, lw=1.8, ls="--", label="model $Y_{int}$ (gas front)")
        if fs_pts.size:
            ax.plot(fs_pts[:, 0], fs_pts[:, 1], "s", ms=5, mfc="none", mec=C_FS,
                    label="experiment $Y_{fs}$ (Fig.9a, digitized)")
        if int_pts.size:
            ax.plot(int_pts[:, 0], int_pts[:, 1], "o", ms=5, mfc="none", mec=C_INT,
                    label="experiment $Y_{int}$ (Fig.9a, digitized)")
        ax.axhline(HR, color="#16a34a", ls=":", lw=1.2)
        ax.axhline(H0, color="0.65", ls=":", lw=1.0)
        ax.set_ylim(0, 1.95)
        ax.set_xlim(*xlim)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Y [m] above riser entrance")
        ax.grid(alpha=0.3)
    a1.axvspan(8.0, 10.0, color="#f3f4f6", zorder=0)
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    a1.set_title("full model trajectory (paper Fig.9a window shaded)", fontsize=10)
    # shifted slope check: align the model's FIRST surge (crossing Y=1.0 m)
    # with the experimental climb through the same level
    if fs_pts.size and first_crossing(t, s["Yfs"], 1.0) is not None:
        t_exp_mid = float(np.interp(1.0, fs_pts[:, 1], fs_pts[:, 0]))
        t_mod_mid = first_crossing(t, s["Yfs"], 1.0)
        dshift = t_exp_mid - t_mod_mid
        a2.plot(t + dshift, s["Yfs"], color=C_MODEL, lw=1.1, ls=":",
                label=f"model $Y_{{fs}}$ shifted {dshift:+.2f} s (slope check)")
        a2.plot(t + dshift, s["Yint"], color=C_MODEL2, lw=1.1, ls=":",
                label=f"model $Y_{{int}}$ shifted {dshift:+.2f} s")
    a2.legend(frameon=False, fontsize=7, loc="upper left")
    a2.set_title("eruption window (paper data t = 8..10 s)", fontsize=10)
    fig.suptitle("Run B-H1 riser trajectories -- $D_r$=16 mm ($D_r/D$=0.32), "
                 "$H_0$=0.66 m, $L_0$=0.61 m", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "caseA_comparison_levels.png", dpi=150)
    plt.close(fig)

    # ---------------- pressure overlay (Fig. 10a) ----------------
    # cycle-average the model pocket head over ~one release-slosh period so the
    # overlay carries the envelope, not the resolved ringing (cf. VW2011 cases);
    # the raw trace stays in the CSV.
    pk_over = s["pocket"] / H0
    pk_avg = np.full_like(pk_over, np.nan)
    for i in range(len(pk_over)):
        mwin = (t >= t[i] - 0.25) & (t <= t[i] + 0.25)
        if np.any(np.isfinite(pk_over[mwin])):
            pk_avg[i] = np.nanmean(pk_over[mwin])
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.fill_between(p10["t"], p10["lo"], p10["hi"], color=C_PAPER_BAND, alpha=0.45,
                    label="Run B-1 PT1, digitized pixel envelope (Fig.10a)")
    ax.plot(p10["t"], p10["med"], color="#374151", lw=1.3,
            label="Run B-1 PT1, digitized median (video series, same condition)")
    ax.plot(t, pk_avg, color=C_MODEL, lw=1.8,
            label="model: trapped-pocket EOS gauge head / $H_0$ (0.5 s cycle-avg)")
    ax.plot(t, pk_over, color=C_MODEL, lw=0.7, alpha=0.35,
            label="model: raw trace")
    ax.axhline(EXP["PT1_peak_over_H0"], color="#16a34a", ls=":", lw=1.2,
               label="paper text: geyser surge $\\approx 1.9\\,H_0$")
    ax.set_xlim(0, 13)
    ax.set_ylim(-1.0, 4.0)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("$H/H_0$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Run B-H1 pocket pressure vs PT1 (pipe crown near the closed end)\n"
                 "model (decoupled two-fluid, frozen copy) vs Cong et al. (2017) Fig.10(a), digitized",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "caseA_comparison_pressure.png", dpi=150)
    plt.close(fig)

    # ---------------- metrics ----------------
    v_fs_exp_chk = v_int_exp_chk = None
    if fs_pts.size >= 4:
        v_fs_exp_chk = float(np.polyfit(fs_pts[:, 0][fs_pts[:, 1] > 0.65],
                                        fs_pts[:, 1][fs_pts[:, 1] > 0.65], 1)[0])
    if int_pts.size >= 4:
        v_int_exp_chk = float(np.polyfit(int_pts[:, 0][int_pts[:, 1] > 0.10],
                                         int_pts[:, 1][int_pts[:, 1] > 0.10], 1)[0])
    m = dict(
        case=dict(run="B-H1", D=case.D, Dr=case.Dr, Dr_over_D=case.Dr / case.D,
                  H0=H0, L0=case.L_down, riser_height=HR,
                  tee_x=case.x_riser, valve_x=case.x_valve,
                  Vair_L=case.V_air * 1000.0,
                  Vstar_air=case.V_air / (0.25 * math.pi * case.Dr ** 2 * H0)),
        model=dict(
            geyser=bool(geyser_m),
            Ta_gas_enters_riser_s=Ta_m,
            t_free_surface_at_rim_s=t_rim_m,
            v_fs_event_mean_mps=v_fs_m,
            v_int_event_fit_mps=v_int_m,
            pocket_surge_over_H0=pk / H0,
            pocket_surge_time_s=float(t[i_pk]),
            pocket_plateau_over_H0=float(np.nanmedian(
                s["pocket"][(t > 2.0) & (t < 7.0)])) / H0,
        ),
        paper=dict(
            geyser=True,
            Ta_s=EXP["Ta"],
            Uf_over_sqrtgD=0.444,
            t_free_surface_at_rim_s=EXP["t_rim"],
            v_fs_mps=EXP["v_fs"],
            v_int_mps=EXP["v_int"],
            PT1_geyser_surge_over_H0=EXP["PT1_peak_over_H0"],
            fig9a_vfs_climb_fit_mps=v_fs_exp_chk,
            fig9a_vint_climb_fit_mps=v_int_exp_chk,
        ),
        notes=[
            "Geometry: tee at x=2.88 m from Table 2's own Ta*Uf kinematics "
            "(8.07 s x 0.444 sqrt(gD) = 2.51 m valve-to-tee); the '~1.8 m' in "
            "the text is the camera field of view.",
            "PT1 datum: the digitized Fig.10(a) trace is Run B-1 (video series, "
            "same nominal condition); its pre-arrival plateau reads ~1.3 H0, "
            "above the static reservoir head -- transducer datum/calibration "
            "uncertainty ~0.3 H0; shape anchors (release oscillation, plateau, "
            "pre-geyser dip, surge) are the meaningful comparison.",
            "Model pocket head is the trapped-pocket EOS gauge head (the gas "
            "pressure PT1 sits in once the pocket surrounds the port).",
        ],
    )
    (OUT / "caseA_comparison_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

    with (OUT / "caseA_model_series.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "Yfs_m", "Yint_m", "pocket_head_m", "tr_head_m", "pj_head_m"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.4f}", f"{s['Yfs'][i]:.4f}", f"{s['Yint'][i]:.4f}",
                        f"{s['pocket'][i]:.4f}", f"{s['tr'][i]:.4f}", f"{s['pj'][i]:.4f}"])

    build_report(m)
    print(json.dumps(m["model"], indent=2))
    print(f"-> {OUT / 'caseA_comparison_levels.png'}")
    print(f"-> {OUT / 'caseA_comparison_pressure.png'}")
    print(f"-> {REPORT}")


def build_report(m: dict):
    mo, pa = m["model"], m["paper"]

    def _f(v, nd=2):
        return "-" if v is None else f"{v:.{nd}f}"

    rows = [
        ("是否喷发（水气混合物射出竖管顶）", "是（高速摄像判定）",
         ("是" if mo.get("geyser") else "否")
         + (f"（水面 t={_f(mo.get('t_free_surface_at_rim_s'))} s 到顶并持续溢出）" if mo.get("geyser") else ""),
         "分支判别正确" if mo.get("geyser") else "分支错误"),
        ("气囊到达竖管 Ta", f"{_f(pa.get('Ta_s'))} s", f"{_f(mo.get('Ta_gas_enters_riser_s'))} s",
         "偏早 ~0.5 s（Benjamin 波速略高于实测 0.44√(gD)）"),
        ("水面到达竖管顶（喷发时刻）", f"{_f(pa.get('t_free_surface_at_rim_s'))} s",
         f"{_f(mo.get('t_free_surface_at_rim_s'))} s", "偏晚 ~2 s（多脉冲爬升，见诚实记录）"),
        ("首个喷发脉冲水面上升速度 v_fs", f"{_f(pa.get('v_fs_mps'))} m/s（Table 2，全程平均）",
         f"{_f(mo.get('v_fs_event_mean_mps'))} m/s", "首脉冲起爬到脉冲峰的平均斜率"),
        ("首个喷发脉冲气核上升速度 v_int", f"{_f(pa.get('v_int_mps'))} m/s（Table 2，全程平均）",
         f"{_f(mo.get('v_int_event_fit_mps'))} m/s", "同段拟合斜率"),
        ("气囊压缩激增 / H0（首个喷发脉冲）", f"约 {_f(pa.get('PT1_geyser_surge_over_H0'), 1)}（正文，PT1）",
         f"{_f(mo.get('pocket_surge_over_H0'))}（t={_f(mo.get('pocket_surge_time_s'), 1)} s）",
         "压缩-喷发机理的量级锚点"),
        ("喷发前气囊压头平台 / H0", "约 1.3（Fig.10a 数字化；基准面存疑 ±0.3）",
         f"{_f(mo.get('pocket_plateau_over_H0'))}", "形态均为长平台"),
    ]
    trs = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>"
                  for a, b, c, d in rows)
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Case A: Cong 2017 Run B-H1（喷发）对比</title>
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
<h1>Case A — Cong, Chan &amp; Lee (2017) Run B-H1（喷发分支，高速摄像）</h1>
<p>工况：<code>D=50 mm</code>、<code>Dr=16 mm</code>（Dr/D=0.32）、<code>H0=0.66 m</code>（上游定水头）、
<code>L0=0.61 m</code>（初始常压气囊，V*air=9.03）。布置：定水头水箱 — 2.88 m 水平管 — T 型三通
（上接 1.8 m 竖管）— 2.51 m 水平管 — 球阀 — 0.61 m 气囊 — 封闭端（三通位置由 Table 2 自身的
Ta×Uf 运动学定出：8.07 s × 0.444√(gD) = 2.51 m 阀-三通距离）。开阀后气囊沿管顶向竖管迁移
（实测 Ta=8.07 s）、部分进入竖管，供给水柱受阻并<b>压缩气囊</b>（实测压头冲至 ~1.9H0），
把竖管水柱整体喷出（t≈9.55 s 喷发）。</p>
<div class="panel">
  <h2 style="margin-top:0">论文中与本工况相关的图表</h2>
  <table>
    <tr><th>论文图表</th><th>内容</th><th>与本工况的关系</th><th>本报告中的对照</th></tr>
    <tr><td><b>Table 2</b>（p7）</td><td>Series B 全部工况参数与结果汇总</td>
        <td><b>B-H1 行</b>：Ta=8.07 s、Uf=0.444√(gD)、v_fs=0.924、v_int=1.231 m/s、GEYSER</td>
        <td>指标对照表 + 三通位置的运动学依据</td></tr>
    <tr><td><b>Fig. 8</b>（p8）</td><td>B-H1 竖管内气囊上升的高速摄像瞬时帧（t=8.67/9.07/9.47/9.55 s）</td>
        <td><b>正是本工况</b></td><td>下方原图收录，可与帧查看器对照</td></tr>
    <tr><td><b>Fig. 9</b>（p9）</td><td>B-H1 的 Yfs/Yint/速度/气囊压头四联图（高速摄像）</td>
        <td><b>正是本工况</b>；(a) 面板已数字化</td>
        <td>水位叠加图</td></tr>
    <tr><td><b>Fig. 10(a)</b>（p10）</td><td>PT1/PT2 压力时程（Run B-1，同名义工况的录像系列）</td>
        <td>同 Dr=16 mm、H0=0.66、L0=0.61 的重复（视频采样）</td>
        <td>压力叠加图</td></tr>
    <tr><td class="muted">Fig. 12/13</td><td class="muted">喷发机理示意 / B-1 水平管前沿与反射波（气量分配）</td>
        <td class="muted">机理背景（前沿受阻→压缩→喷发）</td><td class="muted">正文引用</td></tr>
  </table>
</div>
<div class="panel">
  <h2 style="margin-top:0">论文原图（扫描）</h2>
  <div class="grid2">
    <div><h3 style="margin:4px 0">Fig.9 B-H1 竖管数据（(a) 已数字化）</h3><img src="../reference/paper_scans/fig9_bh1_riser.png"></div>
    <div><h3 style="margin:4px 0">Fig.10 压力时程（上=B-1 喷发）</h3><img src="../reference/paper_scans/fig10_pressure.png"></div>
  </div>
  <h3 style="margin:12px 0 4px 0">Fig.8 B-H1 高速摄像瞬时帧</h3>
  <img src="../reference/paper_scans/fig8_bh1_photos.png">
</div>
<div class="panel">
  <h2 style="margin-top:0">叠加对比</h2>
  <h3>竖管水面与气核前端 Y(t)</h3><img src="caseA_comparison_levels.png">
  <h3>气囊压头 H/H0</h3><img src="caseA_comparison_pressure.png">
  <h3>指标对照</h3>
  <table>
    <tr><th>指标</th><th>论文实验</th><th>模型</th><th>备注</th></tr>
    {trs}
  </table>
  <p class="muted"><b>吻合点</b>：喷发分支判别正确（水气混合物冲至竖管顶并溢出）；气囊到达时刻
  7.58 vs 实测 8.07 s（几何由 Table 2 的 Ta×Uf 运动学定出后偏差收敛到 ~0.5 s）；机理链条与论文
  Fig.12 一致（气囊沿管顶迁移 → 部分进入竖管 → 供给段受阻、气囊被压缩 → 压头激增 →
  水柱整体喷出）；喷发压缩激增与喷发前平台两条压力形态锚点均复现。<br>
  <b>差异点（诚实记录）</b>：① 喷发时刻偏晚 ~2 s 且分多个脉冲——第一脉冲冲至 ~1.2 m 回落、
  后续脉冲到顶，实验为单次光滑加速爬升；一维模型把喷发解析成"压缩-释放"极限环而非单调喷射，
  全程平均速度因此低于实测的单段爬升；② 喷发后压力振荡幅度偏大（±2H0）——刚性水柱回落再压缩
  缺少三维破碎/掺气耗散；③ PT1 数字化平台 ~1.3H0 高于静水库头（传感器基准/标定存疑 ±0.3H0），
  压力对比以形态锚点（释放振荡、平台、喷发前回落、激增）为主。
  模型序列见 <code>caseA_model_series.csv</code>；数字化中间产物见 <code>../data/digitized/</code>。</p>
</div>
__EXTRA_SECTIONS__
</div></body></html>"""
    html = html.replace("__EXTRA_SECTIONS__", build_extra_sections())
    REPORT.write_text(html, encoding="utf-8")


def build_extra_sections() -> str:
    parts = []
    frames_json = OUT / "frames_index.json"
    if frames_json.exists():
        frames = json.loads(frames_json.read_text(encoding="utf-8"))
        for frame in frames:
            frame["file"] = frame["file"].removeprefix("outputs/")
            frame["riserFile"] = frame["riserFile"].removeprefix("outputs/")
        frames_data = json.dumps(frames)
        parts.append("""
<div class="panel">
  <h2 style="margin-top:0">两流体模拟逐帧查看器 — 水平管 + 竖管全场演化</h2>
  <p class="muted">同一冻结求解器的 B-H1 全场演化（水=蓝，气=白/浅灰）：开阀释放振荡 →
  气囊沿管顶向竖管迁移（越过三通继续向水库侧推进） → 竖管进气、气核增长 →
  供给段受阻、气囊压缩 → 压头激增、水柱整体喷出（喷发） → 排空回落。
  <b>左图为真实比例 1:1</b>（管径 D=50 mm、竖管 Dr=16 mm 按真实尺寸），
  竖管细节请看右侧放大同步视图（按当地两流体气含率 α_g 画气核宽度）。
  左右方向键 / 滑块 / 播放均可调帧。</p>
  <div class="meta" style="display:flex;gap:18px;margin:10px 0;font-weight:700;flex-wrap:wrap">
    <span id="vIdx"></span><span id="vTime"></span><span id="vWtop"></span>
    <span id="vItop"></span><span id="vMass"></span><span id="vHead"></span></div>
  <div style="display:grid;grid-template-columns:minmax(0,2.6fr) minmax(220px,1fr);gap:14px;align-items:start">
    <div><h3 style="margin:0 0 8px 0;font-size:15px">全局 1:1 视图</h3>
      <img id="vFrame" style="width:100%">
      <p class="muted" style="margin:6px 0 0 0">水平管按截面含水率显示分层气团推进；左端为定水头水箱。</p></div>
    <div><h3 style="margin:0 0 8px 0;font-size:15px">竖管放大同步视图</h3>
      <img id="vRiser" style="width:100%">
      <p class="muted" style="margin:6px 0 0 0">气核宽度 = 当地 α_g；蓝线=可见水面，红虚线=气相前沿。</p></div>
  </div>
  <div style="margin-top:8px">
    <button id="vPrev" style="padding:8px 14px;margin:6px;border:1px solid #bbb;border-radius:8px;background:#fff;cursor:pointer">上一帧</button>
    <button id="vPlay" style="padding:8px 14px;margin:6px;border:1px solid #bbb;border-radius:8px;background:#fff;cursor:pointer">播放</button>
    <input id="vSlider" type="range" style="width:60%;vertical-align:middle">
    <button id="vNext" style="padding:8px 14px;margin:6px;border:1px solid #bbb;border-radius:8px;background:#fff;cursor:pointer">下一帧</button>
  </div>
  <p class="muted">整段 GIF 备份：<a href="caseA_animation.gif">caseA_animation.gif</a></p>
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
  vWtop.textContent=`竖管可见水位=${f.wtop.toFixed(3)} m`;
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
  <h2 style="margin-top:0">两流体模拟动画</h2>
  <img src="caseA_animation.gif" style="max-width:900px">
</div>""")
    return "".join(parts)


if __name__ == "__main__":
    main()

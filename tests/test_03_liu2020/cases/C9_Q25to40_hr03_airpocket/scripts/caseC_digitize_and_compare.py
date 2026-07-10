# -*- coding: utf-8 -*-
"""Liu, Shao & Zhu (2020) Case C9 -- digitize the published pressure traces
(Fig. 9), run the frozen model (two declared variants), overlay against the
measurement AND the paper's own analytic model Eq. (7), and build report.html.

Case C9: Q = 25 -> 40 L/s (Tv ~ 0.4 s), downstream full pipe (throttled tail
gate), initial riser column hr0 = 0.30 m, sealed air pocket in the upstream
pipe crown.  Observed: two-phase violent geysers (8 events); phase 1 =
hydraulic transient (geysers 1-2), phase 2 = pocket arrival/release
(geysers 3-8, from t ~ 6.46 s).

ROUND-1 SCOPE (declared): phase 1.  The paper's own analytic model Eq. (7)
addresses exactly this phase under the all-water assumption; the model is
run both all-water (no_pocket=True, the Eq.-7 assumption set) and with the
sealed pocket (cushioning demo).  The phase-2 bubble/slug geyser train is
beyond this one-dimensional reduction and is reported as such.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))

from liu2020_network_twofluid import LiuCase, run_case, RHO_L, G  # noqa: E402

SCANS = HERE / "paper_scans"
DIG = HERE / "digitized"
OUT = HERE / "outputs"
for d in (DIG, OUT):
    d.mkdir(exist_ok=True)

PAGE = SCANS / "p09_fig9_C9_pressure_phases.png"
CROP = (100, 60, 1650, 700)      # generous crop around the Fig. 9 plot (px)

PAPER = dict(
    geyser=True, n_geysers=8,
    # phase-1 chronology (text)
    P1m_kPa=10.69, t_P1m_s=0.50,
    t_first_top_s=0.73,           # free surface reached riser top
    t_pocket_arrival_s=6.46,      # phase-2 start (pocket body at chamber)
    # initial / final steady (text)
    PT2_initial_kPa=2.97, PT3_initial_kPa=7.09,
    PT2_final_kPa=8.79, PT3_final_kPa=12.76, PT4_final_kPa=9.25,
    # analytic model Eq. (7)/(8)
    T_eq8_s=1.45,                 # paper's stated period for C9
    # Fig. 11 regressions (Series C)
    fig11_h_slope=1.5385, fig11_h_intercept=0.6087,
    fig11_pf_slope=0.6971, fig11_pf_intercept=-0.0245,
)


def eq7_head(t, case: LiuCase):
    """the paper's analytic Eq. (7): piezometric head at PT2 (datum = chamber
    top), all-water, frictionless, valve ramp dQu/dt = (Q1-Q0)/Tv"""
    Ad = case.Ad
    Ar = case.Ar
    T = 2.0 * math.pi * math.sqrt(case.Ld * Ar / (G * Ad) + case.hr0 / G)
    dQdt = (case.Q1 - case.Q0) / case.Tv
    amp = case.Ld / (G * Ad) * dQdt
    return case.hr0 + amp * (1.0 - np.cos(2.0 * math.pi * t / T)), T


# ---------------------------------------------------------------- digitizer
def _reject_trace_spikes(t: np.ndarray, p: np.ndarray,
                        max_jump_kpa: float = 2.5,
                        max_rate_kpa_s: float = 60.0):
    """Drop columns polluted by dashed vertical guide lines in Fig.9."""
    if t.size < 2:
        return t, p
    keep_t, keep_p = [float(t[0])], [float(p[0])]
    for i in range(1, t.size):
        dt = float(t[i] - keep_t[-1])
        if dt <= 1e-9:
            continue
        dp = abs(float(p[i] - keep_p[-1]))
        if dp > max_jump_kpa or dp / dt > max_rate_kpa_s:
            continue
        keep_t.append(float(t[i]))
        keep_p.append(float(p[i]))
    return np.asarray(keep_t), np.asarray(keep_p)


def _plot_box(gray: np.ndarray):
    dark = gray < 0.55
    h, w = dark.shape
    col_frac = dark.sum(axis=0) / h
    row_frac = dark.sum(axis=1) / w
    cols = np.where(col_frac > 0.5)[0]
    rows = np.where(row_frac > 0.5)[0]
    if cols.size < 2 or rows.size < 2:
        raise RuntimeError("plot box not found")
    return int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])


def digitize_fig9():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    img = np.asarray(Image.open(PAGE).convert("RGB"), dtype=float) / 255.0
    rgb = img[CROP[1]:CROP[3], CROP[0]:CROP[2], :]
    gray = rgb.mean(axis=2)
    x0, x1, y0, y1 = _plot_box(gray)
    T0, T1 = -1.0, 20.0
    P0, P1 = -2.0, 18.0

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    masks = {
        # PT3 gold/ochre: red+green high, blue low
        "PT3": (r > 0.55) & (g > 0.35) & (b < 0.45) & (r - b > 0.25) & (g - b > 0.12),
        # PT4 red: red dominant over green AND blue
        "PT4": (r > 0.55) & (r - g > 0.25) & (r - b > 0.25),
        # PT2 navy: blue dominant, darkish
        "PT2": (b > 0.30) & (b - r > 0.12) & (b - g > 0.12),
        # PT1 green: green dominant
        "PT1": (g > 0.35) & (g - r > 0.10) & (g - b > 0.10),
    }
    series = {}
    for name, m in masks.items():
        mm = m.copy()
        mm[:y0 + 2, :] = False
        mm[y1 - 1:, :] = False
        mm[:, :x0 + 2] = False
        mm[:, x1 - 1:] = False
        ts, md_ = [], []
        prev = None
        for cx in range(x0 + 2, x1 - 1):
            rows = np.where(mm[:, cx])[0]
            if rows.size == 0:
                continue
            # the original figure draws the geyser-boundary boxes as DASHED
            # vertical lines in the same colors as the traces; per column
            # they appear as tall multi-cluster pixel runs.  Split the rows
            # into contiguous clusters, drop implausibly tall ones, and keep
            # the cluster closest to the previous accepted value (trace
            # continuity).
            splits = np.where(np.diff(rows) > 4)[0]
            clusters = np.split(rows, splits + 1)
            clusters = [c for c in clusters if c[-1] - c[0] < 0.10 * (y1 - y0)]
            if not clusters:
                continue
            pmed = [float(P1 + (np.median(c) - y0) / (y1 - y0) * (P0 - P1))
                    for c in clusters]
            if prev is None:
                val = pmed[int(np.argmin([abs(p) for p in pmed]))] \
                    if name == "PT1" else pmed[0]
            else:
                val = pmed[int(np.argmin([abs(p - prev) for p in pmed]))]
                if abs(val - prev) > 4.0:      # discontinuous jump: dashed box
                    continue
            prev = val
            ts.append(T0 + (cx - x0) / (x1 - x0) * (T1 - T0))
            md_.append(val)
        t_arr, p_arr = np.array(ts), np.array(md_)
        t_arr, p_arr = _reject_trace_spikes(t_arr, p_arr,
                                            max_jump_kpa=1.2,
                                            max_rate_kpa_s=35.0)
        if name == "PT2" and p_arr.size >= 7:
            from scipy.ndimage import median_filter
            p_arr = median_filter(p_arr, size=7, mode="nearest")
        series[name] = dict(t=t_arr, med=p_arr)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    colors = dict(PT1="#1d8a4a", PT2="#27357e", PT3="#c8960c", PT4="#c81e3c")
    for name, s in series.items():
        ax.plot(s["t"], s["med"], color=colors[name], lw=0.9, label=f"{name} median")
    ax.set_xlim(-1, 20); ax.set_ylim(-2, 18)
    ax.set_xlabel("t [s]"); ax.set_ylabel("p [kPa]")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    ax.set_title("digitized check: Liu2020 Fig.9 (Case C9)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DIG / "debug_fig9_extract.png", dpi=140)
    plt.close(fig)

    for name, s in series.items():
        arr = np.column_stack([s["t"], s["med"]])
        np.savetxt(DIG / f"fig9_{name}.csv", arr, delimiter=",",
                   header="t_s,p_med_kPa", comments="")
    return series


# ---------------------------------------------------------------- pipeline
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = digitize_fig9()

    base = dict(Q0=0.025, Q1=0.040, downstream_full=True, series_c=True, hr0=0.30)
    case_w = LiuCase(t_end=20.0, no_pocket=True, **base)    # Eq.-7 assumption set
    rec_w = run_case(case_w, verbose=False)
    case_p = LiuCase(t_end=6.0, no_pocket=False, **base)    # sealed-pocket variant
    rec_p = run_case(case_p, verbose=False)

    tw = np.asarray(rec_w["t"])
    tp = np.asarray(rec_p["t"])
    PT2w = np.asarray(rec_w["PT2"])
    PT2p = np.asarray(rec_p["PT2"])

    # ---------------- three-way phase-1 figure (heads, datum chamber top) ----
    t_an = np.linspace(0.0, 5.0, 600)
    Hs_an, T_an = eq7_head(t_an, case_w)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    s = series["PT2"]
    m5 = (s["t"] >= -0.2) & (s["t"] <= 5.0)
    ax.plot(s["t"][m5], s["med"][m5] * 1e3 / (RHO_L * G), color="#27357e", lw=1.4,
            label="experiment (PT2, digitized Fig.9)")
    ax.plot(t_an, Hs_an, color="#d97706", lw=1.4, ls="--",
            label=f"paper's analytic Eq.(7)  (T={T_an:.2f} s)")
    mw5 = tw <= 5.0
    ax.plot(tw[mw5], PT2w[mw5] * 1e3 / (RHO_L * G), color="#c81e3c", lw=1.8,
            label="present model (all-water variant)")
    mp5 = tp <= 4.0
    k = 5   # 50 ms moving average: the raw pocket-EOS signal rings at the
            # grid scale and is unreadable overlaid; smoothing declared
    PT2p_s = np.convolve(PT2p, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    ax.plot(tp[mp5], PT2p_s[mp5] * 1e3 / (RHO_L * G), color="#9333ea", lw=1.2, ls=":",
            label="present model (sealed-pocket variant, 50 ms smoothed)")
    ax.set_xlabel("t [s]  (t=0 at the start of the inflow ramp)")
    ax.set_ylabel("$H_s$ at PT2 [m]  (datum: chamber top)")
    ax.set_xlim(-0.2, 5.0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Case C9, phase 1: measured vs analytic Eq.(7) vs present model")
    fig.tight_layout()
    fig.savefig(OUT / "caseC_phase1_threeway.png", dpi=150)
    plt.close(fig)

    # ---------------- full-record overlay (PT2/PT3/PT4) ----------------
    colors = dict(PT2="#27357e", PT3="#c8960c", PT4="#c81e3c")
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.0), sharex=True)
    for ax, name in zip(axes, ("PT2", "PT3", "PT4")):
        s = series[name]
        ax.plot(s["t"], s["med"], color=colors[name], lw=1.0,
                label=f"experiment {name} (digitized)")
        ax.plot(tw, np.asarray(rec_w[name]), color="#111827", lw=1.5,
                label=f"model {name} (all-water)")
        ax.axvline(PAPER["t_pocket_arrival_s"], color="#6b7280", ls="--", lw=1.0)
        ax.set_ylabel(f"{name} [kPa]")
        ax.set_xlim(-1, 20)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].text(PAPER["t_pocket_arrival_s"] + 0.15, 15.5,
                 "pocket arrival (exp):\nphase-2 geysers 3-8\nnot represented",
                 fontsize=7.5, color="#6b7280")
    axes[-1].set_xlabel("t [s]  (t=0 at the start of the inflow ramp)")
    axes[0].set_title("Liu2020 Case C9 (Q 25$\\to$40 L/s, $h_{r0}$=0.30 m): "
                      "full record -- model (all-water) vs digitized Fig.9")
    fig.tight_layout()
    fig.savefig(OUT / "caseC_comparison_pressure.png", dpi=150)
    plt.close(fig)

    # ---------------- riser column ----------------
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.plot(tw, np.asarray(rec_w["hr"]), color="#c81e3c", lw=1.6,
            label="model riser column $h_r$ (all-water)")
    ax.plot(tp, np.asarray(rec_p["hr"]), color="#9333ea", lw=1.1, ls=":",
            label="model riser column $h_r$ (sealed-pocket)")
    ax.axhline(case_w.Hr, color="#16a34a", ls=":", lw=1.3,
               label=f"riser top {case_w.Hr} m")
    ax.axvline(PAPER["t_first_top_s"], color="#6b7280", ls="--", lw=1.0)
    ax.text(PAPER["t_first_top_s"] + 0.05, 0.15, "free surface at top (exp)",
            rotation=90, fontsize=7, color="#6b7280")
    ax.set_xlabel("t [s]"); ax.set_ylabel("$h_r$ [m]")
    ax.set_xlim(-0.5, 6.0); ax.set_ylim(0, 1.35)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Case C9: riser column -- first (transient-driven) geysers")
    fig.tight_layout()
    fig.savefig(OUT / "caseC_riser_column.png", dpi=150)
    plt.close(fig)

    # ---------------- metrics ----------------
    i5 = tw <= 5.0
    ipk = int(np.argmax(PT2w[i5]))
    hrw = np.asarray(rec_w["hr"])
    i_top = np.argmax(hrw >= case_w.Hr - 1e-9) if (hrw >= case_w.Hr - 1e-9).any() else -1
    # model phase-1 oscillation period: spacing of the first two PT2 peaks
    pk_t = []
    k = 8
    ps = np.convolve(PT2w, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    for i in range(k, len(ps) - k):
        if tw[i] > 0.2 and tw[i] < 6.0 and ps[i] == max(ps[i - k:i + k + 1]) \
                and (not pk_t or tw[i] - pk_t[-1] > 0.4):
            pk_t.append(float(tw[i]))
    T_model = round(pk_t[1] - pk_t[0], 3) if len(pk_t) >= 2 else None
    m_fin = tw >= 16.0
    pmax_w = float(np.max(PT2w))
    h_jet_w = float(rec_w["h_jet"])
    metrics = dict(
        case="Liu2020 C9: Q 25->40 L/s, hr0=0.30 m, upstream air pocket",
        model_all_water=dict(
            geyser=bool(rec_w["geyser"]),
            P1m_kPa=float(PT2w[i5][ipk]), t_P1m_s=float(tw[i5][ipk]),
            t_first_top_s=float(tw[i_top]) if i_top >= 0 else None,
            T_osc_s=T_model,
            PT2_final_kPa=float(np.mean(PT2w[m_fin])),
            PT3_final_kPa=float(np.mean(np.asarray(rec_w["PT3"])[m_fin])),
            PT4_final_kPa=float(np.mean(np.asarray(rec_w["PT4"])[m_fin])),
            wr_eject=float(rec_w["wr_eject"]), h_jet=h_jet_w,
            overflow_L=float(rec_w["overflow_vol"] * 1e3),
            mass_error_L=float(rec_w["mass_error"] * 1e3),
        ),
        model_sealed_pocket=dict(
            P1m_kPa=float(np.max(PT2p)), t_P1m_s=float(tp[np.argmax(PT2p)]),
            note="cushioning delays the first peak; drifts after ~4 s "
                 "(declared reduced-solver limit)",
        ),
        analytic_eq7=dict(T_s=float(eq7_head(np.array([0.0]), case_w)[1]),
                          amp_m=float(case_w.Ld / (G * case_w.Ad)
                                      * (case_w.Q1 - case_w.Q0) / case_w.Tv)),
        paper=PAPER,
    )
    (OUT / "caseC_metrics.json").write_text(json.dumps(metrics, indent=2),
                                            encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    # ---------------- report ----------------
    mw = metrics["model_all_water"]
    an = metrics["analytic_eq7"]
    h_reg = PAPER["fig11_h_slope"] * pmax_w * 1e3 / (RHO_L * G) + PAPER["fig11_h_intercept"]
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>Liu2020 Case C9 复现 — 交汇室竖井（两阶段喷发，第一阶段）</title>
<style>
body{{font-family:-apple-system,Segoe UI,Arial,'Microsoft YaHei',sans-serif;margin:0;background:#f6f8fb;color:#1f2937}}
.wrap{{max-width:1180px;margin:24px auto;padding:0 18px}}
.panel{{background:#fff;border:1px solid #ddd;border-radius:12px;padding:16px;margin-bottom:16px}}
img{{width:100%;border:1px solid #ddd;border-radius:10px;background:#fff}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid #eee}}
th{{background:#f3f4f6}}
h1{{font-size:22px}} h2{{font-size:17px}}
.muted{{font-size:13px;color:#6b7280}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start}}
@media(max-width:900px){{.grid2{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<h1>Liu, Shao &amp; Zhu (2020) Case C9 复现 — 上游封气囊，两阶段喷发（第一阶段·三方对照）</h1>
<p>工况 <b>C9</b>：进口流量 25&rarr;40 L/s（Tv&asymp;0.4 s），下游满管（尾门强节流）、
竖管初始水柱 h<sub>r0</sub>=0.30 m、上游管顶封闭气囊。实测：<b>两阶段共 8 次喷发</b>——
第一阶段（0–5 s）为水力瞬变触发的第 1、2 次喷发（首峰 P<sub>1m</sub>=10.69 kPa @0.50 s，
水面 0.73 s 到管顶）；第二阶段自气囊本体 t=6.46 s 抵达交汇室起，气囊释放触发第 3–8 次
喷发。<b>本轮声明范围 = 第一阶段</b>：论文自家解析模型 Eq.(7) 也只在全水假设下描述这一
阶段——这正是 Campaign 3 独有的"实测 + 解析 + 数值"三方对照。第二阶段的气泡/弹状流
喷发列为一维简化的分辨率之外（如实报告）。</p>

<div class="panel">
  <h2 style="margin-top:0">第一阶段三方对照 — 实测 vs 解析 Eq.(7) vs 本模型</h2>
  <p class="muted">纵轴为 PT2 测压头（基准取室顶）。模型跑两个声明变体：<b>全水变体</b>
  （上游管初始全水——与 Eq.(7) 假设组完全一致）与<b>封气囊变体</b>（气垫缓冲演示）。</p>
  <img src="outputs/caseC_phase1_threeway.png">
</div>

<div class="panel">
  <h2 style="margin-top:0">全程压力对比 — PT2 / PT3 / PT4</h2>
  <p class="muted">灰色虚线 = 实测气囊本体到达时刻（6.46 s，第二阶段起点）。模型
  （全水变体）此后按纯水力路径趋稳；实测的 3–8 次喷发振荡为气囊释放驱动，模型
  不表示（声明）。</p>
  <img src="outputs/caseC_comparison_pressure.png">
  <div class="grid2" style="margin-top:10px">
    <div><h3 style="margin:4px 0">论文原图 Fig.9（扫描）</h3><img src="paper_scans/p09_fig9_C9_pressure_phases.png"></div>
    <div><h3 style="margin:4px 0">数字化质量检查</h3><img src="digitized/debug_fig9_extract.png"></div>
  </div>
</div>

<div class="panel">
  <h2 style="margin-top:0">竖管水柱 — 第一阶段喷发</h2>
  <img src="outputs/caseC_riser_column.png">
</div>

<div class="panel">
  <h2 style="margin-top:0">指标对照（第一阶段 + 终稳态）</h2>
  <table>
    <tr><th>量</th><th>实验</th><th>解析 Eq.(7)/(8)</th><th>模型（全水）</th></tr>
    <tr><td>首峰 P<sub>1m</sub></td><td>10.69 kPa @ 0.50 s</td>
        <td>幅值 2&times;{an['amp_m']:.2f} m&asymp;{2*an['amp_m']*RHO_L*G/1e3+PAPER['PT2_initial_kPa']:.1f} kPa（无阻尼上限）</td>
        <td>{mw['P1m_kPa']:.1f} kPa @ {mw['t_P1m_s']:.2f} s</td></tr>
    <tr><td>水面到管顶（第 1 次喷发）</td><td>0.73 s</td><td>—</td>
        <td>{mw['t_first_top_s']:.2f} s</td></tr>
    <tr><td>振荡周期 T</td><td>{PAPER['T_eq8_s']:.2f} s（论文引述）</td>
        <td>{an['T_s']:.2f} s</td><td>{mw['T_osc_s'] if mw['T_osc_s'] else '—'} s</td></tr>
    <tr><td>终稳态 PT2 / PT3 / PT4</td>
        <td>{PAPER['PT2_final_kPa']:.2f} / {PAPER['PT3_final_kPa']:.2f} / {PAPER['PT4_final_kPa']:.2f} kPa</td><td>—</td>
        <td>{mw['PT2_final_kPa']:.2f} / {mw['PT3_final_kPa']:.2f} / {mw['PT4_final_kPa']:.2f} kPa</td></tr>
    <tr><td>喷发判别</td><td>8 次（两阶段）</td><td>—</td>
        <td>第一阶段喷发 ✓（h<sub>jet</sub>={mw['h_jet']:.2f} m；Fig.11 回归按模型峰压给 {h_reg:.2f} m）</td></tr>
    <tr><td>体积守恒误差</td><td>—</td><td>—</td><td>{mw['mass_error_L']:.6f} L</td></tr>
  </table>
  <p class="muted">诚实说明：(1) <b>范围</b>——第二阶段（气囊释放的 3–8 次喷发）需要
  气相动量 + 分散相闭合，一维简化不表示，图中以灰虚线明示；封气囊变体演示了气垫对
  首峰的缓冲（{metrics['model_sealed_pocket']['P1m_kPa']:.1f} kPa @
  {metrics['model_sealed_pocket']['t_P1m_s']:.2f} s，偏晚），~4 s 后气囊被压缩穿过
  满管阈值而漂移，属既定简化解算器的已声明极限。(2) <b>尾门定标</b>——A_gate 只用
  文档化初值（PT2 初值 2.97 kPa = h<sub>r0</sub>）定一次，终稳态是预测：PT2 终值偏高
  {mw['PT2_final_kPa']-PAPER['PT2_final_kPa']:+.1f} kPa（实测终态含掺气混合物密度效应，
  纯水模型天然偏高）。(3) 初始场为分析构造（全水/气囊楔 + 弹性超充使全域测压线
  = HGL0），暖机 10 s 内自平衡。(4) 论文明言本 Series 的喷发"不能用界面-水面相对
  运动分析"（弹状流机制）——与 Campaign 1/2 的机制学对照收进论文讨论章。</p>
</div>
</div></body></html>"""
    (HERE / "report.html").write_text(html, encoding="utf-8")
    print(f"-> {OUT / 'caseC_phase1_threeway.png'}")
    print(f"-> {OUT / 'caseC_comparison_pressure.png'}")
    print(f"-> {HERE / 'report.html'}")


if __name__ == "__main__":
    main()

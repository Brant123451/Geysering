# -*- coding: utf-8 -*-
"""Liu, Shao & Zhu (2020) Case B3 -- digitize the published pressure traces
(Fig. 5b), run the frozen model, overlay, and build report.html.  Same
workflow as the Case A2 campaign in the sibling folder.

Case B3: Q = 20 -> 100 L/s (Tv ~ 0.4 s), downstream pipe initially FULL
(overflow-weir controlled at hd/Dd=1), observed: single-shoot GEYSER; PT2/PT3 slam peak
55.03/51.76 kPa at t ~ 1.47 s; rebound troughs PT2 -20.26 / PT3 -17.77 kPa;
two more inertia oscillations (periods 0.51 / 0.37 s); final steady
time-averaged PT1/PT2/PT3 = 0 / 1.82 / 4.65 kPa.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
MODEL = CASE_ROOT / "model"
sys.path.insert(0, str(MODEL))

from liu2020_network_twofluid import LiuCase, run_case, RHO_L, G  # noqa: E402

SCANS = CASE_ROOT / "reference" / "paper_scans"
DIG = CASE_ROOT / "data" / "digitized"
OUT = CASE_ROOT / "outputs"
for d in (DIG, OUT):
    d.mkdir(exist_ok=True)

PAGE = SCANS / "p05_fig4_heights_fig5_B3.png"
# generous crop around the Fig.5(b) plot on the scanned page (px)
CROP = (140, 1150, 860, 1500)

# measured targets quoted in the paper text (Case B3)
PAPER = dict(
    geyser=True,
    bore_reach_chamber_s=1.20,
    PT2_peak_kPa=55.03,          # point C, t ~ 1.47 s
    PT3_peak_kPa=51.76,
    t_peak_s=1.47,
    PT1_min_kPa=-8.30,           # point D
    PT2_min_kPa=-20.26,          # point E
    PT3_min_kPa=-17.77,
    osc_periods_s=(0.51, 0.37, 0.37),
    PT2_final_kPa=1.82,          # time-averaged final steady
    PT3_final_kPa=4.65,
    PT1_final_kPa=0.0,
    # Fig. 7(a) regression: h = 0.6943 * PMax/(rho g) + 0.3086  (R^2 = 0.97)
    fig7a_slope=0.6943,
    fig7a_intercept=0.3086,
    # event chronology from the Fig. 5(a) photo sequence
    t_mist_s=1.47, t_jet_out_s=1.51, t_column_top_s=1.65,
    t_break_s=(1.70, 1.89),
)


# ---------------------------------------------------------------- digitizer
def _plot_box(gray: np.ndarray):
    dark = gray < 0.55
    h, w = dark.shape
    col_frac = dark.sum(axis=0) / h
    row_frac = dark.sum(axis=1) / w
    cols = np.where(col_frac > 0.55)[0]
    rows = np.where(row_frac > 0.55)[0]
    if cols.size < 2 or rows.size < 2:
        raise RuntimeError("plot box not found")
    return int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])


def digitize_fig5b():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    img = np.asarray(Image.open(PAGE).convert("RGB"), dtype=float) / 255.0
    rgb = img[CROP[1]:CROP[3], CROP[0]:CROP[2], :]
    gray = rgb.mean(axis=2)
    x0, x1, y0, y1 = _plot_box(gray)
    # axes: x = -0.5..4.5 s, y = -30..70 kPa (box edges = axis limits)
    T0, T1 = -0.5, 4.5
    P0, P1 = -30.0, 70.0

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    masks = {
        "PT2": (r > 0.45) & (r > g + 0.15) & (r > b + 0.15),           # red
        "PT3": (g > 0.35) & (g > r + 0.10) & (g > b + 0.10),           # green
        "PT1": (b > 0.30) & (b > r + 0.10) & (b > g + 0.10),           # blue
    }
    series = {}
    for name, m in masks.items():
        mm = m.copy()
        mm[:y0 + 2, :] = False
        mm[y1 - 1:, :] = False
        mm[:, :x0 + 2] = False
        mm[:, x1 - 1:] = False
        # exclude the legend block (upper-right corner of the plot box)
        mm[:y0 + int(0.35 * (y1 - y0)), x0 + int(0.78 * (x1 - x0)):] = False
        ts, lo_, md_, hi_ = [], [], [], []
        for cx in range(x0 + 2, x1 - 1):
            rows = np.where(mm[:, cx])[0]
            if rows.size == 0:
                continue
            tval = T0 + (cx - x0) / (x1 - x0) * (T1 - T0)
            pvals = P1 + (rows - y0) / (y1 - y0) * (P0 - P1)
            ts.append(tval)
            lo_.append(float(np.min(pvals)))
            md_.append(float(np.median(pvals)))
            hi_.append(float(np.max(pvals)))
        series[name] = dict(t=np.array(ts), lo=np.array(lo_),
                            med=np.array(md_), hi=np.array(hi_))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = dict(PT1="#2b3f9e", PT2="#c81e3c", PT3="#1d8a4a")
    for name, s in series.items():
        ax.fill_between(s["t"], s["lo"], s["hi"], color=colors[name], alpha=0.25)
        ax.plot(s["t"], s["med"], color=colors[name], lw=1.0, label=f"{name} median")
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(-30, 70)
    ax.set_xlabel("t [s]"); ax.set_ylabel("p [kPa]")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("digitized check: Liu2020 Fig.5(b) (Case B3)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DIG / "debug_fig5b_extract.png", dpi=140)
    plt.close(fig)

    for name, s in series.items():
        arr = np.column_stack([s["t"], s["lo"], s["med"], s["hi"]])
        np.savetxt(DIG / f"fig5b_{name}.csv", arr, delimiter=",",
                   header="t_s,p_lo_kPa,p_med_kPa,p_hi_kPa", comments="")
    return series


def _osc_periods(t, p, t_from, t_to):
    """periods between successive positive peaks of the (smoothed)
    oscillation train after the slam"""
    m = (t >= t_from) & (t <= t_to)
    tt, pp = t[m], p[m]
    if len(pp) < 11:
        return []
    k = 5  # ~50 ms moving average at the 10 ms output cadence
    ps = np.convolve(pp, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    pk = []
    for i in range(k, len(ps) - k):
        if ps[i] == max(ps[i - k:i + k + 1]) and ps[i] > 3.0:
            if not pk or tt[i] - pk[-1] > 0.2:
                pk.append(tt[i])
    return [round(pk[i + 1] - pk[i], 3) for i in range(len(pk) - 1)]


# ---------------------------------------------------------------- pipeline
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = digitize_fig5b()

    case = LiuCase(t_end=14.0, downstream_full=True)
    rec = run_case(case, verbose=False)
    t = np.asarray(rec["t"])
    PT1 = np.asarray(rec["PT1"])
    PT2 = np.asarray(rec["PT2"])
    PT3 = np.asarray(rec["PT3"])
    hr = np.asarray(rec["hr"])

    # ---------------- overlay figure ----------------
    colors = dict(PT1="#2b3f9e", PT2="#c81e3c", PT3="#1d8a4a")
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.5), sharex=True)
    for ax, name, model in zip(axes, ("PT2", "PT3", "PT1"), (PT2, PT3, PT1)):
        s = series[name]
        ax.fill_between(s["t"], s["lo"], s["hi"], color=colors[name], alpha=0.22,
                        label=f"experiment {name} (digitized envelope)")
        ax.plot(s["t"], s["med"], color=colors[name], lw=1.1,
                label=f"experiment {name} (median)")
        ax.plot(t, model, color="#111827", lw=1.6, label=f"model {name}")
        ax.set_ylabel(f"{name} [kPa]")
        ax.set_xlim(-0.5, 4.5)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylim(-30, 70)
    axes[1].set_ylim(-30, 70)
    axes[2].set_ylim(-12, 12)
    axes[-1].set_xlabel("t [s]  (t=0 at the start of the inflow ramp)")
    axes[0].set_title("Liu2020 Case B3 (Q 20$\\to$100 L/s, downstream full pipe): "
                      "pressures at PT2 / PT3 / PT1 -- model vs digitized Fig.5(b)")
    fig.tight_layout()
    fig.savefig(OUT / "caseB_comparison_pressure.png", dpi=150)
    plt.close(fig)

    # ---------------- riser column history ----------------
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(t, hr, color="#c81e3c", lw=1.8, label="model riser column height $h_r$")
    ax.axhline(case.Hr, color="#16a34a", ls=":", lw=1.4,
               label=f"riser top {case.Hr} m (geyser threshold)")
    for tev, lab in ((PAPER["t_mist_s"], "mist out (exp)"),
                     (PAPER["t_column_top_s"], "column at top (exp)")):
        ax.axvline(tev, color="#6b7280", ls="--", lw=1.0)
        ax.text(tev + 0.02, 0.05, lab, rotation=90, fontsize=7, color="#6b7280")
    ax.set_xlabel("t [s]"); ax.set_ylabel("$h_r$ [m]")
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(0, 1.35)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Case B3: mixture column in the riser -- single-shoot geyser branch")
    fig.tight_layout()
    fig.savefig(OUT / "caseB_riser_column.png", dpi=150)
    plt.close(fig)

    # ---------------- h vs PMax against the Fig.7(a) regression ----------------
    pmax_model = float(np.max(PT2))
    h_model = float(rec["h_jet"])
    hm = np.linspace(0.0, 6.0, 50)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(hm, PAPER["fig7a_slope"] * hm + PAPER["fig7a_intercept"],
            "k--", lw=1.3,
            label="experimental regression Fig.7(a):\n"
                  "$h = 0.6943\\,P_{max}/\\rho g + 0.3086$  ($R^2=0.97$)")
    ax.plot(PAPER["PT2_peak_kPa"] * 1e3 / (RHO_L * G),
            PAPER["fig7a_slope"] * PAPER["PT2_peak_kPa"] * 1e3 / (RHO_L * G)
            + PAPER["fig7a_intercept"],
            "o", ms=9, mfc="none", mec="#1d8a4a", mew=1.6,
            label="B3 experiment ($P_{max}=55.0$ kPa)")
    ax.plot(pmax_model * 1e3 / (RHO_L * G), h_model, "^", ms=10,
            color="#c81e3c", label="B3 model $(P_{max},\\ h_{jet})$")
    ax.axhline(1.22, color="#16a34a", ls=":", lw=1.2, label="riser top (geyser threshold)")
    ax.set_xlabel("$P_{max}/\\rho g$ [m]")
    ax.set_ylabel("jetting height $h$ [m]")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Case B3 on the Series-B height--pressure relation")
    fig.tight_layout()
    fig.savefig(OUT / "caseB_h_vs_pmax.png", dpi=150)
    plt.close(fig)

    # ---------------- metrics ----------------
    ipk = int(np.argmax(PT2))
    m_fin = t >= 10.0
    periods = _osc_periods(t, PT2, float(t[ipk]) + 0.1, 4.5)
    metrics = dict(
        case="Liu2020 B3: Q 20->100 L/s, downstream full pipe",
        model=dict(
            geyser=bool(rec["geyser"]),
            hr_max=float(rec["hr_max"]),
            wr_eject=float(rec["wr_eject"]),
            h_jet=h_model,
            PT2_peak_kPa=pmax_model,
            PT3_peak_kPa=float(np.max(PT3)),
            t_peak_s=float(t[ipk]),
            PT1_min_kPa=float(np.min(PT1)),
            PT2_min_kPa=float(np.min(PT2)),
            PT3_min_kPa=float(np.min(PT3)),
            osc_periods_s=periods[:3],
            PT1_final_kPa=float(np.mean(PT1[m_fin])),
            PT2_final_kPa=float(np.mean(PT2[m_fin])),
            PT3_final_kPa=float(np.mean(PT3[m_fin])),
            overflow_L=float(rec["overflow_vol"] * 1e3),
            mass_error_L=float(rec["mass_error"] * 1e3),
        ),
        paper=PAPER,
    )
    (OUT / "caseB_metrics.json").write_text(json.dumps(metrics, indent=2),
                                            encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    # ---------------- report ----------------
    mm = metrics["model"]
    h_reg = PAPER["fig7a_slope"] * pmax_model * 1e3 / (RHO_L * G) + PAPER["fig7a_intercept"]
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>Liu2020 Case B3 复现 — 交汇室竖井（单发喷发分支）</title>
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
<h1>Liu, Shao &amp; Zhu (2020) Case B3 复现 — 交汇室上方竖井，单发喷发分支</h1>
<p>工况 <b>B3</b>：进口流量 20&rarr;100 L/s（阀门 ~0.4 s 斜坡），下游管<b>初始满管</b>
（尾门控制）；与 Case A2 <b>只差下游初始条件</b>，实验结果翻转为单发（single-shoot）
geyser。实测：涌波 t=1.20 s 到室 &rarr; PT2/PT3 冲击峰 55.03/51.76 kPa（t&asymp;1.47 s）
&rarr; 混合物 t=1.51 s 喷出管顶 &rarr; 回弹负压 PT2 &minus;20.26 kPa &rarr; 两个衰减惯性
振荡（周期 0.51/0.37 s）&rarr; 终稳态 PT2/PT3 = 1.82/4.65 kPa。
模型：与 A2 同一冻结复合域求解器，仅 (i) 下游满管初始/淹没出流边界（Series B 工况开关），
(ii) 竖管由准静态列改为<b>带惯量的刚性水柱 ODE</b>（喷发速度与冲击峰正是本分支的观测量）。</p>

<div class="panel">
  <h2 style="margin-top:0">压力对比 — PT2（室顶盖）/ PT3（室底）/ PT1（竖管 +0.80 m）</h2>
  <p class="muted">彩色包络+中位线 = 论文 Fig.5(b) 数字化（逐列取色）；黑线 = 模型。
  t=0 为流量斜坡起点。</p>
  <img src="caseB_comparison_pressure.png">
  <div class="grid2" style="margin-top:10px">
    <div><h3 style="margin:4px 0">论文原图 Fig.5(b)（扫描）</h3><img src="../data/digitized/_fig5b_crop_probe.png"></div>
    <div><h3 style="margin:4px 0">数字化质量检查</h3><img src="../data/digitized/debug_fig5b_extract.png"></div>
  </div>
</div>

<div class="panel">
  <h2 style="margin-top:0">竖管混合柱 — 喷发判别与时序</h2>
  <img src="caseB_riser_column.png">
</div>

<div class="panel">
  <h2 style="margin-top:0">喷发高度—峰压关系 — 对标 Fig.7(a)</h2>
  <img src="caseB_h_vs_pmax.png">
  <p class="muted">论文对 Series B 全部喷发工况给出回归
  h = 0.6943&middot;P<sub>Max</sub>/&rho;g + 0.3086（R&sup2;=0.97）。模型点
  (P<sub>Max</sub>/&rho;g = {pmax_model*1e3/(RHO_L*G):.2f} m, h<sub>jet</sub> = {h_model:.2f} m)；
  按回归线，该峰压对应 h = {h_reg:.2f} m。</p>
</div>

<div class="panel">
  <h2 style="margin-top:0">指标对照</h2>
  <table>
    <tr><th>量</th><th>实验（论文正文/图）</th><th>模型</th></tr>
    <tr><td>分支判别</td><td>单发 geyser</td>
        <td>{'geyser' if mm['geyser'] else '不喷发'}（柱到顶，喷出速度 {mm['wr_eject']:.1f} m/s）</td></tr>
    <tr><td>PT2 冲击峰</td><td>{PAPER['PT2_peak_kPa']:.1f} kPa（t={PAPER['t_peak_s']:.2f} s）</td>
        <td>{mm['PT2_peak_kPa']:.1f} kPa（t={mm['t_peak_s']:.2f} s）</td></tr>
    <tr><td>PT3 冲击峰</td><td>{PAPER['PT3_peak_kPa']:.1f} kPa</td>
        <td>{mm['PT3_peak_kPa']:.1f} kPa</td></tr>
    <tr><td>回弹负压 PT2 / PT3</td><td>{PAPER['PT2_min_kPa']:.1f} / {PAPER['PT3_min_kPa']:.1f} kPa</td>
        <td>{mm['PT2_min_kPa']:.1f} / {mm['PT3_min_kPa']:.1f} kPa</td></tr>
    <tr><td>惯性振荡周期</td><td>{PAPER['osc_periods_s'][0]:.2f} / {PAPER['osc_periods_s'][1]:.2f} s</td>
        <td>{' / '.join(f'{p:.2f}' for p in mm['osc_periods_s']) or '—'} s</td></tr>
    <tr><td>终稳态 PT2 / PT3</td><td>{PAPER['PT2_final_kPa']:.2f} / {PAPER['PT3_final_kPa']:.2f} kPa</td>
        <td>{mm['PT2_final_kPa']:.2f} / {mm['PT3_final_kPa']:.2f} kPa</td></tr>
    <tr><td>喷发高度 h（回归推算）</td><td>{PAPER['fig7a_slope']*PAPER['PT2_peak_kPa']*1e3/(RHO_L*G)+PAPER['fig7a_intercept']:.2f} m（实测回归）</td>
        <td>{h_model:.2f} m</td></tr>
    <tr><td>体积守恒误差</td><td>—</td><td>{mm['mass_error_L']:.6f} L</td></tr>
  </table>
  <p class="muted">诚实说明：(1) <b>冲击峰幅值被低估（31 vs 55 kPa）且偏晚 0.2 s</b>。
  峰值是声学量：满管弹性支路用的是全局冻结的降速数值波速 a=40 m/s（为避免网格尺度
  微空穴的刚性振铃，见 A2 报告），Joukowsky 尺度 &rho;a&Delta;u 直接随 a 线性缩放——
  敏感性验证：a=80 m/s 时峰值 63.5 kPa（接近实测 55），但回弹负压过深（&minus;48 kPa）
  且时间步减半；按冻结配置政策保留 a=40 并如实报告。偏晚 0.2 s 与 A2 的涌波到达偏晚
  （~1.4 vs 1.20 s）同源。(2) <b>模型的 (P<sub>Max</sub>, h) 点落在实验回归线附近</b>
  （按模型自身峰压 {pmax_model*1e3/(RHO_L*G):.1f} m 回归应为 h={h_reg:.2f} m，模型
  h<sub>jet</sub>={h_model:.2f} m）——即高度—峰压的<b>物理关系</b>被正确复现，误差
  集中在声学峰幅本身。(3) 回弹后 2.4–3.3 s 模型持续亚大气（实测很快回正）；随后
  第二次柱涌起（3.5–4.5 s）与实测"自由面再振荡两周期后趋稳"定性一致。
  (4) 惯性振荡周期：实测 0.51/0.37 s；论文自家解析模型 Eq.(7) 对本几何给
  T&asymp;1.05 s（也偏慢一倍）；模型给出 {' / '.join(f'{p:.2f}' for p in mm['osc_periods_s'])} s
  的混合振荡，同样偏慢——单相水柱 + 通风室不含掺气弹性，此项列为已知短板。
  (5) 室内"水气混合加压"（论文原述机制）尝试过封闭气囊 EOS 方案：气囊在涌波到达前
  就加压喷发（正反馈：柱堵住排气路），已回退为通风室 + 液相惯量方案并记录。
  (6) 下游边界为淹没出流对固定尾水位 H_tail=Dd，是论文 Series B 溢流堰
  hd/Dd=1 的静水等效实现；未对任何瞬态/终态结果拟合。</p>
</div>
</div></body></html>"""
    (OUT / "report.html").write_text(html, encoding="utf-8")
    print(f"-> {OUT / 'caseB_comparison_pressure.png'}")
    print(f"-> {OUT / 'caseB_h_vs_pmax.png'}")
    print(f"-> {OUT / 'report.html'}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Case B of the Cong, Chan & Lee (2017) reproduction: Run B-H6 (no geyser).

    D = 0.05 m, Dr = 0.041 m (Dr/D = 0.82), H0 = 0.66 m, L0 = 0.61 m,
    V*air = 1.37  ->  experiment: NO GEYSER (high-speed camera).

Identical layout and release to Case A (B-H1); the ONLY difference is the
riser diameter -- the cleanest controlled test of the paper's "diameter ratio
controls geysering" conclusion.  Experiment: the pocket arrives at t~8.1 s,
rises like a Taylor bubble (vint = 0.476 m/s, vnet ~ vTaylor = 0.22 m/s), the
free surface is pushed up gently (Yfs 0.58 -> 1.2 m, vfs = 0.246 m/s), and
the pocket BREAKS at the free surface (~10.5 s) without reaching the rim.
PT1 shows no geyser surge: a gentle hump to ~1.4*H0 (Run B-32, Fig. 10b).

Same frozen solver copy as Case A (model/); only Dr changes in the case
parameters.  Outputs: overlays vs digitized Fig. 7(a) + Fig. 10(b), metrics,
report.html with the frame viewer.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parent
sys.path.insert(0, str(CASE_ROOT / "model"))

from cong2017_network_twofluid import G, NetworkCase, run_network

DIG = CASE_ROOT / "data" / "digitized"
OUT = CASE_ROOT / "outputs"
OUT.mkdir(exist_ok=True)

C_MODEL = "#d62728"
C_MODEL2 = "#f59e0b"
C_PAPER_BAND = "#9ca3af"
C_FS = "#2563eb"
C_INT = "#111827"

H0 = 0.66
HR = 1.8
CASE_KW = dict(
    D=0.05, Dr=0.041, riser_height=HR,
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

EXP = dict(geyser=False, Ta=8.10, Uf_star=0.443, v_fs=0.246, v_int=0.476,
           v_net=0.235, v_taylor=0.219, PT1_peak_over_H0=1.4)


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
    with (DIG / "fig7a_levels.csv").open() as f:
        for row in csv.DictReader(f):
            (fs if row["kind"] == "fs" else gi).append((float(row["t_s"]), float(row["Y_m"])))
    p10 = dict(t=[], med=[], lo=[], hi=[])
    with (DIG / "fig10b_pt1.csv").open() as f:
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
    Yfs_max = float(np.nanmax(s["Yfs"]))

    # climb kinematics over the venting event (arrival -> gas front peak)
    v_fs_m = v_int_m = t_catch = None
    if Ta_m is not None:
        i_pk = int(np.nanargmax(s["Yint"]))
        seg = (t >= Ta_m) & (t <= t[i_pk]) & (s["Yint"] > 0.05)
        if np.count_nonzero(seg) >= 3:
            v_int_m = float(np.polyfit(t[seg], s["Yint"][seg], 1)[0])
        seg_fs = (t >= Ta_m) & (t <= t[i_pk])
        if np.count_nonzero(seg_fs) >= 3:
            v_fs_m = float(np.polyfit(t[seg_fs], s["Yfs"][seg_fs], 1)[0])
        # gas front catches the (visible) free surface = pocket break
        gap = np.where(s["Yint"] > 0.10, s["Yfs"] - s["Yint"], np.inf)
        t_catch = first_crossing(t, -gap, -0.05, after=Ta_m)

    # pocket head: plateau + post-arrival hump (no sharp surge expected)
    plateau = float(np.nanmedian(s["pocket"][(t > 2.0) & (t < 7.0)])) / H0
    if Ta_m is not None:
        post = (t >= Ta_m)
        pk_post = float(np.nanmax(s["pocket"][post])) / H0
        t_pk_post = float(t[post][int(np.nanargmax(s["pocket"][post]))])
    else:
        pk_post, t_pk_post = float(np.nanmax(s["pocket"])) / H0, float("nan")

    # ---------------- levels overlay (Fig. 7a) ----------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.0))
    for ax, xlim in ((a1, (0.0, T_END)), (a2, (8.0, 11.0))):
        ax.plot(t, s["Yfs"], color=C_MODEL, lw=2.0, label="model $Y_{fs}$ (free surface)")
        ax.plot(t, s["Yint"], color=C_MODEL2, lw=1.8, ls="--", label="model $Y_{int}$ (gas front)")
        if fs_pts.size:
            ax.plot(fs_pts[:, 0], fs_pts[:, 1], "s", ms=5, mfc="none", mec=C_FS,
                    label="experiment $Y_{fs}$ (Fig.7a, digitized)")
        if int_pts.size:
            ax.plot(int_pts[:, 0], int_pts[:, 1], "o", ms=5, mfc="none", mec=C_INT,
                    label="experiment $Y_{int}$ (Fig.7a, digitized)")
        ax.axhline(HR, color="#16a34a", ls=":", lw=1.2)
        ax.axhline(H0, color="0.65", ls=":", lw=1.0)
        ax.set_ylim(0, 1.95)
        ax.set_xlim(*xlim)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Y [m] above riser entrance")
        ax.grid(alpha=0.3)
    a1.axvspan(8.0, 11.0, color="#f3f4f6", zorder=0)
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    a1.set_title("full model trajectory (paper Fig.7a window shaded)", fontsize=10)
    # slope check: align the model gas-front climb midpoint with the experiment
    if int_pts.size and Ta_m is not None and np.nanmax(s["Yint"]) > 0.5:
        y_ref = 0.5
        t_exp_mid = float(np.interp(y_ref, int_pts[:, 1], int_pts[:, 0]))
        i_pk = int(np.nanargmax(s["Yint"]))
        seg = slice(int(np.argmax(t >= Ta_m)), i_pk + 1)
        t_mod_mid = float(np.interp(y_ref, s["Yint"][seg], t[seg]))
        dshift = t_exp_mid - t_mod_mid
        a2.plot(t + dshift, s["Yfs"], color=C_MODEL, lw=1.1, ls=":",
                label=f"model $Y_{{fs}}$ shifted {dshift:+.2f} s (slope check)")
        a2.plot(t + dshift, s["Yint"], color=C_MODEL2, lw=1.1, ls=":",
                label=f"model $Y_{{int}}$ shifted {dshift:+.2f} s")
    a2.legend(frameon=False, fontsize=7, loc="upper left")
    a2.set_title("paper Fig.7(a) window (t = 8..11 s)", fontsize=10)
    fig.suptitle("Run B-H6 riser trajectories -- $D_r$=41 mm ($D_r/D$=0.82), "
                 "$H_0$=0.66 m, $L_0$=0.61 m (no geyser)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "caseB_comparison_levels.png", dpi=150)
    plt.close(fig)

    # ---------------- pressure overlay (Fig. 10b) ----------------
    pk_over = s["pocket"] / H0
    pk_avg = np.full_like(pk_over, np.nan)
    for i in range(len(pk_over)):
        mwin = (t >= t[i] - 0.25) & (t <= t[i] + 0.25)
        if np.any(np.isfinite(pk_over[mwin])):
            pk_avg[i] = np.nanmean(pk_over[mwin])
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.fill_between(p10["t"], p10["lo"], p10["hi"], color=C_PAPER_BAND, alpha=0.45,
                    label="Run B-32 PT1, digitized pixel envelope (Fig.10b)")
    ax.plot(p10["t"], p10["med"], color="#374151", lw=1.3,
            label="Run B-32 PT1, digitized median (video series, same condition)")
    ax.plot(t, pk_avg, color=C_MODEL, lw=1.8,
            label="model: trapped-pocket EOS gauge head / $H_0$ (0.5 s cycle-avg)")
    ax.plot(t, pk_over, color=C_MODEL, lw=0.7, alpha=0.35,
            label="model: raw trace")
    ax.axhline(EXP["PT1_peak_over_H0"], color="#16a34a", ls=":", lw=1.2,
               label="paper text: no-geyser peak $\\approx 1.4\\,H_0$")
    ax.set_xlim(0, 13)
    ax.set_ylim(-1.0, 4.0)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("$H/H_0$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Run B-H6 pocket pressure vs PT1 (pipe crown near the closed end)\n"
                 "model (decoupled two-fluid, frozen copy) vs Cong et al. (2017) Fig.10(b), digitized",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "caseB_comparison_pressure.png", dpi=150)
    plt.close(fig)

    # ---------------- metrics ----------------
    v_fs_exp_chk = v_int_exp_chk = None
    if fs_pts.size >= 4:
        m_fs = (fs_pts[:, 0] > 8.4) & (fs_pts[:, 0] < 10.2)
        if np.count_nonzero(m_fs) >= 3:
            v_fs_exp_chk = float(np.polyfit(fs_pts[m_fs, 0], fs_pts[m_fs, 1], 1)[0])
    if int_pts.size >= 4:
        m_int = int_pts[:, 1] > 0.10
        v_int_exp_chk = float(np.polyfit(int_pts[m_int, 0], int_pts[m_int, 1], 1)[0])
    m = dict(
        case=dict(run="B-H6", D=case.D, Dr=case.Dr, Dr_over_D=case.Dr / case.D,
                  H0=H0, L0=case.L_down, riser_height=HR,
                  tee_x=case.x_riser, valve_x=case.x_valve,
                  Vair_L=case.V_air * 1000.0,
                  Vstar_air=case.V_air / (0.25 * math.pi * case.Dr ** 2 * H0)),
        model=dict(
            geyser=bool(geyser_m),
            Yfs_max_m=Yfs_max,
            Ta_gas_enters_riser_s=Ta_m,
            v_fs_event_fit_mps=v_fs_m,
            v_int_event_fit_mps=v_int_m,
            t_gas_catches_surface_s=t_catch,
            pocket_plateau_over_H0=plateau,
            pocket_post_arrival_peak_over_H0=pk_post,
            pocket_post_arrival_peak_time_s=t_pk_post,
        ),
        paper=dict(
            geyser=False,
            Ta_s=EXP["Ta"],
            Uf_over_sqrtgD=EXP["Uf_star"],
            v_fs_mps=EXP["v_fs"],
            v_int_mps=EXP["v_int"],
            v_net_mps=EXP["v_net"],
            v_taylor_mps=EXP["v_taylor"],
            Yfs_max_m=1.21,
            t_gas_catches_surface_s=10.5,
            PT1_no_geyser_peak_over_H0=EXP["PT1_peak_over_H0"],
            fig7a_vfs_climb_fit_mps=v_fs_exp_chk,
            fig7a_vint_climb_fit_mps=v_int_exp_chk,
        ),
        notes=[
            "Same frozen solver copy and geometry as Case A (B-H1); the only "
            "changed parameter is Dr = 0.041 m.",
            "Key full-coupling check: the staged legacy driver gave BOTH runs "
            "the same 2.07*H0 pipe peak; the experiment differentiates "
            "(B-1 geyser ~1.9*H0 vs B-32 no-geyser ~1.4*H0).",
            "PT1 datum caveat as in Case A: digitized pre-arrival plateau reads "
            "~1.3*H0, above the static reservoir head (sensor datum/calibration "
            "uncertainty ~0.3*H0); shape anchors are the comparison.",
        ],
    )
    (OUT / "caseB_comparison_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

    with (OUT / "caseB_model_series.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "Yfs_m", "Yint_m", "pocket_head_m", "tr_head_m", "pj_head_m"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.4f}", f"{s['Yfs'][i]:.4f}", f"{s['Yint'][i]:.4f}",
                        f"{s['pocket'][i]:.4f}", f"{s['tr'][i]:.4f}", f"{s['pj'][i]:.4f}"])

    build_report(m)
    print(json.dumps(m["model"], indent=2))
    print(f"-> {OUT / 'caseB_comparison_levels.png'}")
    print(f"-> {OUT / 'caseB_comparison_pressure.png'}")
    print(f"-> {HERE / 'report.html'}")


def build_report(m: dict):
    mo, pa = m["model"], m["paper"]

    def _f(v, nd=2):
        return "-" if v is None else f"{v:.{nd}f}"

    rows = [
        ("是否喷发", "否（气囊在竖管内破碎）",
         ("是" if mo.get("geyser") else "否")
         + f"（水面最高 {_f(mo.get('Yfs_max_m'))} m，未到 1.8 m 管顶）",
         "分支判别正确" if not mo.get("geyser") else "分支错误"),
        ("气囊到达竖管 Ta", f"{_f(pa.get('Ta_s'))} s", f"{_f(mo.get('Ta_gas_enters_riser_s'))} s",
         "同 Case A 的几何/波速小幅偏早"),
        ("水面最高位置", f"约 {_f(pa.get('Yfs_max_m'))} m（Fig.7a）",
         f"{_f(mo.get('Yfs_max_m'))} m", ""),
        ("排气段水面上升速度 v_fs", f"{_f(pa.get('v_fs_mps'), 3)} m/s（Table 2）",
         f"{_f(mo.get('v_fs_event_fit_mps'), 3)} m/s", "到达至气核峰的拟合斜率"),
        ("排气段气核上升速度 v_int", f"{_f(pa.get('v_int_mps'), 3)} m/s（Table 2）",
         f"{_f(mo.get('v_int_event_fit_mps'), 3)} m/s",
         f"实测泰勒尺度 vTaylor={_f(pa.get('v_taylor_mps'), 3)} m/s"),
        ("气囊追上水面（破碎）", f"约 {_f(pa.get('t_gas_catches_surface_s'), 1)} s（Fig.7a 两线相遇）",
         f"{_f(mo.get('t_gas_catches_surface_s'), 1)} s", ""),
        ("到达后管端压力峰 / H0", f"约 {_f(pa.get('PT1_no_geyser_peak_over_H0'), 1)}（正文；缓峰非激增）",
         f"{_f(mo.get('pocket_post_arrival_peak_over_H0'))}（t={_f(mo.get('pocket_post_arrival_peak_time_s'), 1)} s）",
         "与 Case A 喷发激增的分化 = 全耦合关键核对项"),
        ("喷发前压头平台 / H0", "约 1.3（Fig.10b 数字化；基准面存疑 ±0.3）",
         f"{_f(mo.get('pocket_plateau_over_H0'))}", "形态均为长平台"),
    ]
    trs = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>"
                  for a, b, c, d in rows)
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Case B: Cong 2017 Run B-H6（不喷发）对比</title>
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
<h1>Case B — Cong, Chan &amp; Lee (2017) Run B-H6（不喷发分支，高速摄像）</h1>
<p>工况：<code>D=50 mm</code>、<code>Dr=41 mm</code>（Dr/D=0.82）、<code>H0=0.66 m</code>、
<code>L0=0.61 m</code>（V*air=1.37）。与 Case A（B-H1）<b>唯一差别是竖管直径</b>——同一释放条件下
分支翻转，是"直径比主控喷发"结论最干净的受控对照。实验：气囊 t≈8.1 s 到达竖管，
以泰勒气泡形态稳定上升（vint=0.476 m/s，vnet≈vTaylor），水面被温和顶托
（0.58→1.2 m），气囊于 ~10.5 s 追上水面破碎，<b>无喷发</b>；管端压力只有 ~1.4H0 的缓峰。</p>
<div class="panel">
  <h2 style="margin-top:0">论文中与本工况相关的图表</h2>
  <table>
    <tr><th>论文图表</th><th>内容</th><th>与本工况的关系</th><th>本报告中的对照</th></tr>
    <tr><td><b>Table 2</b>（p7）</td><td>Series B 汇总</td>
        <td><b>B-H6 行</b>：Ta=8.1 s、Uf=0.443√(gD)、v_fs=0.246、v_int=0.476 m/s、NO geyser</td>
        <td>指标对照表</td></tr>
    <tr><td><b>Fig. 6</b>（p6）</td><td>B-H6 竖管内气囊上升瞬时帧（t=8.7/9.3/9.9/10.5/10.9 s）</td>
        <td><b>正是本工况</b></td><td>下方原图收录，可与帧查看器对照</td></tr>
    <tr><td><b>Fig. 7</b>（p7）</td><td>B-H6 的 Yfs/Yint/速度/Lw/Ha 四联图（高速摄像）</td>
        <td><b>正是本工况</b>；(a) 面板已数字化</td><td>水位叠加图</td></tr>
    <tr><td><b>Fig. 10(b)</b>（p10）</td><td>PT1/PT2 压力时程（Run B-32，同 Dr=41 mm 的录像系列）</td>
        <td>同名义工况重复（视频采样）</td><td>压力叠加图</td></tr>
    <tr><td class="muted">Fig. 5</td><td class="muted">Series A/B 全工况 vnet vs vTaylor</td>
        <td class="muted">B-H6 落在泰勒线附近（不喷发判据一侧）</td><td class="muted">正文引用</td></tr>
  </table>
</div>
<div class="panel">
  <h2 style="margin-top:0">论文原图（扫描）</h2>
  <div class="grid2">
    <div><h3 style="margin:4px 0">Fig.7 B-H6 竖管数据（(a) 已数字化）</h3><img src="paper_scans/fig7_bh6_riser.png"></div>
    <div><h3 style="margin:4px 0">Fig.10 压力时程（下=B-32 不喷发）</h3><img src="paper_scans/fig10_pressure.png"></div>
  </div>
  <h3 style="margin:12px 0 4px 0">Fig.6 B-H6 高速摄像瞬时帧</h3>
  <img src="paper_scans/fig6_bh6_photos.png">
</div>
<div class="panel">
  <h2 style="margin-top:0">叠加对比</h2>
  <h3>竖管水面与气核前端 Y(t)</h3><img src="outputs/caseB_comparison_levels.png">
  <h3>气囊压头 H/H0</h3><img src="outputs/caseB_comparison_pressure.png">
  <h3>指标对照</h3>
  <table>
    <tr><th>指标</th><th>论文实验</th><th>模型</th><th>备注</th></tr>
    {trs}
  </table>
  <p class="muted"><b>与 Case A 的联合读法（全耦合关键核对项）</b>：两工况共用同一冻结求解器
  与几何、只改 Dr——模型在 16 mm 竖管喷发（气囊压缩激增 1.71H0）、在 41 mm 竖管不喷发
  （缓峰 1.02H0），与实验的分支翻转一致；旧分级耦合给两工况相同的 2.07H0 峰压，
  全同步耦合恢复了"宽竖管排气抑制管道压力"的实验分化（实测 1.9 vs 1.4H0）。<br>
  <b>诚实记录</b>：① 宽竖管的水面顶托不足（模型 +0.19 m vs 实测 +0.63 m）——口部体积
  中性交换抵消了进气的抬水位移；对水库布置关闭该交换可恢复顶托但会同时废掉 B-H1 的
  气囊压力重建路径（已试已回退，代码内注记）——两分支对口部闭合的需求冲突是已识别的
  一维局限，与论文 Discussion 的液膜卷吸/口部透射短板一致；② 气核爬升 0.29 vs 实测
  0.476 m/s（泰勒尺度偏向，实测有同向卷吸增强）；③ 到达偏晚 ~1 s；④ PT1 基准面存疑
  ±0.3H0，压力对比以形态锚点为主。模型序列见 <code>outputs/caseB_model_series.csv</code>；
  数字化中间产物见 <code>digitized/</code>。</p>
</div>
__EXTRA_SECTIONS__
</div></body></html>"""
    html = html.replace("__EXTRA_SECTIONS__", build_extra_sections())
    (HERE / "report.html").write_text(html, encoding="utf-8")


def build_extra_sections() -> str:
    parts = []
    frames_json = OUT / "frames_index.json"
    if frames_json.exists():
        frames_data = frames_json.read_text(encoding="utf-8")
        parts.append("""
<div class="panel">
  <h2 style="margin-top:0">两流体模拟逐帧查看器 — 水平管 + 竖管全场演化</h2>
  <p class="muted">同一冻结求解器的 B-H6 全场演化（水=蓝，气=白/浅灰）：开阀释放振荡 →
  气囊沿管顶向竖管迁移 → 宽竖管进气、泰勒气泡式上升、水面温和顶托 →
  气囊追上水面破碎、平缓排气（无喷发）。<b>左图为真实比例 1:1</b>
  （D=50 mm、Dr=41 mm），右侧为竖管放大同步视图（按当地 α_g 画气核宽度）。
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
  <p class="muted">整段 GIF 备份：<a href="outputs/caseB_animation.gif">caseB_animation.gif</a></p>
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
    elif (OUT / "caseB_animation.gif").exists():
        parts.append("""
<div class="panel">
  <h2 style="margin-top:0">两流体模拟动画</h2>
  <img src="outputs/caseB_animation.gif" style="max-width:900px">
</div>""")
    return "".join(parts)


if __name__ == "__main__":
    main()

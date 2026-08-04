# -*- coding: utf-8 -*-
"""Case B of the V&W (2011) reproduction: SMALL tower, geysering branch.

    D_t = 12.7 mm  (D_t* = 0.135),  H_a0 = 0.610 m,  Y_fs0 = 0.356 m.

Paper experimental data for this condition (CENTER panels of the 3x3 grids):
  * Fig. 6  -- normalized pressure head H*(T*),  axes T* = 0..5,  H* = 0..1.5
  * Fig. 8  -- normalized levels Y*fs / Y*int,   axes T* = 3..5,  Y* = 0..1

This script reads (or, when scans are available, regenerates) both digitized
panels, runs the frozen per-case network model, and writes comparisons below
the case-root ``outputs`` directory.  The model code in ``model`` is a frozen
snapshot: later edits elsewhere in the repository do not change this case.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
MODEL = CASE_ROOT / "model"
DIGITIZED = CASE_ROOT / "data" / "digitized"
SCANS = CASE_ROOT / "reference" / "paper_scans"
OUTPUTS = CASE_ROOT / "outputs"
sys.path.insert(0, str(MODEL))

from vw2011_network_twofluid import G, NetworkCase, run_network
from PIL import Image

FIG6 = SCANS / "raw_p5_x101_2000x1457.png"   # pressure heads, Dt*=0.135
FIG8 = SCANS / "raw_p7_x121_2145x1534.png"   # levels, Dt*=0.135
PANEL = (1, 1)                              # center: Ha=0.610 m, WL=0.356 m

DIG = DIGITIZED
OUT = OUTPUTS
DIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

C_MODEL = "#d62728"
C_MODEL2 = "#f59e0b"
C_PAPER = "#374151"
C_PAPER_BAND = "#9ca3af"
C_FS = "#1f77b4"
C_INT = "#111827"

CASE = dict(Dr=0.0127, air_head=0.610, init_water_level=0.356, L=0.610)


def check_paths() -> None:
    """Validate migrated paths without running either numerical model."""
    required = [
        MODEL / "digitize_paper_curves.py",
        MODEL / "vw2011_network_twofluid.py",
        DIGITIZED / "fig6_caseB_Hstar_band.csv",
        DIGITIZED / "fig8_caseB_levels.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    result = {
        "case_root": str(CASE_ROOT),
        "model": str(MODEL),
        "digitized": str(DIGITIZED),
        "scans": str(SCANS),
        "outputs": str(OUTPUTS),
        "required_files_ok": not missing,
        "missing_required": missing,
        "raw_scans_available": FIG6.is_file() and FIG8.is_file(),
    }
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(2)


def load_digitized():
    """Load the committed data when raw raster scans are not materialised."""
    pressure = np.genfromtxt(
        DIG / "fig6_caseB_Hstar_band.csv", delimiter=",", names=True
    )
    fig6d = {
        "T": np.atleast_1d(pressure["Tstar"]),
        "med": np.atleast_1d(pressure["Hstar_med"]),
        "lo": np.atleast_1d(pressure["Hstar_min"]),
        "hi": np.atleast_1d(pressure["Hstar_max"]),
    }
    levels = {"fs": [], "int": []}
    with (DIG / "fig8_caseB_levels.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            levels[row["kind"]].append((float(row["Tstar"]), float(row["Ystar"])))
    fig8d = {
        kind: np.asarray(sorted(points), dtype=float).reshape((-1, 2))
        for kind, points in levels.items()
    }
    return fig6d, fig8d


def crop_panel(gray, box, dst: Path, margin: int = 70):
    x0, x1, y0, y1 = box
    H, W = gray.shape
    r0 = max(0, y0 - margin); r1 = min(H, y1 + margin)
    c0 = max(0, x0 - margin); c1 = min(W, x1 + margin)
    Image.fromarray(gray[r0:r1, c0:c1]).save(dst)


def digitize():
    from digitize_paper_curves import (
        digitize_fig6,
        digitize_fig8,
        draw_panel_debug,
        find_panels,
        label_boxes_in_panel,
        load_gray,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---------------- Fig. 6 : pressure head (T* 0..5) ----------------
    g6 = load_gray(FIG6)
    panels6, _ = find_panels(g6)
    draw_panel_debug(g6, panels6, DIG / "debug_fig6_panels.png", center=PANEL)
    box6 = panels6[PANEL[0]][PANEL[1]]
    crop_panel(g6, box6, DIG / "fig6_caseB_panel.png")
    masks6 = label_boxes_in_panel(g6, box6)
    T6, med6, lo6, hi6, bin6 = digitize_fig6(g6, box6, xlim=(0.0, 5.0), ylim=(0.0, 1.5),
                                             label_masks=masks6)
    with (DIG / "fig6_caseB_Hstar_band.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Tstar", "Hstar_med", "Hstar_min", "Hstar_max"])
        for row in zip(T6, med6, lo6, hi6):
            w.writerow([f"{v:.5f}" for v in row])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.imshow(bin6, cmap="gray_r")
    a1.set_title("Fig.6 case-B panel: extracted dark pixels (labels masked)")
    a2.fill_between(T6, lo6, hi6, color=C_PAPER_BAND, alpha=0.45, label="digitized min-max band")
    a2.plot(T6, med6, color=C_INT, lw=1.2, label="digitized median")
    a2.set_xlim(0, 5); a2.set_ylim(0, 1.5); a2.grid(alpha=0.3)
    a2.set_xlabel("T*_ref"); a2.set_ylabel("H*"); a2.legend(frameon=False, fontsize=8)
    a2.set_title("digitized H* (paper Fig.6, Ha0=0.610, WL=0.356)")
    fig.tight_layout(); fig.savefig(DIG / "debug_fig6_extract.png", dpi=140); plt.close(fig)

    # ---------------- Fig. 8 : levels (T* 3..5) ----------------
    g8 = load_gray(FIG8)
    panels8, _ = find_panels(g8)
    draw_panel_debug(g8, panels8, DIG / "debug_fig8_panels.png", center=PANEL)
    box8 = panels8[PANEL[0]][PANEL[1]]
    crop_panel(g8, box8, DIG / "fig8_caseB_panel.png")
    masks8 = label_boxes_in_panel(g8, box8)
    pts8, comps8, bin8 = digitize_fig8(g8, box8, xlim=(3.0, 5.0), ylim=(0.0, 1.0),
                                       label_masks=masks8, reclass_int_below=0.5)
    with (DIG / "fig8_caseB_levels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Tstar", "Ystar", "kind"])
        for tstar, ystar, kind in sorted(pts8):
            w.writerow([f"{tstar:.5f}", f"{ystar:.5f}", kind])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.imshow(bin8, cmap="gray_r")
    for cx, cy, kind, size, hr in comps8:
        a1.plot(cx, cy, "o", ms=4, mfc="none",
                mec=("#ef4444" if kind == "int" else "#2b7fff"), mew=1.0)
    a1.set_title("Fig.8 case-B panel: components (blue=Y*fs, red=Y*int)")
    for kind, color, mk, lbl in (("fs", "#2b7fff", "^", "Y*_fs digitized"),
                                 ("int", "#ef4444", "o", "Y*_int digitized")):
        xs = [p[0] for p in pts8 if p[2] == kind]
        ys = [p[1] for p in pts8 if p[2] == kind]
        a2.plot(xs, ys, mk, ms=4, mfc="none", mec=color, label=lbl)
    a2.set_xlim(3, 5); a2.set_ylim(0, 1.02); a2.grid(alpha=0.3)
    a2.set_xlabel("T*_ref"); a2.set_ylabel("Y*"); a2.legend(frameon=False, fontsize=8)
    a2.set_title("digitized levels (paper Fig.8, Ha0=0.610, WL=0.356)")
    fig.tight_layout(); fig.savefig(DIG / "debug_fig8_extract.png", dpi=140); plt.close(fig)

    print(f"fig6 samples: {len(T6)}   fig8 markers: {len(pts8)} "
          f"(fs={sum(1 for p in pts8 if p[2] == 'fs')}, int={sum(1 for p in pts8 if p[2] == 'int')})")
    fig6d = dict(T=np.array(T6), med=np.array(med6), lo=np.array(lo6), hi=np.array(hi6))
    fig8d = {k: np.array(sorted([(t, y) for t, y, kk in pts8 if kk == k])) for k in ("fs", "int")}
    return fig6d, fig8d


def run_model(t_end: float = 10.5):
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

    if FIG6.is_file() and FIG8.is_file():
        fig6d, fig8d = digitize()
    else:
        print("Raw paper scans are absent; using committed digitized CSV files.")
        fig6d, fig8d = load_digitized()
    case, s = run_model()
    L = case.riser_height
    tsc = L / math.sqrt(G * case.Dr)
    yfs0 = case.init_water_level / L

    # ------------------------------------------------ pressure overlay
    # Cycle-average the model trace over one release-slosh period (~0.4 s): the
    # raw trace carries the resolved column-on-pocket-spring ringing (cf. caseA).
    tr = np.asarray(s["tr"], dtype=float)
    tt = np.asarray(s["t"], dtype=float)
    win = 0.4
    tr_avg = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        mwin = (tt >= tt[i] - win / 2) & (tt <= tt[i] + win / 2)
        if np.any(np.isfinite(tr[mwin])):
            tr_avg[i] = np.nanmean(tr[mwin])
    # Elevation datum: the model tunnel pressure is the INVERT (pipe-bottom)
    # piezometric head; the tower taps the pipe CROWN, one bore D above it.
    # The paper's H* plateau (0.76) matches the crown-referenced reading of the
    # pocket state (static check: isothermal pocket EOS + measured tower level
    # + crown datum close within ~0.02 L); the invert reading sits ~D/L=0.154
    # higher.  Show both: crown-datum as the primary comparison curve.
    dcrown = 0.094 / L
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.fill_between(fig6d["T"], fig6d["lo"], fig6d["hi"], color=C_PAPER_BAND, alpha=0.45,
                    label="V&W(2011) experiment, Fig.6 panel (3 repetitions, digitized band)")
    ax.plot(fig6d["T"], fig6d["med"], color=C_PAPER, lw=1.4,
            label="V&W(2011) experiment, digitized median")
    ax.plot(s["Tstar"], tr_avg - dcrown, color=C_MODEL, lw=2.2,
            label="model: transducer head, cycle-avg, CROWN datum (tap elevation, $-D/L$)")
    ax.plot(s["Tstar"], tr_avg, color=C_MODEL, lw=1.0, alpha=0.45,
            label="model: transducer head, cycle-avg, invert datum (as recorded)")
    ax.plot(s["Tstar"], s["pocket"], color=C_MODEL2, lw=0.9, ls="--", alpha=0.6,
            label="model: upstream air-pocket gauge head (auxiliary, invert)")
    ax.set_xlim(0, 5); ax.set_ylim(0, 1.5)
    ax.set_xlabel(r"$T^*_{ref} = t\,\sqrt{g D_t}/L$")
    ax.set_ylabel(r"$H^* = H/L$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Case B pressure head at the transducer -- "
                 r"$D_t^*=0.135$, $H_{a0}=0.610$ m, $WL_{init}=0.356$ m"
                 "\nmodel (decoupled two-fluid) vs V&W(2011) JHE Fig.6 (digitized)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "caseB_comparison_pressure.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------ levels overlay
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    for ax, xlim in ((a1, (0.0, 5.0)), (a2, (3.0, 5.0))):
        ax.plot(s["Tstar"], s["Yfs"], color=C_MODEL, lw=2.0, label=r"model $Y^*_{fs}$ (free surface)")
        ax.plot(s["Tstar"], s["Yint"], color=C_MODEL2, lw=1.8, ls="--", label=r"model $Y^*_{int}$ (gas front)")
        if fig8d["fs"].size:
            ax.plot(fig8d["fs"][:, 0], fig8d["fs"][:, 1], "^", ms=6, mfc="none", mec=C_FS,
                    label=r"experiment $Y^*_{fs}$ (Fig.8, digitized)")
        if fig8d["int"].size:
            ax.plot(fig8d["int"][:, 0], fig8d["int"][:, 1], "o", ms=6, mfc="none", mec=C_INT,
                    label=r"experiment $Y^*_{int}$ (Fig.8, digitized)")
        ax.axhline(1.0, color="#16a34a", ls=":", lw=1.2)
        ax.axhline(yfs0, color="0.65", ls=":", lw=1.0)
        ax.set_xlim(*xlim); ax.set_ylim(0, 1.05)
        ax.set_xlabel(r"$T^*_{ref} = t\,\sqrt{g D_t}/L$")
        ax.set_ylabel(r"$Y^* = Y/L$")
        ax.grid(alpha=0.3)
    # slope check: replot the model curves time-shifted so the gas-front climb
    # midpoint aligns with the experimental circles (the whole model gas sequence
    # runs ~0.6 T* early -- the crown-current transit bias, same direction as
    # caseA; a rigid shift is legitimate for comparing the climb kinematics)
    if fig8d["int"].size:
        ip = fig8d["int"][np.argsort(fig8d["int"][:, 0])]
        climb = ip[(ip[:, 1] > 0.1) & (ip[:, 1] < 0.8)]
        if climb.shape[0] >= 3:
            y_ref = 0.40
            t_exp = float(np.interp(y_ref, climb[:, 1], climb[:, 0]))
            yint_m = np.asarray(s["Yint"]); ts_m = np.asarray(s["Tstar"])
            rising = yint_m > 0.05
            if rising.any():
                i0 = int(np.argmax(rising))
                seg = slice(i0, int(np.argmax(yint_m)) + 1)
                t_mod = float(np.interp(y_ref, yint_m[seg], ts_m[seg]))
                dshift = t_exp - t_mod
                a2.plot(ts_m + dshift, yint_m, color=C_MODEL2, lw=1.2, ls=":",
                        label=rf"model $Y^*_{{int}}$ shifted +{dshift:.2f} $T^*$ (slope check)")
                a2.plot(ts_m + dshift, s["Yfs"], color=C_MODEL, lw=1.2, ls=":",
                        label=rf"model $Y^*_{{fs}}$ shifted +{dshift:.2f} $T^*$")
                a2.legend(frameon=False, fontsize=7, loc="center left")
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    a1.set_title("full model trajectory (paper window shaded)", fontsize=10)
    a1.axvspan(3.0, 5.0, color="#f3f4f6", zorder=0)
    a2.set_title("paper Fig.8 window ($T^*$ = 3..5)", fontsize=10)
    fig.suptitle("Case B tower free-surface and air-water interface -- "
                 r"$D_t^*=0.135$, $H_{a0}=0.610$ m, $WL_{init}=0.356$ m", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "caseB_comparison_levels.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------ metrics
    fs_pts, int_pts = fig8d["fs"], fig8d["int"]
    dcrown = 0.094 / L
    Ts_m = np.asarray(s["Tstar"])
    Yint_m = np.asarray(s["Yint"])
    Yfs_m = np.asarray(s["Yfs"])
    liftoff = first_crossing(Ts_m, Yint_m, 0.05)
    # V*int: slope of the gas-front climb (0.1..0.8 band), in sqrt(g Dt) units
    # (dY*/dT* IS V* by construction of the non-dimensionalisation)
    Vint_star = None
    if liftoff is not None:
        seg = (Yint_m > 0.10) & (Yint_m < 0.80) & (Ts_m >= liftoff) \
              & (Ts_m <= (first_crossing(Ts_m, Yint_m, 0.80) or Ts_m[-1]))
        if np.count_nonzero(seg) >= 3:
            Vint_star = float(np.polyfit(Ts_m[seg], Yint_m[seg], 1)[0])
    # V*fs: slope of the final free-surface climb (last rise from 0.90 to top)
    Vfs_star = None
    g_T = first_crossing(Ts_m, Yfs_m, 0.995, after=liftoff or 0.0)
    if g_T is not None and liftoff is not None:
        seg = (Ts_m >= liftoff) & (Ts_m <= g_T) & (Yfs_m > 0.86) & (Yfs_m < 0.995)
        if np.count_nonzero(seg) >= 3:
            Vfs_star = float(np.polyfit(Ts_m[seg], Yfs_m[seg], 1)[0])
    pocket_m = np.asarray(s["pocket"])
    comparison_head = tr_avg - dcrown
    collapse_T = first_crossing(
        Ts_m, comparison_head, 0.3, above=False, after=(liftoff or 2.0)
    )
    audited_interface = int_pts[int_pts[:, 0] > 4.0] if int_pts.size else int_pts
    m = dict(
        case=dict(**CASE, Dt_over_D=CASE["Dr"] / 0.094),
        Tstar_scale_s=tsc,
        datum_note=("The paper does not report the tap elevation. Both the raw "
                    "one-dimensional head and the legacy -D/L hypothesis are retained; "
                    "agreement after that shift does not establish the datum."),
        model=dict(
            geyser=bool(np.nanmax(Yfs_m) >= 0.98),
            geyser_Tstar=first_crossing(Ts_m, Yfs_m, 0.98, after=liftoff or 0.0),
            int_liftoff_Tstar=liftoff,
            int_top_Tstar=first_crossing(Ts_m, Yint_m, 0.85),
            Hstar_collapse_Tstar=collapse_T,
            Hstar_plateau_tr=float(np.nanmedian(s["tr"][(Ts_m > 1.0) & (Ts_m < 3.0)])),
            Hstar_plateau_tr_crown=float(np.nanmedian(s["tr"][(Ts_m > 1.0) & (Ts_m < 3.0)])) - dcrown,
            Yfs_plateau_pre_arrival=float(np.nanmedian(Yfs_m[(Ts_m > 1.5) & (Ts_m < (liftoff or 3.0))])),
            Vint_star=Vint_star,
            Vfs_star=Vfs_star,
            Hstar_max=float(np.nanmax(pocket_m)),
            Yfs_max=float(np.nanmax(Yfs_m)),
        ),
        paper=dict(
            geyser=True,
            geyser_Tstar=first_crossing(fs_pts[:, 0], fs_pts[:, 1], 0.97) if fs_pts.size else None,
            int_liftoff_Tstar=float(int_pts[:, 0].min()) if int_pts.size else None,
            int_top_Tstar=first_crossing(
                audited_interface[:, 0], audited_interface[:, 1], 0.85
            ) if audited_interface.size else None,
            int_top_Tstar_legacy_misclassified=first_crossing(
                int_pts[:, 0], int_pts[:, 1], 0.85
            ) if int_pts.size else None,
            Hstar_plateau=float(np.median(fig6d["med"][(fig6d["T"] > 1.0) & (fig6d["T"] < 4.0)])),
            Hstar_drop_Tstar=first_crossing(fig6d["T"], fig6d["med"], 0.3, above=False, after=2.0),
            Yfs_pre_arrival=float(np.median(fs_pts[fs_pts[:, 0] < 3.65][:, 1]))
                            if fs_pts.size and (fs_pts[:, 0] < 3.65).any() else None,
            Vint_star_table2=1.43,
            Vfs_star_table2=0.44,
            Yfs0=yfs0,
        ),
    )
    (OUT / "caseB_comparison_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

    with (OUT / "caseB_model_series.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "Tstar", "Yfs_star", "Yint_star", "pocket_Hstar", "transducer_Hstar"])
        for i in range(len(s["t"])):
            w.writerow([f"{s['t'][i]:.4f}", f"{s['Tstar'][i]:.4f}", f"{s['Yfs'][i]:.4f}",
                        f"{s['Yint'][i]:.4f}", f"{s['pocket'][i]:.4f}", f"{s['tr'][i]:.4f}"])

    build_report(m)
    print(json.dumps(m, indent=2))
    print(f"-> {OUT / 'caseB_comparison_pressure.png'}")
    print(f"-> {OUT / 'caseB_comparison_levels.png'}")
    print(f"-> {OUT / 'report.html'}")


def build_report(m: dict):
    mo, pa = m["model"], m["paper"]
    tsc = m["Tstar_scale_s"]

    def _f(v, nd=2):
        return "—" if v is None else f"{v:.{nd}f}"

    def _sec(v):
        return "—" if v is None else f"{v * tsc:.1f} s"

    rows = [
        ("是否喷发（水面到塔顶）", "是（每次重复都喷发）",
         ("是" if mo.get("geyser") else "否") + f"（水面最高 {_f(mo.get('Yfs_max'))}L）", ""),
        ("塔水位平台（气体进塔前）", _f(pa.get("Yfs_pre_arrival")), _f(mo.get("Yfs_plateau_pre_arrival")),
         "静水位=气囊准静态平衡"),
        ("压头平台 H*（1&lt;T*&lt;4 中位）", _f(pa.get("Hstar_plateau")),
         f"{_f(mo.get('Hstar_plateau_tr_crown'))}（管顶基准）/ {_f(mo.get('Hstar_plateau_tr'))}（管底）",
         "论文读数与管顶基准一致"),
        ("气水界面进入塔（Y*int 离底）", f"T*={_f(pa.get('int_liftoff_Tstar'))}（{_sec(pa.get('int_liftoff_Tstar'))}）",
         f"T*={_f(mo.get('int_liftoff_Tstar'))}（{_sec(mo.get('int_liftoff_Tstar'))}）", "模型偏早 ~0.6 T*"),
        ("气水界面到达塔顶", f"T*={_f(pa.get('int_top_Tstar'))}（{_sec(pa.get('int_top_Tstar'))}）",
         f"T*={_f(mo.get('int_top_Tstar'))}（{_sec(mo.get('int_top_Tstar'))}）", ""),
        ("自由水面到顶（喷发）", f"T*={_f(pa.get('geyser_Tstar'))}（{_sec(pa.get('geyser_Tstar'))}）",
         f"T*={_f(mo.get('geyser_Tstar'))}（{_sec(mo.get('geyser_Tstar'))}）", ""),
        ("压头骤降（气囊排空）", f"T*={_f(pa.get('Hstar_drop_Tstar'))}（{_sec(pa.get('Hstar_drop_Tstar'))}）",
         f"T*={_f(mo.get('Hstar_collapse_Tstar'))}（{_sec(mo.get('Hstar_collapse_Tstar'))}）", "气体贯通塔顶触发"),
        ("界面爬升速度 V*int（√(gDt) 归一）", f"{_f(pa.get('Vint_star_table2'))}（Table 2 平均）",
         _f(mo.get("Vint_star")), "取 Y*int 0.1–0.8 段斜率"),
        ("水面爬升速度 V*fs（√(gDt) 归一）", f"{_f(pa.get('Vfs_star_table2'))}（Table 2 平均）",
         _f(mo.get("Vfs_star")), "取喷发前最后爬升段斜率"),
    ]
    trs = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in rows)
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Case B: V&amp;W(2011) Dt=12.7mm 小塔（喷发）对比</title>
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
<h1>Case B — V&amp;W(2011) 小塔工况（喷发分支）</h1>
<p>工况：<code>Dt=12.7 mm</code>（Dt/D=0.135）、<code>Ha0=0.610 m</code>、<code>WL=0.356 m</code>。
无量纲化：T*=t·√(gDt)/L，H*=H/L，Y*=Y/L（L=0.610&nbsp;m；本工况 T*=1 对应 {tsc:.2f}&nbsp;s）。</p>
<div class="panel">
  <h2 style="margin-top:0">论文中与 Case B 相关的图</h2>
  <table>
    <tr><th>论文图</th><th>内容</th><th>与本工况的关系</th><th>本报告中的对照</th></tr>
    <tr><td><b>Fig.6</b>（p6）</td><td>压头 H*(T*)，Dt*=0.135 的 3×3 面板阵</td>
        <td><b>正是本工况</b>：中心面板 = Ha0=0.610、WL=0.356，3 次重复</td>
        <td>数字化后与模型叠加（下方"压头"图）</td></tr>
    <tr><td><b>Fig.8</b>（p8）</td><td>水面 Y*fs / 界面 Y*int，Dt*=0.135 的 3×3 面板阵，窗口 T*=3–5</td>
        <td><b>正是本工况</b>：中心面板，3 次重复</td>
        <td>数字化后与模型叠加（下方"水位/界面"图）</td></tr>
    <tr><td><b>Table 2</b>（p8）</td><td>各塔径的无量纲爬升速度平均值</td>
        <td>Dt*=0.135 行：V*int=1.43、V*fs=0.44</td>
        <td>对照表"爬升速度"两行</td></tr>
    <tr><td><b>Fig.3</b>（p3）</td><td>喷发实验流动演化示意（阀开→气囊推进→抵塔）</td>
        <td>通用机理图；中间小图注明"开阀后塔水位振荡、幅值随时间衰减"——与模型的释放振荡同源</td>
        <td>下方原图收录</td></tr>
    <tr><td><b>Fig.11</b>（p11）</td><td>论文自家 TPA 模型 vs 实验的五联图，Dt*=0.135</td>
        <td>同塔径但<b>不同初值</b>（Ha0=0.305、WL=0.254）；我们用同一冻结求解器改初值复跑对照</td>
        <td>"Fig.11 对照"栏（红线直接叠画原图）</td></tr>
    <tr><td class="muted">Fig.1/2/9</td><td class="muted">现场喷发照片 / 实验装置图 / 模型变量示意</td>
        <td class="muted">通用背景，与具体工况无关</td><td class="muted">—</td></tr>
    <tr><td class="muted">Fig.4/5/7/10</td><td class="muted">气囊爬升照片、压头、水位、模型对照</td>
        <td class="muted">均为大塔 Dt*=0.607（= caseA 的对标图）</td>
        <td class="muted">见 caseA 报告</td></tr>
    <tr><td class="muted">Fig.12</td><td class="muted">模型预测的水面抬升-初始水位参数图（全部塔径）</td>
        <td class="muted">参数化汇总，非时间序列</td><td class="muted">—</td></tr>
  </table>
</div>
<div class="panel">
  <h2 style="margin-top:0">论文原图（扫描面板）</h2>
  <div class="grid2">
    <div><h3 style="margin:4px 0">Fig.6 本工况面板（压头）</h3><img src="../data/digitized/fig6_caseB_panel.png"></div>
    <div><h3 style="margin:4px 0">Fig.8 本工况面板（水面/界面）</h3><img src="../data/digitized/fig8_caseB_panel.png"></div>
  </div>
  <h3 style="margin:12px 0 4px 0">Fig.3 流动演化示意（论文机理图）</h3>
  <img src="../reference/paper_scans/fig3_schematic.png">
  <p class="muted">注意中间小图的标注："Water level oscillates after valve is opened, but amplitude
  reduces with time"——实验本身就有开阀释放振荡（论文正文第 4 条亦有描述），模型解析出的
  水柱-气囊弹簧振荡与之同源，差异只在衰减速率。</p>
</div>
__PUB_FIGS__
<div class="panel">
  <h2 style="margin-top:0">叠加对比（工作版全量图）</h2>
  <h3>压头 H*(T*)</h3><img src="caseB_comparison_pressure.png">
  <h3>塔内水面与气水界面 Y*(T*)</h3><img src="caseB_comparison_levels.png">
  <h3>事件时刻与量值对照</h3>
  <table>
    <tr><th>指标</th><th>论文实验（数字化）</th><th>模型</th><th>备注</th></tr>
    {trs}
  </table>
  <p class="muted"><b>基准面说明</b>：论文未报告压力孔的周向高程。图中同时保留一维模型原始值及
  历史 -D/L=0.154 假设；后者改善吻合不能反证测点基准，详见三维算例 PAPER_AUDIT.md。<br>
  <b>吻合点</b>：喷发分支（水面到顶）、气体进塔前塔水位平台（模型 ~0.85 vs 实验 ~0.82，
  即气囊准静态平衡位）、压头平台（管顶基准 ~0.78 vs 实验 0.76）、界面爬升-水面爬升-压头骤降的
  相对时序（进塔→喷发→贯通→骤降）。<br>
  <b>差异点</b>：整段气体时序偏早 ~0.5–0.6 T*（水平管 crown current 抵塔偏早，与 caseA 同向）；
  释放振荡衰减偏慢（水柱-气囊弹簧模态，实验 ~1 T* 内衰减）；喷发后模型塔柱随排气回落（实验窗口内
  无 fs 数据可对比）。数字化中间产物见 <code>digitized/</code>；模型无量纲序列见
  <code>outputs/caseB_model_series.csv</code>。</p>
</div>
__EXTRA_SECTIONS__
</div></body></html>"""
    html = html.replace("__EXTRA_SECTIONS__", build_extra_sections())
    html = html.replace("__PUB_FIGS__", build_pub_figs())
    (OUT / "report.html").write_text(html, encoding="utf-8")


def build_pub_figs() -> str:
    """出版级重绘图面板（论文 E:\\Geysering\\paper 实际收录的矢量图源）。
    产物由 caseB_plot_pub.py 生成，存在才嵌入。"""
    pub_imgs = [
        ("caseB_pressure_pub.png",
         "压头 H*：三次重复数字化包络+中位（灰/黑）+ 模型管顶基准 0.4 s 周期平均（红粗）"
         "+ 管底基准原始记录（红浅）"),
        ("caseB_levels_pub.png",
         "水位/界面 Y*（发表窗口 T*=3–5）：实验散点（三角=水面，圆=界面）+ 模型曲线"
         "+ 刚体平移 +0.54 的浅色对照线（模型画至实验覆盖终点）"),
    ]
    have = [(f, cap) for f, cap in pub_imgs if (OUT / f).exists()]
    if not have:
        return ""
    imgs = "\n".join(f'  <h3>{cap}</h3><img src="{f}">' for f, cap in have)
    return f"""
<div class="panel">
  <h2 style="margin-top:0">出版级重绘图（论文实际收录版本）</h2>
  <p class="muted">以下为论文（E:\\Geysering\\paper）收录的自绘矢量图的 PNG 预览，
  由 caseB_plot_pub.py 生成；数据与下方叠加对比图同源（同一冻结求解器输出 +
  同一数字化 CSV）。</p>
{imgs}
</div>"""


def build_extra_sections() -> str:
    """帧查看器（caseB_make_frame_viewer.py）与 Fig.11 对照
    （caseB_fig11_compare.py）的产物，存在才嵌入。"""
    parts = []
    frames_json = OUT / "frames_index.json"
    if frames_json.exists():
        frames = json.loads(frames_json.read_text(encoding="utf-8"))
        for frame in frames:
            for key in ("file", "riserFile"):
                if frame.get(key, "").startswith("outputs/"):
                    frame[key] = frame[key][len("outputs/"):]
        frame_assets_ok = bool(frames) and all(
            (OUT / frame.get(key, "")).is_file()
            for frame in frames
            for key in ("file", "riserFile")
        )
        frames_data = json.dumps(frames, ensure_ascii=False)
    else:
        frame_assets_ok = False
    if frame_assets_ok:
        parts.append("""
<div class="panel">
  <h2 style="margin-top:0">两流体模拟逐帧查看器 — 水平管 + 通风塔全场演化</h2>
  <p class="muted">同一冻结求解器的 Case B 全场演化（水=蓝，气=白/浅灰）：
  阀开释放振荡 → 塔水位抬升到准静态平衡（~0.5L） → 气囊沿管顶推进 →
  抵塔、驱动气核爬升（水面被顶托到塔顶=喷发） → 贯通排气 → 压头塌落、塔柱回落。
  <b>左图为真实比例 1:1</b>（管径 D=94 mm、塔径 Dt=12.7 mm 均按真实尺寸），
  本工况的塔在全局图上只是一根发丝——塔内细节请看右侧放大同步视图
  （按当地两流体气含率 α_g 画气核宽度）。左右方向键 / 滑块 / 播放均可调帧。</p>
  <p class="muted"><b>时间跨度说明</b>：t=0 为开阀时刻，模拟到 9 s 结束——本工况
  1 T* = 1.73 s，论文曲线窗口 T*=5 对应 8.6 s，9 s 已完整覆盖（平衡段→气囊抵塔→
  气核爬升/喷发→贯通排气→压头归零→塔柱回落）。</p>
  <div class="meta" style="display:flex;gap:18px;margin:10px 0;font-weight:700;flex-wrap:wrap">
    <span id="vIdx"></span><span id="vTime"></span><span id="vTstar"></span><span id="vWtop"></span>
    <span id="vItop"></span><span id="vMass"></span><span id="vHead"></span></div>
  <div style="display:grid;grid-template-columns:minmax(0,2.9fr) minmax(220px,1fr);gap:14px;align-items:start">
    <div><h3 style="margin:0 0 8px 0;font-size:15px">全局 1:1 视图</h3>
      <img id="vFrame" style="width:100%">
      <p class="muted" style="margin:6px 0 0 0">水平管按截面含水率显示分层气团推进。</p></div>
    <div><h3 style="margin:0 0 8px 0;font-size:15px">竖管放大同步视图</h3>
      <img id="vRiser" style="width:100%">
      <p class="muted" style="margin:6px 0 0 0">气核宽度 = 当地 α_g（无阈值化）；
      蓝线=可见水面（含气核顶托），红虚线=气相前沿。</p></div>
  </div>
  <div style="margin-top:8px">
    <button id="vPrev" style="padding:8px 14px;margin:6px;border:1px solid #bbb;border-radius:8px;background:#fff;cursor:pointer">上一帧</button>
    <button id="vPlay" style="padding:8px 14px;margin:6px;border:1px solid #bbb;border-radius:8px;background:#fff;cursor:pointer">播放</button>
    <input id="vSlider" type="range" style="width:60%;vertical-align:middle">
    <button id="vNext" style="padding:8px 14px;margin:6px;border:1px solid #bbb;border-radius:8px;background:#fff;cursor:pointer">下一帧</button>
  </div>
  <p class="muted">整段 GIF 备份：<a href="caseB_animation.gif">caseB_animation.gif</a></p>
</div>
<script>
const vFrames=""" + frames_data + """;
const TSC=1.7282;   // seconds per T* for Dt=12.7 mm
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
  vTstar.textContent=`T*=${(f.time/TSC).toFixed(2)}`;
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
    elif (OUT / "caseB_animation.gif").exists():
        parts.append("""
<div class="panel">
  <h2 style="margin-top:0">两流体模拟动画 — 水平管 + 通风塔全场演化</h2>
  <img src="caseB_animation.gif" style="max-width:900px">
</div>""")
    if (
        (OUT / "caseB_fig11_overlay.png").is_file()
        and (OUT / "caseB_fig11_model_panels.png").is_file()
        and (SCANS / "fig11_full.png").is_file()
    ):
        parts.append("""
<div class="panel">
  <h2 style="margin-top:0">Fig.11 对照 — 模型曲线直接叠画在论文原图上</h2>
  <p class="muted">论文 Fig.11 是他们自家 TPA 模型对小塔（Dt*=0.135）的对照，但工况为
  <b>Ha0=0.305 m、WL=0.254 m</b>（非本文件夹的 0.610/0.356）。我们用同一冻结求解器改初值复跑，
  把结果（<b style="color:#e11d48">红线</b>）按论文坐标系直接画进五个面板：黑粗线=论文 TPA 模型，
  散点/细线=实验三次重复。时间轴做了刚体平移对齐 Y*<sub>int</sub> 爬升中点（开阀时刻在实验中为
  手动，论文 T* 原点取法不同；平移量见图内标注）。</p>
  <img src="caseB_fig11_overlay.png">
  <p class="muted">读图：Y*<sub>int</sub> 爬升段红线斜率与论文模型/实验一致（V*int 拟合 ~1.6 vs
  实验散点 1–3 区间）；H* 平台（管顶基准 ~0.33 vs 论文 ~0.38–0.44）与骤降形态一致、模型骤降更陡；
  该工况初始超压余量为负（Ha0=0.305 &lt; WL 静压），我们的模型水面只抬升到 ~0.61L 即被排气打断，
  论文实验/模型在 T*≈4.1 冲顶——小超压喷发分支的判据差异是已知模型局限（与 caseB 主工况
  Ha0=0.610 的强驱动喷发不同源）。</p>
  <div class="grid2" style="margin-top:10px">
    <div><h3 style="margin:4px 0">论文 Fig.11 原图</h3><img src="../reference/paper_scans/fig11_full.png"></div>
    <div><h3 style="margin:4px 0">我们的模型（同轴五联图）</h3><img src="caseB_fig11_model_panels.png"></div>
  </div>
</div>""")
    return "".join(parts)


if __name__ == "__main__":
    if "--check-paths" in sys.argv:
        check_paths()
    else:
        main()

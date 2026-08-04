# -*- coding: utf-8 -*-
"""Digitize the Run B-H1 measurements from the Cong (2017) figure scans:

  * Fig. 9(a)  -- Yfs (red filled squares) and Yint (blue open squares) vs t,
                  axes t = 8..10 s, Y = 0..2.0 m;
  * Fig. 10(a) -- PT1 pressure trace H/H0 vs t (Run B-1, same condition as
                  B-H1 on the video-camera series), axes t = 0..13 s(x ticks
                  0,2,...,12; box spans 0..13), H/H0 = 0..4 (red line = PT1).

Outputs digitized/fig9a_levels.csv, digitized/fig10a_pt1.csv + debug overlays.
Plot-box pixel anchors were measured with _probe_fig_boxes.py.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = Path(__file__).resolve().parent
DIG = HERE / "digitized"
DIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- Fig. 9(a)
# box: x 297..822 px = t 8..10 s ; y 17..416 px = Y 2.0..0.0 m
F9 = dict(x0=297.0, x1=822.0, t0=8.0, t1=10.0, y0=416.0, y1=17.0, v0=0.0, v1=2.0)


def px_to_val(box, px, py):
    t = box["t0"] + (px - box["x0"]) / (box["x1"] - box["x0"]) * (box["t1"] - box["t0"])
    v = box["v0"] + (py - box["y0"]) / (box["y1"] - box["y0"]) * (box["v1"] - box["v0"])
    return t, v


def cluster_points(mask, min_px=4):
    """connected components -> centroids (cx, cy, size)"""
    import scipy.ndimage as ndi
    lab, n = ndi.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if xs.size >= min_px:
            out.append((float(xs.mean()), float(ys.mean()), int(xs.size)))
    return out


def digitize_fig9a():
    img = mpimg.imread(HERE / "paper_scans" / "fig9_bh1_riser.png")
    rgb = img[..., :3]
    H, W = rgb.shape[:2]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # panel (a) interior only
    inbox = np.zeros((H, W), bool)
    inbox[int(F9["y1"]) + 3:int(F9["y0"]) - 1, int(F9["x0"]) + 3:int(F9["x1"]) - 1] = True
    # legend box (top-left of panel a): mask out
    inbox[15:120, 300:470] = False

    red = (r > 0.55) & (g < 0.45) & (b < 0.45) & inbox            # Yfs filled squares
    blue = (b > 0.45) & (b - r > 0.12) & (b - g > 0.08) & inbox   # Yint open squares

    pts_fs = cluster_points(red, min_px=6)
    pts_int = cluster_points(blue, min_px=4)

    rows = []
    for cx, cy, _ in pts_fs:
        t, v = px_to_val(F9, cx, cy)
        rows.append((t, v, "fs"))
    for cx, cy, _ in pts_int:
        t, v = px_to_val(F9, cx, cy)
        rows.append((t, v, "int"))
    rows.sort()
    with (DIG / "fig9a_levels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "Y_m", "kind"])
        for t, v, k in rows:
            w.writerow([f"{t:.4f}", f"{v:.4f}", k])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.imshow(rgb)
    for cx, cy, _ in pts_fs:
        a1.plot(cx, cy, "o", ms=4, mfc="none", mec="#16a34a")
    for cx, cy, _ in pts_int:
        a1.plot(cx, cy, "s", ms=4, mfc="none", mec="#f59e0b")
    a1.set_xlim(250, 900); a1.set_ylim(450, 0)
    a1.set_title("Fig.9(a): detected markers (green=Yfs, orange=Yint)")
    for k, c, m in (("fs", "#dc2626", "s"), ("int", "#2563eb", "o")):
        xs = [t for t, v, kk in rows if kk == k]
        ys = [v for t, v, kk in rows if kk == k]
        a2.plot(xs, ys, m, ms=4, mfc="none", mec=c, label=k)
    a2.set_xlim(8, 10); a2.set_ylim(0, 2)
    a2.grid(alpha=0.3); a2.legend()
    a2.set_title("digitized Fig.9(a): Yfs / Yint vs t")
    fig.tight_layout()
    fig.savefig(DIG / "debug_fig9a.png", dpi=130)
    plt.close(fig)
    nfs = sum(1 for *_x, k in rows if k == "fs")
    print(f"fig9a: {nfs} fs + {len(rows)-nfs} int markers")


# ---------------------------------------------------------------- Fig. 10(a)
# box: x 180..747 px = t 0..13 s ; y 19..440 px = H/H0 4..0
F10 = dict(x0=180.0, x1=747.0, t0=0.0, t1=13.0, y0=440.0, y1=19.0, v0=0.0, v1=4.0)


def digitize_fig10a():
    img = mpimg.imread(HERE / "paper_scans" / "fig10_pressure.png")
    rgb = img[..., :3]
    H, W = rgb.shape[:2]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    inbox = np.zeros((H, W), bool)
    inbox[int(F10["y1"]) + 2:int(F10["y0"]) - 1, int(F10["x0"]) + 2:int(F10["x1"]) - 1] = True
    inbox[25:95, 620:745] = False        # legend
    red = (r > 0.5) & (g < 0.5) & (b < 0.5) & (r - np.maximum(g, b) > 0.10) & inbox

    # per-column median (PT1 trace) -- keeps the sharp geyser spike
    ts, med, lo, hi = [], [], [], []
    for c in range(int(F10["x0"]) + 2, int(F10["x1"]) - 1):
        ys = np.where(red[:, c])[0]
        if ys.size == 0:
            continue
        t, _ = px_to_val(F10, c, 0)
        _, vmed = px_to_val(F10, c, float(np.median(ys)))
        _, vhi = px_to_val(F10, c, float(ys.min()))
        _, vlo = px_to_val(F10, c, float(ys.max()))
        ts.append(t); med.append(vmed); lo.append(vlo); hi.append(vhi)
    with (DIG / "fig10a_pt1.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "HoverH0_med", "HoverH0_min", "HoverH0_max"])
        for row in zip(ts, med, lo, hi):
            w.writerow([f"{v:.4f}" for v in row])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.imshow(red, cmap="gray_r")
    a1.set_title("Fig.10(a): PT1 red-pixel mask")
    a2.fill_between(ts, lo, hi, color="#fca5a5", alpha=0.6, label="pixel envelope")
    a2.plot(ts, med, color="#b91c1c", lw=1.0, label="median")
    a2.set_xlim(0, 13); a2.set_ylim(0, 4)
    a2.grid(alpha=0.3); a2.legend()
    a2.set_title("digitized Fig.10(a): PT1 H/H0 (Run B-1)")
    fig.tight_layout()
    fig.savefig(DIG / "debug_fig10a.png", dpi=130)
    plt.close(fig)
    print(f"fig10a: {len(ts)} columns")


if __name__ == "__main__":
    digitize_fig9a()
    digitize_fig10a()

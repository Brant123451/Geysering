# -*- coding: utf-8 -*-
"""Digitize the Run B-H6 measurements from the Cong (2017) figure scans:

  * Fig. 7(a)  -- Yfs (red filled squares) and Yint (blue open squares) vs t,
                  axes t = 8..11 s, Y = 0..2.0 m  (box px: x 202..813 = 8..11,
                  y 1378..926 = 0..2.0, measured with _probe_boxes.py);
  * Fig. 10(b) -- PT1 pressure trace H/H0 vs t for Run B-32 (video series,
                  same Dr=41 mm condition), axes t = 0..13 s,
                  H/H0 = -0.5..4 (box px: x 180..747, y 970..550).

Outputs data/digitized/fig7a_levels.csv, data/digitized/fig10b_pt1.csv and
debug overlays.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parent
SCANS = CASE_ROOT / "reference" / "paper_scans"
DIG = CASE_ROOT / "data" / "digitized"
DIG.mkdir(parents=True, exist_ok=True)

F7 = dict(x0=202.0, x1=813.0, t0=8.0, t1=11.0, y0=1378.0, y1=926.0, v0=0.0, v1=2.0)
F10 = dict(x0=180.0, x1=747.0, t0=0.0, t1=13.0, y0=970.0, y1=550.0, v0=-0.5, v1=4.0)


def px_to_val(box, px, py):
    t = box["t0"] + (px - box["x0"]) / (box["x1"] - box["x0"]) * (box["t1"] - box["t0"])
    v = box["v0"] + (py - box["y0"]) / (box["y1"] - box["y0"]) * (box["v1"] - box["v0"])
    return t, v


def cluster_points(mask, min_px=4):
    remaining = mask.copy()
    out = []
    height, width = remaining.shape
    for start_y, start_x in np.argwhere(remaining):
        if not remaining[start_y, start_x]:
            continue
        remaining[start_y, start_x] = False
        stack = [(int(start_y), int(start_x))]
        x_sum = 0
        y_sum = 0
        count = 0
        while stack:
            y, x = stack.pop()
            x_sum += x
            y_sum += y
            count += 1
            for neighbour_y, neighbour_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                if (
                    0 <= neighbour_y < height
                    and 0 <= neighbour_x < width
                    and remaining[neighbour_y, neighbour_x]
                ):
                    remaining[neighbour_y, neighbour_x] = False
                    stack.append((neighbour_y, neighbour_x))
        if count >= min_px:
            out.append((x_sum / count, y_sum / count, count))
    return out


def digitize_fig7a():
    img = mpimg.imread(SCANS / "fig7_bh6_riser.png")
    rgb = img[..., :3]
    H, W = rgb.shape[:2]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    inbox = np.zeros((H, W), bool)
    inbox[int(F7["y1"]) + 3:int(F7["y0"]) - 1, int(F7["x0"]) + 3:int(F7["x1"]) - 1] = True
    # legend box (top-left of panel a)
    inbox[930:1080, 205:400] = False

    red = (r > 0.55) & (g < 0.45) & (b < 0.45) & inbox            # Yfs filled squares
    blue = (b > 0.45) & (b - r > 0.12) & (b - g > 0.08) & inbox   # Yint open squares

    pts_fs = cluster_points(red, min_px=6)
    pts_int = cluster_points(blue, min_px=4)

    rows = []
    for cx, cy, _ in pts_fs:
        t, v = px_to_val(F7, cx, cy)
        rows.append((t, v, "fs"))
    for cx, cy, _ in pts_int:
        t, v = px_to_val(F7, cx, cy)
        rows.append((t, v, "int"))
    rows.sort()
    with (DIG / "fig7a_levels.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["t_s", "Y_m", "kind"])
        for t, v, k in rows:
            w.writerow([f"{t:.4f}", f"{v:.4f}", k])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.imshow(rgb)
    for cx, cy, _ in pts_fs:
        a1.plot(cx, cy, "o", ms=4, mfc="none", mec="#16a34a")
    for cx, cy, _ in pts_int:
        a1.plot(cx, cy, "s", ms=4, mfc="none", mec="#f59e0b")
    a1.set_xlim(150, 900); a1.set_ylim(1450, 880)
    a1.set_title("Fig.7(a): detected markers (green=Yfs, orange=Yint)")
    for k, c, m in (("fs", "#dc2626", "s"), ("int", "#2563eb", "o")):
        xs = [t for t, v, kk in rows if kk == k]
        ys = [v for t, v, kk in rows if kk == k]
        a2.plot(xs, ys, m, ms=4, mfc="none", mec=c, label=k)
    a2.set_xlim(8, 11); a2.set_ylim(0, 2)
    a2.grid(alpha=0.3); a2.legend()
    a2.set_title("digitized Fig.7(a): Yfs / Yint vs t")
    fig.tight_layout()
    fig.savefig(DIG / "debug_fig7a.png", dpi=130)
    plt.close(fig)
    nfs = sum(1 for *_x, k in rows if k == "fs")
    print(f"fig7a: {nfs} fs + {len(rows)-nfs} int markers")


def digitize_fig10b():
    img = mpimg.imread(SCANS / "fig10_pressure.png")
    rgb = img[..., :3]
    H, W = rgb.shape[:2]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    inbox = np.zeros((H, W), bool)
    inbox[int(F10["y1"]) + 2:int(F10["y0"]) - 1, int(F10["x0"]) + 2:int(F10["x1"]) - 1] = True
    inbox[555:625, 620:745] = False        # legend
    red = (r > 0.5) & (g < 0.5) & (b < 0.5) & (r - np.maximum(g, b) > 0.10) & inbox

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
    with (DIG / "fig10b_pt1.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["t_s", "HoverH0_med", "HoverH0_min", "HoverH0_max"])
        for row in zip(ts, med, lo, hi):
            w.writerow([f"{v:.4f}" for v in row])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.imshow(red, cmap="gray_r")
    a1.set_title("Fig.10(b): PT1 red-pixel mask (Run B-32)")
    a2.fill_between(ts, lo, hi, color="#fca5a5", alpha=0.6, label="pixel envelope")
    a2.plot(ts, med, color="#b91c1c", lw=1.0, label="median")
    a2.set_xlim(0, 13); a2.set_ylim(-0.5, 4)
    a2.grid(alpha=0.3); a2.legend()
    a2.set_title("digitized Fig.10(b): PT1 H/H0 (Run B-32, no geyser)")
    fig.tight_layout()
    fig.savefig(DIG / "debug_fig10b.png", dpi=130)
    plt.close(fig)
    print(f"fig10b: {len(ts)} columns")


if __name__ == "__main__":
    digitize_fig7a()
    digitize_fig10b()

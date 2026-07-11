# -*- coding: utf-8 -*-
"""Digitize the V&W (2011) JHE experimental curves for the selected case
(D_t = 12.7 mm -> D_t* = 0.135, H_a0 = 0.610 m, WL_init = 0.356 m).

Sources (native-resolution bitmaps extracted from the paper PDF):
  * _paper_figs/raw_p5_x101_2000x1457.png  = Fig. 6  (normalized pressure head H*,
    3x3 panels, x: T*_ref 0..5, y: H* 0..1.5) -- CENTER panel is our case.
  * _paper_figs/raw_p7_x121_2145x1534.png  = Fig. 8  (normalized Y*_fs & Y*_int,
    3x3 panels, x: T*_ref 3..5, y: Y* 0..1)   -- CENTER panel is our case.

Outputs (paper_reference/digitized/):
  * fig6_center_Hstar_band.csv   : Tstar, Hstar_med, Hstar_min, Hstar_max
  * fig8_center_levels.csv       : Tstar, Ystar, kind (fs = filled/x markers, int = open markers)
  * debug_fig6_panels.png / debug_fig8_panels.png   (detected panel boxes)
  * debug_fig6_extract.png / debug_fig8_extract.png (extraction overlay)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

CASE_ROOT = Path(__file__).resolve().parents[1]
MODEL = CASE_ROOT / "model"
DIGITIZED = CASE_ROOT / "data" / "digitized"
SCANS = CASE_ROOT / "reference" / "paper_scans"
OUTPUTS = CASE_ROOT / "outputs"
FIGS = SCANS
OUT = DIGITIZED
OUT.mkdir(parents=True, exist_ok=True)

FIG6 = FIGS / "raw_p5_x101_2000x1457.png"   # pressure heads, Dt*=0.135
FIG8 = FIGS / "raw_p7_x121_2145x1534.png"   # levels, Dt*=0.135


def load_gray(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("L"), dtype=np.uint8)


def _cluster(indices: np.ndarray, gap: int = 4):
    """Group sorted pixel indices into clusters separated by > gap."""
    groups = []
    cur = [int(indices[0])]
    for v in indices[1:]:
        v = int(v)
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)
    return [int(round(np.mean(g))) for g in groups]


def find_panels(gray: np.ndarray, dark_thresh: int = 185):
    """Locate the 3x3 panel plot boxes via dilated connected components.

    The scan breaks border lines into fragments, so the raw binary image is
    dilated (2 px) first; each panel border box then forms one large
    component. The plot-area rectangle is refined inside each component bbox
    by finding the strongest dark row/column near each bbox edge (tick marks
    that stick out are short and cannot win). Returns [row][col] boxes in
    original-image pixel coordinates.
    """
    H, W = gray.shape
    boxes = []
    for thr in (dark_thresh, 200, 210):
        binm0 = gray < thr
        binm = ndimage.binary_dilation(binm0, iterations=2)
        lab, n = ndimage.label(binm)
        slices = ndimage.find_objects(lab)
        boxes = []
        for sl in slices:
            if sl is None:
                continue
            bh = sl[0].stop - sl[0].start
            bw = sl[1].stop - sl[1].start
            if bh < 0.20 * H or bw < 0.22 * W or bh > 0.45 * H or bw > 0.45 * W:
                continue
            ys0, ys1 = sl[0].start, sl[0].stop
            xs0, xs1 = sl[1].start, sl[1].stop
            sub = binm0[ys0:ys1, xs0:xs1]
            h, w = sub.shape
            edge = 24
            colcnt = sub.sum(axis=0)
            rowcnt = sub.sum(axis=1)
            x_left = int(np.argmax(colcnt[:edge]))
            x_right = w - edge + int(np.argmax(colcnt[-edge:]))
            y_top = int(np.argmax(rowcnt[:edge]))
            y_bot = h - edge + int(np.argmax(rowcnt[-edge:]))
            boxes.append((xs0 + x_left, xs0 + x_right, ys0 + y_top, ys0 + y_bot))
        if len(boxes) == 9:
            break
    if len(boxes) != 9:
        raise RuntimeError(f"expected 9 panel boxes, got {len(boxes)}: {boxes}")
    boxes.sort(key=lambda b: (b[2] + b[3]))          # by vertical position
    rows = [sorted(boxes[3 * r:3 * r + 3], key=lambda b: b[0]) for r in range(3)]
    return rows, boxes


def label_boxes_in_panel(gray: np.ndarray, box, dark_thresh: int = 185):
    """Find the in-panel H_air/WL_init label boxes (rectangular frames
    ~160x80 px) and return their bboxes in panel-crop coordinates.

    A genuine label box has a straight frame: its bbox top AND bottom rows are
    almost fully dark. Merged marker clusters of similar size fail that test.
    """
    x0, x1, y0, y1 = box
    sub = gray[y0 + 3:y1 - 2, x0 + 3:x1 - 2] < dark_thresh
    sub = ndimage.binary_dilation(sub, iterations=1)
    lab, n = ndimage.label(sub)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        bh = sl[0].stop - sl[0].start
        bw = sl[1].stop - sl[1].start
        if not (40 <= bh <= 160 and 100 <= bw <= 340):
            continue
        mask = lab[sl] == i
        top_cov = mask[0:3].any(axis=0).sum() / bw
        bot_cov = mask[-3:].any(axis=0).sum() / bw
        if top_cov > 0.8 and bot_cov > 0.8:
            out.append((sl[1].start, sl[1].stop, sl[0].start, sl[0].stop))
    return out


def draw_panel_debug(gray: np.ndarray, panels, out_png: Path, center=(1, 1)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    for r in range(3):
        for c in range(3):
            x0, x1, y0, y1 = panels[r][c]
            ec = "#ef4444" if (r, c) == center else "#2b7fff"
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=ec, lw=1.6))
    ax.set_title("detected panel plot boxes (red = selected case panel)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 6
def digitize_fig6(gray: np.ndarray, box, xlim=(0.0, 5.0), ylim=(0.0, 1.5),
                  dark_thresh: int = 150, label_masks=()):
    """Extract the pressure-head repetition band from one panel.

    label_masks: list of (mx0, mx1, my0, my1) bboxes (panel-crop coords) to
    blank out (the in-panel H_air/WL_init label boxes).
    """
    x0, x1, y0, y1 = box
    sub = gray[y0 + 3:y1 - 2, x0 + 3:x1 - 2]
    binm = sub < dark_thresh
    h, w = binm.shape

    for mx0, mx1, my0, my1 in label_masks:
        binm[max(my0 - 3, 0):my1 + 3, max(mx0 - 3, 0):mx1 + 3] = False

    Ts, med, lo, hi = [], [], [], []
    for px in range(w):
        ys = np.where(binm[:, px])[0]
        if ys.size == 0:
            continue
        tstar = xlim[0] + (px / (w - 1)) * (xlim[1] - xlim[0])
        # pixel row -> H*: row 0 = top = ylim[1]
        vals = ylim[1] - (ys / (h - 1)) * (ylim[1] - ylim[0])
        Ts.append(tstar)
        med.append(float(np.median(vals)))
        lo.append(float(vals.min()))
        hi.append(float(vals.max()))
    return np.array(Ts), np.array(med), np.array(lo), np.array(hi), binm


# ------------------------------------------------------------------ Fig. 8
def digitize_fig8(gray: np.ndarray, box, xlim=(3.0, 5.0), ylim=(0.0, 1.0),
                  dark_thresh: int = 150, label_masks=(),
                  max_cluster_h: int = 140, max_cluster_w: int = 260,
                  reclass_int_below: float = 0.5):
    """Extract Y*fs / Y*int scatter markers from one panel (Fig.7 or Fig.8 style).

    Marker classification: connected dark components; a component whose
    hole-filled area clearly exceeds its own area is an OPEN marker
    (diamond/square/circle = Y*_int); otherwise it is a filled marker or an
    'x' stroke (= Y*_fs).  Merged clusters up to max_cluster_h x max_cluster_w
    are split by distance-transform peaks.  Markers below reclass_int_below
    are forced to 'int' (the free surface never drops that low in these runs).
    """
    x0, x1, y0, y1 = box
    sub = gray[y0 + 3:y1 - 2, x0 + 3:x1 - 2]
    binm = sub < dark_thresh
    h, w = binm.shape

    for mx0, mx1, my0, my1 in label_masks:
        binm[max(my0 - 3, 0):my1 + 3, max(mx0 - 3, 0):mx1 + 3] = False

    def to_data(cx, cy):
        tstar = xlim[0] + (cx / (w - 1)) * (xlim[1] - xlim[0])
        ystar = ylim[1] - (cy / (h - 1)) * (ylim[1] - ylim[0])
        return tstar, ystar

    def center_is_open(cy, cx):
        """Open marker (ring) if the 3x3 neighbourhood of the centre is white."""
        y0w, y1w = max(int(cy) - 1, 0), min(int(cy) + 2, h)
        x0w, x1w = max(int(cx) - 1, 0), min(int(cx) + 2, w)
        return binm[y0w:y1w, x0w:x1w].sum() <= 2

    lab, n = ndimage.label(binm)
    pts = []
    comps = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        mask = lab[sl] == i
        size = int(mask.sum())
        if size < 12:
            continue
        bh = sl[0].stop - sl[0].start
        bw = sl[1].stop - sl[1].start
        oy, ox = sl[0].start, sl[1].start
        if bh < 4 or bw < 4:                    # dust / grid dots
            continue
        if bh <= 30 and bw <= 30:
            # ---- single marker ----
            filled = ndimage.binary_fill_holes(mask)
            hole_ratio = float(filled.sum()) / size
            ys, xs = np.nonzero(mask)
            cx, cy = ox + float(xs.mean()), oy + float(ys.mean())
            kind = "int" if hole_ratio > 1.45 else "fs"
            tstar, ystar = to_data(cx, cy)
            pts.append((tstar, ystar, kind))
            comps.append((cx, cy, kind, size, hole_ratio))
        elif bh <= max_cluster_h and bw <= max_cluster_w:
            # ---- merged marker cluster: split by distance-transform peaks ----
            filled = ndimage.binary_fill_holes(mask)
            dist = ndimage.distance_transform_edt(filled)
            mx = ndimage.maximum_filter(dist, size=7)
            peaks = (dist >= mx - 1e-9) & (dist > 2.5)
            plab, pn = ndimage.label(peaks)
            if pn == 0:
                continue
            cents = ndimage.center_of_mass(peaks, plab, index=range(1, pn + 1))
            # merge peak centres closer than 6 px
            kept = []
            for cy0, cx0 in sorted(cents):
                if all((cy0 - ky) ** 2 + (cx0 - kx) ** 2 > 36.0 for ky, kx in kept):
                    kept.append((cy0, cx0))
            for cy0, cx0 in kept:
                cy, cx = oy + cy0, ox + cx0
                kind = "int" if center_is_open(cy, cx) else "fs"
                tstar, ystar = to_data(cx, cy)
                pts.append((tstar, ystar, kind))
                comps.append((cx, cy, kind, size, -1.0))
        # larger blobs (axis smears etc.) are ignored

    # zero-line placeholders (pre-arrival / post-event zeros) are dropped
    pts = [p for p in pts if p[1] > 0.03]
    # physics-informed reclassification: the free surface starts near
    # Y*fs0 = WL/L and only rises, so every marker below reclass_int_below is
    # necessarily the air-water interface (markers half-cut by the axis crop
    # sometimes fail the hollow test).
    pts = [(t, y, ("int" if y < reclass_int_below else k)) for (t, y, k) in pts]
    return pts, comps, binm


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---------------- Fig. 6 : pressure head ----------------
    g6 = load_gray(FIG6)
    panels6, _ = find_panels(g6)
    draw_panel_debug(g6, panels6, OUT / "debug_fig6_panels.png")
    box6 = panels6[1][1]
    masks6 = label_boxes_in_panel(g6, box6)
    print("fig6 label masks:", masks6)
    T6, med6, lo6, hi6, bin6 = digitize_fig6(g6, box6, label_masks=masks6)
    with (OUT / "fig6_center_Hstar_band.csv").open("w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["Tstar", "Hstar_med", "Hstar_min", "Hstar_max"])
        for row in zip(T6, med6, lo6, hi6):
            wcsv.writerow([f"{v:.5f}" for v in row])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.imshow(bin6, cmap="gray_r")
    a1.set_title("Fig.6 center panel: extracted dark pixels (label box masked)")
    a2.fill_between(T6, lo6, hi6, color="#9ca3af", alpha=0.45, label="digitized min-max band")
    a2.plot(T6, med6, color="#111827", lw=1.2, label="digitized median")
    a2.set_xlim(0, 5); a2.set_ylim(0, 1.5); a2.grid(alpha=0.3)
    a2.set_xlabel("T*_ref"); a2.set_ylabel("H*")
    a2.legend(frameon=False, fontsize=8)
    a2.set_title("digitized H* (paper Fig.6, Ha0=0.610, WL=0.356)")
    fig.tight_layout(); fig.savefig(OUT / "debug_fig6_extract.png", dpi=140); plt.close(fig)

    # ---------------- Fig. 8 : levels ----------------
    g8 = load_gray(FIG8)
    panels8, _ = find_panels(g8)
    draw_panel_debug(g8, panels8, OUT / "debug_fig8_panels.png")
    box8 = panels8[1][1]
    masks8 = label_boxes_in_panel(g8, box8)
    print("fig8 label masks:", masks8)
    pts8, comps8, bin8 = digitize_fig8(g8, box8, label_masks=masks8)
    with (OUT / "fig8_center_levels.csv").open("w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["Tstar", "Ystar", "kind"])
        for tstar, ystar, kind in sorted(pts8):
            wcsv.writerow([f"{tstar:.5f}", f"{ystar:.5f}", kind])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.imshow(bin8, cmap="gray_r")
    for cx, cy, kind, size, hr in comps8:
        a1.plot(cx, cy, "o", ms=4, mfc="none",
                mec=("#ef4444" if kind == "int" else "#2b7fff"), mew=1.0)
    a1.set_title("Fig.8 center panel: components (blue=Y*fs filled/x, red=Y*int open)")
    for kind, color, mk, lbl in (("fs", "#2b7fff", "^", "Y*_fs digitized"),
                                 ("int", "#ef4444", "o", "Y*_int digitized")):
        xs = [p[0] for p in pts8 if p[2] == kind]
        ys = [p[1] for p in pts8 if p[2] == kind]
        a2.plot(xs, ys, mk, ms=4, mfc="none", mec=color, label=lbl)
    a2.set_xlim(3, 5); a2.set_ylim(0, 1.02); a2.grid(alpha=0.3)
    a2.set_xlabel("T*_ref"); a2.set_ylabel("Y*")
    a2.legend(frameon=False, fontsize=8)
    a2.set_title("digitized levels (paper Fig.8, Ha0=0.610, WL=0.356)")
    fig.tight_layout(); fig.savefig(OUT / "debug_fig8_extract.png", dpi=140); plt.close(fig)

    print(f"fig6 samples: {len(T6)}   fig8 markers: {len(pts8)} "
          f"(fs={sum(1 for p in pts8 if p[2]=='fs')}, int={sum(1 for p in pts8 if p[2]=='int')})")
    print(f"outputs -> {OUT}")


if __name__ == "__main__":
    main()

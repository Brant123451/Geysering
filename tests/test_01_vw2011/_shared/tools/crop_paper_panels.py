# -*- coding: utf-8 -*-
"""Save the original Fig.6 / Fig.8 CENTER panels (our case) as standalone PNGs,
with a small margin so axis labels stay visible. Used by report.html so the
reader can eyeball the digitization fidelity against the true scan."""
from pathlib import Path

import numpy as np
from PIL import Image

from digitize_paper_curves import FIG6, FIG8, load_gray, find_panels

HERE = Path(__file__).resolve().parent
OUT = HERE / "paper_reference" / "digitized"


def crop_center(src: Path, dst: Path, margin: int = 70):
    gray = load_gray(src)
    rows, _ = find_panels(gray)
    x0, x1, y0, y1 = rows[1][1]                  # center panel of the 3x3 grid
    H, W = gray.shape
    r0 = max(0, y0 - margin); r1 = min(H, y1 + margin)
    c0 = max(0, x0 - margin); c1 = min(W, x1 + margin)
    Image.fromarray(gray[r0:r1, c0:c1]).save(dst)
    print(f"{src.name} -> {dst.name}  ({r1-r0}x{c1-c0})")


if __name__ == "__main__":
    crop_center(FIG6, OUT / "fig6_center_panel.png")
    crop_center(FIG8, OUT / "fig8_center_panel.png")

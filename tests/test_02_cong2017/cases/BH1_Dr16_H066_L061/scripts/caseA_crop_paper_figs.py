# -*- coding: utf-8 -*-
"""Crop Fig.8 (B-H1 photo frames), Fig.9 (B-H1 riser data) and Fig.10
(pressure traces) out of the Cong (2017) PDF into paper_scans/."""
from pathlib import Path

import fitz


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "tests").is_dir() and (candidate / "references").is_dir():
            return candidate
    raise RuntimeError("repository root not found")


HERE = Path(__file__).resolve()
CASE_ROOT = HERE.parent.parent
REPO_ROOT = find_repo_root(CASE_ROOT)
PDF = REPO_ROOT / "references" / "cong2017.pdf"
OUT = CASE_ROOT / "reference" / "paper_scans"
OUT.mkdir(parents=True, exist_ok=True)

doc = fitz.open(PDF)

def crop(pno, caption, dst, pad_top=30, pad_side=28, pad_below=6, x1_frac=1.0):
    page = doc[pno]
    hits = page.search_for(caption)
    if not hits:
        print(f"caption {caption!r} not found on page {pno+1}")
        return
    cap = hits[0]
    w = page.rect.width
    clip = fitz.Rect(pad_side, pad_top, w * x1_frac - pad_side, cap.y1 + pad_below)
    pix = page.get_pixmap(dpi=220, clip=clip)
    pix.save(str(dst))
    print(f"{dst}  {pix.width}x{pix.height}")

# Fig.8: full-width photo strip on page 8 (caption at bottom)
crop(7, "Fig. 8.", OUT / "fig8_bh1_photos.png")
# Fig.9: 2x2 panel figure occupying the full width of page 9's upper half
crop(8, "Fig. 9.", OUT / "fig9_bh1_riser.png")
# Fig.10: left column of page 10 (pressure traces a/b)
crop(9, "Fig. 10.", OUT / "fig10_pressure.png", x1_frac=0.52)

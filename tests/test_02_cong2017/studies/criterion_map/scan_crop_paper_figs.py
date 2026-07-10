# -*- coding: utf-8 -*-
"""Crop the scan-relevant Cong (2017) figures into paper_scans/:
Fig.5 (vnet vs vTaylor, all runs), Fig.11 (geyser criterion plane),
Table 2 (Series B summary) and Fig.12 (mechanism sketches)."""
from pathlib import Path

import fitz


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "tests").is_dir() and (candidate / "references").is_dir():
            return candidate
    raise RuntimeError("repository root not found")


HERE = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(HERE)
PDF = REPO_ROOT / "references" / "cong2017.pdf"
OUT = HERE / "paper_scans"
OUT.mkdir(exist_ok=True)

doc = fitz.open(PDF)


def crop(pno, caption, dst, pad_top=30, pad_side=28, pad_below=6,
         x0_frac=0.0, x1_frac=1.0, y0=None):
    page = doc[pno]
    hits = page.search_for(caption)
    if not hits:
        print(f"caption {caption!r} not found on page {pno + 1}")
        return
    cap = hits[0]
    w = page.rect.width
    top = pad_top if y0 is None else y0
    clip = fitz.Rect(max(w * x0_frac, pad_side) if x0_frac > 0 else pad_side,
                     top, w * x1_frac - (pad_side if x1_frac == 1.0 else 0),
                     cap.y1 + pad_below)
    pix = page.get_pixmap(dpi=220, clip=clip)
    pix.save(str(dst))
    print(f"{dst}  {pix.width}x{pix.height}")


def crop_table(pno, header, dst, height=260, pad_side=28):
    page = doc[pno]
    hits = page.search_for(header)
    if not hits:
        print(f"header {header!r} not found on page {pno + 1}")
        return
    h = hits[0]
    w = page.rect.width
    clip = fitz.Rect(pad_side, max(h.y0 - 6, 0), w - pad_side, h.y1 + height)
    pix = page.get_pixmap(dpi=220, clip=clip)
    pix.save(str(dst))
    print(f"{dst}  {pix.width}x{pix.height}")


for pno in range(len(doc)):
    txt = doc[pno].get_text()
    for tag in ("Fig. 5.", "Fig. 11.", "Fig. 12.", "Table 2."):
        if tag in txt:
            print(f"{tag} on page {pno + 1}")

crop(5, "Fig. 5.", OUT / "fig5_vnet_vtaylor.png", x1_frac=0.54)
# Fig.11 sits in the RIGHT column of page 10, above its caption at y~381
crop(9, "Fig. 11.", OUT / "fig11_criterion.png", x0_frac=0.5, x1_frac=1.0,
     y0=30)
crop(10, "Fig. 12.", OUT / "fig12_mechanism.png")
crop_table(6, "Table 2.", OUT / "table2_seriesB.png", height=430)

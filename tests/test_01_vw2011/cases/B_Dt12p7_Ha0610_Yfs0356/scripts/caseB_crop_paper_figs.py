# -*- coding: utf-8 -*-
"""Crop Fig.3 (flow-evolution schematic) and Fig.11 (paper's own model vs
experiment, Dt*=0.135) out of the V&W(2011) PDF into paper_scans/."""
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
PDF = REPO_ROOT / "references" / "vasconcelos2011.pdf"
OUT = CASE_ROOT / "reference" / "paper_scans"
OUT.mkdir(parents=True, exist_ok=True)

doc = fitz.open(PDF)

def crop(pno, caption, dst, pad_top=18, pad_side=30, pad_below=6):
    page = doc[pno]
    hits = page.search_for(caption)
    if not hits:
        print(f"caption {caption!r} not found on page {pno+1}")
        return
    cap = hits[0]
    # figure spans from the top text margin down to just below the caption line
    clip = fitz.Rect(pad_side, pad_top, page.rect.width - pad_side, cap.y1 + pad_below)
    pix = page.get_pixmap(dpi=220, clip=clip)
    pix.save(str(dst))
    print(f"{dst}  {pix.width}x{pix.height}")

crop(2, "Fig. 3.", OUT / "fig3_schematic.png")
crop(10, "Fig. 11.", OUT / "fig11_full.png")

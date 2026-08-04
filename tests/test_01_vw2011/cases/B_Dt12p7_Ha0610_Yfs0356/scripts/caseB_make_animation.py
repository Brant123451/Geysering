# -*- coding: utf-8 -*-
"""Case B animation: the same two-fluid simulation rendered as a GIF
(horizontal tunnel + vertical tower, water blue / air white), using the
solver's own make_case_gif renderer.  Output: outputs/caseB_animation.gif
"""
from __future__ import annotations

import sys
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]
MODEL = CASE_ROOT / "model"
DIGITIZED = CASE_ROOT / "data" / "digitized"
SCANS = CASE_ROOT / "reference" / "paper_scans"
OUTPUTS = CASE_ROOT / "outputs"
sys.path.insert(0, str(MODEL))

from vw2011_network_twofluid import NetworkCase, run_network, make_case_gif

OUT = OUTPUTS
OUT.mkdir(exist_ok=True)


def main():
    case = NetworkCase(Dr=0.0127, air_head=0.610, init_water_level=0.356, t_end=9.0)
    rec = run_network(case, verbose=False)
    p = make_case_gif(case, rec, OUT, "B_tmp", fps=12, max_frames=110)
    target = OUT / "caseB_animation.gif"
    if p is not None:
        Path(p).replace(target)
        print(f"-> {target}")
    else:
        print("no frames recorded, gif not produced")


if __name__ == "__main__":
    main()

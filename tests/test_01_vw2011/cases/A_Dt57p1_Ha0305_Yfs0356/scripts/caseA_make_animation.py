# -*- coding: utf-8 -*-
"""Case A animation: the same two-fluid simulation rendered as a GIF
(horizontal tunnel + vertical tower, water blue / air white), using the
solver's own make_case_gif renderer.  Output: outputs/caseA_animation.gif
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
sys.path.insert(0, str(CASE / "model"))

from vw2011_network_twofluid import NetworkCase, run_network, make_case_gif

OUT = CASE / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    case = NetworkCase(
        Dr=0.0571,
        air_head=0.305,
        init_water_level=0.356,
        t_end=13.0,
    )
    rec = run_network(case, verbose=False)
    p = make_case_gif(case, rec, OUT, "A_tmp", fps=12, max_frames=150)
    target = OUT / "caseA_animation.gif"
    if p is not None:
        Path(p).replace(target)
        print(f"-> {target}")
    else:
        print("no frames recorded, gif not produced")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""B-H1 animation GIF via the solver's own make_case_gif renderer.
Output: outputs/caseA_animation.gif
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))

from cong2017_network_twofluid import NetworkCase, run_network, make_case_gif

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

CASE_KW = dict(
    D=0.05, Dr=0.016, riser_height=1.8,
    L_up=2.88, L_mid=2.51, L_down=0.61,
    x_riser_at=2.88,
    pocket_downstream=True,
    reservoir_head=0.66,
    air_head=0.0,
    init_water_level=0.66,
    Hop_cap=10.0,
    x_transducer_at=5.85,
)


def main():
    case = NetworkCase(**CASE_KW, t_end=13.0)
    rec = run_network(case, verbose=False)
    p = make_case_gif(case, rec, OUT, "BH1_tmp", fps=12, max_frames=110)
    target = OUT / "caseA_animation.gif"
    if p is not None:
        Path(p).replace(target)
        print(f"-> {target}")
    else:
        print("no frames recorded, gif not produced")


if __name__ == "__main__":
    main()

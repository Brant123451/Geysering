# -*- coding: utf-8 -*-
"""63-configuration criterion-map sweep with the frozen fully-synchronous solver.

Grid: Dr = 16/21/26/31/36/41/46 mm  x  L0 = 0.61/1.2/1.8 m  x  H0 = 0.66/0.77/0.88 m.
The model classifies each configuration blind; the paper criterion
(geyser iff Dr/D <= 0.62 and V*air >= 3.42) is evaluated for comparison.

Writes outputs/criterion_scan_fullsync.csv incrementally (resumable)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from scan_common import CSV_FIELDS, run_one

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

DR_MM = [16, 21, 26, 31, 36, 41, 46]
L0S = [0.61, 1.2, 1.8]
H0S = [0.66, 0.77, 0.88]
FIELDS = CSV_FIELDS + ["criterion_geyser"]

# optional shard: pass an L0 value (e.g. "0.61") to run only that group,
# writing to its own CSV so three workers can run in parallel.
if len(sys.argv) > 1:
    L0S = [float(sys.argv[1])]
    tag = f"_L0p{str(L0S[0]).replace('.', '')}"
else:
    tag = ""
CSV_PATH = OUT / f"criterion_scan_fullsync{tag}.csv"


def paper_criterion(Dr_over_D: float, Vair_star: float) -> int:
    return int(Dr_over_D <= 0.62 and Vair_star >= 3.42)


def key(row: dict) -> tuple:
    return (float(row["Dr_mm"]), float(row["L0_m"]), float(row["H0_m"]))


def load_done() -> set[tuple]:
    if not CSV_PATH.exists():
        return set()
    with CSV_PATH.open() as f:
        return {key(r) for r in csv.DictReader(f) if not r.get("error")}


def main() -> None:
    done = load_done()
    todo = [(dr, l0, h0) for dr in DR_MM for l0 in L0S for h0 in H0S
            if (float(dr), l0, h0) not in done]
    print(f"{len(done)} done, {len(todo)} to go")
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, (dr, l0, h0) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] Dr={dr} mm L0={l0} H0={h0} ...", flush=True)
            row = run_one(dr, l0, h0)
            row["criterion_geyser"] = paper_criterion(row["Dr_over_D"],
                                                      row.get("Vair_star", 0.0))
            w.writerow(row)
            f.flush()
            agree = ("=" if row.get("geyser_model") is not None
                     and int(row["geyser_model"]) == row["criterion_geyser"]
                     else "!")
            print(f"    model={row.get('geyser_model')} criterion="
                  f"{row['criterion_geyser']} [{agree}] Yfs_max={row.get('Yfs_max_m')}"
                  f" [{row['runtime_s']} s]", flush=True)
    print("criterion sweep complete ->", CSV_PATH)


if __name__ == "__main__":
    main()

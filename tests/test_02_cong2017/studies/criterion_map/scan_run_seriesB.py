# -*- coding: utf-8 -*-
"""Rerun the Series-B high-speed-camera set (B-H1..B-H7) with the frozen
fully-synchronous solver.  Fixed H0=0.66 m, L0=0.61 m; only Dr varies.

Writes outputs/seriesB_fullsync.csv incrementally (resumable)."""
from __future__ import annotations

import csv
from pathlib import Path

from scan_common import CSV_FIELDS, run_one

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
CSV_PATH = OUT / "seriesB_fullsync.csv"

# Table 2 measured values (Series B high-speed camera runs)
MEASURED = {
    "B-H1": dict(Dr_mm=16, Ta=8.07, v_fs=0.924, v_int=1.231, geyser=True),
    "B-H2": dict(Dr_mm=21, Ta=7.84, v_fs=0.768, v_int=1.022, geyser=True),
    "B-H3": dict(Dr_mm=26, Ta=8.18, v_fs=0.657, v_int=0.916, geyser=True),
    "B-H4": dict(Dr_mm=31, Ta=8.14, v_fs=0.207, v_int=0.418, geyser=False),
    "B-H5": dict(Dr_mm=36, Ta=8.10, v_fs=0.261, v_int=0.481, geyser=False),
    "B-H6": dict(Dr_mm=41, Ta=8.10, v_fs=0.246, v_int=0.476, geyser=False),
    "B-H7": dict(Dr_mm=46, Ta=8.22, v_fs=0.203, v_int=0.441, geyser=False),
}
FIELDS = ["run"] + CSV_FIELDS + ["Ta_meas_s", "vfs_meas", "vint_meas",
                                 "geyser_meas", "match"]


def load_done() -> set[str]:
    if not CSV_PATH.exists():
        return set()
    with CSV_PATH.open() as f:
        return {r["run"] for r in csv.DictReader(f) if not r.get("error")}


def main() -> None:
    done = load_done()
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for run, ms in MEASURED.items():
            if run in done:
                print(f"[skip] {run} already done")
                continue
            print(f"[run ] {run}  Dr={ms['Dr_mm']} mm ...", flush=True)
            row = run_one(ms["Dr_mm"], 0.61, 0.66)
            row["run"] = run
            row["Ta_meas_s"] = ms["Ta"]
            row["vfs_meas"] = ms["v_fs"]
            row["vint_meas"] = ms["v_int"]
            row["geyser_meas"] = int(ms["geyser"])
            row["match"] = ("OK" if row.get("geyser_model") is not None
                            and int(row["geyser_model"]) == int(ms["geyser"])
                            else "MISMATCH")
            w.writerow(row)
            f.flush()
            print(f"       geyser={row.get('geyser_model')} ({row['match']}) "
                  f"Ta={row.get('Ta_model_s')} Yfs_max={row.get('Yfs_max_m')} "
                  f"[{row['runtime_s']} s]", flush=True)
    print("seriesB sweep complete ->", CSV_PATH)


if __name__ == "__main__":
    main()

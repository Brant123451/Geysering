#!/usr/bin/env python3
"""Estimate remaining wall time for the 0.12 s screen from the solver log."""
from __future__ import annotations

import re
import time
from pathlib import Path

log = Path(__file__).resolve().parent / "log.compressibleInterFlow"
target = 0.12
pat = re.compile(r"^Time = ([0-9.eE+-]+)\s*$")

if not log.exists():
    print("NO_LOG")
    raise SystemExit(1)

text = log.read_text(errors="replace").splitlines()
times = [float(m.group(1)) for line in text if (m := pat.match(line))]
mtime = log.stat().st_mtime
# Prefer process start from first Time stamp age: use file mtime as "now"
# and estimate elapsed from number of steps is weak; use wall clock of log growth.
# Better: parse ExecutionTime if present.
exec_pat = re.compile(r"ExecutionTime = ([0-9.]+) s")
execs = [float(m.group(1)) for line in text if (m := exec_pat.search(line))]

latest = times[-1] if times else 0.0
n = len(times)
print(f"samples={n}")
print(f"t_phys={latest:.6g} / {target}")
print(f"progress={100.0 * latest / target:.2f}%")

if latest <= 0 or n < 5:
    print("ETA=insufficient_data_early_startup")
    raise SystemExit(0)

# Use last ~20% of samples for rate if available
k = max(5, n // 5)
t0, t1 = times[-k], times[-1]
if execs and len(execs) >= k:
    e0, e1 = execs[-k], execs[-1]
    wall_dt = max(e1 - e0, 1e-9)
    rate = (t1 - t0) / wall_dt  # phys seconds per wall second
    source = "ExecutionTime"
elif execs:
    wall_dt = max(execs[-1], 1e-9)
    rate = latest / wall_dt
    source = "ExecutionTime_from_start"
else:
    # fall back: assume solver has been running since log mtime minus a guess
    print("ETA=no_ExecutionTime_use_prior_4rank_calibration")
    # prior 4-rank: 0.015 phys in 1920 wall-s => 7.8e-6 phys/wall-s
    rate = 0.015 / 1920.0
    source = "prior_4rank"

remain_phys = max(target - latest, 0.0)
if rate <= 0:
    print("ETA=nonpositive_rate")
    raise SystemExit(0)
remain_wall_s = remain_phys / rate
total_wall_s = target / rate
print(f"rate_source={source}")
print(f"rate_phys_per_wall_s={rate:.6g}")
print(f"remain_wall_h={remain_wall_s / 3600.0:.2f}")
print(f"total_wall_h_from_now_plus_done≈{(remain_wall_s)/3600.0:.2f} remaining")
print(f"projected_total_wall_h={total_wall_s / 3600.0:.2f}")
if execs:
    print(f"elapsed_exec_s={execs[-1]:.1f}")

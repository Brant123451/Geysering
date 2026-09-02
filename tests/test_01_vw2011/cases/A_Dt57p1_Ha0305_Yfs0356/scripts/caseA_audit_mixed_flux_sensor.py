#!/usr/bin/env python3
"""Report HLLC admissibility indicators on Case-A mixed-topology faces."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from casea_horizontal_liquid_operator import pressure_potential_wave_state
from vw2011_network_twofluid import (
    G,
    P_ATM,
    R_GAS,
    T_GAS,
    _connected_shallow_water_potential_offsets,
    _horizontal_liquid_parameters_cached,
    _mass_backed_gas_topology,
    _tpa_muscl_faces,
)


def face_indicators(area, discharge, gas_mass, *, diameter, dx, wave_speed):
    area_full = math.pi * diameter**2 / 4.0
    al, ql, ar, qr = _tpa_muscl_faces(area, discharge, area_full, diameter)
    void = np.maximum(area_full - np.clip(area, 0.0, area_full), 0.0)
    support = _mass_backed_gas_topology(
        void,
        gas_mass,
        full_area=area_full,
        cell_width=dx,
        rho_reference=P_ATM / (R_GAS * T_GAS),
        void_floor_fraction=1.0e-4,
        active_void_fraction=5.0e-4,
        topology_density_fraction=0.02,
        resolved_density_fraction=0.50,
    )
    support &= void >= 5.0e-4 * area_full
    support_g = np.concatenate(([support[0]], support, [support[-1]]))
    press_l = ~support_g[:-1]
    press_r = ~support_g[1:]
    mixed = press_l ^ press_r

    area_g = np.concatenate(([area[0]], area, [area[-1]]))
    discharge_g = np.concatenate(([-discharge[0]], discharge, [-discharge[-1]]))
    al[mixed] = area_g[:-1][mixed]
    ar[mixed] = area_g[1:][mixed]
    ql[mixed] = discharge_g[:-1][mixed]
    qr[mixed] = discharge_g[1:][mixed]

    dry = 1.0e-9 * area_full
    ale = np.maximum(al, dry)
    are = np.maximum(ar, dry)
    qle = np.where(al > dry, ql, 0.0)
    qre = np.where(ar > dry, qr, 0.0)
    params = _horizontal_liquid_parameters_cached(
        area_full, diameter, wave_speed, dx, 0.05
    )
    offsets = _connected_shallow_water_potential_offsets(area, support, params)
    offsets_g = np.concatenate(([offsets[0]], offsets, [offsets[-1]]))
    pl = pressure_potential_wave_state(
        ale, ~press_l, params, stratified_potential_offset=offsets_g[:-1]
    )
    pr = pressure_potential_wave_state(
        are, ~press_r, params, stratified_potential_offset=offsets_g[1:]
    )
    ul = np.where(al > dry, qle / ale, 0.0)
    ur = np.where(ar > dry, qre / are, 0.0)
    sl = np.minimum(ul - pl.celerity, ur - pr.celerity)
    sr = np.maximum(ul + pl.celerity, ur + pr.celerity)
    span = np.maximum(sr - sl, 1.0e-12)
    denom = al * (sl - ul) - ar * (sr - ur)
    denom = np.where(np.abs(denom) > 1.0e-14, denom, -1.0e-14)
    sm = (
        pr.potential
        - pl.potential
        + al * ul * (sl - ul)
        - ar * ur * (sr - ur)
    ) / denom
    sm = np.clip(sm, sl, sr)
    dl = np.where(np.abs(sl - sm) > 1.0e-14, sl - sm, -1.0e-14)
    dr = np.where(np.abs(sr - sm) > 1.0e-14, sr - sm, 1.0e-14)
    astar_l = np.maximum(al * (sl - ul) / dl, 0.0)
    astar_r = np.maximum(ar * (sr - ur) / dr, 0.0)
    scale = np.maximum.reduce((al, ar, np.full_like(al, dry)))
    amin = np.maximum(np.minimum(al, ar), dry)
    area_jump = np.abs(ar - al) / np.maximum(ar + al, dry)
    velocity_jump = np.abs(ur - ul) / np.maximum(
        pl.celerity + pr.celerity, 1.0e-12
    )
    star_ratio = np.maximum(astar_l, astar_r) / scale
    star_low_ratio = np.minimum(astar_l, astar_r) / amin
    contact_margin = np.minimum(np.abs(sm - sl), np.abs(sr - sm)) / span
    return {
        "mixed": mixed,
        "al": al,
        "ar": ar,
        "ul": ul,
        "ur": ur,
        "cl": pl.celerity,
        "cr": pr.celerity,
        "sm": sm,
        "area_jump": area_jump,
        "velocity_jump": velocity_jump,
        "star_ratio": star_ratio,
        "star_low_ratio": star_low_ratio,
        "contact_margin": contact_margin,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--t-min", type=float, default=6.3)
    parser.add_argument("--t-max", type=float, default=9.0)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    saved = np.load(args.archive)
    diameter = 0.094
    area_full = math.pi * diameter**2 / 4.0
    rows = []
    for k, time in enumerate(saved["time"]):
        if time < args.t_min or time > args.t_max:
            continue
        fields = face_indicators(
            saved["horizontal_alpha_l_raw"][k] * area_full,
            saved["horizontal_liquid_discharge"][k],
            saved["horizontal_gas_mass"][k],
            diameter=diameter,
            dx=4.0 / saved["horizontal_alpha_l"].shape[1],
            wave_speed=28.0,
        )
        for face in np.flatnonzero(fields["mixed"]):
            rows.append(
                (
                    max(fields["area_jump"][face], fields["velocity_jump"][face]),
                    float(time),
                    int(face),
                    *(float(fields[name][face]) for name in (
                        "al", "ar", "ul", "ur", "cl", "cr", "sm",
                        "area_jump", "velocity_jump", "star_ratio",
                        "star_low_ratio", "contact_margin",
                    )),
                )
            )
    rows.sort(reverse=True)
    print(
        "sensor time face al/A ar/A ul ur cl cr sm areaJump velJump "
        "starMax starMin contactMargin"
    )
    for row in rows[: args.top]:
        sensor, time, face, al, ar, *rest = row
        print(
            f"{sensor:.6g} {time:.9f} {face:d} {al/area_full:.6g} "
            f"{ar/area_full:.6g} " + " ".join(f"{value:.6g}" for value in rest)
        )


if __name__ == "__main__":
    main()

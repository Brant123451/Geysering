"""Evaluate the Case-A frame-187 horizontal interface against the IKH criterion.

The input is the ASCII VTU exported by ``openfoam/2d/export_kh_frame.sh`` at
Time = 9.35 s.  OpenFOAM cell velocities are phase-volume weighted in short
horizontal bins.  The resulting liquid holdup is mapped to the physical
circular pipe before evaluating the inviscid criterion used by the companion
decoupled two-fluid model.

This is a diagnostic, not a proof that every visible wrinkle is Kelvin--
Helmholtz instability.  In particular, the OpenFOAM calculation is planar 2D
whereas the criterion uses the experimental circular section.
"""

from __future__ import annotations

import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CASE = HERE.parent
VTU = CASE / "openfoam/2d/VTK_KH_935/2d_20424/internal.vtu"
CSV_OUT = CASE / "outputs/caseA_frame187_kh_bins.csv"
JSON_OUT = CASE / "outputs/caseA_frame187_kh_summary.json"

D = 0.094
AF = math.pi * D * D / 4.0
RHO_L = 1000.0
R_GAS = 287.0
T_GAS = 293.15
P_ATM = 101325.0
G = 9.80665
X_MIN = 2.45
X_MAX = 3.45
BIN_WIDTH = 0.05


def _array(node: ET.Element, ncomp: int = 1) -> np.ndarray:
    if node.attrib.get("format") != "ascii":
        raise ValueError("Run export_kh_frame.sh with foamToVTK -ascii first")
    values = np.fromstring(node.text or "", sep=" ")
    return values.reshape(-1, ncomp) if ncomp > 1 else values


def _cell_geometry(points: np.ndarray, cells: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    centres = np.empty((len(cells), 2))
    areas = np.empty(len(cells))
    for i, cell in enumerate(cells):
        xy = np.unique(points[cell, :2], axis=0)
        centre = xy.mean(axis=0)
        order = np.argsort(np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0]))
        poly = xy[order]
        x = poly[:, 0]
        y = poly[:, 1]
        centres[i] = centre
        areas[i] = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return centres, areas


def _gamma(alpha_l: float) -> float:
    lo, hi = 1.0e-10, 2.0 * math.pi - 1.0e-10
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fraction = (mid - math.sin(mid)) / (2.0 * math.pi)
        if fraction < alpha_l:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    root = ET.parse(VTU).getroot()
    points = _array(root.find(".//Points/DataArray"), 3)
    arrays = {node.attrib.get("Name"): node for node in root.findall(".//Cells/DataArray")}
    connectivity = _array(arrays["connectivity"]).astype(int)
    offsets = _array(arrays["offsets"]).astype(int)
    cells = [connectivity[i:j] for i, j in zip(np.r_[0, offsets[:-1]], offsets)]
    cell_data = {node.attrib.get("Name"): node for node in root.findall(".//CellData/DataArray")}
    alpha = np.clip(_array(cell_data["alpha.water"]), 0.0, 1.0)
    pressure = _array(cell_data["p"])
    velocity = _array(cell_data["U"], 3)
    centres, areas = _cell_geometry(points, cells)

    pipe = (centres[:, 1] >= -1.0e-10) & (centres[:, 1] <= D + 1.0e-10)
    edges = np.arange(X_MIN, X_MAX + 0.5 * BIN_WIDTH, BIN_WIDTH)
    rows: list[dict[str, float]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        use = pipe & (centres[:, 0] >= left) & (centres[:, 0] < right)
        if not np.any(use):
            continue
        w = areas[use]
        aw = alpha[use]
        gw = 1.0 - aw
        liquid_measure = float(np.sum(w * aw))
        gas_measure = float(np.sum(w * gw))
        if liquid_measure <= 1.0e-14 or gas_measure <= 1.0e-14:
            continue
        alpha_l = float(liquid_measure / np.sum(w))
        u_l = float(np.sum(w * aw * velocity[use, 0]) / liquid_measure)
        u_g = float(np.sum(w * gw * velocity[use, 0]) / gas_measure)
        p_g = float(np.sum(w * gw * pressure[use]) / gas_measure)
        rho_g = max(p_g / (R_GAS * T_GAS), 1.0e-6)
        a_l = min(max(alpha_l * AF, 1.0e-10), AF - 1.0e-10)
        a_g = AF - a_l
        gamma = _gamma(alpha_l)
        zeta = 1.0 / max(D * math.sin(0.5 * gamma), 1.0e-10)
        h_g = (p_g - P_ATM) / (RHO_L * G)
        slip = abs(u_g - u_l)
        lambda_d = (
            2.0 * G * h_g / a_l
            + (RHO_L - rho_g) / RHO_L * G * zeta
            - rho_g / RHO_L * slip * slip / a_g
        )
        critical_sq = (
            RHO_L / rho_g
            * a_g
            * (2.0 * G * h_g / a_l + (RHO_L - rho_g) / RHO_L * G * zeta)
        )
        critical = math.sqrt(critical_sq) if critical_sq > 0.0 else 0.0
        lambda_no_pressure = (
            (RHO_L - rho_g) / RHO_L * G * zeta
            - rho_g / RHO_L * slip * slip / a_g
        )
        critical_no_pressure_sq = (
            RHO_L / rho_g
            * a_g
            * ((RHO_L - rho_g) / RHO_L * G * zeta)
        )
        critical_no_pressure = (
            math.sqrt(critical_no_pressure_sq)
            if critical_no_pressure_sq > 0.0
            else 0.0
        )
        rows.append(
            {
                "x_m": 0.5 * (left + right),
                "alpha_l": alpha_l,
                "u_l_m_s": u_l,
                "u_g_m_s": u_g,
                "slip_m_s": slip,
                "p_g_Pa": p_g,
                "rho_g_kg_m3": rho_g,
                "H_g_m": h_g,
                "critical_slip_m_s": critical,
                "slip_ratio": slip / critical if critical > 0.0 else math.inf,
                "Lambda_d_1_m_s2": lambda_d,
                "critical_slip_no_pressure_m_s": critical_no_pressure,
                "slip_ratio_no_pressure": (
                    slip / critical_no_pressure
                    if critical_no_pressure > 0.0
                    else math.inf
                ),
                "Lambda_d_no_pressure_1_m_s2": lambda_no_pressure,
            }
        )

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "time_s": 9.35,
        "x_interval_m": [X_MIN, X_MAX],
        "bin_width_m": BIN_WIDTH,
        "mapping": "planar OpenFOAM phase velocities; circular-pipe IKH geometry",
        "max_slip_m_s": max(row["slip_m_s"] for row in rows),
        "max_slip_ratio": max(row["slip_ratio"] for row in rows),
        "min_lambda_d_1_m_s2": min(row["Lambda_d_1_m_s2"] for row in rows),
        "unstable_bin_count": sum(row["Lambda_d_1_m_s2"] < 0.0 for row in rows),
        "max_slip_ratio_no_pressure": max(
            row["slip_ratio_no_pressure"] for row in rows
        ),
        "min_lambda_d_no_pressure_1_m_s2": min(
            row["Lambda_d_no_pressure_1_m_s2"] for row in rows
        ),
        "unstable_bin_count_no_pressure": sum(
            row["Lambda_d_no_pressure_1_m_s2"] < 0.0 for row in rows
        ),
        "bin_count": len(rows),
    }
    JSON_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(JSON_OUT)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

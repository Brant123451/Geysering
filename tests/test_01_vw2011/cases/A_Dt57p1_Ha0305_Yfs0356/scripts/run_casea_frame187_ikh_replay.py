"""Controlled frame-187 IKH growth audit with the companion-paper model."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
CASE = HERE.parent
sys.path.insert(0, str(CASE / "model"))

from casea_decoupled_ikh_model import (  # noqa: E402
    ModelParameters,
    advance_ssprk2,
    conserved_to_primitives,
    primitives_to_conserved,
    restoring_coefficient,
    stable_time_step,
)


BIN_CSV = CASE / "outputs/caseA_frame187_kh_bins.csv"
OUT_JSON = CASE / "outputs/caseA_frame187_ikh_replay.json"
OUT_CSV = CASE / "outputs/caseA_frame187_ikh_growth.csv"
OUT_PNG = CASE / "outputs/caseA_frame187_ikh_replay.png"
OUT_PDF = CASE / "outputs/caseA_frame187_ikh_replay.pdf"
PROVENANCE = CASE / "model/casea_decoupled_ikh_provenance.json"
SOURCE_FILES = (
    Path(r"E:/Research/论文/my sci/appendix_b_current_model_en.tex"),
    Path(r"E:/Research/论文/my sci/main_text_current_algorithm.tex"),
    Path(r"E:/Research/论文/my sci/decoupled_model_dispersion_5p1.py"),
    Path(r"E:/Research/The lase case/paper_solver_copy/run_own_model_slug_5_3.py"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_state() -> dict[str, float]:
    with BIN_CSV.open(encoding="utf-8", newline="") as handle:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    # Use the cell closest to loss of hyperbolicity, not the one with merely the
    # largest dimensional slip.  This is the most conservative KH audit.
    return max(rows, key=lambda row: row["slip_ratio"])


def _run(
    *,
    label: str,
    state: dict[str, float],
    velocity_g: float,
    duration: float = 0.12,
    n_cells: int = 160,
    length: float = 1.0,
    perturbation_fraction: float = 2.0e-3,
) -> dict:
    params = ModelParameters()
    dx = length / n_cells
    x = (np.arange(n_cells) + 0.5) * dx
    area_l_base = state["alpha_l"] * params.area_full
    wavelength = 0.25
    area_l = area_l_base * (
        1.0 + perturbation_fraction * np.sin(2.0 * math.pi * x / wavelength)
    )
    density_g = np.full(
        n_cells,
        state["p_g_Pa"] / (params.gas_constant * params.gas_temperature),
    )
    velocity_l = np.full(n_cells, state["u_l_m_s"])
    velocity_g_field = np.full(n_cells, velocity_g)
    conserved = primitives_to_conserved(
        density_g, velocity_g_field, area_l, velocity_l, params
    )

    initial_liquid = float(np.sum(conserved[2]) * dx)
    initial_gas = float(np.sum(conserved[0]) * dx)
    base_amplitude = float(np.std(area_l / params.area_full))
    times = [0.0]
    amplitudes = [base_amplitude]
    minimum_lambda = []
    snapshots = [(0.0, (area_l / params.area_full).copy())]
    time = 0.0
    next_output = 0.002
    steps = 0
    termination_reason = "requested duration reached"
    while time < duration - 1.0e-14:
        dt = min(stable_time_step(conserved, dx, 0.38, params), duration - time)
        previous = conserved
        trial = advance_ssprk2(conserved, dt, dx, params)
        if not np.all(np.isfinite(trial)):
            conserved = previous
            termination_reason = "stopped before non-finite elliptic blow-up"
            break
        conserved = trial
        time += dt
        steps += 1
        if time >= next_output - 1.0e-14 or time >= duration - 1.0e-14:
            rho_g, ug, al, ul = conserved_to_primitives(conserved, params)
            lam = restoring_coefficient(al, ul, rho_g, ug, params)
            amplitude = float(np.std(al / params.area_full))
            times.append(float(time))
            amplitudes.append(amplitude)
            minimum_lambda.append(float(np.min(lam)))
            if len(snapshots) < 7:
                target = len(snapshots) * duration / 6.0
                if time >= target - 0.5 * next_output:
                    snapshots.append((float(time), (al / params.area_full).copy()))
            next_output += 0.002
            if amplitude / max(base_amplitude, 1.0e-30) >= 10.0:
                termination_reason = "nonlinear-amplitude threshold reached"
                break

    rho_g, ug, al, ul = conserved_to_primitives(conserved, params)
    final_liquid = float(np.sum(al) * dx)
    final_gas = float(np.sum((params.area_full - al) * rho_g) * dx)
    amplification = float(amplitudes[-1] / max(amplitudes[0], 1.0e-30))
    return {
        "label": label,
        "x": x,
        "time": np.asarray(times),
        "amplitude": np.asarray(amplitudes),
        "snapshots": snapshots,
        "steps": steps,
        "end_time_s": float(time),
        "termination_reason": termination_reason,
        "velocity_g_m_s": float(velocity_g),
        "initial_amplitude": amplitudes[0],
        "final_amplitude": amplitudes[-1],
        "amplification": amplification,
        "minimum_lambda": float(min(minimum_lambda)) if minimum_lambda else math.nan,
        "liquid_relative_balance_error": (final_liquid - initial_liquid) / initial_liquid,
        "gas_relative_balance_error": (final_gas - initial_gas) / initial_gas,
    }


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state = _selected_state()
    actual = _run(label="Frame 187 state", state=state, velocity_g=state["u_g_m_s"])
    direction = 1.0 if state["u_g_m_s"] >= state["u_l_m_s"] else -1.0
    supercritical_velocity = state["u_l_m_s"] + direction * 1.10 * state["critical_slip_m_s"]
    control = _run(
        label="1.10 × critical-slip control",
        state=state,
        velocity_g=supercritical_velocity,
    )

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "time_s", "holdup_std", "normalized_amplitude"])
        for result in (actual, control):
            for time, amplitude in zip(result["time"], result["amplitude"]):
                writer.writerow(
                    [result["label"], time, amplitude, amplitude / result["initial_amplitude"]]
                )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
    colors = ("#202020", "#c73e1d")
    for result, color in zip((actual, control), colors):
        axes[0].plot(
            result["time"],
            result["amplitude"] / result["initial_amplitude"],
            color=color,
            lw=1.5,
            label=result["label"],
        )
    axes[0].axhline(1.0, color="#777777", lw=0.7, ls=":")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Normalized interface amplitude")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].tick_params(direction="in", top=True, right=True)

    for result, color in zip((actual, control), colors):
        final_time, final_profile = result["snapshots"][-1]
        axes[1].plot(
            result["x"],
            final_profile,
            color=color,
            lw=1.3,
            label=f"{result['label']} ({final_time:.2f} s)",
        )
    axes[1].axhline(state["alpha_l"], color="#777777", lw=0.7, ls=":")
    axes[1].set_xlabel("Local horizontal coordinate (m)")
    axes[1].set_ylabel(r"Liquid holdup $A_l/A_f$")
    axes[1].tick_params(direction="in", top=True, right=True)
    axes[1].legend(frameon=False, fontsize=7.3)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    public_results = []
    for result in (actual, control):
        public_results.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"x", "time", "amplitude", "snapshots"}
            }
        )
    summary = {
        "purpose": "Controlled local IKH growth audit; not a coupled Case-A rerun",
        "frame": 187,
        "time_s": 9.35,
        "selected_x_m": state["x_m"],
        "selected_state": state,
        "results": public_results,
        "interpretation_rule": (
            "Exponential perturbation growth is required for an IKH attribution; "
            "a visible but bounded travelling wave is not sufficient."
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    provenance = {
        "local_module": str((CASE / "model/casea_decoupled_ikh_model.py").resolve()),
        "scope": "Case-A local stratified-branch IKH audit",
        "source_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in SOURCE_FILES
        ],
        "equations": ["A16", "A17", "A30", "A31", "31", "32"],
        "known_boundary": (
            "The full pressurised cut-cell/T-junction solver is not present at the "
            "path referenced by the companion manuscript generator; this module "
            "therefore does not claim a coupled-network replacement."
        ),
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(OUT_JSON)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

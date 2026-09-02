"""Immutable Campaign-2 H1/H3/H6 qualification contract.

The solver never receives the experimental outcome.  That table is retained
only for post-run validation.  All apparatus and closure fields are shared;
the riser diameter is the sole solver input that differs among the three runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ApparatusContract:
    """Paper geometry and the material/ambient contract shared with 2D."""

    pipe_diameter_m: float = 0.050
    tunnel_length_m: float = 6.590
    riser_x_m: float = 3.470
    valve_x_m: float = 5.980
    riser_height_m: float = 1.800
    initial_head_from_invert_m: float = 0.660
    initial_head_from_crown_m: float = 0.610
    liquid_density_kg_m3: float = 998.0
    liquid_dynamic_viscosity_Pa_s: float = 0.001003
    liquid_bulk_modulus_Pa: float = 2.2e9
    atmospheric_pressure_Pa: float = 101_325.0
    gravity_m_s2: float = 9.81
    gas_constant_J_kg_K: float = 287.05
    gas_temperature_K: float = 296.15
    air_molar_mass_kg_kmol: float = 28.965
    air_dynamic_viscosity_Pa_s: float = 1.81e-5
    surface_tension_N_m: float = 0.072
    valve_open_time_s: float = 0.200
    valve_effective_area_law: str = "sin(pi*t/(2*t_open))^2"
    one_d_valve_implementation: str = (
        "global hydraulic-time equivalent during opening; "
        "not yet a local Forchheimer face boundary"
    )


@dataclass(frozen=True)
class SharedOneDimensionalClosure:
    wave_speed_m_s: float = 28.0
    gas_drive_efficiency: float = 1.0
    entry_drive_efficiency: float = 1.0
    gas_escape_efficiency: float = 1.0
    case1_handoff_event: str = "riser_arrival"
    top_liquid_outflow_tolerance_m3: float = 1.0e-9


@dataclass(frozen=True)
class QualificationCase:
    case_id: str
    paper_run: str
    riser_diameter_m: float


APPARATUS = ApparatusContract()
SHARED_CLOSURE = SharedOneDimensionalClosure()
QUALIFICATION_CASES = (
    QualificationCase("BH1", "B-H1", 0.016),
    QualificationCase("BH3", "B-H3", 0.026),
    QualificationCase("BH6", "B-H6", 0.041),
)

# Validation-only evidence.  Do not pass this mapping into a solver call.
EXPERIMENT_GEYSER = {"BH1": True, "BH3": True, "BH6": False}


def solver_contract(case: QualificationCase) -> dict[str, object]:
    """Return a solver-facing contract with no target outcome field."""

    return {
        "apparatus": asdict(APPARATUS),
        "shared_closure": asdict(SHARED_CLOSURE),
        "case": {
            "case_id": case.case_id,
            "paper_run": case.paper_run,
            "riser_diameter_m": case.riser_diameter_m,
        },
    }


def shared_solver_signature(case: QualificationCase) -> dict[str, object]:
    """Fields that must be exactly identical across H1/H3/H6."""

    payload = solver_contract(case)
    return {
        "apparatus": payload["apparatus"],
        "shared_closure": payload["shared_closure"],
    }


__all__ = [
    "APPARATUS",
    "EXPERIMENT_GEYSER",
    "QUALIFICATION_CASES",
    "SHARED_CLOSURE",
    "ApparatusContract",
    "QualificationCase",
    "SharedOneDimensionalClosure",
    "shared_solver_signature",
    "solver_contract",
]

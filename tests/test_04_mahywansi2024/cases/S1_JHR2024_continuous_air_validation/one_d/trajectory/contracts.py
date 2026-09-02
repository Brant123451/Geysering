"""Frozen contracts for S1 canonical one-dimensional trajectory evidence.

The objects in this module contain diagnostics that an accepted physical step
must expose to the observer.  They are intentionally separate from the
horizontal and riser operators: the observer never invents a pressure, gross
boundary flux, zero-storage-node residual, or eruption flag from a rendered
or reduced proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from model.conservation import LedgerEntry
from model.errors import ContractViolation
from model.state import CoupledState


HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT_PATH = HERE.parent / "config" / "COMMON_OBSERVABLES.yaml"

ArtifactRole = Literal["formal_production", "synthetic_contract_test"]

PUBLISHED_PRESSURE_TO_CANONICAL = {
    "P1": "P1_gauge_Pa",
    "P2": "P2_gauge_Pa",
    "P3": "P3_gauge_Pa",
    "P4": "H_upstream_gauge_Pa",
    "P5": "riser_left_gauge_Pa",
    "P6": "riser_right_gauge_Pa",
}


class ObservationContractError(ContractViolation):
    """Raised when a trajectory cannot support a canonical observable."""


def _optional_finite(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ObservationContractError(f"{name} must be finite when supplied")
    return result


def _optional_nonnegative(name: str, value: float | None) -> float | None:
    result = _optional_finite(name, value)
    if result is not None and result < 0.0:
        raise ObservationContractError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class GaugePressurePacket:
    """Published P1--P6 gauge pressures at one accepted physical state."""

    P1: float | None = None
    P2: float | None = None
    P3: float | None = None
    P4: float | None = None
    P5: float | None = None
    P6: float | None = None

    def __post_init__(self) -> None:
        for name in PUBLISHED_PRESSURE_TO_CANONICAL:
            object.__setattr__(
                self, name, _optional_finite(f"{name} gauge pressure", getattr(self, name))
            )

    def canonical(self) -> dict[str, float | None]:
        return {
            canonical: getattr(self, published)
            for published, canonical in PUBLISHED_PRESSURE_TO_CANONICAL.items()
        }


@dataclass(frozen=True, slots=True)
class AcceptedGrossFluxPacket:
    """Native accepted gross phase rates, never reconstructed from net flow."""

    supply_branch_liquid_outflow_m3_s: float | None = None
    supply_branch_gas_inflow_kg_s: float | None = None
    mouth_liquid_outflow_m3_s: float | None = None
    mouth_liquid_inflow_m3_s: float | None = None
    mouth_gas_outflow_kg_s: float | None = None
    mouth_gas_inflow_kg_s: float | None = None
    cumulative_mouth_liquid_outflow_m3: float | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _optional_nonnegative(name, getattr(self, name))
            )


@dataclass(frozen=True, slots=True)
class AcceptedNodePacket:
    """Accepted zero-storage-node residuals and reaction impulse evidence."""

    air_supply_liquid_volume_residual_m3_s: float | None = None
    air_supply_gas_mass_residual_kg_s: float | None = None
    air_supply_momentum_x_residual_N: float | None = None
    air_supply_momentum_z_residual_N: float | None = None
    riser_liquid_volume_residual_m3_s: float | None = None
    riser_gas_mass_residual_kg_s: float | None = None
    riser_momentum_x_residual_N: float | None = None
    riser_momentum_z_residual_N: float | None = None
    node_reaction_impulse_Ns: float | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = _optional_finite(name, getattr(self, name))
            if name == "node_reaction_impulse_Ns" and value is not None and value < 0.0:
                raise ObservationContractError(
                    "node_reaction_impulse_Ns is a cumulative magnitude and must be non-negative"
                )
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class InternalMouthEventPacket:
    """Native 1-D mouth-event state at an accepted common time."""

    active: bool | None = None
    accepted_once: bool | None = None
    onset_s: float | None = None
    acceptance_time_s: float | None = None
    evidence_status: str = "unavailable"

    def __post_init__(self) -> None:
        for name in ("active", "accepted_once"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ObservationContractError(f"mouth event {name} must be bool or None")
        for name in ("onset_s", "acceptance_time_s"):
            value = _optional_nonnegative(name, getattr(self, name))
            object.__setattr__(self, name, value)
        if not self.evidence_status.strip():
            raise ObservationContractError("mouth event evidence_status must be non-empty")


@dataclass(frozen=True, slots=True)
class AcceptedStepDiagnostics:
    """All non-state diagnostics required at one common output time."""

    pressure: GaugePressurePacket = GaugePressurePacket()
    gross_flux: AcceptedGrossFluxPacket = AcceptedGrossFluxPacket()
    nodes: AcceptedNodePacket = AcceptedNodePacket()
    mouth_event: InternalMouthEventPacket = InternalMouthEventPacket()


@dataclass(frozen=True, slots=True)
class CommonAcceptedSample:
    """One accepted state exactly on the unshifted Stage-2 common grid.

    ``ledger_entries_since_previous_sample`` must contain every accepted
    transaction after the preceding common sample and through this sample.
    The first sample at Stage-2 ``t=0`` therefore carries an empty tuple.
    """

    stage2_time_s: float
    state: CoupledState
    diagnostics: AcceptedStepDiagnostics
    ledger_entries_since_previous_sample: tuple[LedgerEntry, ...] = ()

    def __post_init__(self) -> None:
        value = float(self.stage2_time_s)
        if not math.isfinite(value) or value < 0.0:
            raise ObservationContractError(
                "stage2_time_s must be finite and non-negative"
            )
        object.__setattr__(self, "stage2_time_s", value)
        if not isinstance(self.state, CoupledState):
            raise ObservationContractError("sample state must be CoupledState")
        entries = tuple(self.ledger_entries_since_previous_sample)
        if any(not isinstance(entry, LedgerEntry) for entry in entries):
            raise ObservationContractError(
                "sample ledger interval contains a non-LedgerEntry"
            )
        object.__setattr__(self, "ledger_entries_since_previous_sample", entries)


@dataclass(frozen=True, slots=True)
class ObserverContract:
    """Validated subset of ``COMMON_OBSERVABLES.yaml`` used by the exporter."""

    path: Path
    raw: Mapping[str, Any]
    common_dt_s: float
    canonical_series: tuple[str, ...]
    required_profile_fields: tuple[str, ...]
    cfg: Mapping[str, Any]


def load_observer_contract(path: Path | None = None) -> ObserverContract:
    """Load and mechanically audit the frozen common-observable semantics."""

    source = (path or DEFAULT_CONTRACT_PATH).resolve()
    if not source.is_file():
        raise ObservationContractError(f"missing observer contract: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ObservationContractError("COMMON_OBSERVABLES root must be a mapping")
    if int(payload.get("schema_version", -1)) < 2:
        raise ObservationContractError("observer requires COMMON_OBSERVABLES schema >=2")
    if payload.get("time_origin") != "stage_2_air_opening":
        raise ObservationContractError("observer time origin must be Stage-2 opening")
    if payload.get("time_shift_allowed") is not False:
        raise ObservationContractError("observer contract must forbid time shifting")
    dt = float(payload.get("comparison_time_grid_s", math.nan))
    if not math.isclose(dt, 0.10, rel_tol=0.0, abs_tol=1.0e-12):
        raise ObservationContractError("common observer grid must remain exactly 0.10 s")

    canonical = tuple(str(item) for item in payload.get("canonical_series", ()))
    if not canonical or len(canonical) != len(set(canonical)) or canonical[0] != "time_s":
        raise ObservationContractError(
            "canonical series must be unique, non-empty and begin with time_s"
        )
    profiles = tuple(str(item) for item in payload.get("required_profile_fields", ()))
    expected_profiles = {
        "time_s",
        "riser_z_cell_center_m",
        "riser_Aup_m2",
        "riser_Qup_m3_s",
        "riser_Adown_m2",
        "riser_Qdown_m3_s",
        "riser_gas_area_m2",
        "riser_gas_mass_kg_m",
        "riser_gas_velocity_m_s",
    }
    if set(profiles) != expected_profiles:
        raise ObservationContractError("required riser profile field contract drifted")

    semantics = payload.get("source_semantics", {})
    fig8 = semantics.get("fig8_horizontal_slug_velocity", {})
    if fig8.get("quantity") != (
        "water_velocity_magnitude_in_unmixed_middle_part_of_horizontal_slug"
    ):
        raise ObservationContractError("Fig.8 water-velocity semantics are missing")
    forbidden = {str(item) for item in fig8.get("forbidden_substitutions", ())}
    if not {"gas_nose_speed", "gas_front_speed", "slug_nose_speed"} <= forbidden:
        raise ObservationContractError("Fig.8 forbidden-substitution gate drifted")

    published = semantics.get("published_pressure_probes", {}).get("probes", {})
    coordinates = {
        "P1": ([0.0, 0.0], "P1_gauge_Pa", "P1"),
        "P2": ([0.0, 0.30], "P2_gauge_Pa", "P2"),
        "P3": ([0.0, 0.45], "P3_gauge_Pa", "P3"),
        "P4": ([-0.80, 0.0], "H_upstream_gauge_Pa", "H_upstream"),
        "P5": ([-0.10, 0.0], "riser_left_gauge_Pa", "riser_left"),
        "P6": ([0.10, 0.0], "riser_right_gauge_Pa", "riser_right"),
    }
    for name, (xy, output, alias) in coordinates.items():
        record = published.get(name, {})
        if record.get("published_x_y_m") != xy:
            raise ObservationContractError(f"published {name} coordinate drifted")
        if record.get("canonical_output") != output or record.get("engineering_alias") != alias:
            raise ObservationContractError(f"published {name} alias/output drifted")

    cfg = payload.get("one_d_observer_contract")
    if not isinstance(cfg, dict):
        raise ObservationContractError("one_d_observer_contract is missing")
    if cfg.get("scalar_sampling", {}).get("interpolation_allowed") is not False:
        raise ObservationContractError("1-D observer may not interpolate common times")
    production = payload.get("production_trajectory_gate", {})
    if production.get("formal_entry_requires_operator_production_ready") is not True:
        raise ObservationContractError("formal production gate is not enabled")
    if production.get("observer_may_not_set_production_ready") is not True:
        raise ObservationContractError("observer promotion prohibition is missing")
    if production.get("result_marker_written") is not False:
        raise ObservationContractError("observer contract must forbid result markers")
    return ObserverContract(
        path=source,
        raw=payload,
        common_dt_s=dt,
        canonical_series=canonical,
        required_profile_fields=profiles,
        cfg=cfg,
    )


__all__ = [
    "AcceptedGrossFluxPacket",
    "AcceptedNodePacket",
    "AcceptedStepDiagnostics",
    "ArtifactRole",
    "CommonAcceptedSample",
    "DEFAULT_CONTRACT_PATH",
    "GaugePressurePacket",
    "InternalMouthEventPacket",
    "ObservationContractError",
    "ObserverContract",
    "PUBLISHED_PRESSURE_TO_CANONICAL",
    "load_observer_contract",
]

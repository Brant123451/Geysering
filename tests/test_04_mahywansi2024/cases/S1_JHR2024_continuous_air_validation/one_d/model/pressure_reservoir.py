"""Declared 1-D translation of the published 5700 Pa gas boundary.

Mahyawansi et al. publish a pressure boundary, not the valve, supply-line or
reservoir-loss data needed to reproduce the experimental air-delivery system.
This module therefore supplies only a transparent numerical reference: a
stagnant isothermal ideal-gas reservoir coupled to a network node by an HLL
Riemann flux.  It must not be described as a reproduction of the experimental
valve or its opening.

The face normal points from the reservoir into the 1-D network.  Positive mass
flow is consequently inflow to the network and negative mass flow is permitted
as pressure-driven backflow.  The momentum result is the oriented conservative
Euler flux ``rho*u**2 + p`` (convective momentum plus pressure), not a fitted
jet force.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .errors import ContractViolation


PUBLISHED_GAUGE_PRESSURE_PA = 5700.0
DECLARED_ATMOSPHERIC_PRESSURE_PA = 101325.0
DECLARED_REFERENCE_TEMPERATURE_K = 293.15
DECLARED_DRY_AIR_GAS_CONSTANT_J_KG_K = 287.05

PRESSURE_EVIDENCE_STATUS = "published_gauge_pressure"
THERMODYNAMIC_REFERENCE_STATUS = "declared_OpenFOAM_consistent_reference"
VALVE_MODEL_STATUS = "not_reproduced__valve_and_supply_losses_unreported"


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractViolation(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class PressureReservoirFlux:
    """HLL face flux and its area-integrated rates.

    ``mass_flux_kg_m2_s`` and ``mass_flow_kg_s`` are signed positive from the
    reservoir into the network.  ``axial_momentum_pressure_flux_Pa`` is the
    conservative axial-momentum flux density; multiplying it by the caller's
    inlet area gives ``axial_momentum_pressure_rate_N``.
    """

    mass_flux_kg_m2_s: float
    mass_flow_kg_s: float
    axial_momentum_pressure_flux_Pa: float
    axial_momentum_pressure_rate_N: float
    reservoir_absolute_pressure_Pa: float
    node_absolute_pressure_Pa: float
    inlet_area_m2: float

    def __post_init__(self) -> None:
        for name in (
            "mass_flux_kg_m2_s",
            "mass_flow_kg_s",
            "axial_momentum_pressure_flux_Pa",
            "axial_momentum_pressure_rate_N",
            "reservoir_absolute_pressure_Pa",
            "node_absolute_pressure_Pa",
            "inlet_area_m2",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.reservoir_absolute_pressure_Pa <= 0.0:
            raise ContractViolation("reservoir absolute pressure must be positive")
        if self.node_absolute_pressure_Pa <= 0.0:
            raise ContractViolation("node absolute pressure must be positive")
        if self.inlet_area_m2 <= 0.0:
            raise ContractViolation("inlet area must be positive")


@dataclass(frozen=True, slots=True)
class IsothermalIdealGasPressureReservoir:
    """Stagnant-reservoir HLL boundary for the S1 continuous-air reference.

    The 5700 Pa gauge pressure is published.  ``p_atm``, ``T`` and ``R`` are
    declared OpenFOAM-consistent reference values used for the 1-D translation,
    rather than measurements of the experimental supply apparatus.
    """

    gauge_pressure_Pa: float = PUBLISHED_GAUGE_PRESSURE_PA
    atmospheric_pressure_Pa: float = DECLARED_ATMOSPHERIC_PRESSURE_PA
    temperature_K: float = DECLARED_REFERENCE_TEMPERATURE_K
    gas_constant_J_kg_K: float = DECLARED_DRY_AIR_GAS_CONSTANT_J_KG_K
    pressure_evidence_status: str = PRESSURE_EVIDENCE_STATUS
    thermodynamic_reference_status: str = THERMODYNAMIC_REFERENCE_STATUS
    valve_model_status: str = VALVE_MODEL_STATUS

    def __post_init__(self) -> None:
        for name in (
            "gauge_pressure_Pa",
            "atmospheric_pressure_Pa",
            "temperature_K",
            "gas_constant_J_kg_K",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.atmospheric_pressure_Pa <= 0.0:
            raise ContractViolation("atmospheric pressure must be positive")
        if self.reservoir_absolute_pressure_Pa <= 0.0:
            raise ContractViolation("reservoir absolute pressure must be positive")
        if self.temperature_K <= 0.0 or self.gas_constant_J_kg_K <= 0.0:
            raise ContractViolation("temperature and gas constant must be positive")
        for name in (
            "pressure_evidence_status",
            "thermodynamic_reference_status",
            "valve_model_status",
        ):
            if not str(getattr(self, name)).strip():
                raise ContractViolation(f"{name} must be non-empty")

    @property
    def reservoir_absolute_pressure_Pa(self) -> float:
        """Published gauge pressure translated to absolute pressure."""

        return self.atmospheric_pressure_Pa + self.gauge_pressure_Pa

    @property
    def sound_speed_m_s(self) -> float:
        """Isothermal sound speed ``sqrt(R*T)``."""

        return math.sqrt(self.gas_constant_J_kg_K * self.temperature_K)

    def density_from_pressure(self, absolute_pressure_Pa: float) -> float:
        """Return ideal-gas density at the declared common temperature."""

        pressure = _finite("absolute_pressure_Pa", absolute_pressure_Pa)
        if pressure <= 0.0:
            raise ContractViolation("absolute pressure must be positive")
        density = pressure / (self.gas_constant_J_kg_K * self.temperature_K)
        if not math.isfinite(density) or density <= 0.0:
            raise ContractViolation("ideal-gas density must be finite and positive")
        return density

    def evaluate(
        self,
        *,
        node_absolute_pressure_Pa: float,
        node_axial_velocity_m_s: float,
        inlet_area_m2: float,
    ) -> PressureReservoirFlux:
        """Evaluate the isothermal Euler HLL flux at the supply face.

        ``inlet_area_m2`` is intentionally mandatory.  The paper does not
        publish enough valve geometry to infer an effective 1-D flow area, so
        the caller must provide and provenance-track the area used by its own
        discretisation.
        """

        p_left = self.reservoir_absolute_pressure_Pa
        p_right = _finite("node_absolute_pressure_Pa", node_absolute_pressure_Pa)
        u_right = _finite("node_axial_velocity_m_s", node_axial_velocity_m_s)
        area = _finite("inlet_area_m2", inlet_area_m2)
        if p_right <= 0.0:
            raise ContractViolation("node absolute pressure must be positive")
        if area <= 0.0:
            raise ContractViolation("inlet area must be positive")

        c = self.sound_speed_m_s
        rho_left = self.density_from_pressure(p_left)
        rho_right = self.density_from_pressure(p_right)
        u_left = 0.0

        # Isothermal Euler state U=(rho, rho*u), flux
        # F=(rho*u, rho*u^2+p).  Including the stagnant reservoir waves makes
        # s_left < 0 < s_right, but the general HLL form is kept explicit.
        state_left = (rho_left, rho_left * u_left)
        state_right = (rho_right, rho_right * u_right)
        flux_left = (rho_left * u_left, rho_left * u_left * u_left + p_left)
        flux_right = (rho_right * u_right, rho_right * u_right * u_right + p_right)
        s_left = min(u_left - c, u_right - c)
        s_right = max(u_left + c, u_right + c)

        if s_left >= 0.0:
            mass_flux, momentum_flux = flux_left
        elif s_right <= 0.0:
            mass_flux, momentum_flux = flux_right
        else:
            denominator = s_right - s_left
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise ContractViolation("HLL wave-speed interval must be finite and positive")
            mass_flux = (
                s_right * flux_left[0]
                - s_left * flux_right[0]
                + s_left * s_right * (state_right[0] - state_left[0])
            ) / denominator
            momentum_flux = (
                s_right * flux_left[1]
                - s_left * flux_right[1]
                + s_left * s_right * (state_right[1] - state_left[1])
            ) / denominator

        mass_flow = mass_flux * area
        momentum_rate = momentum_flux * area
        for name, value in (
            ("mass flux", mass_flux),
            ("mass flow", mass_flow),
            ("momentum/pressure flux", momentum_flux),
            ("momentum/pressure rate", momentum_rate),
        ):
            if not math.isfinite(value):
                raise ContractViolation(f"computed HLL {name} must be finite")

        return PressureReservoirFlux(
            mass_flux_kg_m2_s=mass_flux,
            mass_flow_kg_s=mass_flow,
            axial_momentum_pressure_flux_Pa=momentum_flux,
            axial_momentum_pressure_rate_N=momentum_rate,
            reservoir_absolute_pressure_Pa=p_left,
            node_absolute_pressure_Pa=p_right,
            inlet_area_m2=area,
        )

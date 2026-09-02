"""Physical constant-head reservoir boundary for the mirrored Case-1 core.

Campaign 2 places the upstream reservoir at physical ``x = 0``.  The
Campaign-1 horizontal solver is used in the mirrored coordinate
``x_m = L - x_physical``, so this reservoir is the *right* boundary of the
Case-1 grid.  The boundary prescribed here is therefore obtained from the
incoming pressurised-pipe ``C+`` characteristic,

``C+ = u + g H / a`` and ``u_b = C+ - g H_r / a``.

Velocity and discharge in this module use the mirrored Case-1 sign.  The
reported physical flow is positive from the reservoir into the horizontal
pipe, hence ``Q_in = -Q_mirrored``.  A commit adds exactly
``Q_in * dt`` to the final Case-1 control volume and records the equal and
opposite reservoir transaction.  No case identifier, geyser outcome, valve
opening law, or fitted multiplier appears here.

This boundary is deliberately independent of the adapter's current global
hydraulic-time approximation for the 0.20 s valve stroke.  A coupled driver
may call :meth:`commit` after each physical horizontal step; the boundary
uses the physical ``dt`` passed to it and never reads or rescales model time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConstantHeadBoundarySolution:
    """Characteristic solution at the physical reservoir face.

    ``physical_inflow_m3_s`` is positive into the horizontal domain.  The
    mirrored quantities have the Case-1 coordinate sign and are therefore
    negative during reservoir inflow.
    """

    reservoir_head_from_invert_m: float
    characteristic_foot_head_m: float
    characteristic_foot_velocity_m_s: float
    incoming_characteristic_m_s: float
    mirrored_boundary_velocity_m_s: float
    mirrored_boundary_discharge_m3_s: float
    physical_inflow_m3_s: float
    boundary_area_m2: float
    physical_dt_s: float

    @property
    def requested_liquid_volume_to_horizontal_m3(self) -> float:
        return float(self.physical_inflow_m3_s * self.physical_dt_s)


@dataclass(frozen=True)
class ConstantHeadBoundaryCommit:
    """State and mass ledger resulting from one reservoir transaction."""

    state: Any
    solution: ConstantHeadBoundarySolution
    liquid_volume_to_horizontal_m3: float
    horizontal_liquid_volume_change_m3: float
    reservoir_liquid_volume_change_m3: float
    mass_balance_residual_m3: float
    cumulative_liquid_to_horizontal_m3: float
    cumulative_reservoir_liquid_change_m3: float
    cumulative_mass_balance_residual_m3: float


class Case1ConstantHeadBoundary:
    """Constant-head reservoir coupled to the mirrored Case-1 right end.

    Parameters
    ----------
    solver:
        A ``Campaign2Case1MirroredHorizontal``-compatible object.  Only its
        Case-1 grid, circular section, and physical/mirrored geometry are
        used; importing the adapter is intentionally unnecessary.
    reservoir_head_from_invert_m:
        Fixed piezometric head measured from the horizontal-pipe invert.
        Cong et al. Campaign 2 uses 0.66 m.

    Notes
    -----
    The adjacent Case-1 cell remains an interior storage cell.  The fixed
    head is a face/ghost condition, not a command to overwrite that complete
    cell with reservoir state.  This is what permits the committed volume to
    equal the characteristic face flux times physical time exactly.
    """

    def __init__(
        self,
        solver: Any,
        *,
        reservoir_head_from_invert_m: float = 0.66,
    ) -> None:
        self.solver = solver
        self.reservoir_head_from_invert_m = float(
            reservoir_head_from_invert_m
        )
        if (
            not math.isfinite(self.reservoir_head_from_invert_m)
            or self.reservoir_head_from_invert_m <= 0.0
        ):
            raise ValueError(
                "reservoir_head_from_invert_m must be positive and finite"
            )

        required = ("section", "dx", "config", "physical_length")
        missing = [name for name in required if not hasattr(solver, name)]
        if missing:
            raise TypeError(
                "solver lacks mirrored Case-1 attributes: "
                + ", ".join(missing)
            )
        if not math.isfinite(float(solver.dx)) or float(solver.dx) <= 0.0:
            raise ValueError("solver.dx must be positive and finite")
        if not math.isclose(
            float(solver.physical_length),
            float(solver.config.length),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("physical and mirrored Case-1 lengths differ")

        section = solver.section
        self._boundary_area_m2 = float(
            section.full_area
            * (
                1.0
                + section.gravity
                * (
                    self.reservoir_head_from_invert_m - section.diameter
                )
                / section.wave_speed**2
            )
        )
        if (
            not math.isfinite(self._boundary_area_m2)
            or self._boundary_area_m2 <= 0.0
        ):
            raise ValueError("fixed reservoir head gives inadmissible area")

        self.cumulative_liquid_to_horizontal_m3 = 0.0
        self.cumulative_reservoir_liquid_change_m3 = 0.0
        self.cumulative_inflow_m3 = 0.0
        self.cumulative_outflow_m3 = 0.0
        self.commit_count = 0

    @property
    def physical_boundary_x_m(self) -> float:
        return 0.0

    @property
    def mirrored_boundary_x_m(self) -> float:
        return float(self.solver.config.length)

    @property
    def boundary_area_m2(self) -> float:
        return self._boundary_area_m2

    @staticmethod
    def _pressurised_head(area: np.ndarray, section: Any) -> np.ndarray:
        """Return elastic pressurised head even for a transient below crown."""

        return section.diameter + section.wave_speed**2 / section.gravity * (
            area / section.full_area - 1.0
        )

    def solve(
        self,
        state: Any,
        dt: float,
    ) -> ConstantHeadBoundarySolution:
        """Solve the incoming pressurised characteristic at fixed head."""

        step = float(dt)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be positive and finite")

        area = np.asarray(state.area, dtype=float)
        discharge = np.asarray(state.discharge, dtype=float)
        if area.ndim != 1 or discharge.shape != area.shape or area.size < 2:
            raise ValueError(
                "Case-1 state must have matching one-dimensional area and "
                "discharge arrays with at least two cells"
            )
        if np.any(~np.isfinite(area[-2:])) or np.any(area[-2:] <= 0.0):
            raise FloatingPointError(
                "reservoir-adjacent pressurised cells are inadmissible"
            )
        if np.any(~np.isfinite(discharge[-2:])):
            raise FloatingPointError(
                "reservoir-adjacent discharge is not finite"
            )

        section = self.solver.section
        acoustic = float(section.wave_speed)
        gravity = float(section.gravity)
        dx = float(self.solver.dx)
        courant = acoustic * step / dx
        if courant > 1.0 + 1.0e-12:
            raise ValueError(
                "constant-head characteristic crosses more than one grid "
                "interval"
            )
        ratio = min(1.0, courant)

        velocity = discharge[-2:] / area[-2:]
        head = self._pressurised_head(area[-2:], section)
        diameter = float(section.diameter)
        friction_factor = float(self.solver.config.darcy_friction)
        friction = (
            friction_factor
            * velocity
            * np.abs(velocity)
            / (2.0 * diameter)
        )

        # C+ arrives at the mirrored right boundary from the domain.  This is
        # the same characteristic-foot interpolation and source integration
        # used by Case-1 ``pressurised_moc_step``.
        foot_velocity = float(
            velocity[-1] + ratio * (velocity[-2] - velocity[-1])
        )
        foot_head = float(head[-1] + ratio * (head[-2] - head[-1]))
        foot_friction = float(
            friction[-1] + ratio * (friction[-2] - friction[-1])
        )
        incoming = float(
            foot_velocity
            + gravity / acoustic * foot_head
            + gravity
            * (float(self.solver.config.bed_slope) - foot_friction)
            * step
        )
        reservoir_characteristic = (
            gravity / acoustic * self.reservoir_head_from_invert_m
        )
        mirrored_velocity = float(incoming - reservoir_characteristic)

        # Make the exact hydrostatic state an exact computational fixed point;
        # this removes only subtraction-level roundoff, not physical flow.
        roundoff_scale = max(
            1.0,
            abs(incoming),
            abs(reservoir_characteristic),
        )
        if abs(mirrored_velocity) <= (
            32.0 * np.finfo(float).eps * roundoff_scale
        ):
            mirrored_velocity = 0.0

        mirrored_discharge = self._boundary_area_m2 * mirrored_velocity
        physical_inflow = -mirrored_discharge
        return ConstantHeadBoundarySolution(
            reservoir_head_from_invert_m=(
                self.reservoir_head_from_invert_m
            ),
            characteristic_foot_head_m=foot_head,
            characteristic_foot_velocity_m_s=foot_velocity,
            incoming_characteristic_m_s=incoming,
            mirrored_boundary_velocity_m_s=mirrored_velocity,
            mirrored_boundary_discharge_m3_s=mirrored_discharge,
            physical_inflow_m3_s=physical_inflow,
            boundary_area_m2=self._boundary_area_m2,
            physical_dt_s=step,
        )

    def commit(
        self,
        state: Any,
        dt: float,
    ) -> ConstantHeadBoundaryCommit:
        """Commit one characteristic reservoir flux to the horizontal state.

        The final mirrored cell receives the signed physical volume
        ``Q_in * dt``.  Its discharge is set to the boundary-face discharge;
        pressure-wave propagation into the remaining branch is still owned by
        the Case-1 MOC step.  A too-large outflow is rejected rather than
        clipped, because clipping would silently break the characteristic
        boundary and its mass ledger.
        """

        solution = self.solve(state, dt)
        requested = (
            solution.requested_liquid_volume_to_horizontal_m3
        )
        area = np.asarray(state.area, dtype=float).copy()
        discharge = np.asarray(state.discharge, dtype=float).copy()
        volume_before = float(np.sum(area) * float(self.solver.dx))
        area[-1] += requested / float(self.solver.dx)
        minimum_area = (
            1.0e-12 * float(self.solver.section.full_area)
        )
        if not math.isfinite(float(area[-1])) or area[-1] <= minimum_area:
            raise FloatingPointError(
                "constant-head outflow would empty the boundary cell; "
                "reduce the physical coupling step"
            )
        discharge[-1] = solution.mirrored_boundary_discharge_m3_s
        updated = replace(state, area=area, discharge=discharge)

        volume_after = float(np.sum(area) * float(self.solver.dx))
        actual = volume_after - volume_before
        reservoir_change = -actual
        residual = actual + reservoir_change
        tolerance = max(1.0e-16, 64.0 * np.finfo(float).eps * max(
            abs(volume_before),
            abs(volume_after),
            abs(requested),
            1.0e-12,
        ))
        if not math.isclose(actual, requested, rel_tol=0.0, abs_tol=tolerance):
            raise FloatingPointError(
                "constant-head commit did not add the characteristic volume"
            )
        if abs(residual) > tolerance:
            raise FloatingPointError(
                "constant-head reservoir/horizontal mass ledger does not close"
            )

        self.cumulative_liquid_to_horizontal_m3 += actual
        self.cumulative_reservoir_liquid_change_m3 += reservoir_change
        self.cumulative_inflow_m3 += max(actual, 0.0)
        self.cumulative_outflow_m3 += max(-actual, 0.0)
        self.commit_count += 1
        cumulative_residual = (
            self.cumulative_liquid_to_horizontal_m3
            + self.cumulative_reservoir_liquid_change_m3
        )
        return ConstantHeadBoundaryCommit(
            state=updated,
            solution=solution,
            liquid_volume_to_horizontal_m3=actual,
            horizontal_liquid_volume_change_m3=actual,
            reservoir_liquid_volume_change_m3=reservoir_change,
            mass_balance_residual_m3=residual,
            cumulative_liquid_to_horizontal_m3=(
                self.cumulative_liquid_to_horizontal_m3
            ),
            cumulative_reservoir_liquid_change_m3=(
                self.cumulative_reservoir_liquid_change_m3
            ),
            cumulative_mass_balance_residual_m3=cumulative_residual,
        )

    def ledger(self) -> dict[str, float | int]:
        """Return the current signed and one-way reservoir volume totals."""

        return {
            "commit_count": self.commit_count,
            "liquid_to_horizontal_m3": (
                self.cumulative_liquid_to_horizontal_m3
            ),
            "reservoir_liquid_change_m3": (
                self.cumulative_reservoir_liquid_change_m3
            ),
            "inflow_to_horizontal_m3": self.cumulative_inflow_m3,
            "outflow_from_horizontal_m3": self.cumulative_outflow_m3,
            "mass_balance_residual_m3": (
                self.cumulative_liquid_to_horizontal_m3
                + self.cumulative_reservoir_liquid_change_m3
            ),
        }

    def provenance(self) -> dict[str, object]:
        return {
            "model": "fixed-head pressurised C+ characteristic boundary",
            "physical_boundary": "x=0 m",
            "mirrored_case1_boundary": "right end, x_m=L",
            "reservoir_head_from_invert_m": (
                self.reservoir_head_from_invert_m
            ),
            "flow_sign": "positive physical flow is reservoir to horizontal",
            "volume_commit": "dV_horizontal=Q_physical_in*dt",
            "valve_hydraulic_time_coupling": "none; physical dt only",
        }


__all__ = [
    "Case1ConstantHeadBoundary",
    "ConstantHeadBoundaryCommit",
    "ConstantHeadBoundarySolution",
]

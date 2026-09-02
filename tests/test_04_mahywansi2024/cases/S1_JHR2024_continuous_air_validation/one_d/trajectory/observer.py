"""Canonical scalar/profile observer for accepted S1 1-D states.

All spatial quantities are calculated from native conservative cell states.
Pressure, boundary gross-flow, node and event quantities are accepted only
through explicit diagnostic packets; absent values remain unavailable and are
never replaced by a visually plausible state proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from model.state import CoupledGeometry

from .contracts import (
    CommonAcceptedSample,
    ObservationContractError,
    ObserverContract,
)


Array = np.ndarray


@dataclass(frozen=True, slots=True)
class CumulativeLedgerResiduals:
    liquid_volume_m3: float = 0.0
    gas_mass_kg: float = 0.0
    horizontal_momentum_Ns: float = 0.0
    vertical_momentum_Ns: float = 0.0


@dataclass(frozen=True, slots=True)
class RiserProfileFrame:
    z_cell_center_m: Array
    Aup_m2: Array
    Qup_m3_s: Array
    Adown_m2: Array
    Qdown_m3_s: Array
    gas_area_m2: Array
    gas_mass_kg_m: Array
    gas_velocity_m_s: Array
    gas_velocity_available: Array


@dataclass(frozen=True, slots=True)
class ObservedFrame:
    row: Mapping[str, float]
    profile: RiserProfileFrame
    unavailable: Mapping[str, str]


class S1CanonicalObserver:
    """Observe one accepted state without changing or advancing the solver."""

    def __init__(self, *, geometry: CoupledGeometry, contract: ObserverContract) -> None:
        self.geometry = geometry
        self.contract = contract
        cfg = contract.cfg
        self.horizontal_edges = self._edges(
            float(cfg["horizontal_x_min_m"]), geometry.horizontal_dx_m
        )
        self.vertical_edges = self._edges(
            float(cfg["riser_bottom_z_m"]), geometry.vertical_dz_m
        )
        self.supply_edges = self._edges(
            float(cfg["supply_branch_bottom_z_m"]), geometry.supply_branch_dz_m
        )
        self.horizontal_centres = 0.5 * (
            self.horizontal_edges[:-1] + self.horizontal_edges[1:]
        )
        self.vertical_centres = 0.5 * (
            self.vertical_edges[:-1] + self.vertical_edges[1:]
        )
        self.supply_centres = 0.5 * (
            self.supply_edges[:-1] + self.supply_edges[1:]
        )
        self._validate_geometry()

    @staticmethod
    def _edges(origin: float, widths: tuple[float, ...]) -> Array:
        return np.concatenate(
            (np.asarray([origin], dtype=float), origin + np.cumsum(widths, dtype=float))
        )

    def _validate_geometry(self) -> None:
        cfg = self.contract.cfg
        tolerance = 1.0e-10
        expected = (
            (self.horizontal_edges[0], float(cfg["horizontal_x_min_m"]), "horizontal x min"),
            (self.horizontal_edges[-1], float(cfg["horizontal_x_max_m"]), "horizontal x max"),
            (self.vertical_edges[0], float(cfg["riser_bottom_z_m"]), "riser bottom"),
            (self.vertical_edges[-1], float(cfg["riser_rim_z_m"]), "riser rim"),
            (self.supply_edges[0], float(cfg["supply_branch_bottom_z_m"]), "supply bottom"),
            (self.supply_edges[-1], float(cfg["supply_branch_top_z_m"]), "supply top"),
        )
        for actual, target, label in expected:
            if not math.isclose(float(actual), target, rel_tol=0.0, abs_tol=tolerance):
                raise ObservationContractError(
                    f"observer {label}={actual:.17g} does not match contract {target:.17g}"
                )
        for label, coordinate in (
            ("air supply tee", float(cfg["air_supply_tee_x_m"])),
            ("riser tee", float(cfg["riser_tee_x_m"])),
        ):
            if float(np.min(np.abs(self.horizontal_edges - coordinate))) > tolerance:
                raise ObservationContractError(
                    f"observer {label} is not an exact horizontal cell face"
                )

    @staticmethod
    def _put(
        row: dict[str, float],
        unavailable: dict[str, str],
        name: str,
        value: float | bool | None,
        reason: str,
    ) -> None:
        if value is None:
            row[name] = math.nan
            unavailable[name] = reason
            return
        result = float(value)
        if not math.isfinite(result):
            row[name] = math.nan
            unavailable[name] = reason
            return
        row[name] = result

    def _supply_connected_gas(
        self, sample: CommonAcceptedSample
    ) -> tuple[Array, float, float, float, float, bool, float]:
        state = sample.state
        cfg = self.contract.cfg
        area = self.geometry.horizontal_area_m2
        supply_area = float(self.geometry.supply_branch_area_m2)
        horizontal_gas_area = np.maximum(
            area - np.asarray(state.horizontal.Al, dtype=float), 0.0
        )
        horizontal_fraction = horizontal_gas_area / area
        horizontal_mass = np.asarray(state.horizontal.Mg, dtype=float)
        gas_min = float(cfg["gas_cell_min_area_fraction"])
        horizontal_mask = (horizontal_fraction >= gas_min - 1.0e-14) & (
            horizontal_mass > 0.0
        )

        supply_gas_area = np.maximum(
            supply_area - np.asarray(state.supply_branch.Al, dtype=float), 0.0
        )
        supply_fraction = supply_gas_area / supply_area
        supply_mass = np.asarray(state.supply_branch.Mg, dtype=float)
        supply_mask = (supply_fraction >= gas_min - 1.0e-14) & (supply_mass > 0.0)
        top_members: list[int] = []
        for index in range(len(supply_mask) - 1, -1, -1):
            if not supply_mask[index]:
                break
            top_members.append(index)
        supply_front = (
            float(self.supply_edges[min(top_members)]) if top_members else math.nan
        )
        reaches_tee = bool(top_members and min(top_members) == 0)
        if not reaches_tee:
            return (
                np.asarray([], dtype=np.int64),
                math.nan,
                math.nan,
                math.nan,
                0.0,
                False,
                supply_front,
            )

        tee = float(cfg["air_supply_tee_x_m"])
        seed = np.flatnonzero(
            horizontal_mask
            & (self.horizontal_edges[:-1] <= tee + 1.0e-12)
            & (self.horizontal_edges[1:] >= tee - 1.0e-12)
        )
        candidates: list[Array] = []
        visited = np.zeros(len(horizontal_mask), dtype=bool)
        for raw_seed in seed:
            if visited[raw_seed]:
                continue
            stack = [int(raw_seed)]
            members: list[int] = []
            while stack:
                index = stack.pop()
                if visited[index] or not horizontal_mask[index]:
                    continue
                visited[index] = True
                members.append(index)
                if index > 0:
                    stack.append(index - 1)
                if index + 1 < len(horizontal_mask):
                    stack.append(index + 1)
            if members:
                candidates.append(np.asarray(sorted(members), dtype=np.int64))
        if not candidates:
            return (
                np.asarray([], dtype=np.int64),
                math.nan,
                math.nan,
                math.nan,
                0.0,
                False,
                supply_front,
            )
        cells = max(
            candidates,
            key=lambda indices: float(
                np.dot(
                    horizontal_gas_area[indices],
                    np.asarray(self.geometry.horizontal_dx_m, dtype=float)[indices],
                )
            ),
        )
        weights = horizontal_gas_area[cells] * np.asarray(
            self.geometry.horizontal_dx_m, dtype=float
        )[cells]
        gas_volume = float(weights.sum())
        if gas_volume <= 0.0:
            raise ObservationContractError(
                "supply-connected gas component has non-positive gas volume"
            )
        tail = float(np.min(self.horizontal_edges[cells]))
        nose = float(np.max(self.horizontal_edges[cells + 1]))
        centroid = float(np.dot(weights, self.horizontal_centres[cells]) / gas_volume)
        riser = float(cfg["gas_arrival_location_x_m"])
        arrival = bool(tail - 1.0e-12 <= riser <= nose + 1.0e-12)
        return cells, tail, nose, centroid, gas_volume, arrival, supply_front

    def _slug(self, sample: CommonAcceptedSample) -> tuple[float, float, float]:
        state = sample.state
        cfg = self.contract.cfg
        area = self.geometry.horizontal_area_m2
        liquid_area = np.asarray(state.horizontal.Al, dtype=float)
        liquid_fraction = np.clip(liquid_area / area, 0.0, 1.0)
        gas_fraction = 1.0 - liquid_fraction
        full = liquid_fraction >= float(cfg["unmixed_water_min_area_fraction"]) - 1.0e-14
        gas_bearing = gas_fraction >= float(cfg["surrounding_gas_min_area_fraction"]) - 1.0e-14
        search_min, search_max = (float(item) for item in cfg["slug_search_x_m"])
        selected = np.flatnonzero(
            (self.horizontal_centres >= search_min - 1.0e-12)
            & (self.horizontal_centres <= search_max + 1.0e-12)
        )
        candidates: list[tuple[float, float, Array]] = []
        cursor = 0
        while cursor < len(selected):
            cell = int(selected[cursor])
            if not full[cell]:
                cursor += 1
                continue
            end = cursor
            while end + 1 < len(selected) and full[int(selected[end + 1])]:
                end += 1
            if cursor > 0 and end + 1 < len(selected):
                left = int(selected[cursor - 1])
                right = int(selected[end + 1])
                run = selected[cursor : end + 1]
                tail = float(self.horizontal_edges[int(run[0])])
                nose = float(self.horizontal_edges[int(run[-1]) + 1])
                if (
                    gas_bearing[left]
                    and gas_bearing[right]
                    and nose - tail
                    >= float(cfg["slug_min_axial_length_m"]) - 1.0e-12
                ):
                    candidates.append((tail, nose, run.copy()))
            cursor = end + 1
        if not candidates:
            return math.nan, math.nan, math.nan
        piv_min, piv_max = (
            float(item) for item in cfg["published_slug_PIV_window_x_m"]
        )
        anchor = 0.5 * (piv_min + piv_max)
        tail, nose, _ = min(
            candidates, key=lambda item: abs(0.5 * (item[0] + item[1]) - anchor)
        )
        if nose < piv_min - 1.0e-12 or tail > piv_max + 1.0e-12:
            return tail, nose, math.nan
        piv = np.flatnonzero(
            full
            & (self.horizontal_centres >= piv_min - 1.0e-12)
            & (self.horizontal_centres <= piv_max + 1.0e-12)
        )
        positive = piv[liquid_area[piv] > 0.0]
        if len(positive) == 0:
            return tail, nose, math.nan
        dx = np.asarray(self.geometry.horizontal_dx_m, dtype=float)[positive]
        weights = liquid_area[positive] * dx
        water_speed = np.abs(
            np.asarray(state.horizontal.Ql, dtype=float)[positive]
            / liquid_area[positive]
        )
        velocity = float(np.dot(weights, water_speed) / weights.sum())
        return tail, nose, velocity

    def _riser_profile(self, sample: CommonAcceptedSample) -> RiserProfileFrame:
        state = sample.state.vertical
        Aup = np.asarray(state.Aup, dtype=float)
        Qup = np.asarray(state.Qup, dtype=float)
        Adown = np.asarray(state.Adown, dtype=float)
        Qdown = np.asarray(state.Qdown, dtype=float)
        mass = np.asarray(state.Mg, dtype=float)
        momentum = np.asarray(state.Jg, dtype=float)
        gas_area = np.maximum(self.geometry.vertical_area_m2 - Aup - Adown, 0.0)
        available = mass > 0.0
        velocity = np.full_like(mass, math.nan, dtype=float)
        velocity[available] = momentum[available] / mass[available]
        return RiserProfileFrame(
            z_cell_center_m=self.vertical_centres.copy(),
            Aup_m2=Aup.copy(),
            Qup_m3_s=Qup.copy(),
            Adown_m2=Adown.copy(),
            Qdown_m3_s=Qdown.copy(),
            gas_area_m2=gas_area,
            gas_mass_kg_m=mass.copy(),
            gas_velocity_m_s=velocity,
            gas_velocity_available=available,
        )

    def _riser_connected_top(self, profile: RiserProfileFrame) -> float:
        fraction = (
            profile.Aup_m2 + profile.Adown_m2
        ) / self.geometry.vertical_area_m2
        minimum = float(self.contract.cfg["riser_wet_min_area_fraction"])
        last = -1
        for index, value in enumerate(fraction):
            if value < minimum - 1.0e-14:
                break
            last = index
        return math.nan if last < 0 else float(self.vertical_edges[last + 1])

    def _riser_section_flows(
        self, profile: RiserProfileFrame
    ) -> tuple[float | None, float | None]:
        target = float(self.contract.cfg["riser_flux_section_z_m"])
        z = profile.z_cell_center_m
        if len(z) < 2 or target < z[0] - 1.0e-12 or target > z[-1] + 1.0e-12:
            return None, None
        return (
            float(np.interp(target, z, profile.Qup_m3_s)),
            float(np.interp(target, z, profile.Qdown_m3_s)),
        )

    def observe(
        self,
        sample: CommonAcceptedSample,
        *,
        cumulative: CumulativeLedgerResiduals,
    ) -> ObservedFrame:
        """Return canonical data and explicit unavailability reasons."""

        self.geometry.validate_state(sample.state)
        row = {name: math.nan for name in self.contract.canonical_series}
        unavailable: dict[str, str] = {}
        row["time_s"] = sample.stage2_time_s

        pressure = sample.diagnostics.pressure.canonical()
        for name, value in pressure.items():
            self._put(
                row,
                unavailable,
                name,
                value,
                "accepted native P1-P6 gauge-pressure diagnostic unavailable",
            )

        _, gas_tail, gas_nose, gas_centroid, gas_volume, arrival, supply_front = (
            self._supply_connected_gas(sample)
        )
        row["horizontal_gas_tail_x_m"] = gas_tail
        row["horizontal_gas_nose_x_m"] = gas_nose
        row["horizontal_gas_centroid_x_m"] = gas_centroid
        row["horizontal_gas_volume_m3"] = gas_volume
        row["gas_arrival_at_riser"] = float(arrival)
        row["supply_branch_gas_front_z_m"] = supply_front

        slug_tail, slug_nose, slug_water_speed = self._slug(sample)
        row["horizontal_slug_tail_x_m"] = slug_tail
        row["horizontal_slug_nose_x_m"] = slug_nose
        row["horizontal_slug_velocity_m_s"] = slug_water_speed

        profile = self._riser_profile(sample)
        row["riser_connected_water_top_z_m"] = self._riser_connected_top(profile)
        q_up, q_down = self._riser_section_flows(profile)
        self._put(
            row,
            unavailable,
            "riser_upward_liquid_flow_m3_s",
            q_up,
            "z=0.30 m lies outside native riser cell-centre interpolation support",
        )
        self._put(
            row,
            unavailable,
            "riser_downward_liquid_flow_m3_s",
            q_down,
            "z=0.30 m lies outside native riser cell-centre interpolation support",
        )

        gross = sample.diagnostics.gross_flux
        for name in (
            "supply_branch_liquid_outflow_m3_s",
            "supply_branch_gas_inflow_kg_s",
            "mouth_liquid_outflow_m3_s",
            "mouth_gas_outflow_kg_s",
            "cumulative_mouth_liquid_outflow_m3",
        ):
            self._put(
                row,
                unavailable,
                name,
                getattr(gross, name),
                "accepted native gross boundary-flux diagnostic unavailable",
            )

        nodes = sample.diagnostics.nodes
        mapping = {
            "air_supply_node_liquid_volume_residual_m3_s": (
                nodes.air_supply_liquid_volume_residual_m3_s
            ),
            "air_supply_node_gas_mass_residual_kg_s": (
                nodes.air_supply_gas_mass_residual_kg_s
            ),
            "riser_node_liquid_volume_residual_m3_s": (
                nodes.riser_liquid_volume_residual_m3_s
            ),
            "riser_node_gas_mass_residual_kg_s": nodes.riser_gas_mass_residual_kg_s,
            "node_reaction_impulse_Ns": nodes.node_reaction_impulse_Ns,
        }
        for name, value in mapping.items():
            self._put(
                row,
                unavailable,
                name,
                value,
                "accepted zero-storage-node diagnostic unavailable",
            )

        row["liquid_volume_residual_m3"] = cumulative.liquid_volume_m3
        row["gas_mass_residual_kg"] = cumulative.gas_mass_kg
        row["horizontal_momentum_residual_Ns"] = cumulative.horizontal_momentum_Ns
        row["vertical_momentum_residual_Ns"] = cumulative.vertical_momentum_Ns

        event = sample.diagnostics.mouth_event
        self._put(
            row,
            unavailable,
            "internal_mouth_event_active",
            event.active,
            "native internal mouth-event state unavailable",
        )

        # Diagnostic-only extras are numeric so the canonical CSV remains a
        # single machine-readable table. Their meanings are frozen in metadata.
        row.update(
            {
                "P4_gauge_Pa": math.nan if pressure["H_upstream_gauge_Pa"] is None else float(pressure["H_upstream_gauge_Pa"]),
                "P5_gauge_Pa": math.nan if pressure["riser_left_gauge_Pa"] is None else float(pressure["riser_left_gauge_Pa"]),
                "P6_gauge_Pa": math.nan if pressure["riser_right_gauge_Pa"] is None else float(pressure["riser_right_gauge_Pa"]),
                "mouth_liquid_inflow_m3_s": math.nan if gross.mouth_liquid_inflow_m3_s is None else gross.mouth_liquid_inflow_m3_s,
                "mouth_gas_inflow_kg_s": math.nan if gross.mouth_gas_inflow_kg_s is None else gross.mouth_gas_inflow_kg_s,
                "riser_mouth_state_Qup_m3_s": float(profile.Qup_m3_s[-1]),
                "riser_mouth_state_Qdown_m3_s": float(profile.Qdown_m3_s[-1]),
                "air_supply_node_momentum_x_residual_N": math.nan if nodes.air_supply_momentum_x_residual_N is None else nodes.air_supply_momentum_x_residual_N,
                "air_supply_node_momentum_z_residual_N": math.nan if nodes.air_supply_momentum_z_residual_N is None else nodes.air_supply_momentum_z_residual_N,
                "riser_node_momentum_x_residual_N": math.nan if nodes.riser_momentum_x_residual_N is None else nodes.riser_momentum_x_residual_N,
                "riser_node_momentum_z_residual_N": math.nan if nodes.riser_momentum_z_residual_N is None else nodes.riser_momentum_z_residual_N,
                "internal_mouth_event_accepted_once": math.nan if event.accepted_once is None else float(event.accepted_once),
                "internal_mouth_event_onset_s": math.nan if event.onset_s is None else event.onset_s,
                "internal_mouth_event_acceptance_time_s": math.nan if event.acceptance_time_s is None else event.acceptance_time_s,
                "derived_plume_height_proxy_m": sample.state.exterior_plume.derived_centroid_height_proxy_m,
            }
        )
        missing_canonical = set(self.contract.canonical_series) - set(row)
        if missing_canonical:
            raise ObservationContractError(
                f"observer omitted canonical fields: {sorted(missing_canonical)}"
            )
        return ObservedFrame(row=row, profile=profile, unavailable=unavailable)


__all__ = [
    "CumulativeLedgerResiduals",
    "ObservedFrame",
    "RiserProfileFrame",
    "S1CanonicalObserver",
]

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from model.conservation import ConservationSnapshot, LedgerEntry
from model.errors import MissingPhysicalClosure
from model.flux import BoundaryExchange
from model.state import (
    CoupledGeometry,
    CoupledState,
    HorizontalState,
    SupplyBranchState,
    TNodeState,
    VerticalState,
)
from trajectory import (
    AcceptedGrossFluxPacket,
    AcceptedNodePacket,
    AcceptedStepDiagnostics,
    CommonAcceptedSample,
    GaugePressurePacket,
    InternalMouthEventPacket,
    ObservationContractError,
    build_canonical_trajectory,
    load_observer_contract,
    write_trajectory_artifacts,
)


CASE = Path(__file__).resolve().parents[2]
COMPARISON = CASE / "comparison"
if str(COMPARISON) not in sys.path:
    sys.path.insert(0, str(COMPARISON))

from observables import (  # noqa: E402
    compare_1d_to_2d,
    load_definitions,
    read_numeric_csv,
    sha256_file,
    two_d_columns,
    validate_profile_npz,
    write_csv,
)


def _geometry() -> CoupledGeometry:
    area = math.pi * 0.0254**2 / 4.0
    return CoupledGeometry(
        horizontal_dx_m=(0.01,) * 310,
        vertical_dz_m=(1.02 / 8.0,) * 8,
        horizontal_area_m2=area,
        vertical_area_m2=area,
        liquid_density_kg_m3=998.4,
        supply_branch_dz_m=(0.1373 / 2.0,) * 2,
        supply_branch_area_m2=area,
        horizontal_elastic_overarea_fraction=0.02,
    )


def _state(time_s: float, geometry: CoupledGeometry) -> CoupledState:
    area = geometry.horizontal_area_m2
    centres = -1.83 + (np.arange(310, dtype=float) + 0.5) * 0.01
    Al = np.full(310, area)
    Ql = np.zeros(310)
    Mg = np.zeros(310)
    Jg = np.zeros(310)

    # Supply-connected gas terminates immediately upstream of an embedded
    # full-bore water slug; a second gas-bearing interval brackets its nose.
    gas = ((centres >= -1.516) & (centres <= -0.854)) | (
        (centres >= -0.746) & (centres <= -0.704)
    )
    Al[gas] = 0.40 * area
    Mg[gas] = 1.2 * (area - Al[gas])
    piv = (centres >= -0.85) & (centres <= -0.75)
    Ql[piv] = 0.85 * Al[piv]

    vertical_count = len(geometry.vertical_dz_m)
    Aup = np.full(vertical_count, 0.55 * area)
    Adown = np.full(vertical_count, 0.15 * area)
    Qup = np.linspace(1.8e-5, 2.5e-5, vertical_count)
    Qdown = np.linspace(0.9e-5, 1.2e-5, vertical_count)
    vertical_gas_area = area - Aup - Adown
    vertical_mass = 1.2 * vertical_gas_area
    vertical_momentum = 0.20 * vertical_mass
    supply_Al = (0.40 * area,) * len(geometry.supply_branch_dz_m)
    supply_Mg = tuple(1.2 * (area - value) for value in supply_Al)
    state = CoupledState(
        time_s=time_s,
        horizontal=HorizontalState(
            Al=tuple(Al), Ql=tuple(Ql), Mg=tuple(Mg), Jg=tuple(Jg)
        ),
        vertical=VerticalState(
            Aup=tuple(Aup),
            Qup=tuple(Qup),
            Adown=tuple(Adown),
            Qdown=tuple(Qdown),
            Mg=tuple(vertical_mass),
            Jg=tuple(vertical_momentum),
        ),
        supply_branch=SupplyBranchState(
            Al=supply_Al,
            Ql=(0.0,) * len(supply_Al),
            Mg=supply_Mg,
            Jg=(0.0,) * len(supply_Al),
        ),
        air_supply_node=TNodeState(),
        riser_node=TNodeState(),
    )
    geometry.validate_state(state)
    return state


def _ledger(before: CoupledState, after: CoupledState, geometry: CoupledGeometry, number: int) -> LedgerEntry:
    return LedgerEntry(
        transaction_id=f"accepted-{number}",
        time_start_s=before.time_s,
        time_end_s=after.time_s,
        before=ConservationSnapshot.from_state(before, geometry),
        after=ConservationSnapshot.from_state(after, geometry),
        boundary=BoundaryExchange(),
        boundary_momentum_x_impulse_kg_m_s=0.0,
        boundary_momentum_z_impulse_kg_m_s=0.0,
        external_force_x_impulse_kg_m_s=0.0,
        external_force_z_impulse_kg_m_s=0.0,
        liquid_volume_residual_m3=number * 1.0e-16,
        gas_mass_residual_kg=-number * 2.0e-16,
        mixture_momentum_x_residual_kg_m_s=number * 3.0e-16,
        mixture_momentum_z_residual_kg_m_s=-number * 4.0e-16,
    )


def _diagnostics(time_s: float, *, complete: bool = True) -> AcceptedStepDiagnostics:
    if not complete:
        return AcceptedStepDiagnostics()
    active = time_s >= 0.1 - 1.0e-12
    return AcceptedStepDiagnostics(
        pressure=GaugePressurePacket(
            P1=100.0 + time_s,
            P2=200.0 + time_s,
            P3=300.0 + time_s,
            P4=400.0 + time_s,
            P5=500.0 + time_s,
            P6=600.0 + time_s,
        ),
        gross_flux=AcceptedGrossFluxPacket(
            supply_branch_liquid_outflow_m3_s=1.0e-5,
            supply_branch_gas_inflow_kg_s=2.0e-5,
            mouth_liquid_outflow_m3_s=4.0e-5 if active else 0.0,
            mouth_liquid_inflow_m3_s=1.0e-5,
            mouth_gas_outflow_kg_s=3.0e-6,
            mouth_gas_inflow_kg_s=2.0e-7,
            cumulative_mouth_liquid_outflow_m3=max(time_s - 0.1, 0.0) * 4.0e-5,
        ),
        nodes=AcceptedNodePacket(
            air_supply_liquid_volume_residual_m3_s=0.0,
            air_supply_gas_mass_residual_kg_s=0.0,
            air_supply_momentum_x_residual_N=0.0,
            air_supply_momentum_z_residual_N=0.0,
            riser_liquid_volume_residual_m3_s=0.0,
            riser_gas_mass_residual_kg_s=0.0,
            riser_momentum_x_residual_N=0.0,
            riser_momentum_z_residual_N=0.0,
            node_reaction_impulse_Ns=0.25 * time_s,
        ),
        mouth_event=InternalMouthEventPacket(
            active=active,
            accepted_once=active,
            onset_s=0.1 if active else None,
            acceptance_time_s=0.2 if time_s >= 0.2 - 1.0e-12 else None,
            evidence_status="synthetic_native_internal_event_contract",
        ),
    )


def _samples(*, complete: bool = True) -> tuple[CoupledGeometry, list[CommonAcceptedSample]]:
    geometry = _geometry()
    states = [_state(time_s, geometry) for time_s in (0.0, 0.1, 0.2)]
    samples = [
        CommonAcceptedSample(
            stage2_time_s=0.0,
            state=states[0],
            diagnostics=_diagnostics(0.0, complete=complete),
        )
    ]
    for index in (1, 2):
        samples.append(
            CommonAcceptedSample(
                stage2_time_s=states[index].time_s,
                state=states[index],
                diagnostics=_diagnostics(states[index].time_s, complete=complete),
                ledger_entries_since_previous_sample=(
                    _ledger(states[index - 1], states[index], geometry, index),
                ),
            )
        )
    return geometry, samples


class _Operator:
    def __init__(self, ready: bool) -> None:
        self.production_ready = ready


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_synthetic_2d(path: Path, defs, rows, level: str, offset: float) -> None:
    unavailable = set(
        str(name)
        for name in defs.data["observable_status"][
            "unavailable_from_alpha_water_and_U_only"
        ]
    )
    gate = defs.result["hard_physics_gate"]["fixed_physical_event_test"]
    minimum_volume = float(gate["minimum_connected_water_volume_m3"])
    output = []
    for index, source in enumerate(rows):
        time_s = float(source["time_s"])
        active = time_s >= 0.1 - 1.0e-12
        row = {name: float(source.get(name, math.nan)) for name in defs.canonical}
        for name in unavailable:
            row[name] = math.nan
        row["P1_gauge_Pa"] += offset
        volume = 2.0 * minimum_volume if active else 0.0
        top = 1.10 if active else math.nan
        row.update(
            {
                "sample_index": index,
                "target_time_s": time_s,
                "actual_time_s": time_s,
                "stage2_elapsed_s": time_s,
                "target_absolute_time_s": time_s,
                "actual_absolute_time_s": time_s,
                "vtk_time_error_s": 0.0,
                "launch_component_count": 1 if active else 0,
                "component_water_volume_m3": volume,
                "volume_over_minimum": volume / minimum_volume,
                "bulk_top_q99_z_m": top,
                "bulk_height_above_rim_m": top - 1.02 if active else math.nan,
                "active_raw": float(active),
                "active_persistent": float(active),
                "internal_mouth_event_active": float(active),
                "water_weighted_uz_m_per_s": 0.3 if active else math.nan,
                "eruption_event_id": 1 if active else math.nan,
            }
        )
        output.append(row)
    write_csv(path, output, two_d_columns(defs))
    metadata = {
        "case_id": defs.data["case_id"],
        "physical_condition_count": 1,
        "mesh_level": level,
        "time_origin": "stage_2_air_opening",
        "time_shift_applied_s": 0.0,
        "common_grid_s": defs.dt,
        "definition_sha256": sha256_file(defs.path),
        "result_acceptance_sha256": sha256_file(defs.result_path),
        "common_observables_sha256": sha256_file(defs.common_path),
        "csv_sha256": sha256_file(path),
        "result_marker_written": False,
        "vtk_sha256": {f"synthetic_{level}.vtu": ("ABC"[len(level) % 3] * 64)},
    }
    path.with_name(path.name + ".metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_contract_freezes_fig8_water_velocity_and_published_P1_to_P6_mapping() -> None:
    contract = load_observer_contract()
    semantics = contract.raw["source_semantics"]
    fig8 = semantics["fig8_horizontal_slug_velocity"]
    assert fig8["quantity"] == "water_velocity_magnitude_in_unmixed_middle_part_of_horizontal_slug"
    assert "gas_nose_speed" in fig8["forbidden_substitutions"]
    probes = semantics["published_pressure_probes"]["probes"]
    assert probes["P4"] == {
        "published_x_y_m": [-0.8, 0.0],
        "model_x_z_m": [-0.8, 0.0],
        "canonical_output": "H_upstream_gauge_Pa",
        "engineering_alias": "H_upstream",
    }
    assert probes["P6"]["published_x_y_m"] == [0.1, 0.0]


def test_canonical_observer_exports_native_water_slug_and_two_stream_profiles(tmp_path: Path) -> None:
    geometry, samples = _samples()
    trajectory = build_canonical_trajectory(
        samples,
        geometry=geometry,
        stage2_origin_absolute_s=0.0,
        artifact_role="synthetic_contract_test",
    )
    row = trajectory.rows[0]
    assert row["horizontal_gas_tail_x_m"] == pytest.approx(-1.52)
    assert row["horizontal_gas_nose_x_m"] == pytest.approx(-0.85)
    assert row["horizontal_slug_tail_x_m"] == pytest.approx(-0.85)
    assert row["horizontal_slug_nose_x_m"] == pytest.approx(-0.75)
    assert row["horizontal_slug_velocity_m_s"] == pytest.approx(0.85)
    assert row["gas_arrival_at_riser"] == 0.0
    assert row["riser_connected_water_top_z_m"] == pytest.approx(1.02)
    assert row["H_upstream_gauge_Pa"] == pytest.approx(400.0)
    assert trajectory.rows[-1]["liquid_volume_residual_m3"] == pytest.approx(3e-16)
    assert trajectory.rows[-1]["horizontal_momentum_residual_Ns"] == pytest.approx(9e-16)

    artifacts = write_trajectory_artifacts(trajectory, tmp_path / "artifacts")
    fields, csv_rows = read_numeric_csv(artifacts.canonical_csv)
    assert set(trajectory.contract.canonical_series) <= set(fields)
    assert [row["time_s"] for row in csv_rows] == pytest.approx([0.0, 0.1, 0.2])
    profile_report = validate_profile_npz(
        artifacts.riser_profiles_npz,
        load_definitions(COMPARISON / "OBSERVABLE_DEFINITIONS.yaml"),
    )
    assert profile_report["Qup_Qdown_stored_as_separate_keys"] is True
    with np.load(artifacts.riser_profiles_npz, allow_pickle=False) as archive:
        assert archive["riser_Aup_m2"].shape == (3, 8)
        assert archive["riser_Qup_m3_s"].shape == (3, 8)
        assert archive["riser_Qdown_m3_s"].shape == (3, 8)
        assert archive["riser_gas_velocity_available"].all()
        assert np.allclose(archive["riser_gas_velocity_m_s"], 0.2)
    metadata = json.loads(artifacts.metadata_json.read_text(encoding="utf-8"))
    assert metadata["result_marker_written"] is False
    assert metadata["time_shift_applied_s"] == 0.0
    assert metadata["fig8_velocity_semantics"]["quantity"].startswith("water_velocity")
    assert not any((artifacts.metadata_json.parent / name).exists() for name in (
        "RESULT_ACCEPTED", "RUN_COMPLETE_UNVALIDATED", "ERUPTION_ACCEPTED"
    ))


def test_formal_mode_requires_ready_operator_and_complete_native_diagnostics() -> None:
    geometry, complete = _samples()
    blocked = _Operator(False)
    with pytest.raises(MissingPhysicalClosure, match="production_ready"):
        build_canonical_trajectory(
            complete,
            geometry=geometry,
            stage2_origin_absolute_s=0.0,
            artifact_role="formal_production",
            operator=blocked,
        )
    assert blocked.production_ready is False

    geometry, incomplete = _samples(complete=False)
    ready = _Operator(True)
    with pytest.raises(ObservationContractError, match="unavailable required diagnostics"):
        build_canonical_trajectory(
            incomplete,
            geometry=geometry,
            stage2_origin_absolute_s=0.0,
            artifact_role="formal_production",
            operator=ready,
        )
    assert ready.production_ready is True


def test_missing_native_diagnostics_are_explicit_only_in_synthetic_metadata(tmp_path: Path) -> None:
    geometry, incomplete = _samples(complete=False)
    trajectory = build_canonical_trajectory(
        incomplete,
        geometry=geometry,
        stage2_origin_absolute_s=0.0,
        artifact_role="synthetic_contract_test",
    )
    assert "P1_gauge_Pa" in trajectory.unavailable_by_time["0"]
    assert "mouth_gas_outflow_kg_s" in trajectory.unavailable_by_time["0"]
    artifacts = write_trajectory_artifacts(trajectory, tmp_path)
    metadata = json.loads(artifacts.metadata_json.read_text(encoding="utf-8"))
    assert metadata["unavailable_by_time"]["0"]["P1_gauge_Pa"].startswith("accepted native")
    with artifacts.canonical_csv.open(newline="", encoding="utf-8") as stream:
        first = next(csv.DictReader(stream))
    assert first["P1_gauge_Pa"] == ""


def test_common_times_and_every_ledger_interval_fail_closed() -> None:
    geometry, samples = _samples()
    shifted = list(samples)
    shifted[1] = CommonAcceptedSample(
        stage2_time_s=0.11,
        state=shifted[1].state,
        diagnostics=shifted[1].diagnostics,
        ledger_entries_since_previous_sample=shifted[1].ledger_entries_since_previous_sample,
    )
    with pytest.raises(ObservationContractError, match="contiguous exact"):
        build_canonical_trajectory(
            shifted,
            geometry=geometry,
            stage2_origin_absolute_s=0.0,
            artifact_role="synthetic_contract_test",
        )
    missing = list(samples)
    missing[1] = CommonAcceptedSample(
        stage2_time_s=0.1,
        state=missing[1].state,
        diagnostics=missing[1].diagnostics,
        ledger_entries_since_previous_sample=(),
    )
    with pytest.raises(ObservationContractError, match="omits its accepted conservation ledgers"):
        build_canonical_trajectory(
            missing,
            geometry=geometry,
            stage2_origin_absolute_s=0.0,
            artifact_role="synthetic_contract_test",
        )


def test_exported_csv_and_npz_run_through_full_common_comparator(tmp_path: Path) -> None:
    geometry, samples = _samples()
    trajectory = build_canonical_trajectory(
        samples,
        geometry=geometry,
        stage2_origin_absolute_s=0.0,
        artifact_role="synthetic_contract_test",
    )
    artifacts = write_trajectory_artifacts(trajectory, tmp_path / "one_d")
    defs = load_definitions(COMPARISON / "OBSERVABLE_DEFINITIONS.yaml")
    _, rows = read_numeric_csv(artifacts.canonical_csv)
    meshes = {}
    for level, offset in (("coarse", 0.03), ("medium_refine", 0.01), ("refined", 0.0)):
        path = tmp_path / f"{level}.csv"
        _write_synthetic_2d(path, defs, rows, level, offset)
        meshes[level] = path
    result = compare_1d_to_2d(
        one_d_csv=artifacts.canonical_csv,
        one_d_profile_npz=artifacts.riser_profiles_npz,
        mesh_csvs=meshes,
        defs=defs,
    )
    assert result["hard_eruption_gate"]["status"] == "PASS_ERUPTION_BRANCH_MATCH"
    assert result["time_alignment"]["time_shift_applied_s"] == 0.0
    assert result["one_d_profile_validation"]["Qup_Qdown_stored_as_separate_keys"] is True
    assert result["one_d_profile_to_canonical_consistency"][
        "Qup_profile_vs_canonical_scalar"
    ]["maximum_absolute_difference"] == pytest.approx(0.0)
    assert result["result_marker_written"] is False

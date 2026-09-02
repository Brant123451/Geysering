"""Regression guards for the promoted Case-B shock-fitted closure."""

from __future__ import annotations

import inspect
import csv
import json
from pathlib import Path

import numpy as np

from vw2011_network_twofluid import NetworkCase, run_network


CASE_ROOT = Path(__file__).resolve().parents[1]


def test_selected_caseb_parameters_are_explicit() -> None:
    case = NetworkCase()
    assert case.tower_entry_alpha_min == 0.08
    assert case.slug_train_core_factor == 0.5
    assert case.slug_glug_resistance_scale == 1.50
    assert case.mouth_coverage_alpha == 0.10


def test_selected_closure_has_no_remote_nose_deposition() -> None:
    source = inspect.getsource(run_network)
    assert "Mgr_new[k_top]" not in source
    assert "Mgrs_new[k_top]" not in source
    assert "Jgrs_new[k_top]" not in source
    assert "Mgr_new[0] += dm" in source
    assert "Mgrs_new[0] += dm" in source


def test_native_diagnostics_and_grid_rim_tolerance() -> None:
    case = NetworkCase(t_end=0.04)
    record = run_network(case, verbose=False)
    n = len(record["t"])
    required = (
        "wtop",
        "itop",
        "slug_front_z",
        "field_itop",
        "field_itop_alpha010",
        "field_itop_alpha020",
        "liquid_exchange_balance_m3",
        "slug_transfer_mismatch",
        "gas_inventory",
        "gas_vented_cumulative",
        "water_rim_latched",
        "slug_rim_latched",
    )
    assert all(len(record[key]) == n for key in required)
    assert np.max(np.abs(record["liquid_exchange_balance_m3"])) < 1.0e-20

    closure = record["caseb_selected_closure"]
    assert closure["Yint_primary_observer"] == "shock-fitted slug-train front"
    assert closure["numerical_rim_tolerance_basis"] == (
        "one vertical finite-volume cell"
    )
    assert np.isclose(
        closure["numerical_rim_tolerance_m"], record["dz"]
    )
    assert np.isclose(
        closure["numerical_rim_tolerance_star"],
        record["dz"] / case.riser_height,
    )


def test_canonical_output_passes_shape_and_conservation_audit() -> None:
    metrics_path = CASE_ROOT / "outputs" / "caseB_comparison_metrics.json"
    series_path = CASE_ROOT / "outputs" / "caseB_model_series.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["status"] == "formal_caseB_selected_closure"
    assert metrics["time_shift_applied"] is False
    assert metrics["curve_fit_applied"] is False
    assert metrics["level_curvature"]["model_Yfs_geometric"][
        "convex_accelerating"
    ] is True
    assert metrics["level_curvature"]["model_Yint_primary"][
        "convex_accelerating"
    ] is True
    assert metrics["numerical_rim_event"][
        "primary_Yint_artificial_collapse"
    ] is False
    assert metrics["numerical_rim_event"][
        "primary_Yint_post_latch_min"
    ] == 1.0
    audit = metrics["conservation_audit"]
    assert abs(audit["liquid_relative_change"]) < 2.0e-4
    assert audit["T_mouth_exchange_identity_max_abs_m3"] < 1.0e-20
    assert audit["gas_budget_max_abs_relative_error"] < 2.0e-5
    assert audit["pocket_to_riser_transfer_mismatch_max_abs_kg"] < 1.0e-14

    with series_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    required_columns = {
        "Yfs_star",
        "Yint_star",
        "Yint_star_dynamic_front",
        "Yint_star_field_alpha010",
        "Yint_star_field_alpha020",
        "Yint_star_field_alpha050",
        "liquid_exchange_balance_m3",
        "gas_budget_relative_error",
        "slug_rim_latched",
        "numerical_rim_tolerance_star",
    }
    assert required_columns.issubset(reader.fieldnames or [])
    latched = [row for row in rows if row["slug_rim_latched"] == "1"]
    assert latched
    assert min(float(row["Yint_star"]) for row in latched) == 1.0
    water_latched = [row for row in rows if row["water_rim_latched"] == "1"]
    assert water_latched
    tolerance = float(water_latched[0]["numerical_rim_tolerance_star"])
    assert float(water_latched[0]["Yfs_star"]) >= 1.0 - tolerance - 1.0e-10

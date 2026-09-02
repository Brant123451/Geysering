"""Tests for the read-only Case-A T-junction acceptance audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


CASE = Path(__file__).resolve().parent.parent
SCRIPTS = CASE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from caseA_accept_1d_tjunction_against_2d import audit  # noqa: E402


def _write_fixture(
    tmp_path: Path, *, failing: bool
) -> tuple[Path, Path, Path, Path]:
    time = np.linspace(8.5, 9.2, 15)
    z = (np.arange(10, dtype=float) + 0.5) * 0.610 / 10.0
    reference_height = np.full(time.size, 0.10)
    reference_flux = np.linspace(-2.0e-5, 2.0e-5, time.size)
    reference_gross_up = 8.0e-5 + 0.5 * reference_flux
    reference_gross_down = 8.0e-5 - 0.5 * reference_flux
    reference_bottom_inventory = np.linspace(2.04e-4, 2.22e-4, time.size)
    model_height = np.full(time.size, 0.01 if failing else 0.10)
    alpha_l = np.repeat((model_height / 0.610)[:, None], z.size, axis=1)

    horizontal_mass = np.full((time.size, 4), 1.0e-4)
    vertical_mass = np.full((time.size, 3), 8.0e-5)
    fields_path = tmp_path / "fields.npz"
    np.savez(
        fields_path,
        time=time,
        z=z,
        alpha_l=alpha_l,
        horizontal_gas_mass=horizontal_mass,
        horizontal_gas_momentum=10.0 * horizontal_mass,
        vertical_gas_mass=vertical_mass,
        vertical_gas_momentum=5.0 * vertical_mass,
    )

    liquid_ledger = np.full(time.size, 0.025)
    gas_ledger = np.full(time.size, 0.005)
    diagnostic_speed = np.full(time.size, 200.0 if failing else 20.0)
    model_gross_up = reference_gross_up.copy()
    model_gross_down = reference_gross_down.copy()
    model_bottom_inventory = reference_bottom_inventory.copy()
    if failing:
        liquid_ledger[-1] += 1.0e-5
        gas_ledger[-1] += 1.0e-5
        model_gross_up[:] = -1.0e-6
        model_gross_down[:] = 1.0e-6
        model_bottom_inventory[:] = 2.0e-5
    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "t": time.tolist(),
                "junction_vertical_liquid_flux": reference_flux.tolist(),
                "junction_gross_upward_liquid_flux": model_gross_up.tolist(),
                "junction_gross_downward_liquid_flux": model_gross_down.tolist(),
                "twostream_bottom_0p1m_inventory": model_bottom_inventory.tolist(),
                "total_liquid_including_escape": liquid_ledger.tolist(),
                "total_gas_mass_including_atmosphere": gas_ledger.tolist(),
                "horizontal_gas_mass_error": np.zeros(time.size).tolist(),
                "horizontal_gas_maximum_velocity": diagnostic_speed.tolist(),
                "coupled_gas_maximum_velocity": diagnostic_speed.tolist(),
            }
        ),
        encoding="utf-8",
    )

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "trace": [
                    {
                        "time_s": float(t),
                        "riser_equivalent_liquid_height_m": float(h),
                        "volume_derivative_flux_m3_s": float(q),
                    }
                    for t, h, q in zip(time, reference_height, reference_flux)
                ]
            }
        ),
        encoding="utf-8",
    )

    mouth_reference_path = tmp_path / "mouth_reference.json"
    mouth_reference_path.write_text(
        json.dumps(
            {
                "trace": [
                    {
                        "time_s": float(t),
                        "mouth_gross_up_m3_s": float(q_up),
                        "mouth_gross_down_m3_s": float(-q_down),
                        "bottom_0p10m_equivalent_liquid_volume_m3": float(volume),
                    }
                    for t, q_up, q_down, volume in zip(
                        time,
                        reference_gross_up,
                        reference_gross_down,
                        reference_bottom_inventory,
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    return fields_path, diagnostics_path, baseline_path, mouth_reference_path


def test_matching_conservative_fixture_passes(tmp_path: Path) -> None:
    fields, diagnostics, baseline, mouth_reference = _write_fixture(
        tmp_path, failing=False
    )
    result = audit(
        fields,
        diagnostics,
        baseline,
        mouth_reference_path=mouth_reference,
    )

    assert result["status"] == "PASS"
    assert result["failed_checks"] == []
    assert result["provenance"]["solver_imported"] is False
    assert result["provenance"]["result_prescription"] is False
    assert result["riser_hold_up"]["comparison"]["mae"] < 1.0e-12
    assert result["junction_net_liquid_flux"]["comparison"]["rmse"] < 1.0e-12
    gross = result["junction_gross_liquid_exchange"]
    assert gross["comparison_downward_uses_positive_magnitude"] is True
    assert gross["upward"]["integral_ratio"] == 1.0
    assert gross["downward_magnitude"]["integral_ratio"] == 1.0
    assert gross["net_decomposition_closure"]["maximum_absolute_error_m3_s"] < 1.0e-18
    assert result["bottom_0p1m_liquid_inventory"]["comparison"]["mae"] < 1.0e-18


def test_bad_hold_up_speed_and_ledgers_fail(tmp_path: Path) -> None:
    fields, diagnostics, baseline, mouth_reference = _write_fixture(
        tmp_path, failing=True
    )
    result = audit(
        fields,
        diagnostics,
        baseline,
        mouth_reference_path=mouth_reference,
    )

    assert result["status"] == "FAIL"
    failed = set(result["failed_checks"])
    assert "riser_hold_up_mae" in failed
    assert "riser_hold_up_mean_ratio" in failed
    assert "resolved_gas_speed_guard" in failed
    assert "liquid_conservation_ledger" in failed
    assert "gas_conservation_ledger" in failed
    assert "junction_gross_flux_sign_convention" in failed
    assert "junction_gross_upward_integral_ratio" in failed
    assert "junction_gross_downward_integral_ratio" in failed
    assert "junction_simultaneous_gross_exchange" in failed
    assert "junction_gross_net_decomposition_closure" in failed
    assert "bottom_0p1m_inventory_mae" in failed
    assert "bottom_0p1m_inventory_mean_ratio" in failed


def test_positive_raw_2d_downward_flux_is_rejected(tmp_path: Path) -> None:
    fields, diagnostics, baseline, mouth_reference = _write_fixture(
        tmp_path, failing=False
    )
    payload = json.loads(mouth_reference.read_text(encoding="utf-8"))
    payload["trace"][0]["mouth_gross_down_m3_s"] = 1.0e-5
    mouth_reference.write_text(json.dumps(payload), encoding="utf-8")

    try:
        audit(
            fields,
            diagnostics,
            baseline,
            mouth_reference_path=mouth_reference,
        )
    except ValueError as error:
        assert "gross-down reference violates its negative sign convention" in str(
            error
        )
    else:
        raise AssertionError("audit accepted an invalid 2-D gross-down sign")

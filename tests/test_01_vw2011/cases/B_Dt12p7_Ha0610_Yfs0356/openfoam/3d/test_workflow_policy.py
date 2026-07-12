#!/usr/bin/env python3
"""Regression tests for Case-B evidence and resume policies."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import postprocess as postprocess_module
from postprocess import (
    apply_acceptance,
    is_baseline_full,
    is_baseline_full_physics,
    is_canonical_hold,
    parse_solver_diagnostics,
    should_update_hold_evidence,
    update_sensitivity_csv,
)
from resume_manifest import ENVIRONMENT_KEYS, read_manifest, shell_exports


HERE = Path(__file__).resolve().parent


def baseline_manifest(mesh: str = "base") -> dict:
    return {
        "stage": "full",
        "end_time_s": 10.5,
        "mesh_preset": mesh,
        "valve_mode": "opening",
        "valve_representation": "dissipativeResistance",
        "valve_open_time_s": 0.25,
        "valve_seal_speed_m_per_s": 1.0,
        "initial_air_head_m": 0.610,
        "gas_equation_of_state": "perfectGas",
        "solver": "compressibleInterFlow",
        "two_phase_flow_commit": (
            "de9826f9ffb24f4b635ac97fd388ebd560cfc174"
        ),
        "advection_scheme": "isoAdvection",
        "reconstruction_scheme": "plicRDF",
        "reconstruction_iterations": 5,
        "reconstruction_tolerance": 1e-6,
        "interpolate_normal": False,
        "curvature_model": "RDF",
        "curvature_value_per_m": 0.0,
        "curvature_from_trace": True,
        "max_co": 0.30,
        "max_alpha_co": 0.20,
        "max_capillary_num": 1.0,
        "max_delta_t_s": 0.00025,
        "time_control": "runTime",
        "field_write_interval_s": 0.10,
        "c_alpha": 1.0,
        "n_alpha_bounds": 5,
        "n_alpha_corr": 1,
        "n_alpha_subcycles": 2,
        "n_outer_correctors": 1,
        "n_pressure_correctors": 2,
        "n_non_orthogonal_correctors": 0,
        "alpha_clip": False,
    }


class BaselinePolicyTests(unittest.TestCase):
    def test_only_base_mesh_is_canonical(self) -> None:
        base = baseline_manifest("base")
        refined = baseline_manifest("refined")
        self.assertTrue(is_baseline_full_physics(base))
        self.assertTrue(is_baseline_full_physics(refined))
        self.assertTrue(is_baseline_full(base))
        self.assertFalse(is_baseline_full(refined))

    def test_numerical_sensitivity_is_not_baseline(self) -> None:
        manifest = baseline_manifest()
        manifest["valve_seal_speed_m_per_s"] = 0.5
        self.assertFalse(is_baseline_full_physics(manifest))
        manifest = baseline_manifest()
        manifest["max_delta_t_s"] = 0.0005
        self.assertFalse(is_baseline_full_physics(manifest))
        manifest = baseline_manifest()
        manifest["curvature_model"] = "fitParaboloid"
        self.assertFalse(is_baseline_full_physics(manifest))
        manifest = baseline_manifest()
        manifest["curvature_from_trace"] = False
        self.assertFalse(is_baseline_full_physics(manifest))
        manifest = baseline_manifest()
        manifest["interpolate_normal"] = True
        self.assertFalse(is_baseline_full_physics(manifest))
        manifest = baseline_manifest()
        manifest["reconstruction_iterations"] = 10
        self.assertFalse(is_baseline_full_physics(manifest))
        manifest = baseline_manifest()
        manifest["reconstruction_tolerance"] = 1e-8
        self.assertFalse(is_baseline_full_physics(manifest))
        manifest = baseline_manifest()
        manifest["n_outer_correctors"] = 2
        self.assertFalse(is_baseline_full_physics(manifest))
        manifest = baseline_manifest()
        manifest["time_control"] = "adjustableRunTime"
        self.assertFalse(is_baseline_full_physics(manifest))

    def test_acceptance_can_only_complete_base_mesh(self) -> None:
        for mesh, expected in (("base", "complete"), ("refined", "incomplete")):
            metrics = {
                "run_configuration": baseline_manifest(mesh),
                "mesh_sensitivity": {"available": True},
                "openfoam_3d": {
                    "end_Tstar": 6.1,
                    "geyser": True,
                    "liquid_mass_error_pct_max_abs": 0.1,
                    "gas_mass_error_pct_max_abs": 0.1,
                    "geyser_height_censored_by_domain": False,
                },
            }
            apply_acceptance(metrics, hold_passed=True)
            self.assertEqual(
                metrics["openfoam_3d"]["completion_status"], expected
            )


class HoldEvidenceTests(unittest.TestCase):
    def test_canonical_hold_requires_baseline_controls(self) -> None:
        manifest = baseline_manifest()
        manifest.update(
            stage="hold",
            valve_mode="closed",
            valve_representation="conformalNoSlipBaffle",
            end_time_s=1.0,
        )
        self.assertTrue(is_canonical_hold(manifest))
        manifest["valve_representation"] = "dissipativeResistance"
        self.assertFalse(is_canonical_hold(manifest))
        manifest["valve_representation"] = "conformalNoSlipBaffle"
        manifest["mesh_preset"] = "refined"
        self.assertFalse(is_canonical_hold(manifest))

    def test_passed_hold_cannot_be_downgraded(self) -> None:
        existing = {"hold_test": {"duration_s": 1.0, "passed": True}}
        candidate = {"hold_test": {"duration_s": 2.0, "passed": False}}
        self.assertFalse(should_update_hold_evidence(existing, candidate))

    def test_longer_failed_diagnostic_replaces_shorter_only(self) -> None:
        existing = {"hold_test": {"duration_s": 0.01, "passed": False}}
        shorter = {"hold_test": {"duration_s": 0.001, "passed": False}}
        longer = {"hold_test": {"duration_s": 1.0, "passed": False}}
        self.assertFalse(should_update_hold_evidence(existing, shorter))
        self.assertTrue(should_update_hold_evidence(existing, longer))


class ResumeManifestTests(unittest.TestCase):
    def test_all_materialised_controls_are_exported(self) -> None:
        manifest = baseline_manifest()
        manifest["stage"] = "hold"
        manifest["valve_mode"] = "closed"
        manifest["valve_representation"] = "conformalNoSlipBaffle"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            restored = read_manifest(path)
        exports = shell_exports(restored)
        for environment_key in ENVIRONMENT_KEYS.values():
            self.assertIn(f"export {environment_key}=", exports)

    def test_missing_control_is_rejected(self) -> None:
        manifest = baseline_manifest()
        del manifest["max_co"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "missing controls"):
                read_manifest(path)


class InitialFieldPolicyTests(unittest.TestCase):
    def test_cut_cells_use_mixture_density_for_reduced_pressure(self) -> None:
        pressure = (HERE / "system" / "setExprFieldsDict").read_text()
        reduced = (
            HERE / "system" / "setExprFieldsReducedPressureDict"
        ).read_text()
        allrun = (HERE / "Allrun").read_text()

        self.assertNotIn("mixtureReducedPressure", pressure)
        self.assertIn("mixtureReducedPressure", reduced)
        self.assertIn("(1 - alpha.water)*(p/(287.058*293.15))", reduced)
        self.assertNotIn("reducedWaterPressure", reduced)
        self.assertNotIn("reducedAirPressure", reduced)
        pressure_call = "setExprFields -dict system/setExprFieldsDict.runtime"
        reduced_call = (
            "setExprFields -dict "
            "system/setExprFieldsReducedPressureDict.runtime"
        )
        self.assertLess(allrun.index(pressure_call), allrun.index(reduced_call))

    def test_reduced_pressure_reloads_final_absolute_pressure(self) -> None:
        pressure = (HERE / "system" / "setExprFieldsDict").read_text()
        reduced = (
            HERE / "system" / "setExprFieldsReducedPressureDict"
        ).read_text()

        self.assertIn("readFields (alpha.water);", pressure)
        self.assertIn("readFields (alpha.water p p_rgh T);", reduced)
        self.assertIn("keepPatches true;", reduced)


class TwoPhaseFlowDeckTests(unittest.TestCase):
    def test_rdf_geometric_vof_is_the_default(self) -> None:
        settings = (HERE / "system" / "runSettings.default").read_text()
        solution = (HERE / "system" / "fvSolution").read_text()
        prepare = (HERE / "prepare_run.py").read_text()
        alpha_correctors = (
            HERE / "system" / "alphaCorrectors.default"
        ).read_text()
        pimple_correctors = (
            HERE / "system" / "pimpleCorrectors.default"
        ).read_text()
        surface_forces = (
            HERE / "constant" / "surfaceForces.default"
        ).read_text()
        self.assertIn("advectionScheme         isoAdvection;", settings)
        self.assertIn("reconstructionScheme    plicRDF;", settings)
        self.assertIn("iterations              5;", settings)
        self.assertIn("tol                     1e-6;", settings)
        self.assertIn("interpolateNormal       false;", settings)
        self.assertNotIn("iterations      5;", solution)
        self.assertNotIn("tol             1e-6;", solution)
        self.assertNotIn("interpolateNormal true;", solution)
        self.assertIn("CASEB_INTERPOLATE_NORMAL", prepare)
        self.assertIn("CASEB_RECONSTRUCTION_ITERATIONS", prepare)
        self.assertIn("CASEB_RECONSTRUCTION_TOL", prepare)
        self.assertIn("CASEB_CURVATURE_VALUE", prepare)
        self.assertIn("CASEB_CURV_FROM_TR", prepare)
        self.assertIn('"constantCurvature"', prepare)
        self.assertIn("clip                    false;", settings)
        self.assertIn("surfaceTensionForceModel    RDF;", surface_forces)
        self.assertIn("nAlphaCorr      1;", alpha_correctors)
        self.assertIn("nAlphaSubCycles 2;", alpha_correctors)
        self.assertIn("nOuterCorrectors         1;", pimple_correctors)
        self.assertIn("nCorrectors              2;", pimple_correctors)
        self.assertIn("nNonOrthogonalCorrectors 0;", pimple_correctors)

    def test_compressible_inter_flow_entrypoints_are_consistent(self) -> None:
        control = (HERE / "system" / "controlDict").read_text()
        allrun = (HERE / "Allrun").read_text()
        resume = (HERE / "Allrun.resume").read_text()
        self.assertIn("application     compressibleInterFlow;", control)
        self.assertIn("mpirun -np \"$NP\" compressibleInterFlow", allrun)
        self.assertIn("mpirun -np \"$NP\" compressibleInterFlow", resume)
        self.assertNotIn("log.compressibleInterFoam", allrun + resume)

    def test_closed_mode_uses_conformal_baffle_and_runtime_output(self) -> None:
        mesh = (HERE / "make_mesh.py").read_text()
        allrun = (HERE / "Allrun").read_text()
        allclean = (HERE / "Allclean").read_text()
        baffles = (HERE / "system" / "createBafflesDict.hold").read_text()
        control = (HERE / "system" / "controlDict").read_text()
        run_control = (HERE / "system" / "runControl.default").read_text()
        valve = (HERE / "constant" / "fvOptions").read_text()
        self.assertIn('setPhysicalName(2, valve_group, "valvePlane")', mesh)
        self.assertIn("createBaffles -overwrite", allrun)
        self.assertIn("constant/cellToRegion", allclean)
        self.assertIn("zoneName    valvePlane;", baffles)
        self.assertIn("valveWallUpstream", baffles)
        self.assertIn("valveWallDownstream", baffles)
        self.assertIn('if (mode != "closed")', valve)
        self.assertNotIn("adjustableRunTime", control + run_control)
        self.assertIn("writeControl    runTime;", run_control)
        for field in ("U", "p", "p_rgh", "alpha.water", "T", "T.air", "T.water"):
            text = (HERE / "0.orig" / field).read_text()
            self.assertIn('"valveWall.*"', text)

    def test_phase_models_and_temperatures_are_explicit(self) -> None:
        thermo = (HERE / "constant" / "thermophysicalProperties").read_text()
        self.assertEqual(thermo.count("type        pureMovingPhaseModel;"), 2)
        for phase in ("water", "air"):
            temperature = (HERE / "0.orig" / f"T.{phase}").read_text()
            self.assertIn(f"object      T.{phase};", temperature)

    def test_accounting_uses_solver_phase_density_and_flux(self) -> None:
        control = (HERE / "system" / "controlDict").read_text()
        self.assertIn('"thermo:rho.water"', control)
        self.assertIn('"thermo:rho.air"', control)
        self.assertIn('"alphaPhi.water"', control)
        self.assertIn("CASEB_BOUNDS", control)
        self.assertIn("alphaAtMax", control)
        self.assertIn("CASEB_FORCE_BALANCE", control)
        self.assertIn("hydrostaticForceResidual", control)
        self.assertIn("surfaceTensionForce", control)
        self.assertNotIn("min(max(alpha[cellI]", control)
        self.assertNotIn("fieldFunctionObjects", control)

    def test_solver_diagnostics_capture_rdf_screening_fields(self) -> None:
        diagnostics = parse_solver_diagnostics(
            "\n".join(
                (
                    "Courant Number mean: 0.01 max: 0.3",
                    "Interface Courant Number mean: 0.02 max: 0.2",
                    "Capillary Number: 0.04",
                    "deltaT = 1.5e-05",
                    "Phase-1 volume fraction = 0.2  "
                    "Min(alpha.water) = -2e-09  "
                    "Max(alpha.water) - 1 = 3e-10",
                    "CASEB_BOUNDS Min(alpha.water) = -1e-10 "
                    "Max(alpha.water) = 1 max(mag(U)) = 0.25 "
                    "at location (3.516 0.403 0.0) "
                    "alphaAtMax = 0.4 rhoAtMax = 400 "
                    "KAtMax = -12 pAtMax = 101325 "
                    "p_rghAtMax = 105270 proc = 2 cell = 42",
                    "CASEB_FORCE_BALANCE "
                    "maxHydrostatic(Pa/m) = 10 signed = -10 "
                    "at location (3.5 0.4 0) "
                    "maxSurface(Pa/m) = 20 signed = 20 "
                    "at location (3.51 0.41 0.01) "
                    "maxTotal(Pa/m) = 25 signed = -25 "
                    "at location (3.52 0.42 -0.01)",
                    "ExecutionTime = 12.5 s  ClockTime = 13 s",
                )
            )
        )
        self.assertEqual(diagnostics["max_courant_number"], 0.3)
        self.assertEqual(diagnostics["maximum_alpha_water"], 1.0000000003)
        self.assertEqual(diagnostics["max_velocity_m_per_s"], 0.25)
        self.assertEqual(
            diagnostics["max_velocity_location_m"], [3.516, 0.403, 0.0]
        )
        self.assertEqual(diagnostics["alpha_water_at_max_velocity"], 0.4)
        self.assertEqual(diagnostics["curvature_at_max_velocity_per_m"], -12)
        self.assertEqual(diagnostics["max_velocity_processor"], 2)
        self.assertEqual(diagnostics["max_velocity_local_cell"], 42)
        self.assertEqual(
            diagnostics["max_hydrostatic_force_residual_pa_per_m"], 10
        )
        self.assertEqual(diagnostics["max_surface_tension_force_pa_per_m"], 20)
        self.assertEqual(diagnostics["max_total_force_residual_pa_per_m"], 25)


class SensitivityIndexTests(unittest.TestCase):
    def test_full_run_rows_are_upserted_by_configuration(self) -> None:
        metrics = {
            "run_configuration": baseline_manifest(),
            "openfoam_3d": {
                "end_Tstar": 6.0,
                "completion_status": "incomplete",
                "geyser": True,
            },
        }
        original_outputs = postprocess_module.OUTPUTS
        with tempfile.TemporaryDirectory() as directory:
            postprocess_module.OUTPUTS = Path(directory)
            try:
                update_sensitivity_csv(metrics)
                metrics["openfoam_3d"]["end_Tstar"] = 6.1
                update_sensitivity_csv(metrics)
                with (
                    Path(directory) / "openfoam_3d_sensitivity.csv"
                ).open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
            finally:
                postprocess_module.OUTPUTS = original_outputs
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["end_Tstar"]), 6.1)


if __name__ == "__main__":
    unittest.main()

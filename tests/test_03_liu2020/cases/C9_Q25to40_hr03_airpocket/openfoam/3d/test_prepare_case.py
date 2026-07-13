#!/usr/bin/env python3
"""Focused regression tests for C9 case generation."""

from __future__ import annotations

import json
import unittest

import prepare_case


class PerfectFluidEosTests(unittest.TestCase):
    def test_reference_density_and_wave_speed_are_preserved(self) -> None:
        density = 998.2
        pressure = 101325.0
        temperature = 293.15
        wave_speed = 305.0
        bulk_modulus = density * wave_speed**2

        fluid_constant, rho0, generated_speed = (
            prepare_case.perfect_fluid_eos_parameters(
                bulk_modulus,
                density,
                pressure,
                temperature,
            )
        )

        self.assertAlmostEqual(generated_speed, wave_speed)
        self.assertAlmostEqual(
            rho0 + pressure / (fluid_constant * temperature),
            density,
        )
        self.assertAlmostEqual(
            density * fluid_constant * temperature,
            bulk_modulus,
        )

    def test_invalid_bulk_modulus_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            prepare_case.perfect_fluid_eos_parameters(
                0.0,
                998.2,
                101325.0,
                293.15,
            )


class PocketBodyTracerTests(unittest.TestCase):
    def setUp(self) -> None:
        with prepare_case.PARAM_FILE.open(encoding="utf-8") as stream:
            self.parameters = json.load(stream)
        self.model = self.parameters["model"]
        self.pocket = self.model["pocket_profiles"]["base"]

    def test_tracer_excludes_the_thin_crown_layer(self) -> None:
        dictionary = prepare_case.generate_set_fields(
            self.parameters["paper"],
            self.model,
            self.pocket,
        )

        self.assertEqual(
            dictionary.count("volScalarFieldValue pocketBodyTracer 1"),
            1,
        )
        self.assertIn("volScalarFieldValue pocketBodyTracer 0", dictionary)
        tracer_index = dictionary.index(
            "volScalarFieldValue pocketBodyTracer 1"
        )
        containing_region = dictionary[
            dictionary.rfind("boxToCell", 0, tracer_index) : tracer_index
        ]
        self.assertIn(
            f"box ({self.pocket['tail_x_m']:.8g}",
            containing_region,
        )
        self.assertIn(
            f"({self.pocket['body_nose_x_m']:.8g} 0.12",
            containing_region,
        )

    def test_tracer_uses_air_phase_mass_flux_and_inventory(self) -> None:
        dictionary = prepare_case.make_control_dict(
            "compressibleInterFoam",
            1.25,
            0.1,
            0.35,
            0.2,
            5e-4,
            self.parameters["paper"],
            self.model,
            self.pocket,
        )

        expected_order = [
            "tracerPhysicalAirMassDensity",
            "tracerAirDensityFaces",
            "tracerAirVolumeFlux",
            "tracerAirMassFlux",
            "pocketBodyTracerTransport",
            "matrixPocketBodyTracerMass",
            "totalPocketBodyTracerMassSource",
            "inletPocketBodyTracerMassFlux",
            "gatePocketBodyTracerMassFlux",
            "atmospherePocketBodyTracerMassFlux",
        ]
        indices = [dictionary.index(name) for name in expected_order]
        self.assertEqual(indices, sorted(indices))
        self.assertIn(
            'expression      "phi - alphaPhi0.water";',
            dictionary,
        )
        self.assertIn(
            'expression      "airDensityFaceForTracer * airVolumeFluxForTracer";',
            dictionary,
        )
        self.assertIn("dimensions      [1 0 -1 0 0 0 0];", dictionary)
        self.assertNotIn("fields          (rhoPhi waterMassFluxForTracer);", dictionary)
        self.assertIn("type            boundedPhaseMassTransport;", dictionary)
        self.assertIn('libs            ("libboundedPhaseMassTransport.so");', dictionary)
        self.assertIn("phi             airMassFluxForTracer;", dictionary)
        self.assertIn("alpha           alpha.air;", dictionary)
        self.assertIn("phaseRho        thermo:rho.air;", dictionary)
        self.assertIn("p               p_rgh;", dictionary)
        self.assertIn(
            "carrierFluxResult correctedAirMassFluxForTracer;",
            dictionary,
        )
        self.assertIn("rhoResult       alphaRhoAirForTracer;", dictionary)
        self.assertIn("sigmaResult     pocketBodyTracerSigma;", dictionary)
        self.assertIn("fluxResult      pocketBodyTracerMassFlux;", dictionary)
        self.assertIn("sourceResult    pocketBodyTracerMassSource;", dictionary)
        self.assertIn("boundsTolerance 1e-3;", dictionary)
        self.assertIn("continuityTolerance 1e-4;", dictionary)
        self.assertIn("nCorr           1;", dictionary)
        self.assertIn("nNonOrthCorr    1;", dictionary)
        self.assertIn("nProjectionCorr 2;", dictionary)
        self.assertIn("fields          (pocketBodyTracerMassSource);", dictionary)
        self.assertIn("fields          (pocketBodyTracerSigma);", dictionary)
        self.assertNotIn("clearPocketBodyTracerOutsideAir", dictionary)
        self.assertNotIn("boundPocketBodyTracer", dictionary)
        self.assertEqual(
            dictionary.count("weightField     alphaRhoAirPhysicalForTracer;"),
            4,
        )
        self.assertEqual(
            dictionary.count("weightField     alphaRhoAirForTracer;"),
            0,
        )
        self.assertIn("sigma := sigmaOld - dt*div(flux(phi, s))", dictionary)
        self.assertNotIn("Sp(fvc::ddt(alpha,rho)", dictionary)
        self.assertEqual(
            dictionary.count("fields          (pocketBodyTracerMassFlux);"),
            3,
        )
        self.assertNotIn(
            "phi             rhoPhi;\n        rho             rho;",
            dictionary,
        )


if __name__ == "__main__":
    unittest.main()

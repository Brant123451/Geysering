"""Exact-conservation checks for the Case-A T topology event."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import sys
import unittest

import numpy as np


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_shockfit_network import build_case_a_shockfit_solver  # noqa: E402
from casea_topology_event import (  # noqa: E402
    EventNotAtJunction,
    InactiveLegacyInterfaceError,
    TopologyEventError,
    assert_exact_event_identity,
    create_post_t_topology_event,
)


class CaseATopologyEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = build_case_a_shockfit_solver(dx=0.040)
        initial = self.solver.case_b_initial_state()
        # Exercise exact bit preservation with nonuniform fields, signed zero,
        # and a gas state different from the initial event-free inventory.
        area = np.linspace(
            0.07 * self.solver.section.full_area,
            1.01 * self.solver.section.full_area,
            initial.area.size,
            dtype=np.float64,
        )
        discharge = np.linspace(
            -2.3e-3, 3.7e-3, initial.discharge.size, dtype=np.float64
        )
        discharge[0] = -0.0
        gas = initial.gas.with_volume(0.73 * initial.gas.volume)
        self.event_state = replace(
            initial,
            time=6.4375,
            area=area,
            discharge=discharge,
            gas=gas,
            air_pressure_abs=gas.pressure_abs,
            interface_x=self.solver.junction_face_x,
            interface_speed=0.381,
        )

    def create_event(self):
        return create_post_t_topology_event(
            self.event_state,
            junction_face_x=self.solver.junction_face_x,
            dx=self.solver.dx,
        )

    def test_event_preserves_every_physical_quantity_exactly(self) -> None:
        before_area_bits = self.event_state.area.tobytes()
        before_discharge_bits = self.event_state.discharge.tobytes()
        liquid_before = float(np.sum(self.event_state.area) * self.solver.dx)
        momentum_before = float(
            np.sum(self.event_state.discharge) * self.solver.dx
        )

        post = self.create_event()

        self.assertEqual(post.horizontal.area.tobytes(), before_area_bits)
        self.assertEqual(
            post.horizontal.discharge.tobytes(), before_discharge_bits
        )
        self.assertEqual(post.horizontal.liquid_volume, liquid_before)
        self.assertEqual(post.horizontal.discharge_integral, momentum_before)
        self.assertEqual(post.horizontal.time, self.event_state.time)
        self.assertEqual(post.horizontal.gas_mass, self.event_state.gas.mass)
        self.assertEqual(post.horizontal.gas_volume, self.event_state.gas.volume)
        self.assertEqual(
            post.horizontal.gas_eos_pressure_abs,
            self.event_state.gas.pressure_abs,
        )
        self.assertEqual(
            post.horizontal.air_pressure_abs,
            self.event_state.air_pressure_abs,
        )
        self.assertIs(post.horizontal.gas, self.event_state.gas)
        assert_exact_event_identity(
            self.event_state, post, dx=self.solver.dx
        )

    def test_fields_are_frozen_bit_copies_not_a_remapped_solution(self) -> None:
        post = self.create_event()
        self.assertFalse(np.shares_memory(
            post.horizontal.area, self.event_state.area
        ))
        self.assertFalse(np.shares_memory(
            post.horizontal.discharge, self.event_state.discharge
        ))
        self.assertFalse(post.horizontal.area.flags.writeable)
        self.assertFalse(post.horizontal.discharge.flags.writeable)
        with self.assertRaises(ValueError):
            post.horizontal.area[0] *= 0.5
        with self.assertRaises(ValueError):
            post.horizontal.discharge[0] = 1.0

    def test_legacy_interface_is_inactive_and_cannot_pass_the_t(self) -> None:
        post = self.create_event()
        legacy = post.legacy_interface
        self.assertFalse(legacy.active)
        self.assertEqual(legacy.position, self.solver.junction_face_x)
        with self.assertRaises(InactiveLegacyInterfaceError):
            legacy.advance(1.0e-3)
        with self.assertRaises(ValueError):
            replace(
                legacy,
                position=np.nextafter(
                    self.solver.junction_face_x, np.inf
                ),
            )

    def test_east_and_vertical_fronts_start_at_zero_and_are_independent(self) -> None:
        post = self.create_event()
        self.assertIsNot(post.east_front, post.vertical_front)
        self.assertEqual(post.east_front.position, 0.0)
        self.assertEqual(post.vertical_front.position, 0.0)

        east_moved = post.with_east_front_position(0.117)
        self.assertEqual(east_moved.east_front.position, 0.117)
        self.assertEqual(east_moved.vertical_front.position, 0.0)
        self.assertEqual(
            east_moved.legacy_interface.position,
            self.solver.junction_face_x,
        )

        both_moved = east_moved.with_vertical_front_position(0.063)
        self.assertEqual(both_moved.east_front.position, 0.117)
        self.assertEqual(both_moved.vertical_front.position, 0.063)
        self.assertEqual(
            both_moved.legacy_interface.position,
            self.solver.junction_face_x,
        )

    def test_pre_event_and_crossed_states_are_rejected(self) -> None:
        before = replace(
            self.event_state,
            interface_x=self.solver.junction_face_x - 1.0e-5,
        )
        crossed = replace(
            self.event_state,
            interface_x=np.nextafter(
                self.solver.junction_face_x, np.inf
            ),
        )
        nonfinite = replace(self.event_state, interface_x=float("nan"))
        for invalid in (before, crossed, nonfinite):
            with self.subTest(interface_x=invalid.interface_x):
                with self.assertRaises(EventNotAtJunction):
                    create_post_t_topology_event(
                        invalid,
                        junction_face_x=self.solver.junction_face_x,
                        dx=self.solver.dx,
                    )

    def test_nonlinear_location_tolerance_snaps_only_metadata(self) -> None:
        near = replace(
            self.event_state,
            interface_x=self.solver.junction_face_x - 2.0e-11,
        )
        post = create_post_t_topology_event(
            near,
            junction_face_x=self.solver.junction_face_x,
            dx=self.solver.dx,
            location_tolerance=3.0e-11,
        )
        self.assertEqual(
            post.legacy_interface.position, self.solver.junction_face_x
        )
        self.assertEqual(post.horizontal.area.tobytes(), near.area.tobytes())
        self.assertEqual(
            post.horizontal.discharge.tobytes(), near.discharge.tobytes()
        )
        self.assertEqual(post.horizontal.gas_volume, near.gas.volume)

    def test_event_identity_audit_detects_a_field_edit(self) -> None:
        post = self.create_event()
        changed_area = np.array(post.horizontal.area, copy=True)
        changed_area[3] = np.nextafter(changed_area[3], np.inf)
        changed_area.setflags(write=False)
        changed = replace(
            post,
            horizontal=replace(post.horizontal, area=changed_area),
        )
        with self.assertRaisesRegex(TopologyEventError, "area bits"):
            assert_exact_event_identity(
                self.event_state, changed, dx=self.solver.dx
            )

    def test_event_state_contains_no_arbitrary_tee_storage(self) -> None:
        post = self.create_event()
        names = {item.name for item in fields(post)}
        self.assertNotIn("tee_total_volume", names)
        self.assertNotIn("tee_storage_volume", names)
        self.assertFalse(hasattr(post, "tee_total_volume"))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from campaign2_global_budget import Campaign2GlobalBudget


class Campaign2GlobalBudgetTests(unittest.TestCase):
    def test_internal_tee_exchange_does_not_enter_global_balance(self) -> None:
        ledger = Campaign2GlobalBudget(1.0, 0.10)
        ledger.book_internal_tee(
            liquid_to_riser_m3=0.03,
            gas_to_riser_kg=0.004,
        )
        audit = ledger.audit(
            final_horizontal_liquid_m3=0.60,
            final_vertical_liquid_m3=0.40,
            final_horizontal_gas_kg=0.06,
            final_vertical_gas_kg=0.04,
        )
        self.assertEqual(audit["liquid_residual_m3"], 0.0)
        self.assertEqual(audit["gas_residual_kg"], 0.0)

    def test_reservoir_and_open_top_fluxes_close_both_phase_budgets(self) -> None:
        ledger = Campaign2GlobalBudget(1.0, 0.10)
        ledger.book_reservoir_liquid(0.08)
        ledger.book_top_liquid(0.03)
        ledger.book_top_gas(0.015)
        audit = ledger.audit(
            final_horizontal_liquid_m3=0.65,
            final_vertical_liquid_m3=0.40,
            final_horizontal_gas_kg=0.055,
            final_vertical_gas_kg=0.030,
        )
        self.assertAlmostEqual(audit["liquid_residual_m3"], 0.0, places=15)
        self.assertAlmostEqual(audit["gas_residual_kg"], 0.0, places=15)

    def test_unbooked_liquid_creation_is_visible(self) -> None:
        ledger = Campaign2GlobalBudget(1.0, 0.10)
        audit = ledger.audit(
            final_horizontal_liquid_m3=0.70,
            final_vertical_liquid_m3=0.35,
            final_horizontal_gas_kg=0.06,
            final_vertical_gas_kg=0.04,
        )
        self.assertAlmostEqual(audit["liquid_residual_m3"], 0.05, places=15)

    def test_top_cannot_import_liquid(self) -> None:
        ledger = Campaign2GlobalBudget(1.0, 0.10)
        with self.assertRaisesRegex(ValueError, "cannot donate liquid"):
            ledger.book_top_liquid(-1.0e-6)


if __name__ == "__main__":
    unittest.main()

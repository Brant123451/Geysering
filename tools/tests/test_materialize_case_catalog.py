import csv
import tempfile
import unittest
from pathlib import Path

from tools.materialize_case_catalog import load_series_rows, write_single_row_csv


class MaterializeCaseCatalogTests(unittest.TestCase):
    def test_load_series_rows_indexes_by_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "series.csv"
            source.write_text(
                "run,Dr_mm,H0_m\nB-H1,16,0.66\nB-H2,21,0.66\n",
                encoding="utf-8",
            )

            rows = load_series_rows(source)

            self.assertEqual("21", rows["B-H2"]["Dr_mm"])

    def test_write_single_row_csv_preserves_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "case.csv"
            write_single_row_csv(
                target,
                {"run": "B-H2", "Dr_mm": "21", "H0_m": "0.66"},
            )

            with target.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([{"run": "B-H2", "Dr_mm": "21", "H0_m": "0.66"}], rows)


if __name__ == "__main__":
    unittest.main()

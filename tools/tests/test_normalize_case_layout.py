import tempfile
import unittest
from pathlib import Path

from tools.normalize_case_layout import normalize_case


class NormalizeCaseLayoutTests(unittest.TestCase):
    def test_normalizes_existing_case_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            (case / "digitized").mkdir(parents=True)
            (case / "digitized" / "curve.csv").write_text("x,y\n", encoding="utf-8")
            (case / "paper_scans").mkdir()
            (case / "paper_scans" / "figure.png").write_bytes(b"png")
            (case / "model").mkdir()
            (case / "outputs").mkdir()
            (case / "runner.py").write_text("print('run')\n", encoding="utf-8")
            (case / "report.html").write_text("<html></html>\n", encoding="utf-8")
            (case / "openfoam_2d_caseA").mkdir()
            (case / "README.md").write_text("# Case\n", encoding="utf-8")

            normalize_case(
                case,
                openfoam_dirs={"openfoam_2d_caseA": "2d"},
                dry_run=False,
            )

            self.assertTrue((case / "data" / "digitized" / "curve.csv").is_file())
            self.assertTrue((case / "reference" / "paper_scans" / "figure.png").is_file())
            self.assertTrue((case / "scripts" / "runner.py").is_file())
            self.assertTrue((case / "outputs" / "report.html").is_file())
            self.assertTrue((case / "openfoam" / "2d").is_dir())
            self.assertTrue((case / "config").is_dir())
            self.assertTrue((case / "README.md").is_file())

    def test_normalization_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            case.mkdir()

            normalize_case(case, dry_run=False)
            normalize_case(case, dry_run=False)

            for name in ("config", "data", "model", "scripts", "reference", "outputs"):
                self.assertTrue((case / name).is_dir())


if __name__ == "__main__":
    unittest.main()

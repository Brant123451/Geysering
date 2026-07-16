import tempfile
import unittest
from pathlib import Path

from tools.validate_layout import EXPECTED_CASES, validate_case, validate_repository


def create_complete_case(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "README.md").write_text("# Case\n", encoding="utf-8")
    (path / "manifest.yaml").write_text("id: case\n", encoding="utf-8")
    for name in ("config", "data", "model", "scripts", "reference", "outputs"):
        (path / name).mkdir()


class LayoutValidatorTests(unittest.TestCase):
    def test_case_requires_contract_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            case.mkdir()

            errors = validate_case(case)

            self.assertIn("missing README.md", errors)
            self.assertIn("missing manifest.yaml", errors)
            self.assertIn("missing directory: data", errors)

    def test_complete_minimal_case_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            create_complete_case(case)

            self.assertEqual([], validate_case(case))

    def test_repository_requires_all_expected_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            errors = validate_repository(Path(tmp))

            self.assertTrue(
                any("test_01_vw2011/cases/A_Dt57p1_Ha0305_Yfs0356" in error for error in errors)
            )

    def test_repository_accepts_complete_target_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for test_id, case_ids in EXPECTED_CASES.items():
                test_root = root / "tests" / test_id
                (test_root / "_shared").mkdir(parents=True)
                (test_root / "README.md").write_text(f"# {test_id}\n", encoding="utf-8")
                for case_id in case_ids:
                    create_complete_case(test_root / "cases" / case_id)

            self.assertEqual([], validate_repository(root))

    def test_repository_rejects_root_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for test_id, case_ids in EXPECTED_CASES.items():
                test_root = root / "tests" / test_id
                (test_root / "_shared").mkdir(parents=True)
                (test_root / "README.md").write_text(f"# {test_id}\n", encoding="utf-8")
                for case_id in case_ids:
                    create_complete_case(test_root / "cases" / case_id)
            (root / "_tmp_probe.py").write_text("pass\n", encoding="utf-8")

            errors = validate_repository(root)

            self.assertIn("forbidden temporary file: _tmp_probe.py", errors)


if __name__ == "__main__":
    unittest.main()

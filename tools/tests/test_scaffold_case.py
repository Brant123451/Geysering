import json
import tempfile
import unittest
from pathlib import Path

from tools.scaffold_case import scaffold_case


class ScaffoldCaseTests(unittest.TestCase):
    def test_scaffold_creates_contract_manifest_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            model_source = Path(tmp) / "solver.py"
            model_source.write_text("VALUE = 1\n", encoding="utf-8")

            scaffold_case(
                case,
                manifest={"id": "BH2", "test": "test_02_cong2017"},
                config={"Dr_mm": 21, "H0_m": 0.66},
                model_sources=[model_source],
            )

            for name in ("config", "data", "model", "scripts", "reference", "outputs"):
                self.assertTrue((case / name).is_dir())
            manifest = json.loads((case / "manifest.yaml").read_text(encoding="utf-8"))
            config = json.loads((case / "config" / "case.json").read_text(encoding="utf-8"))
            self.assertEqual("BH2", manifest["id"])
            self.assertEqual(21, config["Dr_mm"])
            self.assertEqual("VALUE = 1\n", (case / "model" / "solver.py").read_text(encoding="utf-8"))

    def test_scaffold_does_not_overwrite_different_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            (case / "model").mkdir(parents=True)
            (case / "model" / "solver.py").write_text("old\n", encoding="utf-8")
            model_source = Path(tmp) / "solver.py"
            model_source.write_text("new\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "model collision"):
                scaffold_case(
                    case,
                    manifest={"id": "case"},
                    config={},
                    model_sources=[model_source],
                )


if __name__ == "__main__":
    unittest.main()

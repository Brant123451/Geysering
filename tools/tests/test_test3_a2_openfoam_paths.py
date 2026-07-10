import importlib.util
import unittest
from pathlib import Path


class Test3A2OpenFOAMPathTests(unittest.TestCase):
    def test_postprocessor_resolves_case_owned_assets(self) -> None:
        root = Path(__file__).resolve().parents[2]
        script = (
            root
            / "tests"
            / "test_03_liu2020"
            / "cases"
            / "A2_Q20to100_openchannel_nogeyser"
            / "openfoam"
            / "3d"
            / "postprocess_compare.py"
        )
        spec = importlib.util.spec_from_file_location("test3_a2_postprocess", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        case_root = script.parents[2]
        self.assertEqual(case_root, module.CASE_ROOT)
        self.assertEqual(case_root / "model", module.MODEL)
        self.assertEqual(case_root / "data" / "digitized", module.DIGITIZED)


if __name__ == "__main__":
    unittest.main()

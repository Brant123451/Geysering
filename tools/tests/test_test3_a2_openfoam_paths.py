import importlib.util
import json
import unittest
from pathlib import Path


class Test3A2OpenFOAMPathTests(unittest.TestCase):
    @property
    def root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def cases(self) -> Path:
        return self.root / "tests" / "test_03_liu2020" / "cases"

    def test_postprocessor_resolves_case_owned_assets(self) -> None:
        script = (
            self.cases
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

    def test_case_scripts_use_canonical_case_directories(self) -> None:
        scripts = (
            self.cases
            / "A2_Q20to100_openchannel_nogeyser"
            / "scripts"
            / "caseA_digitize_and_compare.py",
            self.cases
            / "A2_Q20to100_openchannel_nogeyser"
            / "scripts"
            / "caseA_make_frame_viewer.py",
            self.cases
            / "A2_Q20to100_openchannel_nogeyser"
            / "scripts"
            / "caseA_steady_profile_check.py",
            self.cases
            / "B3_Q20to100_fullpipe_geyser"
            / "scripts"
            / "caseB_digitize_and_compare.py",
            self.cases
            / "C9_Q25to40_hr03_airpocket"
            / "scripts"
            / "caseC_digitize_and_compare.py",
        )
        for script in scripts:
            with self.subTest(script=script):
                text = script.read_text(encoding="utf-8")
                self.assertIn(
                    'CASE_ROOT = Path(__file__).resolve().parents[1]',
                    text,
                )
                self.assertIn('MODEL = CASE_ROOT / "model"', text)
                self.assertIn('OUT = CASE_ROOT / "outputs"', text)
                self.assertNotIn('HERE / "model"', text)
                self.assertNotIn('HERE / "outputs"', text)

        for script in (scripts[0], scripts[3], scripts[4]):
            with self.subTest(digitizer=script):
                text = script.read_text(encoding="utf-8")
                self.assertIn(
                    'SCANS = CASE_ROOT / "reference" / "paper_scans"',
                    text,
                )
                self.assertIn('DIG = CASE_ROOT / "data" / "digitized"', text)

    def test_manifest_entrypoints_exist(self) -> None:
        for case in sorted(self.cases.iterdir()):
            if not case.is_dir():
                continue
            manifest = json.loads(
                (case / "manifest.yaml").read_text(encoding="utf-8")
            )
            for entrypoint in manifest["entrypoints"]:
                with self.subTest(case=case.name, entrypoint=entrypoint):
                    self.assertTrue((case / entrypoint).is_file())

    def test_postprocessor_uses_existing_digitized_csv_names(self) -> None:
        case = self.cases / "A2_Q20to100_openchannel_nogeyser"
        script = case / "openfoam" / "3d" / "postprocess_compare.py"
        text = script.read_text(encoding="utf-8")
        self.assertNotIn("_median.csv", text)
        for probe in ("PT2", "PT3"):
            self.assertTrue(
                (case / "data" / "digitized" / f"fig3_{probe}.csv").is_file()
            )

    def test_generated_reports_link_from_outputs_directory(self) -> None:
        scripts = (
            self.cases
            / "A2_Q20to100_openchannel_nogeyser"
            / "scripts"
            / "caseA_digitize_and_compare.py",
            self.cases
            / "B3_Q20to100_fullpipe_geyser"
            / "scripts"
            / "caseB_digitize_and_compare.py",
            self.cases
            / "C9_Q25to40_hr03_airpocket"
            / "scripts"
            / "caseC_digitize_and_compare.py",
        )
        for script in scripts:
            with self.subTest(script=script):
                text = script.read_text(encoding="utf-8")
                self.assertIn('(OUT / "report.html").write_text', text)
                self.assertNotIn('src="outputs/', text)
                self.assertNotIn('src="paper_scans/', text)
                self.assertNotIn('src="digitized/', text)

    def test_test3_docs_use_repository_paths(self) -> None:
        docs = [
            self.root
            / "tests"
            / "test_03_liu2020"
            / "_shared"
            / "metadata"
            / "paper_reference"
            / "paper_parameters_Liu2020_JHE.md",
            self.cases / "B3_Q20to100_fullpipe_geyser" / "README.md",
            self.cases / "C9_Q25to40_hr03_airpocket" / "README.md",
        ]
        for path in docs:
            with self.subTest(path=path):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn(r"E:\Useless\Anaconda", text)
                self.assertNotIn("papers/liu2020.pdf", text)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SUFFIXES = {".py", ".ps1", ".sh"}
LEGACY_MARKERS = (
    r"E:\Geysering",
    r"D:\tests\Geysering",
    "Vasconcelos_Wright_2011_Geysering",
    "Cong_Chan_Lee_2017_Geyser_Horizontal_Pipe_Vertical_Shaft",
    "Liu_Shao_Zhu_2020_Junction_Chamber_Geyser",
)


class StalePathTests(unittest.TestCase):
    def test_production_scripts_do_not_reference_legacy_absolute_paths(self) -> None:
        violations: list[str] = []
        for path in sorted((ROOT / "tests").rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            relative = path.relative_to(ROOT)
            if "_archive" in relative.parts or "_staging_legacy" in relative.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for marker in LEGACY_MARKERS:
                if marker in text:
                    violations.append(f"{relative.as_posix()}: {marker}")
        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from tools.migrate_geysering_layout import apply_plan, safe_move


class SafeMoveTests(unittest.TestCase):
    def test_dry_run_changes_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "old.txt"
            source.write_text("data", encoding="utf-8")

            result = safe_move(root, "old.txt", "new/new.txt", dry_run=True)

            self.assertEqual("move", result.action)
            self.assertTrue(source.exists())
            self.assertFalse((root / "new" / "new.txt").exists())

    def test_move_creates_parent_and_moves_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "old.txt").write_text("data", encoding="utf-8")

            result = safe_move(root, "old.txt", "new/new.txt", dry_run=False)

            self.assertEqual("move", result.action)
            self.assertFalse((root / "old.txt").exists())
            self.assertEqual("data", (root / "new" / "new.txt").read_text(encoding="utf-8"))

    def test_identical_target_is_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.txt").write_text("same", encoding="utf-8")
            (root / "target.txt").write_text("same", encoding="utf-8")

            result = safe_move(root, "source.txt", "target.txt", dry_run=True)

            self.assertEqual("duplicate", result.action)

    def test_different_target_collision_aborts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.txt").write_text("source", encoding="utf-8")
            (root / "target.txt").write_text("target", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "content collision"):
                safe_move(root, "source.txt", "target.txt", dry_run=True)

    def test_move_cannot_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.txt").write_text("source", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "outside repository"):
                safe_move(root, "source.txt", "../escaped.txt", dry_run=True)

    def test_apply_plan_creates_directories_moves_and_writes_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.txt").write_text("source", encoding="utf-8")
            ledger = root / "ledger.csv"
            plan = {
                "directories": ["tests/test_01/cases"],
                "moves": [
                    {
                        "source": "source.txt",
                        "target": "tests/test_01/cases/source.txt",
                    }
                ],
            }

            results = apply_plan(root, plan, dry_run=False, ledger_path=ledger)

            self.assertEqual("move", results[0].action)
            self.assertTrue((root / "tests" / "test_01" / "cases").is_dir())
            self.assertTrue((root / "tests" / "test_01" / "cases" / "source.txt").is_file())
            self.assertIn("source.txt", ledger.read_text(encoding="utf-8"))

    def test_apply_plan_skips_missing_optional_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {
                "moves": [
                    {
                        "source": "missing.txt",
                        "target": "target.txt",
                        "required": False,
                    }
                ]
            }

            results = apply_plan(root, plan, dry_run=True)

            self.assertEqual("skip_missing", results[0].action)


if __name__ == "__main__":
    unittest.main()

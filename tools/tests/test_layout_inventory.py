import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.layout_inventory import inventory, write_inventory


class LayoutInventoryTests(unittest.TestCase):
    def test_inventory_skips_git_and_hashes_small_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "case").mkdir()
            (root / "case" / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "index").write_bytes(b"ignored")

            rows = inventory(root, hash_limit=1024)

            self.assertEqual(["case/data.csv"], [row["path"] for row in rows])
            self.assertEqual(64, len(str(rows[0]["sha256"])))

    def test_inventory_leaves_large_file_hash_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "large.bin").write_bytes(b"12345")

            rows = inventory(root, hash_limit=4)

            self.assertEqual("", rows[0]["sha256"])

    def test_write_inventory_creates_csv_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            (root / "alpha").mkdir()
            (root / "alpha" / "a.txt").write_text("a", encoding="utf-8")
            (root / "b.txt").write_text("bb", encoding="utf-8")
            csv_path = Path(tmp) / "inventory.csv"
            summary_path = Path(tmp) / "summary.json"

            write_inventory(root, csv_path, summary_path, hash_limit=1024)

            with csv_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(2, len(rows))
            self.assertEqual(2, summary["file_count"])
            self.assertEqual(3, summary["total_bytes"])
            self.assertEqual({"alpha": 1, "(root)": 1}, summary["files_by_top_level"])


if __name__ == "__main__":
    unittest.main()

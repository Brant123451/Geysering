"""Run Cong 2017 B-H6 with this Case's frozen model."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scan_common import run_one

CASE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((CASE_ROOT / "config" / "case.json").read_text(encoding="utf-8"))
RESULT = run_one(CONFIG["Dr_mm"], CONFIG["L0_m"], CONFIG["H0_m"])
RESULT["run"] = CONFIG["run"]
output = CASE_ROOT / "outputs" / "rerun_result.json"
output.write_text(json.dumps(RESULT, indent=2) + "\n", encoding="utf-8")
print(output)

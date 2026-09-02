from __future__ import annotations

import sys
from pathlib import Path


COMPARISON = Path(__file__).resolve().parents[1]
if str(COMPARISON) not in sys.path:
    sys.path.insert(0, str(COMPARISON))

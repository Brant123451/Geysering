#!/usr/bin/env python3
"""Compatibility entry point for the B-H3 native-time 1D--2D viewer."""
from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "comparison"))

from build_html import main  # noqa: E402


if __name__ == "__main__":
    main()

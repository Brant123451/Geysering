#!/usr/bin/env python3
"""Convert a mouth-driver CSV into OpenFOAM uniformFixedValue tables."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parents[1]
SLIT_WIDTH_M = 0.0017159574


def pick(row: dict[str, str], *names: str) -> float:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return float(row[name])
    raise KeyError(f"CSV needs one of these columns: {', '.join(names)}")


def optional_pick(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return float(row[name])
    return None


def load_rows(path: Path) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t = pick(row, "local_time_s", "t_local_s")
            alpha = pick(
                row,
                "forcing_alpha_uniform",
                "alpha_area_mean",
                "alpha_mean",
                "alpha_water",
            )
            uy = pick(
                row,
                "forcing_Uy_uniform_m_per_s",
                "alpha_weighted_Uy_m_per_s",
                "Uy_alpha_weighted",
                "Uy_m_per_s",
            )
            if not all(math.isfinite(x) for x in (t, alpha, uy)):
                raise ValueError(f"Non-finite CSV row: {row}")
            if alpha < -1.0e-10 or alpha > 1.0 + 1.0e-10:
                raise ValueError(f"Water fraction outside [0,1]: {row}")
            alpha = min(1.0, max(0.0, alpha))
            line_flux = optional_pick(
                row,
                "water_line_flux_m2_per_s",
                "water_line_flux_m2_s",
            )
            if line_flux is not None:
                if not math.isfinite(line_flux):
                    raise ValueError(f"Non-finite water line flux: {row}")
                represented_flux = SLIT_WIDTH_M * alpha * uy
                tolerance = 1.0e-12 + 1.0e-7 * max(abs(line_flux), abs(represented_flux))
                if abs(line_flux - represented_flux) > tolerance:
                    raise ValueError(
                        "Uniform alpha/U do not preserve the supplied liquid line flux: "
                        f"q={line_flux:.12g}, W*alpha*Uy={represented_flux:.12g}, row={row}"
                    )
            rows.append((t, alpha, uy))
    if len(rows) < 2:
        raise ValueError("Driver CSV needs at least two rows")
    if abs(rows[0][0]) > 1.0e-12:
        raise ValueError("Driver CSV must start at local_time_s=0")
    if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError("Driver times must be strictly increasing")
    return rows


def write_u(path: Path, rows: list[tuple[float, float, float]]) -> None:
    lines = [
        "towerInlet",
        "{",
        "    type uniformFixedValue;",
        "    uniformValue",
        "    {",
        "        type table;",
        "        outOfBounds clamp;",
        "        interpolationScheme linear;",
        "        values",
        "        (",
    ]
    lines.extend(f"            ({t:.9g} (0 {uy:.9g} 0))" for t, _, uy in rows)
    lines.extend(
        [
            "        );",
            "    }",
            "    value uniform (0 0 0);",
            "}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_alpha(path: Path, rows: list[tuple[float, float, float]]) -> None:
    lines = [
        "towerInlet",
        "{",
        "    type uniformInletOutlet;",
        "    phi phi;",
        "    uniformInletValue",
        "    {",
        "        type table;",
        "        outOfBounds clamp;",
        "        interpolationScheme linear;",
        "        values",
        "        (",
    ]
    lines.extend(f"            ({t:.9g} {min(1.0, max(0.0, alpha)):.9g})" for t, alpha, _ in rows)
    lines.extend(
        [
            "        );",
            "    }",
            "    value uniform 0;",
            "}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "0.orig" / "includes")
    args = parser.parse_args()
    rows = load_rows(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_u(args.output_dir / "towerInlet_U", rows)
    write_alpha(args.output_dir / "towerInlet_alpha.water", rows)
    print(f"generated OpenFOAM tables from {args.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

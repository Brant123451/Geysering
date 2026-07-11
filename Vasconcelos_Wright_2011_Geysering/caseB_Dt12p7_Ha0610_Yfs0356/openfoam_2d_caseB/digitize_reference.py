#!/usr/bin/env python3
"""Reproduce the Case B data digitized from V&W (2011), Figs. 6 and 8."""
from __future__ import annotations

import csv
import hashlib
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter


HERE = Path(__file__).resolve().parent
DIGITIZED = HERE / "digitized"

FIG6_URL = "https://attachments.academia-assets.com/67996050/27/img/1.jpg"
FIG8_URL = "https://attachments.academia-assets.com/67996050/29/img/1.jpg"
FIG6_SHA256 = "3406a4fbf78bd304c5e38b7ab14b616f33525b8c19aa7392f17030f17c0cfd5c"
FIG8_SHA256 = "6d19fc8c9fbf7971a33a6f36ef57a98a2751219fa0567616b41e21e12b997d8e"

# Figure 6, middle panel: T*=0..5 and H*=0..1.5.
P_X0, P_X5 = 255.0, 478.0
P_Y0, P_Y15 = 323.0, 177.0

# Figure 8, middle panel: T*=3..5 and Y*=0..1.
L_X3, L_X5 = 263.0, 475.0
L_Y0, L_Y1 = 306.0, 173.0


def fetch(url: str, expected_sha256: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "caseB-digitizer/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(f"Unexpected SHA-256 for {url}: {digest}")
    target.write_bytes(payload)


def pressure_from_figure(path: Path) -> None:
    image = np.asarray(Image.open(path).convert("L"))
    x_pixels = np.arange(260, 477)
    grid_x = np.array([255, 300, 344, 389, 433, 478])
    rows: list[tuple[float, float, float]] = []

    for x in x_pixels:
        if np.min(np.abs(grid_x - x)) <= 2:
            rows.append((np.nan, np.nan, np.nan))
            continue
        if x < 422:
            low, high = 228, 270
        elif x < 455:
            low, high = 228, 322
        else:
            low, high = 305, 326
        yy, _ = np.where(image[low : high + 1, x - 2 : x + 3] < 175)
        y_values = yy + low
        # Exclude the printed horizontal grid; the H*=0 trace is retained.
        y_values = y_values[
            ~np.isin(y_values, [176, 177, 178, 224, 225, 226, 273, 274, 275])
        ]
        if len(y_values) >= 3:
            # Pixel y increases downwards: q85 is the lower H* envelope.
            rows.append(tuple(np.quantile(y_values, [0.85, 0.50, 0.15])))
        else:
            rows.append((np.nan, np.nan, np.nan))

    pixel_y = np.asarray(rows)
    for column in range(3):
        valid = np.isfinite(pixel_y[:, column])
        pixel_y[:, column] = np.interp(
            x_pixels, x_pixels[valid], pixel_y[valid, column]
        )
        pixel_y[:, column] = median_filter(
            pixel_y[:, column], size=5, mode="nearest"
        )
    pixel_y[:, 1] = np.clip(pixel_y[:, 1], pixel_y[:, 2], pixel_y[:, 0])

    tstar = (x_pixels - P_X0) * 5.0 / (P_X5 - P_X0)
    hstar = (P_Y0 - pixel_y) * 1.5 / (P_Y0 - P_Y15)
    hstar = np.clip(hstar, 0.0, 1.5)
    # Values within the two-pixel baseline thickness represent H*=0.
    hstar[hstar < 0.04] = 0.0

    output = DIGITIZED / "fig6_caseB_pressure_envelope.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "Tstar",
                "Hstar_min",
                "Hstar_med",
                "Hstar_max",
                "pixel_x",
                "pixel_y_q85",
                "pixel_y_median",
                "pixel_y_q15",
            ]
        )
        writer.writerows(
            np.column_stack((tstar, hstar, x_pixels, pixel_y))
        )


def levels_from_pixels() -> None:
    source = DIGITIZED / "fig8_caseB_level_pixels.csv"
    output = DIGITIZED / "fig8_caseB_levels.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    with output.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "kind",
            "run",
            "symbol",
            "Tstar",
            "Ystar",
            "pixel_x",
            "pixel_y",
            "Tstar_uncertainty",
            "Ystar_uncertainty",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            x = float(row["pixel_x"])
            y = float(row["pixel_y"])
            writer.writerow(
                {
                    "kind": row["kind"],
                    "run": row["run"],
                    "symbol": row["symbol"],
                    "Tstar": 3.0 + (x - L_X3) * 2.0 / (L_X5 - L_X3),
                    "Ystar": (L_Y0 - y) / (L_Y0 - L_Y1),
                    "pixel_x": row["pixel_x"],
                    "pixel_y": row["pixel_y"],
                    "Tstar_uncertainty": 2.0 * 2.0 / (L_X5 - L_X3),
                    "Ystar_uncertainty": 2.0 / (L_Y0 - L_Y1),
                }
            )


def main() -> None:
    DIGITIZED.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="caseB-digitize-") as tmp:
        fig6 = Path(tmp) / "figure6.jpg"
        fig8 = Path(tmp) / "figure8.jpg"
        fetch(FIG6_URL, FIG6_SHA256, fig6)
        fetch(FIG8_URL, FIG8_SHA256, fig8)
        # Fetching Fig. 8 and checking its hash ties the retained pixel picks
        # to the exact source raster; level conversion itself uses the CSV.
        pressure_from_figure(fig6)
    levels_from_pixels()
    print("CASE_B_DIGITIZATION_DONE")


if __name__ == "__main__":
    main()

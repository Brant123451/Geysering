#!/usr/bin/env python3
"""Render lightweight SVG frames from the geometry-aligned B-H3 1D run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
NPZ = HERE / "model_1d" / "paper_layout_1d_frames.npz"
OUT = HERE / "model_1d" / "frames"
MANIFEST = HERE / "model_1d" / "frames.json"
PIPE_LENGTH = 6.59
PIPE_D = 0.050
RISER_X = 3.47
RISER_D = 0.026
RISER_H = 1.80
DISPLAY_TOP = 3.05
BLUE = "#2778d8"
AIR = "#f7f9fc"
WALL = "#263442"


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Microsoft YaHei,sans-serif;fill:#263442}</style>',
        f'<text x="18" y="24" font-size="16" font-weight="600">{title}</text>',
    ]


def full_svg(time_s: float, xt: np.ndarray, alt: np.ndarray, dx: float,
             zr: np.ndarray, alr: np.ndarray, agr: np.ndarray,
             wtop: float, itop: float, head: float) -> str:
    width, height = 1300, 520
    left, right, top, bottom = 54.0, 24.0, 42.0, 38.0
    xscale = (width - left - right) / PIPE_LENGTH
    zscale = (height - top - bottom) / (DISPLAY_TOP + PIPE_D)
    xpix = lambda x: left + x * xscale
    zpix = lambda z: top + (DISPLAY_TOP - z) * zscale
    rows = svg_header(width, height, f"Project 1D model · B-H3 · t = {time_s:.2f} s")
    rows.append(f'<text x="18" y="45" font-size="12" fill="#66717e">Yfs={wtop:.3f} m · Yint={itop:.3f} m · pocket head={head:.3f} m</text>')

    # Air background and area-fraction water layer in the horizontal pipe.
    rows.append(f'<rect x="{xpix(0):.2f}" y="{zpix(0):.2f}" width="{PIPE_LENGTH*xscale:.2f}" height="{PIPE_D*zscale:.2f}" fill="{AIR}"/>')
    for x, fraction in zip(xt, alt):
        f = float(np.clip(fraction, 0.0, 1.0))
        if f <= 0.003:
            continue
        # The horizontal-pipe crown is z=0 and its invert is z=-PIPE_D.
        # A liquid fraction f therefore starts at z=-PIPE_D*(1-f), not
        # z=-PIPE_D*f.  The latter sends a full-pipe cell below the pipe and
        # out of the SVG viewport.
        rows.append(
            f'<rect x="{xpix(x-0.5*dx):.2f}" y="{zpix(-PIPE_D*(1.0-f)):.2f}" '
            f'width="{max(dx*xscale+0.15,0.3):.2f}" height="{PIPE_D*f*zscale+0.15:.2f}" fill="{BLUE}"/>'
        )

    # Riser volume fractions: resolved gas core centred between symmetric films.
    rx0 = xpix(RISER_X - 0.5 * RISER_D)
    rw = RISER_D * xscale
    rows.append(f'<rect x="{rx0:.2f}" y="{zpix(RISER_H):.2f}" width="{rw:.2f}" height="{RISER_H*zscale:.2f}" fill="{AIR}"/>')
    dz = float(np.median(np.diff(zr)))
    for z, liquid, gas in zip(zr, alr, agr):
        if z > RISER_H or liquid <= 0.002:
            continue
        gas_fraction = float(np.clip(gas, 0.0, 0.98))
        film = 0.5 * (1.0 - gas_fraction) * rw
        yy = zpix(z + 0.5 * dz)
        hh = dz * zscale + 0.15
        if gas_fraction < 0.01:
            rows.append(f'<rect x="{rx0:.2f}" y="{yy:.2f}" width="{rw:.2f}" height="{hh:.2f}" fill="{BLUE}"/>')
        else:
            rows.append(f'<rect x="{rx0:.2f}" y="{yy:.2f}" width="{film:.2f}" height="{hh:.2f}" fill="{BLUE}"/>')
            rows.append(f'<rect x="{rx0+rw-film:.2f}" y="{yy:.2f}" width="{film:.2f}" height="{hh:.2f}" fill="{BLUE}"/>')

    # Apparatus outline and paper locations.
    y0, yb = zpix(0), zpix(-PIPE_D)
    rows.extend([
        f'<path d="M {xpix(0):.2f} {y0:.2f} H {rx0:.2f} M {rx0+rw:.2f} {y0:.2f} H {xpix(PIPE_LENGTH):.2f} '
        f'M {xpix(0):.2f} {yb:.2f} H {xpix(PIPE_LENGTH):.2f} M {xpix(0):.2f} {yb:.2f} V {y0:.2f} '
        f'M {xpix(PIPE_LENGTH):.2f} {yb:.2f} V {y0:.2f} M {rx0:.2f} {y0:.2f} V {zpix(RISER_H):.2f} '
        f'M {rx0+rw:.2f} {y0:.2f} V {zpix(RISER_H):.2f}" fill="none" stroke="{WALL}" stroke-width="1.2"/>',
        f'<line x1="{rx0-5:.2f}" x2="{rx0+rw+5:.2f}" y1="{zpix(RISER_H):.2f}" y2="{zpix(RISER_H):.2f}" stroke="#d23b31" stroke-dasharray="4 3"/>',
        f'<line x1="{xpix(5.98):.2f}" x2="{xpix(5.98):.2f}" y1="{yb-5:.2f}" y2="{y0+5:.2f}" stroke="#111827" stroke-dasharray="3 3"/>',
        f'<text x="{xpix(RISER_X):.2f}" y="{zpix(RISER_H)-7:.2f}" text-anchor="middle" font-size="11">riser x=3.47 m, rim=1.80 m</text>',
        f'<text x="{xpix(5.98):.2f}" y="{yb+22:.2f}" text-anchor="middle" font-size="11">valve x=5.98 m</text>',
        f'<text x="{xpix(6.285):.2f}" y="{yb+38:.2f}" text-anchor="middle" font-size="11">L0=0.61 m atmospheric pocket</text>',
        f'<text x="{left:.2f}" y="{height-9:.2f}" font-size="11" fill="#66717e">Blue: water; white: air. Pipe and riser use the paper dimensions; external plume is outside the 1D domain.</text>',
        '</svg>',
    ])
    return "\n".join(rows)


def zoom_svg(time_s: float, zr: np.ndarray, alr: np.ndarray, agr: np.ndarray,
             wtop: float, itop: float) -> str:
    width, height = 330, 720
    x0, rw = 108.0, 114.0
    top, bottom = 54.0, 42.0
    scale = (height - top - bottom) / RISER_H
    zpix = lambda z: top + (RISER_H - z) * scale
    rows = svg_header(width, height, f"1D riser zoom · t={time_s:.2f} s")
    rows.append(f'<rect x="{x0}" y="{top}" width="{rw}" height="{RISER_H*scale:.2f}" fill="{AIR}"/>')
    dz = float(np.median(np.diff(zr)))
    for z, liquid, gas in zip(zr, alr, agr):
        if z > RISER_H or liquid <= 0.002:
            continue
        gf = float(np.clip(gas, 0.0, 0.98))
        film = 0.5 * (1.0 - gf) * rw
        yy, hh = zpix(z + 0.5 * dz), dz * scale + 0.25
        if gf < 0.01:
            rows.append(f'<rect x="{x0}" y="{yy:.2f}" width="{rw}" height="{hh:.2f}" fill="{BLUE}"/>')
        else:
            rows.append(f'<rect x="{x0}" y="{yy:.2f}" width="{film:.2f}" height="{hh:.2f}" fill="{BLUE}"/>')
            rows.append(f'<rect x="{x0+rw-film:.2f}" y="{yy:.2f}" width="{film:.2f}" height="{hh:.2f}" fill="{BLUE}"/>')
    rows.extend([
        f'<rect x="{x0}" y="{top}" width="{rw}" height="{RISER_H*scale:.2f}" fill="none" stroke="{WALL}" stroke-width="1.3"/>',
        f'<line x1="{x0-10}" x2="{x0+rw+10}" y1="{top}" y2="{top}" stroke="#d23b31" stroke-dasharray="5 4"/>',
        f'<line x1="{x0}" x2="{x0+rw}" y1="{zpix(itop):.2f}" y2="{zpix(itop):.2f}" stroke="#ec7a19" stroke-dasharray="3 3"/>',
        f'<text x="18" y="48" font-size="12">Yfs = {wtop:.3f} m</text>',
        f'<text x="18" y="66" font-size="12">Yint = {itop:.3f} m</text>',
        f'<text x="{x0+rw/2}" y="{height-12}" text-anchor="middle" font-size="11" fill="#66717e">gas core centred for 1D volume-fraction display</text>',
        '</svg>',
    ])
    return "\n".join(rows)


def main() -> None:
    data = np.load(NPZ, allow_pickle=False)
    OUT.mkdir(parents=True, exist_ok=True)
    times = np.arange(0.0, 13.0001, 0.05)
    src_t = data["frames_t"]
    indices = np.asarray([int(np.argmin(np.abs(src_t - t))) for t in times])
    manifest: list[dict[str, float | str]] = []
    for frame_no, index in enumerate(indices):
        time_s = float(src_t[index])
        wtop = float(data["wtop"][index])
        itop = float(data["itop"][index])
        head = float(data["pocket_head"][index])
        full = OUT / f"full_{frame_no:04d}.svg"
        zoom = OUT / f"zoom_{frame_no:04d}.svg"
        full.write_text(full_svg(
            time_s, data["xt"], data["frames_alt"][index], float(data["dx"][0]),
            data["zr"], data["frames_alr"][index], data["frames_agr"][index],
            wtop, itop, head,
        ), encoding="utf-8")
        zoom.write_text(zoom_svg(
            time_s, data["zr"], data["frames_alr"][index], data["frames_agr"][index],
            wtop, itop,
        ), encoding="utf-8")
        manifest.append({
            "file": f"model_1d/frames/{full.name}",
            "riserFile": f"model_1d/frames/{zoom.name}",
            "time": time_s,
            "Yfs": wtop,
            "Yint": itop,
            "head": head,
        })
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} 1D frame pairs and {MANIFEST}")


if __name__ == "__main__":
    main()

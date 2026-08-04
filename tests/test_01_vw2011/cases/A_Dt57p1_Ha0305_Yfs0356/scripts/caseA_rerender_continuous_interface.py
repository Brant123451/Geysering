"""Re-render an archived Case-A 1D solution with a continuous interface.

The solver stores finite-volume cell averages.  Drawing one rectangle per
cell makes a resolved long wave look serrated even when the cell averages are
well behaved.  This utility converts liquid area fraction to the physically
consistent circular-pipe depth and reconstructs a continuous interface through
the cell-centre values.  The default shape-preserving cubic has zero slope at
resolved extrema, so one-cell crests are rounded instead of triangular.  It
introduces no new extrema and does not average, filter, or alter archived state.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle


HERE = Path(__file__).resolve().parent
CASE = HERE.parent
OUT = CASE / "outputs"

D = 0.094
DT = 0.0571
L = 4.006
TOWER_X = 3.516
TOWER_H = 0.610


def circular_depth_fraction(area_fraction: np.ndarray) -> np.ndarray:
    """Invert circular-segment area fraction by monotone interpolation."""

    fraction = np.clip(np.asarray(area_fraction, dtype=float), 0.0, 1.0)
    theta = np.linspace(0.0, 2.0 * math.pi, 8193)
    area = (theta - np.sin(theta)) / (2.0 * math.pi)
    depth = 0.5 * (1.0 - np.cos(0.5 * theta))
    return np.interp(fraction, area, depth)


def shape_preserving_cubic(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
) -> np.ndarray:
    """Evaluate a Fritsch--Carlson cubic Hermite curve without overshoot."""

    nodes = np.asarray(x, dtype=float)
    values = np.asarray(y, dtype=float)
    query = np.asarray(x_new, dtype=float)
    if nodes.ndim != 1 or values.shape != nodes.shape or nodes.size < 2:
        raise ValueError("cubic interface reconstruction requires equal 1-D arrays")
    spacing = np.diff(nodes)
    if np.any(spacing <= 0.0):
        raise ValueError("interface reconstruction nodes must be strictly increasing")
    secant = np.diff(values) / spacing
    derivative = np.zeros_like(values)
    if values.size == 2:
        derivative[:] = secant[0]
    else:
        for index in range(1, values.size - 1):
            left = secant[index - 1]
            right = secant[index]
            if left == 0.0 or right == 0.0 or left * right <= 0.0:
                derivative[index] = 0.0
            else:
                weight_left = 2.0 * spacing[index] + spacing[index - 1]
                weight_right = spacing[index] + 2.0 * spacing[index - 1]
                derivative[index] = (weight_left + weight_right) / (
                    weight_left / left + weight_right / right
                )
        derivative[0] = (
            (2.0 * spacing[0] + spacing[1]) * secant[0]
            - spacing[0] * secant[1]
        ) / (spacing[0] + spacing[1])
        derivative[-1] = (
            (2.0 * spacing[-1] + spacing[-2]) * secant[-1]
            - spacing[-1] * secant[-2]
        ) / (spacing[-1] + spacing[-2])
        for endpoint, local_secant, adjacent_secant in (
            (0, secant[0], secant[1]),
            (-1, secant[-1], secant[-2]),
        ):
            if derivative[endpoint] * local_secant <= 0.0:
                derivative[endpoint] = 0.0
            elif (
                local_secant * adjacent_secant < 0.0
                and abs(derivative[endpoint]) > 3.0 * abs(local_secant)
            ):
                derivative[endpoint] = 3.0 * local_secant

    interval = np.searchsorted(nodes, query, side="right") - 1
    interval = np.clip(interval, 0, nodes.size - 2)
    width = nodes[interval + 1] - nodes[interval]
    coordinate = np.clip((query - nodes[interval]) / width, 0.0, 1.0)
    h00 = 2.0 * coordinate**3 - 3.0 * coordinate**2 + 1.0
    h10 = coordinate**3 - 2.0 * coordinate**2 + coordinate
    h01 = -2.0 * coordinate**3 + 3.0 * coordinate**2
    h11 = coordinate**3 - coordinate**2
    result = (
        h00 * values[interval]
        + h10 * width * derivative[interval]
        + h01 * values[interval + 1]
        + h11 * width * derivative[interval + 1]
    )
    return np.clip(result, np.min(values), np.max(values))


def compact_cubic_bspline(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
) -> np.ndarray:
    """Reconstruct cell averages with a non-negative compact cubic B-spline.

    The kernel has support over four cell centres, forms a C2-continuous
    surface and has unit partition.  Consequently constants and the global
    liquid level are retained, while no value can exceed the local data range.
    Constant ghost values are used only within two cells of a closed end.
    """

    nodes = np.asarray(x, dtype=float)
    values = np.asarray(y, dtype=float)
    query = np.asarray(x_new, dtype=float)
    if nodes.ndim != 1 or values.shape != nodes.shape or nodes.size < 2:
        raise ValueError("B-spline interface reconstruction requires equal 1-D arrays")
    spacing = np.diff(nodes)
    cell_width = float(np.median(spacing))
    if cell_width <= 0.0 or not np.allclose(spacing, cell_width, rtol=1.0e-8):
        raise ValueError("B-spline interface reconstruction requires a uniform grid")

    ghost_nodes = np.r_[nodes[0] - 2.0 * cell_width,
                        nodes[0] - cell_width,
                        nodes,
                        nodes[-1] + cell_width,
                        nodes[-1] + 2.0 * cell_width]
    ghost_values = np.r_[values[0], values[0], values, values[-1], values[-1]]
    distance = np.abs(
        (query[:, None] - ghost_nodes[None, :]) / cell_width
    )
    weights = np.zeros_like(distance)
    inner = distance < 1.0
    outer = (distance >= 1.0) & (distance < 2.0)
    weights[inner] = (
        2.0 / 3.0 - distance[inner] ** 2 + 0.5 * distance[inner] ** 3
    )
    weights[outer] = (2.0 - distance[outer]) ** 3 / 6.0
    total = np.sum(weights, axis=1)
    result = np.sum(weights * ghost_values[None, :], axis=1) / np.maximum(
        total, 1.0e-15
    )
    return np.clip(result, np.min(values), np.max(values))


def draw_outline(ax) -> None:
    riser_left = TOWER_X - 0.5 * DT
    riser_right = TOWER_X + 0.5 * DT
    wall = dict(color="0.35", linewidth=0.8, zorder=10)
    ax.plot([0, L], [-D, -D], **wall)
    ax.plot([0, 0], [-D, 0], **wall)
    ax.plot([L, L], [-D, 0], **wall)
    ax.plot([0, riser_left], [0, 0], **wall)
    ax.plot([riser_right, L], [0, 0], **wall)
    ax.plot([riser_left, riser_left], [0, TOWER_H], **wall)
    ax.plot([riser_right, riser_right], [0, TOWER_H], **wall)


def draw_riser_phases(
    ax,
    *,
    x_left: float,
    width: float,
    water_top: float,
    gas_nose: float,
    z: np.ndarray,
    dz: float,
    gas_fraction: np.ndarray,
    water_color: str,
    air_color: str,
) -> tuple[float, float]:
    """Map the two axial 1-D fronts to a clipped riser section."""

    top = float(np.clip(water_top, 0.0, TOWER_H))
    nose = float(np.clip(gas_nose, 0.0, top))
    if top > 0.0:
        ax.add_patch(Rectangle(
            (x_left, 0.0), width, top,
            facecolor=water_color, edgecolor="none", zorder=3,
        ))
    for zi, fraction in zip(z, np.clip(gas_fraction, 0.0, 1.0)):
        cell_bottom = max(float(zi - 0.5 * dz), 0.0)
        cell_top = min(float(zi + 0.5 * dz), nose)
        if fraction <= 0.01 or cell_top <= cell_bottom:
            continue
        gas_width = math.sqrt(float(fraction)) * width
        ax.add_patch(Rectangle(
            (x_left + 0.5 * (width - gas_width), cell_bottom),
            gas_width, cell_top - cell_bottom,
            facecolor=air_color, edgecolor="none", zorder=4,
        ))
    if top - nose > 1.0e-9:
        ax.plot(
            [x_left, x_left + width], [top, top],
            color="#1d4ed8", linewidth=0.7, zorder=5,
        )
    return top, nose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-variant", required=True)
    parser.add_argument("--output-variant", required=True)
    parser.add_argument(
        "--interpolation",
        choices=("linear", "pchip", "bspline"),
        default="bspline",
        help="Continuous reconstruction used only to draw archived cell averages.",
    )
    args = parser.parse_args()

    source_fields = OUT / f"vertical_fields_{args.source_variant}.npz"
    source_index = OUT / f"frames_index_{args.source_variant}.json"
    source_diagnostics = OUT / f"solver_diagnostics_{args.source_variant}.json"
    output_frames = OUT / f"frames_{args.output_variant}"
    output_riser_frames = OUT / f"riser_frames_{args.output_variant}"
    output_index = OUT / f"frames_index_{args.output_variant}.json"

    fields = np.load(source_fields)
    alpha_l = np.asarray(fields["horizontal_alpha_l"], dtype=float)
    riser_alpha_l = np.asarray(fields["alpha_l"], dtype=float)
    riser_alpha_g = np.asarray(fields["alpha_g"], dtype=float)
    z = np.asarray(fields["z"], dtype=float)
    times = np.asarray(fields["time"], dtype=float)
    if source_index.is_file():
        manifest = json.loads(source_index.read_text(encoding="utf-8"))
    else:
        if not source_diagnostics.is_file():
            raise FileNotFoundError(
                "source has neither a frame manifest nor solver diagnostics"
            )
        diagnostics = json.loads(source_diagnostics.read_text(encoding="utf-8"))
        diagnostic_time = np.asarray(diagnostics["t"], dtype=float)
        if diagnostic_time.size < 2:
            raise ValueError("solver diagnostics require at least two time samples")
        initial_dz = float(np.median(np.diff(z))) if z.size > 1 else TOWER_H
        initial_equivalent = float(np.sum(riser_alpha_l[0]) * initial_dz)
        equivalent_offset = initial_equivalent - 0.356
        manifest = []
        for frame_index, time_s in enumerate(times):
            material_height = float(np.interp(
                time_s, diagnostic_time, diagnostics["wtop"]
            ))
            liquid_equivalent = max(
                float(np.sum(riser_alpha_l[frame_index]) * initial_dz)
                - equivalent_offset,
                0.0,
            )
            manifest.append({
                "time": float(time_s),
                "wtop": liquid_equivalent,
                "materialHeight": material_height,
                "itop": float(np.interp(
                    time_s, diagnostic_time, diagnostics["itop"]
                )),
                "coreMassMg": 1.0e6 * float(np.interp(
                    time_s, diagnostic_time, diagnostics["core_mass"]
                )),
                "head": float(np.interp(
                    time_s, diagnostic_time, diagnostics["pocket_head"]
                )),
            })
    if (
        alpha_l.shape[0] != len(manifest)
        or riser_alpha_l.shape != riser_alpha_g.shape
        or riser_alpha_l.shape[0] != len(manifest)
        or times.size != len(manifest)
    ):
        raise ValueError("field archive and manifest have inconsistent frames")

    output_frames.mkdir(parents=True, exist_ok=True)
    output_riser_frames.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })
    water = "#2b7fff"
    air = "#f2f4f8"
    handles = [
        Patch(facecolor=water, label="water"),
        Patch(facecolor=air, edgecolor="0.5", label="air"),
    ]

    n_cells = alpha_l.shape[1]
    dx = L / n_cells
    dz = float(np.median(np.diff(z))) if z.size > 1 else TOWER_H
    centers = (np.arange(n_cells, dtype=float) + 0.5) * dx
    dense_x = np.linspace(0.0, L, 8 * n_cells + 1)
    new_manifest = []

    for index, (frame, fraction, vertical_l, vertical_g, time_s) in enumerate(
        zip(manifest, alpha_l, riser_alpha_l, riser_alpha_g, times)
    ):
        depth = D * circular_depth_fraction(fraction)
        node_x = np.r_[0.0, centers, L]
        node_depth = np.r_[depth[0], depth, depth[-1]]
        if args.interpolation == "linear":
            dense_depth = np.interp(dense_x, node_x, node_depth)
        elif args.interpolation == "pchip":
            dense_depth = shape_preserving_cubic(node_x, node_depth, dense_x)
        else:
            # Reconstruct the conserved liquid-area fraction first and only
            # then map it to circular-pipe depth.  Smoothing depth directly
            # would not preserve the finite-volume liquid inventory.
            dense_fraction = compact_cubic_bspline(centers, fraction, dense_x)
            dense_depth = D * circular_depth_fraction(dense_fraction)
        surface_y = -D + dense_depth

        fig, ax = plt.subplots(figsize=(14.0, 3.6))
        ax.add_patch(Rectangle((0, -D), L, D, facecolor=air, edgecolor="none"))
        ax.fill_between(
            dense_x,
            -D,
            surface_y,
            color=water,
            linewidth=0.0,
            antialiased=True,
            zorder=2,
        )
        riser_left = TOWER_X - 0.5 * DT
        ax.add_patch(Rectangle(
            (riser_left, 0), DT, TOWER_H,
            facecolor=air, edgecolor="none", zorder=1,
        ))
        riser_right = TOWER_X + 0.5 * DT
        # ``wtop`` and ``itop`` are axial material fronts.  Drawing an entire
        # mixed finite-volume cell as a central gas core lets the core cross
        # ``itop`` and creates a false notch in the bulk free surface.  Start
        # from the continuous liquid column and overpaint the gas core only
        # below its resolved nose.  The liquid slug between itop and wtop is
        # therefore full-width and the 1-D bulk free surface is horizontal.
        # Older archived manifests used ``wtop`` for liquid-equivalent height
        # and stored the actual occupied-column top as ``materialHeight``.
        # Current manifests use ``wtop`` for the latter.  Prefer the explicit
        # legacy field when it is present so re-rendering remains faithful.
        water_top = float(np.clip(
            frame.get("materialHeight", frame["wtop"]), 0.0, TOWER_H
        ))
        gas_nose = float(np.clip(frame.get("itop", 0.0), 0.0, water_top))
        draw_riser_phases(
            ax,
            x_left=riser_left,
            width=DT,
            water_top=water_top,
            gas_nose=gas_nose,
            z=z,
            dz=dz,
            gas_fraction=vertical_g,
            water_color=water,
            air_color=air,
        )
        draw_outline(ax)
        ax.text(
            0.01,
            0.95,
            f"Time = {float(time_s):.2f} s    riser liquid-trace top = "
            f"{water_top:.3f} m",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
        )
        ax.set_xlim(-0.05, L + 0.05)
        ax.set_ylim(-D - 0.04, TOWER_H + 0.10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("horizontal distance [m]")
        ax.set_ylabel("height [m]")
        ax.set_title(
            "Case A (true scale 1:1) -- D=94 mm, Dt=57.1 mm "
            "(Dt/D=0.61), Ha0=0.305 m, WL0=0.356 m, L=0.61 m",
            fontsize=9,
        )
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        fig.tight_layout()
        output_path = output_frames / f"frame_{index:04d}.png"
        fig.savefig(output_path, dpi=130)
        plt.close(fig)

        zoom_fig, zoom_ax = plt.subplots(figsize=(2.6, 6.2))
        zoom_ax.add_patch(Rectangle(
            (0.0, 0.0), 1.0, TOWER_H,
            facecolor=air, edgecolor="none", zorder=1,
        ))
        draw_riser_phases(
            zoom_ax,
            x_left=0.0,
            width=1.0,
            water_top=water_top,
            gas_nose=gas_nose,
            z=z,
            dz=dz,
            gas_fraction=vertical_g,
            water_color=water,
            air_color=air,
        )
        zoom_ax.plot([0.0, 0.0], [0.0, TOWER_H], color="0.35", lw=0.8)
        zoom_ax.plot([1.0, 1.0], [0.0, TOWER_H], color="0.35", lw=0.8)
        zoom_ax.set_xlim(0.0, 1.0)
        zoom_ax.set_ylim(0.0, TOWER_H)
        zoom_ax.set_xticks([])
        zoom_ax.set_ylabel("height above pipe crown [m]", fontsize=8)
        zoom_ax.set_title(f"riser zoom\nTime = {float(time_s):.2f} s", fontsize=9)
        for spine in ("top", "right"):
            zoom_ax.spines[spine].set_visible(False)
        zoom_fig.tight_layout()
        zoom_path = output_riser_frames / f"riser_{index:04d}.png"
        zoom_fig.savefig(zoom_path, dpi=110)
        plt.close(zoom_fig)

        updated = dict(frame)
        updated["file"] = output_path.relative_to(CASE).as_posix()
        updated["riserFile"] = zoom_path.relative_to(CASE).as_posix()
        updated["materialHeight"] = water_top
        updated["horizontalDisplay"] = (
            "circular-depth compact cubic B-spline reconstruction of cell averages"
            if args.interpolation == "bspline"
            else (
                "circular-depth shape-preserving cubic reconstruction of cell averages"
                if args.interpolation == "pchip"
                else "circular-depth piecewise-linear reconstruction of cell averages"
            )
        )
        new_manifest.append(updated)

    output_index.write_text(
        json.dumps(new_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Re-rendered {len(new_manifest)} frames -> {output_frames}")
    print(f"Wrote {output_index}")


if __name__ == "__main__":
    main()

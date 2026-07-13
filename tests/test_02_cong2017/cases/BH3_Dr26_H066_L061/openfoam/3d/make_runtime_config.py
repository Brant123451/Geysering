#!/usr/bin/env python3
"""Generate small OpenFOAM include files for one declared run variant."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=13.0)
    parser.add_argument("--initial-delta-t", type=float, default=1.0e-6)
    parser.add_argument("--max-co", type=float, default=0.25)
    parser.add_argument("--max-alpha-co", type=float, default=0.15)
    parser.add_argument("--max-delta-t", type=float, default=5.0e-4)
    parser.add_argument("--c-alpha", type=float, default=1.0)
    parser.add_argument("--alpha-smooth-curvature", type=int, default=0)
    parser.add_argument("--sample-interval", type=float, default=0.005)
    parser.add_argument("--write-interval", type=float, default=0.05)
    parser.add_argument("--surface-tension", type=float, default=0.072)
    parser.add_argument(
        "--n-hat-gradient-scheme",
        choices=("gauss-linear", "least-squares"),
        default="gauss-linear",
    )
    return parser.parse_args()


def locations(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [start + i * step for i in range(count + 1)]


def write_locations(path: Path, elevations: list[float]) -> None:
    text = "\n".join(f"            (3.47 0 {z:.6f})" for z in elevations)
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    positive = (
        args.end_time,
        args.initial_delta_t,
        args.max_co,
        args.max_alpha_co,
        args.max_delta_t,
        args.c_alpha,
        args.sample_interval,
        args.write_interval,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all runtime controls must be positive")
    if args.alpha_smooth_curvature < 0:
        raise ValueError("alpha curvature smoothing iterations cannot be negative")
    if args.surface_tension < 0:
        raise ValueError("surface tension cannot be negative")
    if args.initial_delta_t > args.max_delta_t:
        raise ValueError("initial deltaT cannot exceed maxDeltaT")

    system = Path("system")
    system.mkdir(exist_ok=True)
    constant = Path("constant")
    constant.mkdir(exist_ok=True)
    (system / "runControl.runtime").write_text(
        "\n".join(
            (
                f"eventEndTime {args.end_time:.12g};",
                f"initialDeltaT {args.initial_delta_t:.12g};",
                f"maxCoValue {args.max_co:.12g};",
                f"maxAlphaCoValue {args.max_alpha_co:.12g};",
                f"maxDeltaTValue {args.max_delta_t:.12g};",
                f"interfaceCompression {args.c_alpha:.12g};",
                f"alphaSmoothCurvature {args.alpha_smooth_curvature};",
                f"sampleInterval {args.sample_interval:.12g};",
                f"fieldWriteInterval {args.write_interval:.12g};",
                "",
            )
        ),
        encoding="utf-8",
    )
    (constant / "surfaceTension.runtime").write_text(
        f"surfaceTensionValue {args.surface_tension:.12g};\n",
        encoding="utf-8",
    )
    n_hat_scheme = {
        "gauss-linear": "Gauss linear",
        "least-squares": "leastSquares",
    }[args.n_hat_gradient_scheme]
    (system / "gradientSchemes.runtime").write_text(
        f"nHatGradientScheme {n_hat_scheme};\n",
        encoding="utf-8",
    )
    write_locations(
        system / "riserProbeLocations.runtime",
        locations(0.060, 1.840, 0.010),
    )
    write_locations(
        system / "plumeProbeLocations.runtime",
        locations(1.860, 2.980, 0.020),
    )
    print(
        "runtime "
        f"end={args.end_time:g} maxCo={args.max_co:g} "
        f"maxAlphaCo={args.max_alpha_co:g} maxDeltaT={args.max_delta_t:g} "
        f"cAlpha={args.c_alpha:g} "
        f"alphaSmoothCurvature={args.alpha_smooth_curvature} "
        f"sigma={args.surface_tension:g} "
        f"nHatGradient={args.n_hat_gradient_scheme}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Emit piecewise-constant Darcy--Forchheimer valve opening stages."""

from __future__ import annotations

import argparse


CLOSED_DARCY_M2 = 1.0e12
MINIMUM_AREA_FRACTION = 0.01
SEAL_RELEASE_FRACTION = 0.02
VALVE_ZONE_LENGTH_M = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opening-start", type=float, required=True)
    parser.add_argument("--opening-duration", type=float, required=True)
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--stages", type=int, default=20)
    return parser.parse_args()


def area_fraction(time_s: float, start_s: float, duration_s: float) -> float:
    if time_s <= start_s:
        return 0.0
    if time_s >= start_s + duration_s:
        return 1.0
    normalized = (time_s - start_s) / duration_s
    return normalized * normalized * (3.0 - 2.0 * normalized)


def resistance(area: float) -> tuple[float, float]:
    seal_weight = max(1.0 - area / SEAL_RELEASE_FRACTION, 0.0)
    darcy = CLOSED_DARCY_M2 * seal_weight * seal_weight
    if area < MINIMUM_AREA_FRACTION or area >= 1.0:
        return darcy, 0.0
    loss_coefficient = ((1.0 - area) / area) ** 2
    forchheimer = loss_coefficient / VALVE_ZONE_LENGTH_M
    return darcy, forchheimer


def emit(
    end_time_s: float,
    area: float,
    phase: str,
) -> None:
    darcy, forchheimer = resistance(area)
    print(
        f"{end_time_s:.12g}\t{darcy:.12g}\t{forchheimer:.12g}"
        f"\t{area:.12g}\t{phase}"
    )


def main() -> None:
    args = parse_args()
    if args.end_time <= 0:
        raise ValueError("end-time must be positive")
    if args.opening_start < 0:
        raise ValueError("opening-start must be non-negative")
    if args.opening_duration <= 0:
        raise ValueError("opening-duration must be positive")
    if args.stages < 1:
        raise ValueError("stages must be positive")

    opening_end = args.opening_start + args.opening_duration
    if args.opening_start > 0:
        closed_end = min(args.opening_start, args.end_time)
        emit(closed_end, 0.0, "closed")
        if closed_end >= args.end_time:
            return

    stage_width = args.opening_duration / args.stages
    for index in range(args.stages):
        stage_start = args.opening_start + index * stage_width
        if stage_start >= args.end_time:
            return
        stage_end = min(stage_start + stage_width, args.end_time)
        midpoint = 0.5 * (stage_start + stage_end)
        area = area_fraction(
            midpoint,
            args.opening_start,
            args.opening_duration,
        )
        emit(stage_end, area, "opening")
        if stage_end >= args.end_time:
            return

    if args.end_time > opening_end:
        emit(args.end_time, 1.0, "open")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the Liu2020 B3 three-dimensional fluid domain with Gmsh.

The experimental apparatus is the audited A2 geometry.  B3 changes only the
downstream initial/boundary condition.  A compact, water-filled numerical
plenum represents the pressurised feed tank and supplies the reported flow
without an open overflow bypass.  An external atmosphere box above the
*physical* riser rim permits a jet to rise beyond the 1.22 m riser instead of
being deleted by a pressure boundary at the rim.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import gmsh


# Paper geometry [m].
LU, DU, SLOPE = 5.80, 0.20, 0.01
LC, WC, HC, DROP = 0.30, 0.30, 0.45, 0.18
LD, DD = 5.95, 0.28
DR, HR = 0.06, 1.22

RU, RD, RR = DU / 2.0, DD / 2.0, DR / 2.0
X_UPSTREAM = -LU
X_DOWNSTREAM = LC + LD
X_RISER, Y_RISER = LC / 2.0, 0.0
Z_RISER_RIM = HC + HR

# Numerical inlet plenum.  Its top is only just above the upstream pipe crown
# and is a wall: the paper's feed tank is pressurised, not an open overflow.
HEADBOX_X0, HEADBOX_X1 = X_UPSTREAM - 0.35, X_UPSTREAM
HEADBOX_Y0, HEADBOX_Y1 = -0.15, 0.15
HEADBOX_Z0, HEADBOX_Z1 = 0.188, 0.45

# Open atmosphere outside the physical riser.  The expected B3 regression
# height is ~4.21 m above the lid (z~4.66 m), leaving ~0.59 m top clearance.
PLUME_WIDTH = 0.60
PLUME_Z0, PLUME_Z1 = Z_RISER_RIM - 0.001, 5.25
PLUME_X0, PLUME_X1 = X_RISER - PLUME_WIDTH / 2, X_RISER + PLUME_WIDTH / 2
PLUME_Y0, PLUME_Y1 = Y_RISER - PLUME_WIDTH / 2, Y_RISER + PLUME_WIDTH / 2
BOOLEAN_OVERLAP = 0.002


@dataclass(frozen=True)
class MeshProfile:
    pipe: float
    junction: float
    riser: float
    free_surface: float
    jet: float
    far: float


PROFILES = {
    # Fast topology/BC smoke mesh.
    "smoke": MeshProfile(0.065, 0.032, 0.017, 0.045, 0.065, 0.160),
    # Production baseline selected after strict checkMesh and Q0 smoke tests.
    "baseline": MeshProfile(0.050, 0.022, 0.013, 0.035, 0.050, 0.120),
    # Stable critical-region refinement of the production baseline.
    "refined": MeshProfile(0.045, 0.020, 0.0115, 0.030, 0.045, 0.110),
}


def z_axis_upstream(x: float) -> float:
    return DROP + RU - SLOPE * x


def close(value: float, target: float, tolerance: float = 2e-5) -> bool:
    return abs(value - target) <= tolerance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("b3_3d.msh"))
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), default="baseline",
        help="Named mesh used by the baseline and sensitivity runs",
    )
    parser.add_argument("--pipe-size", type=float)
    parser.add_argument("--junction-size", type=float)
    parser.add_argument("--riser-size", type=float)
    parser.add_argument("--free-surface-size", type=float)
    parser.add_argument("--jet-size", type=float)
    parser.add_argument("--far-size", type=float)
    return parser.parse_args()


def selected_profile(args: argparse.Namespace) -> MeshProfile:
    base = PROFILES[args.profile]
    values = {
        field: getattr(args, f"{field}_size") or getattr(base, field)
        for field in ("pipe", "junction", "riser", "free_surface", "jet", "far")
    }
    if min(values.values()) <= 0:
        raise ValueError("All mesh sizes must be positive")
    if values["far"] < max(values["pipe"], values["jet"]):
        raise ValueError("far-size must not be smaller than pipe/jet sizes")
    return MeshProfile(**values)


def add_box_field(
    profile_size: float,
    far_size: float,
    bounds: tuple[float, float, float, float, float, float],
    thickness: float,
) -> int:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    tag = gmsh.model.mesh.field.add("Box")
    for name, value in (
        ("VIn", profile_size),
        ("VOut", far_size),
        ("XMin", xmin),
        ("XMax", xmax),
        ("YMin", ymin),
        ("YMax", ymax),
        ("ZMin", zmin),
        ("ZMax", zmax),
        ("Thickness", thickness),
    ):
        gmsh.model.mesh.field.setNumber(tag, name, value)
    return tag


def main() -> None:
    args = parse_args()
    profile = selected_profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.model.add("Liu2020_B3_Q20to100_fullpipe_geyser_3D")
        occ = gmsh.model.occ

        headbox = occ.addBox(
            HEADBOX_X0,
            HEADBOX_Y0,
            HEADBOX_Z0,
            HEADBOX_X1 - HEADBOX_X0,
            HEADBOX_Y1 - HEADBOX_Y0,
            HEADBOX_Z1 - HEADBOX_Z0,
        )

        up_x0 = X_UPSTREAM - BOOLEAN_OVERLAP
        up_x1 = BOOLEAN_OVERLAP
        upstream = occ.addCylinder(
            up_x0,
            0.0,
            z_axis_upstream(up_x0),
            up_x1 - up_x0,
            0.0,
            z_axis_upstream(up_x1) - z_axis_upstream(up_x0),
            RU,
        )

        chamber = occ.addBox(0.0, -WC / 2, 0.0, LC, WC, HC)

        downstream = occ.addCylinder(
            LC - BOOLEAN_OVERLAP,
            0.0,
            RD,
            X_DOWNSTREAM - (LC - BOOLEAN_OVERLAP),
            0.0,
            0.0,
            RD,
        )

        riser = occ.addCylinder(
            X_RISER,
            Y_RISER,
            HC - BOOLEAN_OVERLAP,
            0.0,
            0.0,
            HR + 2 * BOOLEAN_OVERLAP,
            RR,
        )

        plume = occ.addBox(
            PLUME_X0,
            PLUME_Y0,
            PLUME_Z0,
            PLUME_X1 - PLUME_X0,
            PLUME_Y1 - PLUME_Y0,
            PLUME_Z1 - PLUME_Z0,
        )

        fused, _ = occ.fuse(
            [(3, headbox)],
            [(3, upstream), (3, chamber), (3, downstream), (3, riser), (3, plume)],
            removeObject=True,
            removeTool=True,
        )
        occ.synchronize()

        volumes = [tag for dim, tag in fused if dim == 3]
        if len(volumes) != 1:
            raise RuntimeError(f"Expected one connected fluid volume, got {volumes}")

        inlet_surfaces: list[int] = []
        outlet_surfaces: list[int] = []
        atmosphere_surfaces: list[int] = []
        wall_surfaces: list[int] = []

        boundaries = gmsh.model.getBoundary(
            [(3, volumes[0])], combined=True, oriented=False, recursive=False
        )
        for dim, tag in boundaries:
            if dim != 2:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)

            on_inlet = close(zmin, HEADBOX_Z0) and close(zmax, HEADBOX_Z0)
            on_outlet = close(xmin, X_DOWNSTREAM) and close(xmax, X_DOWNSTREAM)
            plume_surface = (
                zmin >= PLUME_Z0 - 2e-5
                and (
                    close(zmin, PLUME_Z0)
                    and close(zmax, PLUME_Z0)
                    or close(zmin, PLUME_Z1)
                    and close(zmax, PLUME_Z1)
                    or close(xmin, PLUME_X0)
                    and close(xmax, PLUME_X0)
                    or close(xmin, PLUME_X1)
                    and close(xmax, PLUME_X1)
                    or close(ymin, PLUME_Y0)
                    and close(ymax, PLUME_Y0)
                    or close(ymin, PLUME_Y1)
                    and close(ymax, PLUME_Y1)
                )
            )

            if on_inlet:
                inlet_surfaces.append(tag)
            elif on_outlet:
                outlet_surfaces.append(tag)
            elif plume_surface:
                atmosphere_surfaces.append(tag)
            else:
                wall_surfaces.append(tag)

        if (
            not inlet_surfaces
            or not outlet_surfaces
            or not atmosphere_surfaces
            or not wall_surfaces
        ):
            raise RuntimeError(
                "Boundary classification failed: "
                f"inlet={inlet_surfaces}, outlet={outlet_surfaces}, "
                f"atmosphere={atmosphere_surfaces}, walls={wall_surfaces}"
            )

        for dim, tags, name in (
            (3, volumes, "fluid"),
            (2, inlet_surfaces, "inlet"),
            (2, outlet_surfaces, "outlet"),
            (2, atmosphere_surfaces, "atmosphere"),
            (2, wall_surfaces, "walls"),
        ):
            physical = gmsh.model.addPhysicalGroup(dim, tags)
            gmsh.model.setPhysicalName(dim, physical, name)

        fields = [
            add_box_field(
                profile.pipe,
                profile.far,
                (HEADBOX_X0, X_DOWNSTREAM, -0.17, 0.17, -0.02, 0.48),
                2 * profile.pipe,
            ),
            add_box_field(
                profile.junction,
                profile.far,
                (-0.40, 0.70, -0.17, 0.17, -0.02, 0.52),
                2 * profile.junction,
            ),
            add_box_field(
                profile.junction,
                profile.far,
                (HEADBOX_X0, -5.50, -0.17, 0.17, 0.15, 0.65),
                2 * profile.junction,
            ),
            add_box_field(
                profile.junction,
                profile.far,
                (5.75, X_DOWNSTREAM + 0.01, -0.17, 0.17, -0.02, 0.32),
                2 * profile.junction,
            ),
            add_box_field(
                profile.riser,
                profile.far,
                (
                    X_RISER - 0.04,
                    X_RISER + 0.04,
                    -0.04,
                    0.04,
                    HC - 0.03,
                    Z_RISER_RIM + 0.04,
                ),
                2 * profile.riser,
            ),
            add_box_field(
                profile.free_surface,
                profile.far,
                (HEADBOX_X0, 0.02, -0.17, 0.17, 0.24, 0.47),
                2 * profile.free_surface,
            ),
            add_box_field(
                profile.jet,
                profile.far,
                (
                    X_RISER - 0.11,
                    X_RISER + 0.11,
                    -0.11,
                    0.11,
                    Z_RISER_RIM - 0.03,
                    PLUME_Z1,
                ),
                2 * profile.jet,
            ),
        ]
        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

        gmsh.option.setNumber("Mesh.MeshSizeMin", min(profile.__dict__.values()))
        gmsh.option.setNumber("Mesh.MeshSizeMax", profile.far)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 18)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)

        gmsh.model.mesh.generate(3)
        gmsh.write(str(args.output))

        elements = gmsh.model.mesh.getElements(3)[1]
        n_cells = sum(len(tags) for tags in elements)
        fluid_volume = occ.getMass(3, volumes[0])
        outlet_area = sum(occ.getMass(2, tag) for tag in outlet_surfaces)
        print(f"profile={args.profile}")
        print(f"mesh={args.output}")
        print(f"cells_3d={n_cells}")
        print(f"fluid_volume_m3={fluid_volume:.9g}")
        print(f"outlet_area_m2={outlet_area:.9g}")
        print(f"expected_outlet_area_m2={math.pi * RD * RD:.9g}")
        print(f"physical_riser_rim_z_m={Z_RISER_RIM}")
        print(f"plume_top_z_m={PLUME_Z1}")
        print(f"pipe_size_m={profile.pipe}")
        print(f"junction_size_m={profile.junction}")
        print(f"riser_size_m={profile.riser}")
        print(f"free_surface_size_m={profile.free_surface}")
        print(f"jet_size_m={profile.jet}")
        print(f"far_size_m={profile.far}")
        print(f"boundary_surfaces_inlet={len(inlet_surfaces)}")
        print(f"boundary_surfaces_outlet={len(outlet_surfaces)}")
        print(f"boundary_surfaces_atmosphere={len(atmosphere_surfaces)}")
        print(f"boundary_surfaces_walls={len(wall_surfaces)}")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()

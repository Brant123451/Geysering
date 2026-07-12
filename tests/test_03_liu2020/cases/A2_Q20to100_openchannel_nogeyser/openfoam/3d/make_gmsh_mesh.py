#!/usr/bin/env python3
"""Generate a conformal tetrahedral mesh of Liu2020 Case A2 with Gmsh.

The journal article omits the receiving-tank dimensions, but the open-access
thesis describing the same apparatus reports a 0.57 x 0.61 x 0.89 m tank and a
0.30 m diameter, 0.40 m high movable circular overflow weir.  They are included
here so that the downstream stage evolves instead of being clamped at hd.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gmsh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("base", "refined"), default="base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def near(a: float, b: float, tol: float = 2.0e-5) -> bool:
    return abs(a - b) <= tol


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Journal-article dimensions [m].
    lu, du, slope = 5.80, 0.20, 0.01
    lc, wc, hc, drop = 0.30, 0.30, 0.45, 0.18
    ld, dd = 5.95, 0.28
    dr, hr = 0.06, 1.22
    ru, rd, rr = du / 2.0, dd / 2.0, dr / 2.0
    z_up = lambda x: drop + ru - slope * x

    # Liu (2018 thesis), Sec. 3.1: receiving tank and circular movable weir.
    # The experimenters adjusted the crest to obtain hd=0.070 m at Q0.  A
    # standard circular sharp-crested estimate gives 0.051 m head at 20 L/s,
    # hence zcrest=0.019 m.  This datum is calibrated only to the reported
    # initial stage, never to the transient pressure or no-geyser outcome.
    tank_l, tank_w, tank_h = 0.57, 0.61, 0.89
    weir_d, weir_h = 0.30, 0.40
    weir_ro, weir_ri = weir_d / 2.0, weir_d / 2.0 - 0.010
    weir_crest = 0.019
    tank_z0, tank_z1 = weir_crest - weir_h, weir_crest - weir_h + tank_h
    tank_x0, tank_x1 = lc + ld, lc + ld + tank_l
    tank_y0, tank_y1 = -tank_w / 2.0, tank_w / 2.0
    weir_x, weir_y = (tank_x0 + tank_x1) / 2.0, 0.0

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.NumThreads", max(1, args.threads))
        gmsh.model.add(f"Liu2020_A2_{args.profile}")
        occ = gmsh.model.occ

        chamber = occ.addBox(0.0, -wc / 2.0, 0.0, lc, wc, hc)
        # The exact tangent between the circular downstream invert and chamber
        # floor is a zero-angle tetrahedral singularity.  A 20 x 80 x 8 mm
        # local recess under the outlet mouth regularises that one point while
        # preserving the reported 0.30 x 0.30 x 0.45 m chamber everywhere else.
        # Added volume: 12.8 mL (0.0019% of the full fluid domain).  Its 8 mm
        # depth also keeps this numerical regularization from setting the
        # global transient time step.
        tangent_recess = occ.addBox(lc - 0.020, -0.040, -0.008, 0.020, 0.080, 0.008)

        overlap = 0.002
        x0u, x1u = -lu, overlap
        upstream = occ.addCylinder(
            x0u,
            0.0,
            z_up(x0u),
            x1u - x0u,
            0.0,
            z_up(x1u) - z_up(x0u),
            ru,
        )
        downstream = occ.addCylinder(
            lc - overlap,
            0.0,
            rd,
            ld + overlap,
            0.0,
            0.0,
            rd,
        )
        riser = occ.addCylinder(
            lc / 2.0,
            0.0,
            hc - overlap,
            0.0,
            0.0,
            hr + overlap,
            rr,
        )
        tank = occ.addBox(
            tank_x0 - overlap,
            tank_y0,
            tank_z0,
            tank_x1 - tank_x0 + overlap,
            tank_y1 - tank_y0,
            tank_z1 - tank_z0,
        )
        weir_outer = occ.addCylinder(
            weir_x, weir_y, tank_z0, 0.0, 0.0, weir_h, weir_ro
        )
        weir_inner = occ.addCylinder(
            weir_x, weir_y, tank_z0, 0.0, 0.0, weir_h, weir_ri
        )
        weir_wall, _ = occ.cut(
            [(3, weir_outer)],
            [(3, weir_inner)],
            removeObject=True,
            removeTool=True,
        )

        fused, _ = occ.fuse(
            [(3, chamber)],
            [
                (3, upstream),
                (3, downstream),
                (3, riser),
                (3, tangent_recess),
                (3, tank),
            ],
            removeObject=True,
            removeTool=True,
        )
        fluid, _ = occ.cut(
            fused,
            weir_wall,
            removeObject=True,
            removeTool=True,
        )
        occ.removeAllDuplicates()
        occ.synchronize()
        volumes = [tag for dim, tag in fluid if dim == 3]
        if len(volumes) != 1:
            volumes = [tag for dim, tag in gmsh.model.getEntities(3)]
        if len(volumes) != 1:
            raise RuntimeError(f"expected one connected fluid volume, got {volumes}")

        patch_surfaces: dict[str, list[int]] = {
            "walls": [],
            "riserWall": [],
            "inlet": [],
            "tankAtmosphere": [],
            "riserOutlet": [],
            "weirOutlet": [],
        }
        boundary = gmsh.model.getBoundary([(3, volumes[0])], oriented=False)
        for dim, tag in boundary:
            if dim != 2:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            if near(xmin, -lu) and near(xmax, -lu):
                patch = "inlet"
            elif near(zmin, hc + hr) and near(zmax, hc + hr):
                patch = "riserOutlet"
            elif (
                near(zmin, tank_z1)
                and near(zmax, tank_z1)
                and xmin >= tank_x0 - overlap - 1.0e-4
            ):
                patch = "tankAtmosphere"
            elif (
                near(zmin, tank_z0)
                and near(zmax, tank_z0)
                and xmin >= weir_x - weir_ri - 1.0e-4
                and xmax <= weir_x + weir_ri + 1.0e-4
                and ymin >= weir_y - weir_ri - 1.0e-4
                and ymax <= weir_y + weir_ri + 1.0e-4
            ):
                patch = "weirOutlet"
            elif (
                zmin >= hc - 1.0e-4
                and zmax > hc + 0.5
                and xmin >= lc / 2.0 - rr - 1.0e-4
                and xmax <= lc / 2.0 + rr + 1.0e-4
            ):
                patch = "riserWall"
            else:
                patch = "walls"
            patch_surfaces[patch].append(tag)

        for patch, surfaces in patch_surfaces.items():
            if not surfaces:
                raise RuntimeError(f"no Gmsh surfaces classified for patch {patch}")
            group = gmsh.model.addPhysicalGroup(2, surfaces)
            gmsh.model.setPhysicalName(2, group, patch)
        fluid_group = gmsh.model.addPhysicalGroup(3, volumes)
        gmsh.model.setPhysicalName(3, fluid_group, "fluid")

        if args.profile == "base":
            size_max, size_chamber, size_riser = 0.050, 0.018, 0.012
            size_upstream, size_downstream = 0.028, 0.040
            size_tank, size_weir = 0.040, 0.018
        else:
            # Uniform 15% reduction in the governing target sizes.  This is a
            # systematic resolution sensitivity while keeping the complete
            # 18.4 s (-4 to 14.4 s) transient tractable on four MPI ranks.
            size_max, size_chamber, size_riser = 0.0425, 0.0153, 0.0102
            size_upstream, size_downstream = 0.0238, 0.034
            size_tank, size_weir = 0.034, 0.0153

        gmsh.option.setNumber("Mesh.MeshSizeMin", size_riser * 0.75)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
        # Diameter-specific fields below retain ~22/26 circumferential cells
        # on the reported pipes without over-refining the much smaller riser.
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 10)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)

        chamber_box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(chamber_box, "VIn", size_chamber)
        gmsh.model.mesh.field.setNumber(chamber_box, "VOut", size_max)
        gmsh.model.mesh.field.setNumber(chamber_box, "XMin", -0.20)
        gmsh.model.mesh.field.setNumber(chamber_box, "XMax", 0.50)
        gmsh.model.mesh.field.setNumber(chamber_box, "YMin", -0.16)
        gmsh.model.mesh.field.setNumber(chamber_box, "YMax", 0.16)
        gmsh.model.mesh.field.setNumber(chamber_box, "ZMin", -0.01)
        gmsh.model.mesh.field.setNumber(chamber_box, "ZMax", 0.50)

        riser_box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(riser_box, "VIn", size_riser)
        gmsh.model.mesh.field.setNumber(riser_box, "VOut", size_max)
        gmsh.model.mesh.field.setNumber(riser_box, "XMin", 0.115)
        gmsh.model.mesh.field.setNumber(riser_box, "XMax", 0.185)
        gmsh.model.mesh.field.setNumber(riser_box, "YMin", -0.035)
        gmsh.model.mesh.field.setNumber(riser_box, "YMax", 0.035)
        gmsh.model.mesh.field.setNumber(riser_box, "ZMin", 0.44)
        gmsh.model.mesh.field.setNumber(riser_box, "ZMax", 1.68)

        upstream_box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(upstream_box, "VIn", size_upstream)
        gmsh.model.mesh.field.setNumber(upstream_box, "VOut", size_max)
        gmsh.model.mesh.field.setNumber(upstream_box, "XMin", -5.82)
        gmsh.model.mesh.field.setNumber(upstream_box, "XMax", -0.19)
        gmsh.model.mesh.field.setNumber(upstream_box, "YMin", -0.105)
        gmsh.model.mesh.field.setNumber(upstream_box, "YMax", 0.105)
        gmsh.model.mesh.field.setNumber(upstream_box, "ZMin", 0.07)
        gmsh.model.mesh.field.setNumber(upstream_box, "ZMax", 0.46)

        downstream_box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(downstream_box, "VIn", size_downstream)
        gmsh.model.mesh.field.setNumber(downstream_box, "VOut", size_max)
        gmsh.model.mesh.field.setNumber(downstream_box, "XMin", 0.49)
        gmsh.model.mesh.field.setNumber(downstream_box, "XMax", 6.26)
        gmsh.model.mesh.field.setNumber(downstream_box, "YMin", -0.145)
        gmsh.model.mesh.field.setNumber(downstream_box, "YMax", 0.145)
        gmsh.model.mesh.field.setNumber(downstream_box, "ZMin", -0.01)
        gmsh.model.mesh.field.setNumber(downstream_box, "ZMax", 0.29)

        tank_box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(tank_box, "VIn", size_tank)
        gmsh.model.mesh.field.setNumber(tank_box, "VOut", size_max)
        gmsh.model.mesh.field.setNumber(tank_box, "XMin", tank_x0 - 0.01)
        gmsh.model.mesh.field.setNumber(tank_box, "XMax", tank_x1 + 0.01)
        gmsh.model.mesh.field.setNumber(tank_box, "YMin", tank_y0 - 0.01)
        gmsh.model.mesh.field.setNumber(tank_box, "YMax", tank_y1 + 0.01)
        gmsh.model.mesh.field.setNumber(tank_box, "ZMin", tank_z0 - 0.01)
        gmsh.model.mesh.field.setNumber(tank_box, "ZMax", tank_z1 + 0.01)

        weir_box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(weir_box, "VIn", size_weir)
        gmsh.model.mesh.field.setNumber(weir_box, "VOut", size_max)
        gmsh.model.mesh.field.setNumber(weir_box, "XMin", weir_x - weir_ro - 0.04)
        gmsh.model.mesh.field.setNumber(weir_box, "XMax", weir_x + weir_ro + 0.04)
        gmsh.model.mesh.field.setNumber(weir_box, "YMin", weir_y - weir_ro - 0.04)
        gmsh.model.mesh.field.setNumber(weir_box, "YMax", weir_y + weir_ro + 0.04)
        gmsh.model.mesh.field.setNumber(weir_box, "ZMin", weir_crest - 0.05)
        gmsh.model.mesh.field.setNumber(weir_box, "ZMax", weir_crest + 0.18)

        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            minimum,
            "FieldsList",
            [
                chamber_box,
                riser_box,
                upstream_box,
                downstream_box,
                tank_box,
                weir_box,
            ],
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("", force=True, niter=10)
        gmsh.model.mesh.optimize("Netgen", force=True, niter=20)
        gmsh.model.mesh.optimize("Relocate3D", force=True, niter=20)
        gmsh.write(str(args.output))

        entities = gmsh.model.mesh.getElements(3)[1]
        n_cells = sum(len(tags) for tags in entities)
        print(f"profile={args.profile} tetrahedra={n_cells} output={args.output}")
        for patch, surfaces in patch_surfaces.items():
            area = sum(occ.getMass(2, tag) for tag in surfaces)
            print(f"patch={patch} surfaces={len(surfaces)} area={area:.8g} m2")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()

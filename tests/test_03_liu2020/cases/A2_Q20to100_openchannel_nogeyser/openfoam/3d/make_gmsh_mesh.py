#!/usr/bin/env python3
"""Generate a conformal tetrahedral mesh of Liu2020 Case A2 with Gmsh.

The OpenCASCADE Boolean union removes the overlapping/STL-junction ambiguity
of the original pilot.  The reported rig is represented with a clearly
identified numerical upstream reservoir extension and a 6.4 mL local
tetrahedral regularization.  Unreported tank/weir dimensions are not invented:
the reported hd/Dd condition is imposed at the end of the 5.95 m pipe.
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

    # Paper dimensions [m].
    lu, du, slope = 5.80, 0.20, 0.01
    lc, wc, hc, drop = 0.30, 0.30, 0.45, 0.18
    ld, dd = 5.95, 0.28
    dr, hr = 0.06, 1.22
    ru, rd, rr = du / 2.0, dd / 2.0, dr / 2.0
    z_up = lambda x: drop + ru - slope * x

    # Numerical upstream reservoir extension (not reported by Liu et al.).
    hb_x0, hb_x1 = -lu - 0.35, -lu
    hb_y0, hb_y1 = -0.15, 0.15
    hb_z0, hb_z1 = 0.188, 1.10

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.NumThreads", max(1, args.threads))
        gmsh.model.add(f"Liu2020_A2_{args.profile}")
        occ = gmsh.model.occ

        headbox = occ.addBox(
            hb_x0, hb_y0, hb_z0, hb_x1 - hb_x0, hb_y1 - hb_y0, hb_z1 - hb_z0
        )
        chamber = occ.addBox(0.0, -wc / 2.0, 0.0, lc, wc, hc)
        # The exact tangent between the circular downstream invert and chamber
        # floor is a zero-angle tetrahedral singularity.  A 20 x 80 x 4 mm
        # local recess under the outlet mouth regularises that one point while
        # preserving the reported 0.30 x 0.30 x 0.45 m chamber everywhere else.
        # Added volume: 6.4 mL (0.00094% of the full fluid domain).
        tangent_recess = occ.addBox(lc - 0.020, -0.040, -0.004, 0.020, 0.080, 0.004)

        overlap = 0.002
        x0u, x1u = -lu - overlap, overlap
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

        fused, _ = occ.fuse(
            [(3, chamber)],
            [
                (3, headbox),
                (3, upstream),
                (3, downstream),
                (3, riser),
                (3, tangent_recess),
            ],
            removeObject=True,
            removeTool=True,
        )
        occ.removeAllDuplicates()
        occ.synchronize()
        volumes = [tag for dim, tag in fused if dim == 3]
        if len(volumes) != 1:
            volumes = [tag for dim, tag in gmsh.model.getEntities(3)]
        if len(volumes) != 1:
            raise RuntimeError(f"expected one connected fluid volume, got {volumes}")

        patch_surfaces: dict[str, list[int]] = {
            "walls": [],
            "riserWall": [],
            "inlet": [],
            "headboxAtmosphere": [],
            "riserOutlet": [],
            "outlet": [],
        }
        boundary = gmsh.model.getBoundary([(3, volumes[0])], oriented=False)
        for dim, tag in boundary:
            if dim != 2:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            if near(zmin, hb_z0) and near(zmax, hb_z0) and xmax < -5.79:
                patch = "inlet"
            elif near(zmin, hb_z1) and near(zmax, hb_z1) and xmax < -5.79:
                patch = "headboxAtmosphere"
            elif near(zmin, hc + hr) and near(zmax, hc + hr):
                patch = "riserOutlet"
            elif near(xmin, lc + ld) and near(xmax, lc + ld):
                patch = "outlet"
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
            size_max, size_chamber, size_riser, curvature = 0.050, 0.018, 0.009, 22
        else:
            size_max, size_chamber, size_riser, curvature = 0.036, 0.013, 0.0065, 30

        gmsh.option.setNumber("Mesh.MeshSizeMin", size_riser * 0.75)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
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

        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [chamber_box, riser_box])
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

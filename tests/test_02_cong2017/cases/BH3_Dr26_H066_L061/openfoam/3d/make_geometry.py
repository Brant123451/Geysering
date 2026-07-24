#!/usr/bin/env python3
"""Generate the boundary-fitted Cong2017 B-H3 volume mesh and review surfaces.

The geometry is a connected OpenCASCADE volume, not intersecting shell
approximations:

* circular 50 mm horizontal pipe;
* circular 26 mm vertical riser and true three-dimensional tee opening;
* expanded external atmosphere above the physical 1.8 m riser.

The initial horizontal free surface and Valve #4 cross-section are embedded as
conformal internal mesh surfaces. Optional diagnostics either refine both
edges of the declared 15 mm VOF transition or extrude a 24-layer triangular
prism band around it; a further option vertically layers the external air to
remove thin acoustic-boundary tetrahedra. The valve surface lets createBaffles
split one exact pipe cross-section instead of a jagged band of tetrahedron
faces.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import gmsh


PIPE_DIAMETER = 0.050
PIPE_LENGTH = 6.590
PIPE_INVERT_Z = 0.0
PIPE_RADIUS = PIPE_DIAMETER / 2.0
PIPE_AXIS_Z = PIPE_INVERT_Z + PIPE_RADIUS
TEE_X = 3.470
PHYSICAL_RISER_HEIGHT = 1.800
PHYSICAL_RIM_Z = PIPE_INVERT_Z + PIPE_DIAMETER + PHYSICAL_RISER_HEIGHT
COMPUTATIONAL_TOP_Z = 3.000
INITIAL_FREE_SURFACE_Z = 0.660
INITIAL_INTERFACE_THICKNESS = 0.015
INITIAL_INTERFACE_LOWER_Z = (
    INITIAL_FREE_SURFACE_Z - INITIAL_INTERFACE_THICKNESS / 2.0
)
INITIAL_INTERFACE_UPPER_Z = (
    INITIAL_FREE_SURFACE_Z + INITIAL_INTERFACE_THICKNESS / 2.0
)
VALVE_X = 5.980
BOOLEAN_OVERLAP = 0.001
PRISM_BOTTOM_Z = 0.630
PRISM_TOP_Z = 0.690
PRISM_LAYER_HEIGHT = 0.0025
ATMOSPHERE_PRISM_LAYER_HEIGHT = 0.0125
ATMOSPHERE_PRISM_LAYERS = 92
PRISM_STAGES = (
    (INITIAL_INTERFACE_LOWER_Z, 9),
    (INITIAL_FREE_SURFACE_Z, 3),
    (INITIAL_INTERFACE_UPPER_Z, 3),
    (PRISM_TOP_Z, 9),
)
# Independent geometric target for the exact physical domain: the circular
# pipe/riser union, plus the 0.30 m square atmosphere beginning at z=1.85 m.
PRISM_REFERENCE_VOLUME_M3 = 0.11739557248480555
PRISM_CAD_RELATIVE_TOLERANCE = 5.0e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("constant/triSurface"),
    )
    parser.add_argument("--riser-diameter", type=float, default=0.026)
    parser.add_argument("--atmosphere-width", type=float, default=0.300)
    parser.add_argument("--mesh-output", type=Path, default=Path("bh3.msh"))
    parser.add_argument("--pipe-size", type=float, default=0.012)
    parser.add_argument("--riser-size", type=float, default=0.005)
    parser.add_argument("--atmosphere-size", type=float, default=0.030)
    parser.add_argument("--curvature-elements", type=int, default=40)
    parser.add_argument("--align-interface-band", action="store_true")
    parser.add_argument("--interface-size", type=float)
    parser.add_argument("--prism-interface-band", action="store_true")
    parser.add_argument(
        "--prism-atmosphere-layers",
        type=int,
        default=0,
    )
    return parser.parse_args()


def close(value: float, target: float, tolerance: float = 2.0e-6) -> bool:
    return abs(value - target) <= tolerance


def export_patch(output: Path, name: str, surfaces: list[int]) -> None:
    if not surfaces:
        raise RuntimeError(f"Patch {name!r} has no surfaces")
    gmsh.model.removePhysicalGroups()
    group = gmsh.model.addPhysicalGroup(2, surfaces)
    gmsh.model.setPhysicalName(2, group, name)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.write(str(output / f"{name}.stl"))


def single_entity(
    entities: list[int],
    description: str,
) -> int:
    if len(entities) != 1:
        raise RuntimeError(
            f"Expected one {description}, found {len(entities)}: {entities}"
        )
    return entities[0]


def circular_surface_at_z(
    occ,
    surface_tags: list[int],
    elevation: float,
    radius: float,
) -> int:
    expected_area = math.pi * radius**2
    matches = []
    for tag in surface_tags:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
        center_x, center_y, center_z = occ.getCenterOfMass(2, tag)
        if (
            close(zmin, elevation)
            and close(zmax, elevation)
            and close(center_x, TEE_X)
            and close(center_y, 0.0)
            and close(center_z, elevation)
            and abs(occ.getMass(2, tag) / expected_area - 1.0) < 1.0e-8
            and close(xmin, TEE_X - radius)
            and close(xmax, TEE_X + radius)
            and close(ymin, -radius)
            and close(ymax, radius)
        ):
            matches.append(tag)
    return single_entity(matches, f"riser disk at z={elevation}")


def extrusion_entities_at_z(
    occ,
    extrusion: list[tuple[int, int]],
    elevation: float,
    radius: float,
) -> tuple[int, int]:
    volume = single_entity(
        [tag for dim, tag in extrusion if dim == 3],
        f"extruded volume ending at z={elevation}",
    )
    top = circular_surface_at_z(
        occ,
        [tag for dim, tag in extrusion if dim == 2],
        elevation,
        radius,
    )
    return volume, top


def build_prism_geometry(
    occ,
    args: argparse.Namespace,
    riser_radius: float,
) -> tuple[list[int], dict[str, object]]:
    """Build the prism profile without any Boolean after the first extrusion."""
    pipe = occ.addCylinder(
        0.0,
        0.0,
        PIPE_AXIS_Z,
        PIPE_LENGTH,
        0.0,
        0.0,
        PIPE_RADIUS,
    )
    lower_riser = occ.addCylinder(
        TEE_X,
        0.0,
        PIPE_AXIS_Z,
        0.0,
        0.0,
        PRISM_BOTTOM_Z - PIPE_AXIS_Z,
        riser_radius,
    )
    lower_fluid, _ = occ.fuse([(3, pipe)], [(3, lower_riser)])

    # This valve fragment is deliberately the final Boolean operation. All
    # following volumes are made from shared faces by geometric extrusion.
    valve_disk = occ.addDisk(
        VALVE_X,
        0.0,
        PIPE_AXIS_Z,
        PIPE_RADIUS,
        PIPE_RADIUS,
        zAxis=[1.0, 0.0, 0.0],
        xAxis=[0.0, 1.0, 0.0],
    )
    lower_parts, _ = occ.fragment(
        lower_fluid,
        [(2, valve_disk)],
        removeObject=True,
        removeTool=True,
    )
    occ.synchronize()

    lower_volumes = [tag for dim, tag in lower_parts if dim == 3]
    if len(lower_volumes) != 2:
        raise RuntimeError(
            "Valve fragmentation must split the lower fluid into two volumes"
        )
    lower_boundary = gmsh.model.getBoundary(
        [(3, tag) for tag in lower_volumes],
        combined=True,
        oriented=False,
        recursive=False,
    )
    lower_top = circular_surface_at_z(
        occ,
        [tag for dim, tag in lower_boundary if dim == 2],
        PRISM_BOTTOM_Z,
        riser_radius,
    )

    prism_volumes: list[int] = []
    prism_interfaces = [lower_top]
    current_top = lower_top
    current_z = PRISM_BOTTOM_Z
    for target_z, layers in PRISM_STAGES:
        extrusion = occ.extrude(
            [(2, current_top)],
            0.0,
            0.0,
            target_z - current_z,
            numElements=[layers],
            recombine=True,
        )
        occ.synchronize()
        prism_volume, current_top = extrusion_entities_at_z(
            occ,
            extrusion,
            target_z,
            riser_radius,
        )
        prism_volumes.append(prism_volume)
        prism_interfaces.append(current_top)
        current_z = target_z

    # No layer specification: the upper physical riser is tetrahedral.
    upper_extrusion = occ.extrude(
        [(2, current_top)],
        0.0,
        0.0,
        PHYSICAL_RIM_Z - PRISM_TOP_Z,
    )
    occ.synchronize()
    upper_volume, rim_disk = extrusion_entities_at_z(
        occ,
        upper_extrusion,
        PHYSICAL_RIM_Z,
        riser_radius,
    )

    # Make the external floor directly as a square face with a circular hole.
    # Reusing the rim disk edge makes the floor and central opening conformal;
    # this is a wire/face construction, not a post-extrusion Boolean.
    half_width = args.atmosphere_width / 2.0
    outer_points = [
        occ.addPoint(TEE_X - half_width, -half_width, PHYSICAL_RIM_Z),
        occ.addPoint(TEE_X + half_width, -half_width, PHYSICAL_RIM_Z),
        occ.addPoint(TEE_X + half_width, half_width, PHYSICAL_RIM_Z),
        occ.addPoint(TEE_X - half_width, half_width, PHYSICAL_RIM_Z),
    ]
    outer_lines = [
        occ.addLine(outer_points[index], outer_points[(index + 1) % 4])
        for index in range(4)
    ]
    outer_wire = occ.addWire(outer_lines, checkClosed=True)
    occ.synchronize()
    rim_edges = [
        tag
        for dim, tag in gmsh.model.getBoundary(
            [(2, rim_disk)],
            combined=False,
            oriented=True,
            recursive=False,
        )
        if dim == 1
    ]
    # A top face produced by OCC extrusion has the opposite loop orientation
    # from a new coplanar outer wire. Reverse it so it is a hole, not an island.
    inner_wire = occ.addWire([-tag for tag in rim_edges], checkClosed=True)
    external_floor = occ.addPlaneSurface([outer_wire, inner_wire])
    occ.synchronize()
    expected_floor_area = args.atmosphere_width**2 - math.pi * riser_radius**2
    floor_area = occ.getMass(2, external_floor)
    if abs(floor_area / expected_floor_area - 1.0) > 1.0e-10:
        raise RuntimeError(
            "External floor is not the square-minus-riser annulus: "
            f"area={floor_area:.17g}, expected={expected_floor_area:.17g}"
        )

    # Extrude the central disk and annulus together. Their shared circular edge
    # produces one internal cylindrical face. The optional vertical layers
    # remove thin top-boundary tetrahedra from the acoustic-outflow diagnostic.
    atmosphere_layer_options: dict[str, object] = {}
    if args.prism_atmosphere_layers:
        atmosphere_layer_options = {
            "numElements": [args.prism_atmosphere_layers],
            "recombine": True,
        }
    atmosphere_extrusion = occ.extrude(
        [(2, rim_disk), (2, external_floor)],
        0.0,
        0.0,
        COMPUTATIONAL_TOP_Z - PHYSICAL_RIM_Z,
        **atmosphere_layer_options,
    )
    occ.synchronize()
    atmosphere_volumes = [
        tag for dim, tag in atmosphere_extrusion if dim == 3
    ]
    if len(atmosphere_volumes) != 2:
        raise RuntimeError(
            "Central disk and external annulus must produce two atmosphere volumes"
        )

    volumes = [
        *lower_volumes,
        *prism_volumes,
        upper_volume,
        *atmosphere_volumes,
    ]
    model_volumes = [tag for dim, tag in gmsh.model.getEntities(3)]
    if set(volumes) != set(model_volumes) or len(volumes) != len(set(volumes)):
        raise RuntimeError(
            f"Tracked fluid volumes {volumes} do not match CAD volumes {model_volumes}"
        )

    shared_face_levels = []
    for surface, elevation in zip(
        prism_interfaces,
        (PRISM_BOTTOM_Z, *(target for target, _ in PRISM_STAGES)),
        strict=True,
    ):
        adjacent = {
            int(tag) for tag in gmsh.model.getAdjacencies(2, surface)[0]
        }
        if len(adjacent) != 2:
            raise RuntimeError(
                f"Prism interface z={elevation} has {len(adjacent)} adjacent volumes"
            )
        shared_face_levels.append(elevation)
    rim_adjacent = {
        int(tag) for tag in gmsh.model.getAdjacencies(2, rim_disk)[0]
    }
    if len(rim_adjacent) != 2:
        raise RuntimeError(
            f"Physical rim opening has {len(rim_adjacent)} adjacent volumes"
        )

    fluid_volume = sum(occ.getMass(3, tag) for tag in volumes)
    cad_relative_error = fluid_volume / PRISM_REFERENCE_VOLUME_M3 - 1.0
    if abs(cad_relative_error) > PRISM_CAD_RELATIVE_TOLERANCE:
        raise RuntimeError(
            "Prism CAD volume differs from the physical-domain reference: "
            f"volume={fluid_volume:.17g}, reference={PRISM_REFERENCE_VOLUME_M3:.17g}, "
            f"relative_error={cad_relative_error:.17g}"
        )

    return volumes, {
        "prism_volumes": prism_volumes,
        "prism_interfaces": prism_interfaces,
        "atmosphere_volumes": atmosphere_volumes,
        "atmosphere_prism_layers": args.prism_atmosphere_layers,
        "shared_face_levels": shared_face_levels,
        "rim_shared": True,
        "reference_volume_m3": PRISM_REFERENCE_VOLUME_M3,
        "cad_relative_error": cad_relative_error,
    }


def mesh_element_counts() -> tuple[dict[str, int], dict[str, set[int]]]:
    counts: dict[str, int] = {}
    nodes_by_name: dict[str, set[int]] = {}
    element_types, element_tags, node_tags = gmsh.model.mesh.getElements(3)
    for element_type, tags, nodes in zip(
        element_types,
        element_tags,
        node_tags,
        strict=True,
    ):
        name = gmsh.model.mesh.getElementProperties(int(element_type))[0]
        counts[name] = counts.get(name, 0) + len(tags)
        nodes_by_name.setdefault(name, set()).update(int(tag) for tag in nodes)
    return counts, nodes_by_name


def prism_mesh_levels(
    nodes_by_name: dict[str, set[int]],
) -> tuple[int, list[float]]:
    prism_node_tags = {
        tag
        for name, tags in nodes_by_name.items()
        if name.startswith("Prism")
        for tag in tags
    }
    return len(prism_node_tags), node_levels(prism_node_tags)


def node_levels(node_tags: set[int]) -> list[float]:
    all_node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    z_by_tag = {
        int(tag): float(coordinates[3 * index + 2])
        for index, tag in enumerate(all_node_tags)
    }
    raw_levels = sorted(z_by_tag[tag] for tag in node_tags)
    levels: list[float] = []
    for elevation in raw_levels:
        if not levels or abs(elevation - levels[-1]) > 1.0e-10:
            levels.append(elevation)
    return levels


def prism_elements_in_volumes(
    volumes: list[int],
    label: str,
) -> tuple[int, set[int]]:
    count = 0
    node_tags: set[int] = set()
    for volume in volumes:
        types, tags, node_blocks = gmsh.model.mesh.getElements(3, int(volume))
        names = [
            gmsh.model.mesh.getElementProperties(int(item))[0]
            for item in types
        ]
        if not names or any(not name.startswith("Prism") for name in names):
            raise RuntimeError(
                f"{label} volume {volume} contains element types {names}"
            )
        count += sum(len(block) for block in tags)
        for nodes in node_blocks:
            node_tags.update(int(tag) for tag in nodes)
    return count, node_tags


def main() -> None:
    args = parse_args()
    if not 0 < args.riser_diameter < PIPE_DIAMETER:
        raise ValueError("riser-diameter must be between zero and pipe diameter")
    if args.atmosphere_width <= 4 * args.riser_diameter:
        raise ValueError("external atmosphere must span more than four riser diameters")
    if min(args.pipe_size, args.riser_size, args.atmosphere_size) <= 0:
        raise ValueError("mesh sizes must be positive")
    if args.riser_size > args.pipe_size:
        raise ValueError("riser-size must not exceed pipe-size")
    if args.curvature_elements < 20:
        raise ValueError("curvature-elements must be at least 20")
    if args.interface_size is not None and not (
        0 < args.interface_size <= args.riser_size
    ):
        raise ValueError("interface-size must be positive and no larger than riser-size")
    if args.interface_size is not None and not args.align_interface_band:
        raise ValueError("interface-size requires --align-interface-band")
    if args.prism_interface_band and (
        args.align_interface_band or args.interface_size is not None
    ):
        raise ValueError(
            "prism-interface-band is a separate profile from align-interface-band"
        )
    if args.prism_atmosphere_layers and not args.prism_interface_band:
        raise ValueError(
            "prism-atmosphere-layers requires prism-interface-band"
        )
    if args.prism_atmosphere_layers < 0:
        raise ValueError("prism-atmosphere-layers cannot be negative")
    if args.prism_atmosphere_layers and not math.isclose(
        (COMPUTATIONAL_TOP_Z - PHYSICAL_RIM_Z)
        / args.prism_atmosphere_layers,
        ATMOSPHERE_PRISM_LAYER_HEIGHT,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "prism-atmosphere-layers must produce exact "
            f"{ATMOSPHERE_PRISM_LAYER_HEIGHT:g} m layers"
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("*.stl"):
        old.unlink()

    riser_radius = args.riser_diameter / 2.0
    atmosphere_min_x = TEE_X - args.atmosphere_width / 2.0
    atmosphere_max_x = TEE_X + args.atmosphere_width / 2.0
    atmosphere_min_y = -args.atmosphere_width / 2.0
    atmosphere_max_y = args.atmosphere_width / 2.0

    gmsh.initialize()
    try:
        gmsh.model.add("Cong2017_BH3_3D_fluid")
        occ = gmsh.model.occ

        prism_metadata: dict[str, object] | None = None
        if args.prism_interface_band:
            volumes, prism_metadata = build_prism_geometry(
                occ,
                args,
                riser_radius,
            )
        else:
            pipe = occ.addCylinder(
                0.0,
                0.0,
                PIPE_AXIS_Z,
                PIPE_LENGTH,
                0.0,
                0.0,
                PIPE_RADIUS,
            )
            riser = occ.addCylinder(
                TEE_X,
                0.0,
                PIPE_AXIS_Z,
                0.0,
                0.0,
                PHYSICAL_RIM_Z - PIPE_AXIS_Z + BOOLEAN_OVERLAP,
                riser_radius,
            )
            atmosphere = occ.addBox(
                atmosphere_min_x,
                atmosphere_min_y,
                PHYSICAL_RIM_Z,
                args.atmosphere_width,
                args.atmosphere_width,
                COMPUTATIONAL_TOP_Z - PHYSICAL_RIM_Z,
            )

            apparatus, _ = occ.fuse([(3, pipe)], [(3, riser)])
            fused_fluid, _ = occ.fuse(apparatus, [(3, atmosphere)])
            free_surface = occ.addRectangle(
                TEE_X - args.atmosphere_width / 2.0,
                -args.atmosphere_width / 2.0,
                INITIAL_FREE_SURFACE_Z,
                args.atmosphere_width,
                args.atmosphere_width,
            )
            fragment_surfaces = [(2, free_surface)]
            if args.align_interface_band:
                for elevation in (
                    INITIAL_INTERFACE_LOWER_Z,
                    INITIAL_INTERFACE_UPPER_Z,
                ):
                    band_edge = occ.addRectangle(
                        TEE_X - args.atmosphere_width / 2.0,
                        -args.atmosphere_width / 2.0,
                        elevation,
                        args.atmosphere_width,
                        args.atmosphere_width,
                    )
                    fragment_surfaces.append((2, band_edge))
            valve_disk = occ.addDisk(
                VALVE_X,
                0.0,
                PIPE_AXIS_Z,
                PIPE_RADIUS,
                PIPE_RADIUS,
                zAxis=[1.0, 0.0, 0.0],
                xAxis=[0.0, 1.0, 0.0],
            )
            fluid, _ = occ.fragment(
                fused_fluid,
                [*fragment_surfaces, (2, valve_disk)],
                removeObject=True,
                removeTool=True,
            )
            occ.synchronize()

            volumes = [tag for dim, tag in fluid if dim == 3]
            if len(volumes) < 3:
                raise RuntimeError(
                    "Free-surface and valve fragmentation did not partition the fluid"
                )

        patches: dict[str, list[int]] = {
            "inlet": [],
            "closedEnd": [],
            "walls": [],
            "riserWall": [],
            "atmosphere": [],
        }
        boundaries = gmsh.model.getBoundary(
            [(3, tag) for tag in volumes],
            combined=True,
            oriented=False,
            recursive=False,
        )
        for dim, tag in boundaries:
            if dim != 2:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)

            if close(xmin, 0.0) and close(xmax, 0.0):
                patches["inlet"].append(tag)
                continue
            if close(xmin, PIPE_LENGTH) and close(xmax, PIPE_LENGTH):
                patches["closedEnd"].append(tag)
                continue

            above_rim = zmax > PHYSICAL_RIM_Z + 1.0e-5
            external_side = (
                (close(xmin, atmosphere_min_x) and close(xmax, atmosphere_min_x))
                or (close(xmin, atmosphere_max_x) and close(xmax, atmosphere_max_x))
                or (close(ymin, atmosphere_min_y) and close(ymax, atmosphere_min_y))
                or (close(ymin, atmosphere_max_y) and close(ymax, atmosphere_max_y))
                or (close(zmin, COMPUTATIONAL_TOP_Z) and close(zmax, COMPUTATIONAL_TOP_Z))
            )
            if above_rim and external_side:
                patches["atmosphere"].append(tag)
                continue

            confined_to_riser = (
                xmin >= TEE_X - riser_radius - 2.0e-5
                and xmax <= TEE_X + riser_radius + 2.0e-5
                and ymin >= -riser_radius - 2.0e-5
                and ymax <= riser_radius + 2.0e-5
                and zmax <= PHYSICAL_RIM_Z + 2.0e-5
                and zmax > PIPE_DIAMETER + 0.02
            )
            if confined_to_riser:
                patches["riserWall"].append(tag)
            else:
                patches["walls"].append(tag)

        # Physical groups are written directly into the volume mesh so
        # gmshToFoam receives exact patch names without a cut-cell background.
        for name, surfaces in patches.items():
            group = gmsh.model.addPhysicalGroup(2, surfaces)
            gmsh.model.setPhysicalName(2, group, name)
        fluid_group = gmsh.model.addPhysicalGroup(3, volumes)
        gmsh.model.setPhysicalName(3, fluid_group, "fluid")

        pipe_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(pipe_field, "VIn", args.pipe_size)
        gmsh.model.mesh.field.setNumber(pipe_field, "VOut", args.atmosphere_size)
        gmsh.model.mesh.field.setNumber(pipe_field, "XMin", -0.01)
        gmsh.model.mesh.field.setNumber(pipe_field, "XMax", PIPE_LENGTH + 0.01)
        gmsh.model.mesh.field.setNumber(pipe_field, "YMin", -0.03)
        gmsh.model.mesh.field.setNumber(pipe_field, "YMax", 0.03)
        gmsh.model.mesh.field.setNumber(pipe_field, "ZMin", -0.01)
        gmsh.model.mesh.field.setNumber(pipe_field, "ZMax", 0.06)

        riser_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(riser_field, "VIn", args.riser_size)
        gmsh.model.mesh.field.setNumber(riser_field, "VOut", args.atmosphere_size)
        gmsh.model.mesh.field.setNumber(
            riser_field, "XMin", TEE_X - riser_radius - 0.005
        )
        gmsh.model.mesh.field.setNumber(
            riser_field, "XMax", TEE_X + riser_radius + 0.005
        )
        gmsh.model.mesh.field.setNumber(
            riser_field, "YMin", -riser_radius - 0.005
        )
        gmsh.model.mesh.field.setNumber(
            riser_field, "YMax", riser_radius + 0.005
        )
        gmsh.model.mesh.field.setNumber(riser_field, "ZMin", 0.035)
        gmsh.model.mesh.field.setNumber(
            riser_field, "ZMax", PHYSICAL_RIM_Z + 0.02
        )

        tee_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(tee_field, "VIn", args.riser_size)
        gmsh.model.mesh.field.setNumber(tee_field, "VOut", args.atmosphere_size)
        gmsh.model.mesh.field.setNumber(tee_field, "XMin", TEE_X - 0.05)
        gmsh.model.mesh.field.setNumber(tee_field, "XMax", TEE_X + 0.05)
        gmsh.model.mesh.field.setNumber(tee_field, "YMin", -0.035)
        gmsh.model.mesh.field.setNumber(tee_field, "YMax", 0.035)
        gmsh.model.mesh.field.setNumber(tee_field, "ZMin", -0.005)
        gmsh.model.mesh.field.setNumber(tee_field, "ZMax", 0.10)

        valve_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(valve_field, "VIn", args.riser_size)
        gmsh.model.mesh.field.setNumber(valve_field, "VOut", args.atmosphere_size)
        gmsh.model.mesh.field.setNumber(valve_field, "XMin", VALVE_X - 0.03)
        gmsh.model.mesh.field.setNumber(valve_field, "XMax", VALVE_X + 0.03)
        gmsh.model.mesh.field.setNumber(valve_field, "YMin", -0.03)
        gmsh.model.mesh.field.setNumber(valve_field, "YMax", 0.03)
        gmsh.model.mesh.field.setNumber(valve_field, "ZMin", -0.005)
        gmsh.model.mesh.field.setNumber(valve_field, "ZMax", 0.055)

        mesh_fields = [pipe_field, riser_field, tee_field, valve_field]
        if args.interface_size is not None:
            interface_field = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(
                interface_field, "VIn", args.interface_size
            )
            gmsh.model.mesh.field.setNumber(
                interface_field, "VOut", args.atmosphere_size
            )
            gmsh.model.mesh.field.setNumber(
                interface_field, "XMin", TEE_X - riser_radius - 0.005
            )
            gmsh.model.mesh.field.setNumber(
                interface_field, "XMax", TEE_X + riser_radius + 0.005
            )
            gmsh.model.mesh.field.setNumber(
                interface_field, "YMin", -riser_radius - 0.005
            )
            gmsh.model.mesh.field.setNumber(
                interface_field, "YMax", riser_radius + 0.005
            )
            gmsh.model.mesh.field.setNumber(interface_field, "ZMin", 0.630)
            gmsh.model.mesh.field.setNumber(interface_field, "ZMax", 0.690)
            mesh_fields.append(interface_field)

        minimum_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            minimum_field,
            "FieldsList",
            mesh_fields,
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum_field)

        minimum_size = min(
            args.riser_size,
            args.interface_size
            if args.interface_size is not None
            else args.riser_size,
        )
        gmsh.option.setNumber("Mesh.MeshSizeMin", minimum_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", args.atmosphere_size)
        if args.interface_size is not None:
            # The global minimum must permit the local 2.5 mm band, but it
            # must not also release the curvature field to refine the entire
            # 6.59 m pipe below the normal refined-profile 4 mm floor.
            def spatial_size_floor(
                dim: int,
                tag: int,
                x: float,
                y: float,
                z: float,
                proposed_size: float,
            ) -> float:
                del dim, tag
                in_interface_band = (
                    math.hypot(x - TEE_X, y) <= riser_radius + 1.0e-5
                    and 0.630 - 1.0e-8 <= z <= 0.690 + 1.0e-8
                )
                local_floor = (
                    args.interface_size
                    if in_interface_band
                    else args.riser_size
                )
                return max(proposed_size, local_floor)

            gmsh.model.mesh.setSizeCallback(spatial_size_floor)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber(
            "Mesh.MeshSizeFromCurvature", args.curvature_elements
        )
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        # The classic Delaunay tetrahedralizer avoids the under-determined
        # boundary tets produced by HXT for this long pipe/box union.
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.mesh.generate(3)

        element_counts, nodes_by_name = mesh_element_counts()
        prism_count = sum(
            count
            for name, count in element_counts.items()
            if name.startswith("Prism")
        )
        tetrahedron_count = sum(
            count
            for name, count in element_counts.items()
            if name.startswith("Tetrahedron")
        )
        prism_node_count, prism_levels = prism_mesh_levels(nodes_by_name)
        interface_prism_node_count = prism_node_count
        interface_prism_levels = prism_levels
        atmosphere_prism_count = 0
        atmosphere_prism_node_count = 0
        atmosphere_prism_levels: list[float] = []
        if args.prism_interface_band:
            if prism_metadata is None:
                raise RuntimeError("Prism metadata was not initialized")
            slab_count, slab_nodes = prism_elements_in_volumes(
                prism_metadata["prism_volumes"],
                "Interface prism slab",
            )
            interface_prism_node_count = len(slab_nodes)
            interface_prism_levels = node_levels(slab_nodes)
            if args.prism_atmosphere_layers:
                atmosphere_prism_count, atmosphere_nodes = (
                    prism_elements_in_volumes(
                        prism_metadata["atmosphere_volumes"],
                        "Atmosphere prism",
                    )
                )
                atmosphere_prism_node_count = len(atmosphere_nodes)
                atmosphere_prism_levels = node_levels(atmosphere_nodes)
            if (
                prism_count <= 0
                or prism_count != slab_count + atmosphere_prism_count
            ):
                raise RuntimeError(
                    "Prism count mismatch: "
                    f"global={prism_count}, slab={slab_count}, "
                    f"atmosphere={atmosphere_prism_count}"
                )
            expected_levels = [
                PRISM_BOTTOM_Z + index * PRISM_LAYER_HEIGHT
                for index in range(25)
            ]
            if len(interface_prism_levels) != 25 or any(
                abs(actual - expected) > 1.0e-10
                for actual, expected in zip(
                    interface_prism_levels,
                    expected_levels,
                    strict=True,
                )
            ):
                raise RuntimeError(
                    "Expected 25 exact 2.5 mm interface prism levels, found "
                    f"{interface_prism_levels}"
                )
            for key_level in (
                INITIAL_INTERFACE_LOWER_Z,
                INITIAL_FREE_SURFACE_Z,
                INITIAL_INTERFACE_UPPER_Z,
            ):
                if not any(
                    abs(level - key_level) <= 1.0e-10
                    for level in interface_prism_levels
                ):
                    raise RuntimeError(
                        f"Required prism interface z={key_level} is absent"
                    )
            if args.prism_atmosphere_layers:
                expected_atmosphere_levels = [
                    PHYSICAL_RIM_Z
                    + index * ATMOSPHERE_PRISM_LAYER_HEIGHT
                    for index in range(args.prism_atmosphere_layers + 1)
                ]
                if len(atmosphere_prism_levels) != len(
                    expected_atmosphere_levels
                ) or any(
                    abs(actual - expected) > 1.0e-10
                    for actual, expected in zip(
                        atmosphere_prism_levels,
                        expected_atmosphere_levels,
                        strict=True,
                    )
                ):
                    raise RuntimeError(
                        "Atmosphere prism levels do not form exact "
                        f"{ATMOSPHERE_PRISM_LAYER_HEIGHT:g} m "
                        f"layers: {atmosphere_prism_levels}"
                    )
        elif prism_count:
            raise RuntimeError(
                f"Non-prism profile unexpectedly generated {prism_count} prisms"
            )

        args.mesh_output.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(args.mesh_output))

        # Export review/debug surfaces from the exact same volume mesh.
        for name, surfaces in patches.items():
            export_patch(output, name, surfaces)

        fluid_volume = sum(occ.getMass(3, tag) for tag in volumes)
        analytic_pocket = math.pi * PIPE_DIAMETER**2 * 0.61 / 4.0
        print(f"output_dir={output}")
        print(f"mesh_output={args.mesh_output.resolve()}")
        print(f"fluid_volume_m3={fluid_volume:.12g}")
        print(f"analytic_initial_pocket_m3={analytic_pocket:.12g}")
        print(f"pipe_diameter_m={PIPE_DIAMETER}")
        print(f"riser_diameter_m={args.riser_diameter}")
        print(f"circular_area_ratio={(args.riser_diameter / PIPE_DIAMETER) ** 2:.9g}")
        print(f"physical_rim_z_m={PHYSICAL_RIM_Z}")
        print(f"computational_top_z_m={COMPUTATIONAL_TOP_Z}")
        print(f"conformal_initial_free_surface_z_m={INITIAL_FREE_SURFACE_Z}")
        print(
            "interface_band_aligned="
            f"{args.align_interface_band or args.prism_interface_band}"
        )
        print(
            "conformal_interface_band_edges_z_m="
            f"{INITIAL_INTERFACE_LOWER_Z},{INITIAL_INTERFACE_UPPER_Z}"
        )
        print(f"interface_size_m={args.interface_size}")
        print(
            "spatial_mesh_size_floor="
            f"{args.interface_size is not None}"
        )
        print(f"conformal_valve_plane_x_m={VALVE_X}")
        print(f"fluid_partitions={len(volumes)}")
        print(f"cells_3d={sum(element_counts.values())}")
        print(f"tetrahedron_count={tetrahedron_count}")
        print(f"prism_count={prism_count}")
        print(f"gmsh_version={gmsh.option.getString('General.Version')}")
        print(f"prism_profile={args.prism_interface_band}")
        if args.prism_interface_band:
            if prism_metadata is None:
                raise RuntimeError("Prism metadata was lost before audit")
            print(f"total_prism_node_count={prism_node_count}")
            print(f"prism_node_count={interface_prism_node_count}")
            print(f"prism_layer_count={len(interface_prism_levels)}")
            print(
                "prism_layer_z_m="
                + ",".join(
                    f"{level:.10g}" for level in interface_prism_levels
                )
            )
            print("prism_key_layers_asserted=True")
            print(
                "atmosphere_prism_layer_count="
                f"{args.prism_atmosphere_layers}"
            )
            print(
                "atmosphere_prism_cell_count="
                f"{atmosphere_prism_count}"
            )
            print(
                "atmosphere_prism_node_count="
                f"{atmosphere_prism_node_count}"
            )
            print(
                "atmosphere_prism_layer_z_m="
                + ",".join(
                    f"{level:.10g}" for level in atmosphere_prism_levels
                )
            )
            print(
                "prism_shared_face_count="
                f"{len(prism_metadata['shared_face_levels'])}"
            )
            print("prism_shared_faces_asserted=True")
            print(f"prism_rim_shared_asserted={prism_metadata['rim_shared']}")
            print(
                "fluid_reference_volume_m3="
                f"{prism_metadata['reference_volume_m3']:.12g}"
            )
            print(
                "cad_to_reference_volume_relative_error="
                f"{prism_metadata['cad_relative_error']:.12g}"
            )
        print(f"pipe_size_m={args.pipe_size}")
        print(f"riser_size_m={args.riser_size}")
        print(f"atmosphere_size_m={args.atmosphere_size}")
        print(f"curvature_elements_per_2pi={args.curvature_elements}")
        for name, surfaces in patches.items():
            print(f"{name}_surfaces={len(surfaces)}")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()

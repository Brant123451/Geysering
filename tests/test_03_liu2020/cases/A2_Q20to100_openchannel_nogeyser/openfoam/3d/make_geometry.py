# -*- coding: utf-8 -*-
"""Watertight multi-STL geometry for the Liu2020 Case A2 rig (3D interFoam).

Coordinates match the 1D composite-domain model exactly:
  x = 0 at the chamber upstream wall, chamber occupies x in [0, 0.3];
  z = 0 at the chamber floor (= downstream pipe invert).

Pieces (each a separate STL -> its own OpenFOAM patch):
  walls.stl              headbox shell + pipes + chamber shell
  riserWall.stl          riser tube
  inlet.stl              numerical headbox bottom (flow-rate inlet)
  headboxAtmosphere.stl  numerical headbox open top
  riserOutlet.stl        physical riser open top
  outlet.stl             reported downstream-pipe end

All pipe/wall rings share identical vertices.  There are no overlapping lips
or penetrations: the combined geometry.stl is a genuinely closed surface that
can be checked independently and meshed by cartesianMesh.
"""
import argparse
from collections import defaultdict, deque
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument(
    "--case-dir",
    type=Path,
    default=HERE / "case",
    help="OpenFOAM case receiving constant/triSurface (default: source case)",
)
args = parser.parse_args()
OUT = args.case_dir.resolve() / "constant" / "triSurface"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "atmosphere.stl").unlink(missing_ok=True)  # retired combined patch

NSEG = 64              # pipe circumference segments
NSEG_R = 48            # riser circumference segments

# ---- rig dimensions (LiuCase) ----
Lu, Du, slope = 5.80, 0.20, 0.01
Lc, Wc, Hc, drop = 0.30, 0.30, 0.45, 0.18
dr, Hr = 0.06, 1.22
Ld, Dd = 5.95, 0.28

ru, rd, rr = Du / 2, Dd / 2, dr / 2
x_up0 = -Lu                       # upstream pipe start (headbox face)
zax_up = lambda x: (drop + ru) - slope * x   # upstream pipe axis elevation
zax_dn = rd                       # downstream pipe axis elevation
x_dn1 = Lc + Ld                   # downstream pipe end (outfall face)
xr, yr = Lc / 2, 0.0              # riser axis
z_lid = Hc
z_rtop = Hc + Hr

# The paper does not report the upstream tank dimensions.  This is explicitly
# a numerical boundary extension, not a reconstructed apparatus dimension.
# The reported upstream length stops at this tank face exactly.
# headbox
HB = dict(x0=x_up0 - 0.35, x1=x_up0, y0=-0.15, y1=0.15, z0=drop + ru - 0.10 - slope * x_up0 - 0.05,
          z1=1.10)


def tri_block(tris):
    return np.asarray(tris, dtype=np.float64)


def quad(a, b, c, d):
    """two triangles for quad a-b-c-d (ccw)"""
    return [[a, b, c], [a, c, d]]


def write_stl(path, solids):
    """solids: dict name -> (n,3,3) triangle array; ASCII STL"""
    with open(path, "w") as f:
        for name, tris in solids.items():
            f.write(f"solid {name}\n")
            for t in tris:
                n = np.cross(t[1] - t[0], t[2] - t[0])
                nn = np.linalg.norm(n)
                if nn < 1e-16:
                    continue
                n = n / nn
                f.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                f.write("  outer loop\n")
                for v in t:
                    f.write(f"   vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
                f.write("  endloop\n endfacet\n")
            f.write(f"endsolid {name}\n")


def orient_closed_surface(solids):
    """Orient all connected triangles consistently and outward.

    Patch seams participate in the same edge graph, so orientation is fixed
    globally without losing the five physical region names.
    """
    names = []
    flat = []
    for name, tris in solids.items():
        for tri in tris:
            names.append(name)
            flat.append(np.asarray(tri, dtype=float))

    edge_uses = defaultdict(list)
    for ti, tri in enumerate(flat):
        keys = [tuple(np.round(v, 12)) for v in tri]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((keys[i], keys[j])))
            direction = 1 if (keys[i], keys[j]) == key else -1
            edge_uses[key].append((ti, direction))

    bad = [edge for edge, uses in edge_uses.items() if len(uses) != 2]
    if bad:
        raise RuntimeError(f"surface is not closed: {len(bad)} unmatched/nonmanifold edges")

    neighbours = defaultdict(list)
    for uses in edge_uses.values():
        (a, da), (b, db) = uses
        same_direction = da == db
        neighbours[a].append((b, same_direction))
        neighbours[b].append((a, same_direction))

    flip = {0: False}
    queue = deque([0])
    while queue:
        a = queue.popleft()
        for b, toggle in neighbours[a]:
            candidate = flip[a] ^ toggle
            if b in flip and flip[b] != candidate:
                raise RuntimeError("surface orientation graph is inconsistent")
            if b not in flip:
                flip[b] = candidate
                queue.append(b)
    if len(flip) != len(flat):
        raise RuntimeError("surface has disconnected components")

    for i, do_flip in flip.items():
        if do_flip:
            flat[i] = flat[i][[0, 2, 1]]

    signed_volume = sum(
        np.dot(t[0], np.cross(t[1], t[2])) for t in flat
    ) / 6.0
    if signed_volume < 0:
        flat = [t[[0, 2, 1]] for t in flat]

    oriented = {}
    for name in solids:
        oriented[name] = np.asarray(
            [tri for label, tri in zip(names, flat) if label == name],
            dtype=float,
        )
    return oriented, abs(signed_volume)


def tube(p0, p1, r, nseg):
    """open cylinder tube from p0 to p1 (axis), radius r"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    ax = ax / np.linalg.norm(ax)
    # orthonormal frame
    ref = np.array([0.0, 0.0, 1.0]) if abs(ax[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ax, ref); e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    th = np.linspace(0, 2 * np.pi, nseg + 1)
    ring0 = p0[None, :] + r * (np.cos(th)[:, None] * e1[None, :] + np.sin(th)[:, None] * e2[None, :])
    ring1 = ring0 + (p1 - p0)[None, :]
    tris = []
    for i in range(nseg):
        tris += quad(ring0[i], ring0[i + 1], ring1[i + 1], ring1[i])
    return tri_block(tris)


def tube_x(p0, p1, r, nseg):
    """Open tube with circular rings in x=constant planes.

    The upstream axis has a 1:100 slope.  Using vertical end rings makes the
    STL exactly conformal with the vertical tank/chamber faces; the resulting
    0.005% difference from a plane-normal section is far below either mesh
    resolution while preserving the specified circular flow area at each x.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    th = np.linspace(0, 2 * np.pi, nseg + 1)

    def ring(c):
        return np.stack(
            [
                np.full_like(th, c[0]),
                c[1] + r * np.cos(th),
                c[2] + r * np.sin(th),
            ],
            axis=1,
        )

    ring0, ring1 = ring(p0), ring(p1)
    tris = []
    for i in range(nseg):
        tris += quad(ring0[i], ring1[i], ring1[i + 1], ring0[i + 1])
    return tri_block(tris)


def disk(center, r, nseg, normal_up=True, zconst=None):
    """triangle fan disk in a plane z=const (riser top)"""
    c = np.asarray(center, float)
    th = np.linspace(0, 2 * np.pi, nseg + 1)
    ring = np.stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th),
                     np.full_like(th, c[2])], axis=1)
    tris = []
    for i in range(nseg):
        if normal_up:
            tris.append([c, ring[i], ring[i + 1]])
        else:
            tris.append([c, ring[i + 1], ring[i]])
    return tri_block(tris)


def disk_x(center, r, nseg, normal_positive=True):
    """Triangle fan disk normal to x (downstream pipe outlet)."""
    c = np.asarray(center, float)
    th = np.linspace(0, 2 * np.pi, nseg + 1)
    ring = np.stack(
        [
            np.full_like(th, c[0]),
            c[1] + r * np.cos(th),
            c[2] + r * np.sin(th),
        ],
        axis=1,
    )
    tris = []
    for i in range(nseg):
        if normal_positive:
            tris.append([c, ring[i], ring[i + 1]])
        else:
            tris.append([c, ring[i + 1], ring[i]])
    return tri_block(tris)


def rect(a, b, c, d):
    return tri_block(quad(np.asarray(a, float), np.asarray(b, float),
                          np.asarray(c, float), np.asarray(d, float)))


def rect_with_hole(plane_axis, plane_val, u0, u1, v0, v1, cu, cv, r, nseg):
    """Rectangle with a circular hole and only four outer boundary edges.

    Each rectangle side and its corresponding circular arc form one polygon.
    Fan triangulation keeps neighboring box faces exactly conformal (the old
    radial triangulation left T-junctions along every rectangle edge).
    """
    th = np.linspace(0, 2 * np.pi, nseg + 1)[:-1]
    tris = []

    def to3(u, v):
        if plane_axis == "x":
            return np.array([plane_val, u, v])
        if plane_axis == "y":
            return np.array([u, plane_val, v])
        return np.array([u, v, plane_val])   # z-plane

    circ = [(cu + r * np.cos(a), cv + r * np.sin(a)) for a in th]
    # Counter-clockwise rectangle corners, beginning at lower right.
    outer = [(u1, v0), (u1, v1), (u0, v1), (u0, v0)]
    corner_idx = []
    for u, v in outer:
        angle = np.arctan2(v - cv, u - cu) % (2 * np.pi)
        corner_idx.append(int(np.rint(angle * nseg / (2 * np.pi))) % nseg)

    for side in range(4):
        a, b = outer[side], outer[(side + 1) % 4]
        ia, ib = corner_idx[side], corner_idx[(side + 1) % 4]
        ids = [ia]
        while ids[-1] != ib:
            ids.append((ids[-1] + 1) % nseg)
        arc = [circ[i] for i in ids]
        # Polygon order: outer A->B, then inner arc B->A (clockwise).
        poly = [a, b, *reversed(arc)]
        for i in range(1, len(poly) - 1):
            tris.append([to3(*poly[0]), to3(*poly[i]), to3(*poly[i + 1])])
    return tri_block(tris)


def box_faces(x0, x1, y0, y1, z0, z1, skip=()):
    """axis box; skip faces by name in {x0,x1,y0,y1,z0,z1}"""
    p = lambda X, Y, Z: np.array([X, Y, Z])
    out = []
    if "x0" not in skip:
        out += quad(p(x0, y0, z0), p(x0, y0, z1), p(x0, y1, z1), p(x0, y1, z0))
    if "x1" not in skip:
        out += quad(p(x1, y0, z0), p(x1, y1, z0), p(x1, y1, z1), p(x1, y0, z1))
    if "y0" not in skip:
        out += quad(p(x0, y0, z0), p(x1, y0, z0), p(x1, y0, z1), p(x0, y0, z1))
    if "y1" not in skip:
        out += quad(p(x0, y1, z0), p(x0, y1, z1), p(x1, y1, z1), p(x1, y1, z0))
    if "z0" not in skip:
        out += quad(p(x0, y0, z0), p(x0, y1, z0), p(x1, y1, z0), p(x1, y0, z0))
    if "z1" not in skip:
        out += quad(p(x0, y0, z1), p(x1, y0, z1), p(x1, y1, z1), p(x0, y1, z1))
    return tri_block(out)


# ================= build the pieces =================
walls = []

# ---- headbox: sides + holed downstream face (top=atmosphere, bottom=inlet)
walls.append(box_faces(HB["x0"], HB["x1"], HB["y0"], HB["y1"], HB["z0"], HB["z1"],
                       skip=("x1", "z0", "z1")))
walls.append(rect_with_hole("x", HB["x1"], HB["y0"], HB["y1"], HB["z0"], HB["z1"],
                            0.0, zax_up(x_up0), ru, NSEG))

# ---- upstream pipe tube: exact paper length, ring-matched at both ends
p0 = np.array([x_up0, 0.0, zax_up(x_up0)])
p1 = np.array([0.0, 0.0, zax_up(0.0)])
walls.append(tube_x(p0, p1, ru, NSEG))

# ---- chamber shell ----
# upstream wall (x=0) with pipe hole
walls.append(rect_with_hole("x", 0.0, -Wc / 2, Wc / 2, 0.0, Hc,
                            0.0, drop + ru, ru, NSEG))
# downstream wall (x=Lc) with pipe hole tangent to the floor
walls.append(rect_with_hole("x", Lc, -Wc / 2, Wc / 2, 0.0, Hc,
                            0.0, zax_dn, rd, NSEG))
# lid with riser hole
walls.append(rect_with_hole("z", Hc, 0.0, Lc, -Wc / 2, Wc / 2,
                            xr, yr, rr, NSEG_R))
# floor + side walls
walls.append(box_faces(0.0, Lc, -Wc / 2, Wc / 2, 0.0, Hc,
                       skip=("x0", "x1", "z1")))

# ---- downstream pipe tube ----
walls.append(tube_x([Lc, 0.0, zax_dn], [x_dn1, 0.0, zax_dn], rd, NSEG))

walls_tris = np.concatenate(walls, axis=0)

# ---- riser tube (separate STL: finer refinement level) ----
riser_tris = tube([xr, yr, z_lid], [xr, yr, z_rtop], rr, NSEG_R)

# ---- inlet: headbox bottom ----
inlet_tris = rect([HB["x0"], HB["y0"], HB["z0"]], [HB["x1"], HB["y0"], HB["z0"]],
                  [HB["x1"], HB["y1"], HB["z0"]], [HB["x0"], HB["y1"], HB["z0"]])

# ---- distinct atmospheric openings (separate overflow accounting) ----
headbox_atmo_tris = rect(
    [HB["x0"], HB["y0"], HB["z1"]],
    [HB["x0"], HB["y1"], HB["z1"]],
    [HB["x1"], HB["y1"], HB["z1"]],
    [HB["x1"], HB["y0"], HB["z1"]],
)
riser_outlet_tris = disk([xr, yr, z_rtop], rr, NSEG_R)

# ---- outlet: exact end of the reported downstream pipe.  The physical
# overflow-weir/tank geometry was not reported; its hd/Dd=1/4 control is
# represented by the phase/pressure boundary condition, not invented solids.
outlet_tris = disk_x([x_dn1, 0.0, zax_dn], rd, NSEG)

pieces, enclosed_volume = orient_closed_surface(
    {
        "walls": walls_tris,
        "riserWall": riser_tris,
        "inlet": inlet_tris,
        "headboxAtmosphere": headbox_atmo_tris,
        "riserOutlet": riser_outlet_tris,
        "outlet": outlet_tris,
    }
)
for name, tris in pieces.items():
    write_stl(OUT / f"{name}.stl", {name: tris})
write_stl(OUT / "geometry.stl", pieces)

for f in (
    "walls",
    "riserWall",
    "inlet",
    "headboxAtmosphere",
    "riserOutlet",
    "outlet",
    "geometry",
):
    p = OUT / f"{f}.stl"
    print(f"{f}.stl  {p.stat().st_size/1e6:.2f} MB")
print("bounding box: x[%.2f, %.2f] y[-0.15,0.15] z[%.2f, %.2f]"
      % (HB["x0"], x_dn1, HB["z0"], z_rtop))
print(f"closed fluid volume: {enclosed_volume:.8f} m3")

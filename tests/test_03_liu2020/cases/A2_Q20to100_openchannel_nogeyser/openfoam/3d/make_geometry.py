# -*- coding: utf-8 -*-
"""Watertight multi-STL geometry for the Liu2020 Case A2 rig (3D interFoam).

Coordinates match the 1D composite-domain model exactly:
  x = 0 at the chamber upstream wall, chamber occupies x in [0, 0.3];
  z = 0 at the chamber floor (= downstream pipe invert).

Pieces (each a separate STL -> its own snappy patch):
  walls.stl      headbox shell + both pipe tubes + chamber shell (with the
                 three circular openings) + outfall shell
  riserWall.stl  riser tube (own STL for finer surface refinement)
  inlet.stl      headbox bottom (flow-rate inlet)
  atmosphere.stl headbox top + riser top disk + outfall top
  outlet.stl     outfall far face + outfall bottom

Watertightness policy at pipe/wall junctions: every wall hole has radius
(pipe_r - 1 mm) and the pipe tube PENETRATES 2 cm past the wall, so the
1 mm annular lip overlaps the tube -- the flood fill cannot leak, and the
sub-grid lip is swallowed by the mesh resolution.
"""
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "case" / "constant" / "triSurface"
OUT.mkdir(parents=True, exist_ok=True)

EPS_LIP = 0.001        # hole radius deficit vs pipe radius [m]
PEN = 0.02             # tube penetration past each wall [m]
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

# headbox
HB = dict(x0=x_up0 - 0.35, x1=x_up0, y0=-0.15, y1=0.15, z0=drop + ru - 0.10 - slope * x_up0 - 0.05,
          z1=1.10)
# outfall box
OF = dict(x0=x_dn1, x1=x_dn1 + 0.40, y0=-0.15, y1=0.15, z0=-0.25, z1=0.45)


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


def rect(a, b, c, d):
    return tri_block(quad(np.asarray(a, float), np.asarray(b, float),
                          np.asarray(c, float), np.asarray(d, float)))


def rect_with_hole(plane_axis, plane_val, u0, u1, v0, v1, cu, cv, r, nseg):
    """rectangle [u0,u1]x[v0,v1] in the plane {plane_axis=plane_val} with a
    circular hole (center (cu,cv), radius r).  Ring triangulation: sample the
    circle and the rectangle boundary by the SAME angles from the hole
    center (rectangle is convex -> radial parametrization is bijective)."""
    th = np.linspace(0, 2 * np.pi, nseg + 1)[:-1]
    tris = []

    def to3(u, v):
        if plane_axis == "x":
            return np.array([plane_val, u, v])
        if plane_axis == "y":
            return np.array([u, plane_val, v])
        return np.array([u, v, plane_val])   # z-plane

    def rect_pt(a):
        du, dv = np.cos(a), np.sin(a)
        tmax = np.inf
        if du > 1e-12:
            tmax = min(tmax, (u1 - cu) / du)
        elif du < -1e-12:
            tmax = min(tmax, (u0 - cu) / du)
        if dv > 1e-12:
            tmax = min(tmax, (v1 - cv) / dv)
        elif dv < -1e-12:
            tmax = min(tmax, (v0 - cv) / dv)
        return cu + tmax * du, cv + tmax * dv

    circ = [(cu + r * np.cos(a), cv + r * np.sin(a)) for a in th]
    outer = [rect_pt(a) for a in th]
    n = len(th)
    for i in range(n):
        j = (i + 1) % n
        c0, c1 = circ[i], circ[j]
        o0, o1 = outer[i], outer[j]
        t1 = [to3(*c0), to3(*o0), to3(*o1)]
        t2 = [to3(*c0), to3(*o1), to3(*c1)]
        tris += [t1, t2]
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
                            0.0, zax_up(x_up0), ru - EPS_LIP, NSEG))

# ---- upstream pipe tube (penetrates headbox and chamber walls)
p0 = np.array([x_up0 - PEN, 0.0, zax_up(x_up0 - PEN)])
p1 = np.array([0.0 + PEN, 0.0, zax_up(0.0 + PEN)])
walls.append(tube(p0, p1, ru, NSEG))

# ---- chamber shell ----
# upstream wall (x=0) with pipe hole
walls.append(rect_with_hole("x", 0.0, -Wc / 2, Wc / 2, 0.0, Hc,
                            0.0, drop + ru, ru - EPS_LIP, NSEG))
# downstream wall (x=Lc) with pipe hole (tangent to the floor -> lip handles it)
walls.append(rect_with_hole("x", Lc, -Wc / 2, Wc / 2, 0.0, Hc,
                            0.0, zax_dn, rd - EPS_LIP, NSEG))
# lid with riser hole
walls.append(rect_with_hole("z", Hc, 0.0, Lc, -Wc / 2, Wc / 2,
                            xr, yr, rr - EPS_LIP, NSEG_R))
# floor + side walls
walls.append(box_faces(0.0, Lc, -Wc / 2, Wc / 2, 0.0, Hc,
                       skip=("x0", "x1", "z1")))

# ---- downstream pipe tube ----
walls.append(tube([Lc - PEN, 0.0, zax_dn], [x_dn1 + PEN, 0.0, zax_dn], rd, NSEG))

# ---- outfall box: near face holed; top=atmo, far+bottom=outlet ----
walls.append(rect_with_hole("x", OF["x0"], OF["y0"], OF["y1"], OF["z0"], OF["z1"],
                            0.0, zax_dn, rd - EPS_LIP, NSEG))
walls.append(box_faces(OF["x0"], OF["x1"], OF["y0"], OF["y1"], OF["z0"], OF["z1"],
                       skip=("x0", "x1", "z0", "z1")))

# ---- tailwater weir plate (paper Series A: weir-controlled open-channel
# tailwater, hd = Dd/4 at Q0; 1D model: sharp-crested weir, crest AT the
# downstream pipe invert z=0).  Solid 2 cm plate across the outfall box,
# crest z=0: rating gives hd~0.088 m at 20 L/s, ~0.26 m at 100 L/s --
# identical closure to the 1D weir_Cd=0.62 law ----
W_X0, W_X1 = OF["x0"] + 0.23, OF["x0"] + 0.25
walls.append(box_faces(W_X0, W_X1, OF["y0"], OF["y1"], OF["z0"], 0.0,
                       skip=("y0", "y1")))
# pool bottom (upstream of the plate) is SOLID -- an open bottom there would
# drain the weir pool from below and defeat the tailwater control
walls.append(rect([OF["x0"], OF["y0"], OF["z0"]], [OF["x0"], OF["y1"], OF["z0"]],
                  [W_X1, OF["y1"], OF["z0"]], [W_X1, OF["y0"], OF["z0"]]))

walls_tris = np.concatenate(walls, axis=0)

# ---- riser tube (separate STL: finer refinement level) ----
riser_tris = tube([xr, yr, z_lid - PEN], [xr, yr, z_rtop], rr, NSEG_R)

# ---- inlet: headbox bottom ----
inlet_tris = rect([HB["x0"], HB["y0"], HB["z0"]], [HB["x1"], HB["y0"], HB["z0"]],
                  [HB["x1"], HB["y1"], HB["z0"]], [HB["x0"], HB["y1"], HB["z0"]])

# ---- atmosphere: headbox top + riser top disk + outfall top ----
atmo = [rect([HB["x0"], HB["y0"], HB["z1"]], [HB["x0"], HB["y1"], HB["z1"]],
             [HB["x1"], HB["y1"], HB["z1"]], [HB["x1"], HB["y0"], HB["z1"]]),
        disk([xr, yr, z_rtop], rr, NSEG_R),
        rect([OF["x0"], OF["y0"], OF["z1"]], [OF["x0"], OF["y1"], OF["z1"]],
             [OF["x1"], OF["y1"], OF["z1"]], [OF["x1"], OF["y0"], OF["z1"]])]
atmo_tris = np.concatenate(atmo, axis=0)

# ---- outlet: outfall far face + bottom DOWNSTREAM of the weir plate ----
outl = [rect([OF["x1"], OF["y0"], OF["z0"]], [OF["x1"], OF["y1"], OF["z0"]],
             [OF["x1"], OF["y1"], OF["z1"]], [OF["x1"], OF["y0"], OF["z1"]]),
        rect([W_X1, OF["y0"], OF["z0"]], [W_X1, OF["y1"], OF["z0"]],
             [OF["x1"], OF["y1"], OF["z0"]], [OF["x1"], OF["y0"], OF["z0"]])]
outlet_tris = np.concatenate(outl, axis=0)

write_stl(OUT / "walls.stl", {"walls": walls_tris})
write_stl(OUT / "riserWall.stl", {"riserWall": riser_tris})
write_stl(OUT / "inlet.stl", {"inlet": inlet_tris})
write_stl(OUT / "atmosphere.stl", {"atmosphere": atmo_tris})
write_stl(OUT / "outlet.stl", {"outlet": outlet_tris})

for f in ("walls", "riserWall", "inlet", "atmosphere", "outlet"):
    p = OUT / f"{f}.stl"
    print(f"{f}.stl  {p.stat().st_size/1e6:.2f} MB")
print("bounding box: x[%.2f, %.2f] y[-0.16,0.16] z[%.2f, %.2f]"
      % (HB["x0"], OF["x1"], OF["z0"], z_rtop))

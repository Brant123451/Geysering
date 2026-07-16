#!/usr/bin/env python3
"""Render true front-view frames from OpenFOAM y=0 cuttingPlane VTP (alpha.water)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
import vtk
from vtk.util.numpy_support import vtk_to_numpy

WATER = "#2f7fdb"
AIR = "#dfe7ef"
BG = "#0a1520"
PIPE_LEN = 6.59
EXT_TOP = 3.025
INVERT = -0.025


def read_vtp(path: Path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()
    points = vtk_to_numpy(polydata.GetPoints().GetData())
    polys = polydata.GetPolys()
    polys.InitTraversal()
    id_list = vtk.vtkIdList()
    faces: list[list[int]] = []
    while polys.GetNextCell(id_list):
        faces.append([id_list.GetId(i) for i in range(id_list.GetNumberOfIds())])
    alpha = None
    pdata = polydata.GetPointData()
    for i in range(pdata.GetNumberOfArrays()):
        arr = pdata.GetArray(i)
        if arr is None:
            continue
        name = arr.GetName() or ""
        if "alpha" in name.lower():
            alpha = vtk_to_numpy(arr)
            break
    if alpha is None:
        cdata = polydata.GetCellData()
        for i in range(cdata.GetNumberOfArrays()):
            arr = cdata.GetArray(i)
            if arr is None:
                continue
            name = arr.GetName() or ""
            if "alpha" in name.lower():
                alpha = vtk_to_numpy(arr)
                break
    if alpha is None:
        raise RuntimeError(f"No alpha field in {path}")
    return points, faces, alpha


def render_frame(vtp: Path, out_png: Path) -> None:
    points, faces, alpha = read_vtp(vtp)
    if len(alpha) == len(points):
        cell_alpha = np.array([float(alpha[face].mean()) for face in faces])
    elif len(alpha) == len(faces):
        cell_alpha = np.asarray(alpha, dtype=float)
    else:
        raise RuntimeError(
            f"alpha length {len(alpha)} matches neither points {len(points)} "
            f"nor faces {len(faces)}"
        )
    verts = [points[face][:, [0, 2]] for face in faces]
    colors = [WATER if a >= 0.5 else AIR for a in cell_alpha]
    fig, ax = plt.subplots(figsize=(22, 10), dpi=140)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.add_collection(PolyCollection(verts, facecolors=colors, edgecolors="none"))
    ax.set_xlim(-0.05, PIPE_LEN + 0.05)
    ax.set_ylim(INVERT - 0.05, EXT_TOP + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=BG, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cut-root",
        type=Path,
        required=True,
        help="postProcessing/frontViewCut directory",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="BH6_real_y0")
    parser.add_argument("--copy-artifacts", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    times = sorted(
        [p for p in args.cut_root.iterdir() if p.is_dir()],
        key=lambda p: float(p.name),
    )
    written = []
    for tdir in times:
        vtp = tdir / "yMid.vtp"
        if not vtp.exists():
            continue
        out = args.out_dir / f"{args.prefix}_t{tdir.name}.png"
        render_frame(vtp, out)
        written.append(out)
        if args.copy_artifacts:
            args.copy_artifacts.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, args.copy_artifacts / out.name)
        print(f"wrote {out}")
    print(f"frames={len(written)}")


if __name__ == "__main__":
    main()

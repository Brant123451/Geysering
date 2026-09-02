#!/usr/bin/env python3
"""Apply the Mahyawansi Table-1 total-pressure profiles to ``p_rgh``.

This utility is deliberately independent of OpenFOAM executables.  It reads an
ASCII ``polyMesh`` to obtain patch-face centres, converts the published total
pressure equations to OpenFOAM-v2512 ``prghTotalPressure`` dictionaries, and
writes a deterministic JSON audit.  No value is inferred from a previous run.

The frozen densities are the values used by this case's two-dimensional
``thermophysicalProperties``/``setExprFields`` translation.  They are not
rounded literature material properties.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import re
import stat
import tempfile
from typing import Mapping, Sequence


SCHEMA_VERSION = 1
PATM_PA = 101_325.0
G_M_S2 = 9.81
RHO_WATER_KG_M3 = 998.4
RHO_AIR_KG_M3 = 1.204317575
HREF_M = 0.0
AMBIENT_RIM_Z_M = 1.02

# Exact reduced-pressure constants implied by the frozen translation.  These
# are hard gates, not convenient rounded display values.
WATER_INLET_ZERO_U_PRGH_PA = 107_064.462144
WATER_OUTLET_ZERO_U_PRGH_PA = 107_044.873536
AMBIENT_ZERO_U_PRGH_PA = 101_337.05064251897

FORMULA_TOLERANCE_PA = 1.0e-8
GEOMETRY_TOLERANCE_M = 1.0e-10


class ConversionError(RuntimeError):
    """Raised when a source, mesh, field, or formula gate is not satisfied."""


@dataclass(frozen=True)
class BoundaryPatch:
    name: str
    n_faces: int
    start_face: int


@dataclass(frozen=True)
class FieldPatchBlock:
    name: str
    name_start: int
    brace_start: int
    brace_end: int


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    density_kg_m3: float
    equation_id: str
    equation_gauge: str
    head_m: float | None
    expected_zero_u_prgh_pa: float

    def p0_gauge_pa(self, z_m: float) -> float:
        if self.head_m is None and self.equation_gauge == "0":
            return 0.0
        if self.head_m is None:
            raise ConversionError(f"profile {self.name!r} has no pressure head")
        return self.density_kg_m3 * G_M_S2 * (self.head_m - z_m)


PROFILE_SPECS: tuple[ProfileSpec, ...] = (
    ProfileSpec(
        name="waterInlet",
        density_kg_m3=RHO_WATER_KG_M3,
        equation_id="water_inlet_table1",
        equation_gauge="rho_w*g*(0.586-z)",
        head_m=0.586,
        expected_zero_u_prgh_pa=WATER_INLET_ZERO_U_PRGH_PA,
    ),
    ProfileSpec(
        name="waterOutlet",
        density_kg_m3=RHO_WATER_KG_M3,
        equation_id="water_outlet_table1",
        equation_gauge="rho_w*g*(0.584-z)",
        head_m=0.584,
        expected_zero_u_prgh_pa=WATER_OUTLET_ZERO_U_PRGH_PA,
    ),
    ProfileSpec(
        name="ambientFloor",
        density_kg_m3=RHO_AIR_KG_M3,
        equation_id="ambient_floor_table1",
        equation_gauge="0",
        head_m=None,
        expected_zero_u_prgh_pa=AMBIENT_ZERO_U_PRGH_PA,
    ),
    ProfileSpec(
        name="ambientSides",
        density_kg_m3=RHO_AIR_KG_M3,
        equation_id="ambient_sides_table1",
        equation_gauge="rho_air*g*(1.02-z)",
        head_m=AMBIENT_RIM_Z_M,
        expected_zero_u_prgh_pa=AMBIENT_ZERO_U_PRGH_PA,
    ),
    ProfileSpec(
        name="ambientTop",
        density_kg_m3=RHO_AIR_KG_M3,
        equation_id="ambient_top_table1",
        equation_gauge="rho_air*g*(1.02-z)",
        head_m=AMBIENT_RIM_Z_M,
        expected_zero_u_prgh_pa=AMBIENT_ZERO_U_PRGH_PA,
    ),
)
PROFILE_BY_NAME: Mapping[str, ProfileSpec] = {
    profile.name: profile for profile in PROFILE_SPECS
}
TARGET_PATCH_NAMES = tuple(profile.name for profile in PROFILE_SPECS)
PRESERVED_STAGE1_PATCH = "airInlet"

_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_FLOAT_RE = re.compile(_FLOAT_PATTERN)
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:+-]*")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_utf8(path: Path) -> tuple[bytes, str]:
    if not path.exists() or not path.is_file() or path.is_symlink():
        raise ConversionError(f"required regular, non-symlink file is missing: {path}")
    data = path.read_bytes()
    if b"\x00" in data:
        raise ConversionError(f"NUL byte/binary content is not accepted: {path}")
    try:
        return data, data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConversionError(f"file is not UTF-8 ASCII-compatible text: {path}") from exc


def _mask_comments_and_strings(text: str, source: Path | str) -> str:
    """Mask comments and quoted strings while preserving character offsets."""

    chars = list(text)
    n = len(text)
    i = 0
    while i < n:
        if text.startswith("//", i):
            end = text.find("\n", i + 2)
            if end < 0:
                end = n
            for j in range(i, end):
                chars[j] = " "
            i = end
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise ConversionError(f"unterminated block comment in {source}")
            end += 2
            for j in range(i, end):
                if chars[j] not in "\r\n":
                    chars[j] = " "
            i = end
            continue
        if text[i] == '"':
            j = i + 1
            escaped = False
            while j < n:
                ch = text[j]
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    break
                j += 1
            if j >= n:
                raise ConversionError(f"unterminated quoted string in {source}")
            for k in range(i, j + 1):
                if chars[k] not in "\r\n":
                    chars[k] = " "
            i = j + 1
            continue
        i += 1
    return "".join(chars)


def _match_balanced(
    clean: str, start: int, opening: str, closing: str, source: Path | str
) -> int:
    if start >= len(clean) or clean[start] != opening:
        raise ConversionError(f"expected {opening!r} in {source} at offset {start}")
    depth = 0
    for pos in range(start, len(clean)):
        token = clean[pos]
        if token == opening:
            depth += 1
        elif token == closing:
            depth -= 1
            if depth == 0:
                return pos
            if depth < 0:
                break
    raise ConversionError(f"unbalanced {opening}{closing} delimiters in {source}")


def _skip_space(clean: str, pos: int, end: int | None = None) -> int:
    limit = len(clean) if end is None else end
    while pos < limit and clean[pos].isspace():
        pos += 1
    return pos


def _foam_header_end(
    text: str,
    clean: str,
    path: Path,
    *,
    expected_object: str,
    expected_class: str,
) -> int:
    hits = list(re.finditer(r"\bFoamFile\b", clean))
    if len(hits) != 1:
        raise ConversionError(f"expected exactly one FoamFile header in {path}")
    open_pos = clean.find("{", hits[0].end())
    if open_pos < 0:
        raise ConversionError(f"FoamFile dictionary has no opening brace in {path}")
    close_pos = _match_balanced(clean, open_pos, "{", "}", path)
    body = clean[open_pos + 1 : close_pos]

    def one_entry(key: str) -> str:
        matches = re.findall(rf"\b{re.escape(key)}\s+([^;\s]+)\s*;", body)
        if len(matches) != 1:
            raise ConversionError(f"FoamFile {key!r} is missing or duplicated in {path}")
        return matches[0]

    file_format = one_entry("format")
    if file_format != "ascii":
        raise ConversionError(f"only OpenFOAM ASCII files are accepted; {path} is {file_format!r}")
    if one_entry("object") != expected_object:
        raise ConversionError(f"unexpected FoamFile object in {path}; expected {expected_object!r}")
    if one_entry("class") != expected_class:
        raise ConversionError(f"unexpected FoamFile class in {path}; expected {expected_class!r}")
    return close_pos + 1


def _counted_list(
    clean: str, after: int, path: Path
) -> tuple[int, str, int, int]:
    pos = _skip_space(clean, after)
    while pos < len(clean) and clean[pos] == ";":
        pos = _skip_space(clean, pos + 1)
    count_match = re.match(r"\d+", clean[pos:])
    if not count_match:
        raise ConversionError(f"counted list is missing its declared size in {path}")
    count = int(count_match.group(0))
    pos += count_match.end()
    pos = _skip_space(clean, pos)
    if pos >= len(clean) or clean[pos] != "(":
        raise ConversionError(f"counted list has no opening parenthesis in {path}")
    close_pos = _match_balanced(clean, pos, "(", ")", path)
    trailing = clean[close_pos + 1 :]
    if trailing.replace(";", "").strip():
        raise ConversionError(f"unexpected tokens after counted list in {path}")
    return count, clean[pos + 1 : close_pos], pos, close_pos


def _number_paren_tokens(body: str, path: Path) -> list[str]:
    tokens: list[str] = []
    pos = 0
    while pos < len(body):
        if body[pos].isspace():
            pos += 1
            continue
        if body[pos] in "()":
            tokens.append(body[pos])
            pos += 1
            continue
        match = _FLOAT_RE.match(body, pos)
        if not match:
            excerpt = body[pos : pos + 32].replace("\n", "\\n")
            raise ConversionError(f"invalid numeric-list token in {path}: {excerpt!r}")
        tokens.append(match.group(0))
        pos = match.end()
    return tokens


def _parse_points(path: Path) -> tuple[list[tuple[float, float, float]], bytes]:
    raw, text = _read_utf8(path)
    clean = _mask_comments_and_strings(text, path)
    header_end = _foam_header_end(
        text, clean, path, expected_object="points", expected_class="vectorField"
    )
    declared, body, _, _ = _counted_list(clean, header_end, path)
    tokens = _number_paren_tokens(body, path)
    points: list[tuple[float, float, float]] = []
    pos = 0
    for _ in range(declared):
        if pos >= len(tokens) or tokens[pos] != "(":
            raise ConversionError(f"point list has fewer than {declared} entries in {path}")
        pos += 1
        if pos + 3 >= len(tokens):
            raise ConversionError(f"truncated point vector in {path}")
        values = tokens[pos : pos + 3]
        if any(value in "()" for value in values) or tokens[pos + 3] != ")":
            raise ConversionError(f"each point must contain exactly three scalars in {path}")
        vector = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in vector):
            raise ConversionError(f"non-finite point coordinate in {path}")
        points.append((vector[0], vector[1], vector[2]))
        pos += 4
    if pos != len(tokens):
        raise ConversionError(f"point list contains more than its declared {declared} entries in {path}")
    return points, raw


def _parse_faces(path: Path, n_points: int) -> tuple[list[tuple[int, ...]], bytes]:
    raw, text = _read_utf8(path)
    clean = _mask_comments_and_strings(text, path)
    header_end = _foam_header_end(
        text, clean, path, expected_object="faces", expected_class="faceList"
    )
    declared, body, _, _ = _counted_list(clean, header_end, path)
    tokens = _number_paren_tokens(body, path)
    faces: list[tuple[int, ...]] = []
    pos = 0
    for _ in range(declared):
        if pos >= len(tokens) or tokens[pos] in "()":
            raise ConversionError(f"face list has fewer than {declared} entries in {path}")
        n_vertices_float = float(tokens[pos])
        n_vertices = int(n_vertices_float)
        if n_vertices_float != n_vertices or n_vertices < 3:
            raise ConversionError(f"invalid face vertex count {tokens[pos]!r} in {path}")
        pos += 1
        if pos >= len(tokens) or tokens[pos] != "(":
            raise ConversionError(f"face vertex list has no opening parenthesis in {path}")
        pos += 1
        if pos + n_vertices >= len(tokens):
            raise ConversionError(f"truncated face vertex list in {path}")
        indices: list[int] = []
        for token in tokens[pos : pos + n_vertices]:
            if token in "()":
                raise ConversionError(f"face has fewer vertices than declared in {path}")
            value_float = float(token)
            value = int(value_float)
            if value_float != value:
                raise ConversionError(f"non-integer point index {token!r} in {path}")
            if value < 0 or value >= n_points:
                raise ConversionError(
                    f"face point index {value} is outside [0,{n_points}) in {path}"
                )
            indices.append(value)
        pos += n_vertices
        if tokens[pos] != ")":
            raise ConversionError(f"face has more vertices than declared in {path}")
        pos += 1
        faces.append(tuple(indices))
    if pos != len(tokens):
        raise ConversionError(f"face list contains more than its declared {declared} entries in {path}")
    return faces, raw


def _parse_boundary(path: Path, n_faces: int) -> tuple[dict[str, BoundaryPatch], bytes]:
    raw, text = _read_utf8(path)
    clean = _mask_comments_and_strings(text, path)
    header_end = _foam_header_end(
        text,
        clean,
        path,
        expected_object="boundary",
        expected_class="polyBoundaryMesh",
    )
    declared, body, _, _ = _counted_list(clean, header_end, path)
    patches: dict[str, BoundaryPatch] = {}
    pos = 0
    for _ in range(declared):
        pos = _skip_space(body, pos)
        name_match = _WORD_RE.match(body, pos)
        if not name_match:
            raise ConversionError(f"boundary list has fewer than {declared} patch entries in {path}")
        name = name_match.group(0)
        if name in patches:
            raise ConversionError(f"duplicate mesh patch {name!r} in {path}")
        pos = _skip_space(body, name_match.end())
        if pos >= len(body) or body[pos] != "{":
            raise ConversionError(f"mesh patch {name!r} has no dictionary in {path}")
        close_pos = _match_balanced(body, pos, "{", "}", path)
        patch_body = body[pos + 1 : close_pos]

        def integer_entry(key: str) -> int:
            matches = re.findall(rf"\b{re.escape(key)}\s+(\d+)\s*;", patch_body)
            if len(matches) != 1:
                raise ConversionError(
                    f"mesh patch {name!r} {key!r} is missing or duplicated in {path}"
                )
            return int(matches[0])

        n_patch_faces = integer_entry("nFaces")
        start_face = integer_entry("startFace")
        if start_face < 0 or n_patch_faces < 0 or start_face + n_patch_faces > n_faces:
            raise ConversionError(
                f"mesh patch {name!r} face range [{start_face},"
                f"{start_face + n_patch_faces}) is outside [0,{n_faces})"
            )
        patches[name] = BoundaryPatch(name, n_patch_faces, start_face)
        pos = close_pos + 1
    if body[pos:].strip():
        raise ConversionError(f"boundary list contains more than its declared {declared} patches in {path}")

    ranges = sorted(
        (patch.start_face, patch.start_face + patch.n_faces, patch.name)
        for patch in patches.values()
        if patch.n_faces
    )
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise ConversionError(
                f"mesh patch face ranges overlap: {previous[2]!r} and {current[2]!r}"
            )
    return patches, raw


def _find_field_patch_blocks(
    text: str, path: Path
) -> tuple[dict[str, FieldPatchBlock], str]:
    clean = _mask_comments_and_strings(text, path)
    _foam_header_end(
        text, clean, path, expected_object="p_rgh", expected_class="volScalarField"
    )
    hits = list(re.finditer(r"\bboundaryField\b", clean))
    if len(hits) != 1:
        raise ConversionError(f"expected exactly one boundaryField dictionary in {path}")
    open_pos = _skip_space(clean, hits[0].end())
    if open_pos >= len(clean) or clean[open_pos] != "{":
        raise ConversionError(f"boundaryField has no opening brace in {path}")
    boundary_close = _match_balanced(clean, open_pos, "{", "}", path)

    patches: dict[str, FieldPatchBlock] = {}
    pos = open_pos + 1
    while True:
        pos = _skip_space(clean, pos, boundary_close)
        if pos >= boundary_close:
            break
        name_match = _WORD_RE.match(clean, pos)
        if not name_match:
            raise ConversionError(f"unexpected token in boundaryField at offset {pos} in {path}")
        name = name_match.group(0)
        if name in patches:
            raise ConversionError(f"duplicate field patch {name!r} in {path}")
        brace_start = _skip_space(clean, name_match.end(), boundary_close)
        if brace_start >= boundary_close or clean[brace_start] != "{":
            raise ConversionError(f"field patch {name!r} has no dictionary in {path}")
        brace_end = _match_balanced(clean, brace_start, "{", "}", path)
        if brace_end > boundary_close:
            raise ConversionError(f"field patch {name!r} escapes boundaryField in {path}")
        patches[name] = FieldPatchBlock(name, name_match.start(), brace_start, brace_end)
        pos = brace_end + 1
    return patches, clean


def _single_type(patch_body: str, source: Path | str) -> str:
    clean = _mask_comments_and_strings(patch_body, source)
    matches = re.findall(r"\btype\s+([^;\s]+)\s*;", clean)
    if len(matches) != 1:
        raise ConversionError(f"patch type is missing or duplicated in {source}")
    return matches[0]


def _nonuniform_scalar_entry(
    patch_body: str,
    entry: str,
    source: Path | str,
    *,
    required: bool,
) -> list[float] | None:
    clean = _mask_comments_and_strings(patch_body, source)
    entry_hits = list(re.finditer(rf"\b{re.escape(entry)}\b", clean))
    if not entry_hits:
        if required:
            raise ConversionError(f"entry {entry!r} is missing in {source}")
        return None
    if len(entry_hits) != 1:
        raise ConversionError(f"entry {entry!r} is duplicated in {source}")
    pos = _skip_space(clean, entry_hits[0].end())
    prefix = re.match(r"nonuniform\s+List<scalar>", clean[pos:])
    if not prefix:
        uniform = re.match(rf"uniform\s+({_FLOAT_PATTERN})\s*;", clean[pos:])
        if uniform:
            return [float(uniform.group(1))]
        raise ConversionError(f"entry {entry!r} has an unsupported value form in {source}")
    pos += prefix.end()
    pos = _skip_space(clean, pos)
    count_match = re.match(r"\d+", clean[pos:])
    if not count_match:
        raise ConversionError(f"entry {entry!r} has no list count in {source}")
    declared = int(count_match.group(0))
    pos += count_match.end()
    pos = _skip_space(clean, pos)
    if pos >= len(clean) or clean[pos] != "(":
        raise ConversionError(f"entry {entry!r} has no opening parenthesis in {source}")
    close_pos = _match_balanced(clean, pos, "(", ")", source)
    tokens = _number_paren_tokens(clean[pos + 1 : close_pos], Path(str(source)))
    if any(token in "()" for token in tokens):
        raise ConversionError(f"entry {entry!r} contains nested parentheses in {source}")
    values = [float(token) for token in tokens]
    if len(values) != declared:
        raise ConversionError(
            f"entry {entry!r} declares {declared} values but contains {len(values)} in {source}"
        )
    if not all(math.isfinite(value) for value in values):
        raise ConversionError(f"entry {entry!r} contains a non-finite value in {source}")
    tail = _skip_space(clean, close_pos + 1)
    if tail >= len(clean) or clean[tail] != ";":
        raise ConversionError(f"entry {entry!r} list has no terminating semicolon in {source}")
    return values


def _face_centres(
    points: Sequence[tuple[float, float, float]],
    faces: Sequence[tuple[int, ...]],
    patch: BoundaryPatch,
) -> list[tuple[float, float, float]]:
    centres: list[tuple[float, float, float]] = []
    for face_i in range(patch.start_face, patch.start_face + patch.n_faces):
        face = faces[face_i]
        scale = 1.0 / len(face)
        centres.append(
            (
                sum(points[point_i][0] for point_i in face) * scale,
                sum(points[point_i][1] for point_i in face) * scale,
                sum(points[point_i][2] for point_i in face) * scale,
            )
        )
    return centres


def _fmt_scalar(value: float) -> str:
    if not math.isfinite(value):
        raise ConversionError("refusing to render a non-finite scalar")
    return format(value, ".17g")


def _render_scalar_list(entry: str, values: Sequence[float], indent: str, nl: str) -> str:
    lines = [
        f"{indent}{entry:<17} nonuniform List<scalar>",
        f"{indent}{len(values)}",
        f"{indent}(",
    ]
    lines.extend(f"{indent}    {_fmt_scalar(value)}" for value in values)
    lines.extend((f"{indent})", f"{indent};"))
    return nl.join(lines)


def _render_patch_dictionary(
    p0_values: Sequence[float],
    zero_u_prgh_values: Sequence[float],
    base_indent: str,
    nl: str,
) -> str:
    inner = base_indent + "    "
    chunks = [
        "{",
        f"{inner}{'type':<17} prghTotalPressure;",
        _render_scalar_list("p0", p0_values, inner, nl),
        _render_scalar_list("value", zero_u_prgh_values, inner, nl),
        f"{base_indent}}}",
    ]
    return nl.join(chunks)


def _patch_body(text: str, block: FieldPatchBlock) -> str:
    return text[block.brace_start + 1 : block.brace_end]


def _expanded(values: Sequence[float], n_faces: int, source: Path | str) -> list[float]:
    if len(values) == 1:
        return [values[0]] * n_faces
    if len(values) != n_faces:
        raise ConversionError(
            f"scalar list contains {len(values)} values for a {n_faces}-face patch in {source}"
        )
    return list(values)


def _max_abs_error(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if len(lhs) != len(rhs):
        raise ConversionError(f"cannot compare lists with lengths {len(lhs)} and {len(rhs)}")
    return max((abs(a - b) for a, b in zip(lhs, rhs)), default=0.0)


def _atomic_write_if_changed(path: Path, data: bytes, *, source_mode: int | None = None) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise ConversionError(f"atomic-write target is not a regular file: {path}")
        if path.read_bytes() == data:
            return
        mode = stat.S_IMODE(path.stat().st_mode)
    else:
        mode = 0o644 if source_mode is None else source_mode
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _inside(root: Path, candidate: Path, label: str) -> Path:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConversionError(f"{label} must remain inside case directory {root}") from exc
    return resolved


def apply_total_pressure_profiles(
    case_dir: str | os.PathLike[str],
    *,
    field: str | os.PathLike[str] = "0/p_rgh",
    audit: str | os.PathLike[str] = "total_pressure_profile_audit.json",
) -> dict[str, object]:
    """Validate a case, atomically rewrite ``p_rgh``, and emit its audit.

    All five Table-1 water/ambient patches are always converted together.  The
    Stage-1 ``airInlet`` patch is required and byte-preserved.
    """

    root = Path(case_dir).resolve()
    if not root.exists() or not root.is_dir() or root.is_symlink():
        raise ConversionError(f"case directory is missing, not a directory, or a symlink: {root}")
    field_path = _inside(root, root / Path(field), "field path")
    audit_path = _inside(root, root / Path(audit), "audit path")
    mesh_dir = root / "constant" / "polyMesh"
    protected_inputs = {
        field_path,
        (mesh_dir / "points").resolve(),
        (mesh_dir / "faces").resolve(),
        (mesh_dir / "boundary").resolve(),
    }
    if audit_path in protected_inputs:
        raise ConversionError("audit path must not overwrite p_rgh or an input mesh file")

    points, points_raw = _parse_points(mesh_dir / "points")
    faces, faces_raw = _parse_faces(mesh_dir / "faces", len(points))
    mesh_patches, boundary_raw = _parse_boundary(mesh_dir / "boundary", len(faces))
    for required in (*TARGET_PATCH_NAMES, PRESERVED_STAGE1_PATCH):
        if required not in mesh_patches:
            raise ConversionError(f"required mesh patch {required!r} is missing")
        if mesh_patches[required].n_faces <= 0:
            raise ConversionError(f"required mesh patch {required!r} has no faces")

    field_raw, field_text = _read_utf8(field_path)
    field_patches, _ = _find_field_patch_blocks(field_text, field_path)
    for required in (*TARGET_PATCH_NAMES, PRESERVED_STAGE1_PATCH):
        if required not in field_patches:
            raise ConversionError(f"required p_rgh patch {required!r} is missing in {field_path}")

    preserved_before = field_text[
        field_patches[PRESERVED_STAGE1_PATCH].brace_start :
        field_patches[PRESERVED_STAGE1_PATCH].brace_end + 1
    ]
    if _single_type(
        _patch_body(field_text, field_patches[PRESERVED_STAGE1_PATCH]),
        f"{field_path}:{PRESERVED_STAGE1_PATCH}",
    ) != "fixedFluxPressure":
        raise ConversionError(
            "Stage-1 airInlet must remain fixedFluxPressure in p_rgh; "
            "the converter will not reinterpret this patch"
        )
    preserved_value = _nonuniform_scalar_entry(
        _patch_body(field_text, field_patches[PRESERVED_STAGE1_PATCH]),
        "value",
        f"{field_path}:{PRESERVED_STAGE1_PATCH}",
        required=False,
    )
    if preserved_value is not None:
        _expanded(
            preserved_value,
            mesh_patches[PRESERVED_STAGE1_PATCH].n_faces,
            f"{field_path}:{PRESERVED_STAGE1_PATCH}:value",
        )

    profiles: dict[str, dict[str, object]] = {}
    calculated: dict[str, tuple[list[float], list[float]]] = {}
    for spec in PROFILE_SPECS:
        mesh_patch = mesh_patches[spec.name]
        field_block = field_patches[spec.name]
        original_body = _patch_body(field_text, field_block)
        original_type = _single_type(original_body, f"{field_path}:{spec.name}")
        if original_type not in {"fixedValue", "prghTotalPressure"}:
            raise ConversionError(
                f"refusing to replace unexpected {spec.name!r} p_rgh type {original_type!r}"
            )
        if original_type == "fixedValue":
            _expanded(
                _nonuniform_scalar_entry(
                    original_body,
                    "value",
                    f"{field_path}:{spec.name}",
                    required=True,
                )
                or [],
                mesh_patch.n_faces,
                f"{field_path}:{spec.name}:value",
            )

        centres = _face_centres(points, faces, mesh_patch)
        z_values = [centre[2] for centre in centres]
        if spec.name == "ambientFloor" and any(
            abs(z_m - AMBIENT_RIM_Z_M) > GEOMETRY_TOLERANCE_M for z_m in z_values
        ):
            raise ConversionError(
                "ambientFloor face centres are not at the frozen riser-rim elevation z=1.02 m"
            )
        p0_gauge = [spec.p0_gauge_pa(z_m) for z_m in z_values]
        p0_abs = [PATM_PA + value for value in p0_gauge]
        # v2512: p_rgh = p0 - rho*((g & Cf)-ghRef) at U=0.  For
        # g=(0,0,-9.81) and hRef=0 this becomes p0 + rho*g*z.
        zero_u_prgh = [
            p0 + spec.density_kg_m3 * G_M_S2 * z_m
            for p0, z_m in zip(p0_abs, z_values)
        ]
        constant_error = max(
            abs(value - spec.expected_zero_u_prgh_pa) for value in zero_u_prgh
        )
        if constant_error > FORMULA_TOLERANCE_PA:
            raise ConversionError(
                f"{spec.name} zero-U reduced-pressure formula error "
                f"{constant_error:.12g} Pa exceeds {FORMULA_TOLERANCE_PA:.12g} Pa"
            )

        if original_type == "prghTotalPressure":
            existing_p0 = _expanded(
                _nonuniform_scalar_entry(
                    original_body,
                    "p0",
                    f"{field_path}:{spec.name}",
                    required=True,
                )
                or [],
                mesh_patch.n_faces,
                f"{field_path}:{spec.name}:p0",
            )
            if _max_abs_error(existing_p0, p0_abs) > FORMULA_TOLERANCE_PA:
                raise ConversionError(
                    f"existing prghTotalPressure profile for {spec.name!r} does not match Table 1"
                )
            existing_value = _nonuniform_scalar_entry(
                original_body,
                "value",
                f"{field_path}:{spec.name}",
                required=False,
            )
            if existing_value is not None and _max_abs_error(
                _expanded(
                    existing_value,
                    mesh_patch.n_faces,
                    f"{field_path}:{spec.name}:value",
                ),
                zero_u_prgh,
            ) > FORMULA_TOLERANCE_PA:
                raise ConversionError(
                    f"existing zero-U p_rgh values for {spec.name!r} are inconsistent"
                )

        calculated[spec.name] = (p0_abs, zero_u_prgh)
        profiles[spec.name] = {
            "equation_id": spec.equation_id,
            "published_gauge_total_pressure_formula_pa": spec.equation_gauge,
            "openfoam_p0_absolute_formula_pa": f"Patm+({spec.equation_gauge})",
            "density_kg_m3": spec.density_kg_m3,
            "n_faces": mesh_patch.n_faces,
            "start_face": mesh_patch.start_face,
            "face_center_z_m": {"min": min(z_values), "max": max(z_values)},
            "p0_gauge_pa": {"min": min(p0_gauge), "max": max(p0_gauge)},
            "p0_absolute_pa": {"min": min(p0_abs), "max": max(p0_abs)},
            "zero_u_p_rgh_pa": {"min": min(zero_u_prgh), "max": max(zero_u_prgh)},
            "expected_zero_u_p_rgh_pa": spec.expected_zero_u_prgh_pa,
            "zero_u_consistency_max_abs_error_pa": constant_error,
        }

    nl = "\r\n" if "\r\n" in field_text else "\n"
    replacements: list[tuple[int, int, str]] = []
    for spec in PROFILE_SPECS:
        block = field_patches[spec.name]
        line_start = field_text.rfind("\n", 0, block.name_start) + 1
        base_indent = field_text[line_start : block.name_start]
        if base_indent.strip():
            raise ConversionError(f"unexpected text before patch name {spec.name!r} in {field_path}")
        p0_abs, zero_u_prgh = calculated[spec.name]
        rendered = _render_patch_dictionary(p0_abs, zero_u_prgh, base_indent, nl)
        replacements.append((block.brace_start, block.brace_end + 1, rendered))

    candidate = field_text
    for start, end, rendered in sorted(replacements, reverse=True):
        candidate = candidate[:start] + rendered + candidate[end:]

    # Reparse the rendered dictionary and verify declared counts, values, and
    # formulas before a single byte is replaced on disk.
    candidate_patches, _ = _find_field_patch_blocks(candidate, field_path)
    for spec in PROFILE_SPECS:
        body = _patch_body(candidate, candidate_patches[spec.name])
        if _single_type(body, f"candidate:{spec.name}") != "prghTotalPressure":
            raise ConversionError(f"rendered type verification failed for {spec.name}")
        p0_written = _expanded(
            _nonuniform_scalar_entry(body, "p0", f"candidate:{spec.name}", required=True)
            or [],
            mesh_patches[spec.name].n_faces,
            f"candidate:{spec.name}:p0",
        )
        value_written = _expanded(
            _nonuniform_scalar_entry(body, "value", f"candidate:{spec.name}", required=True)
            or [],
            mesh_patches[spec.name].n_faces,
            f"candidate:{spec.name}:value",
        )
        expected_p0, expected_value = calculated[spec.name]
        p0_error = _max_abs_error(p0_written, expected_p0)
        value_error = _max_abs_error(value_written, expected_value)
        if max(p0_error, value_error) > FORMULA_TOLERANCE_PA:
            raise ConversionError(
                f"rendered formula error for {spec.name} exceeds {FORMULA_TOLERANCE_PA} Pa"
            )
        profiles[spec.name]["rendered_p0_max_abs_error_pa"] = p0_error
        profiles[spec.name]["rendered_zero_u_p_rgh_max_abs_error_pa"] = value_error

    preserved_after_block = candidate_patches[PRESERVED_STAGE1_PATCH]
    preserved_after = candidate[
        preserved_after_block.brace_start : preserved_after_block.brace_end + 1
    ]
    if preserved_after != preserved_before:
        raise ConversionError("Stage-1 airInlet p_rgh dictionary changed during conversion")

    candidate_bytes = candidate.encode("utf-8")
    audit_document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "case_dir": str(root),
        "field": str(field_path.relative_to(root)).replace("\\", "/"),
        "openfoam_boundary_condition": {
            "version": "OpenFOAM v2512",
            "type": "prghTotalPressure",
            "zero_velocity_equation": "p_rgh=p0-rho*((g.Cf)-ghRef)",
            "inflow_dynamic_term": "-0.5*rho*neg(phi)*magSqr(U)",
            "gravity_vector_m_s2": [0.0, 0.0, -G_M_S2],
            "hRef_m": HREF_M,
            "p0_semantics": "absolute total pressure; nonuniform List<scalar> in patch-face order",
        },
        "evidence": {
            "pressure_equations": "Mahyawansi et al. (2024), Table 1 (published)",
            "boundary_type": "paper method: inlet/outlet are total-pressure boundaries (published)",
            "openfoam_translation": "OpenFOAM v2512 prghTotalPressure updateCoeffs (solver translation)",
            "density_source": (
                "frozen 2-D thermophysicalProperties/setExprFields translation; "
                "not rounded literature values"
            ),
        },
        "constants": {
            "Patm_pa": PATM_PA,
            "g_m_s2": G_M_S2,
            "rho_water_kg_m3": RHO_WATER_KG_M3,
            "rho_air_kg_m3": RHO_AIR_KG_M3,
            "ambient_rim_z_m": AMBIENT_RIM_Z_M,
            "formula_tolerance_pa": FORMULA_TOLERANCE_PA,
        },
        "mesh": {
            "n_points": len(points),
            "n_faces": len(faces),
            "files_sha256": {
                "constant/polyMesh/points": _sha256(points_raw),
                "constant/polyMesh/faces": _sha256(faces_raw),
                "constant/polyMesh/boundary": _sha256(boundary_raw),
            },
        },
        "profiles": profiles,
        "stage1_air_inlet": {
            "patch": PRESERVED_STAGE1_PATCH,
            "p_rgh_type": "fixedFluxPressure",
            "policy": "required and byte-preserved; not converted before Stage 2",
            "dictionary_sha256": _sha256(preserved_before.encode("utf-8")),
        },
        "field_sha256": _sha256(candidate_bytes),
        "write_policy": "validate-all-then-atomic-os-replace; deterministic and idempotent",
    }
    audit_bytes = (
        json.dumps(audit_document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    # All parsing and formula gates have passed.  Each artifact is now replaced
    # atomically, and unchanged reruns perform no replacement at all.
    field_mode = stat.S_IMODE(field_path.stat().st_mode)
    _atomic_write_if_changed(field_path, candidate_bytes, source_mode=field_mode)
    _atomic_write_if_changed(audit_path, audit_bytes)
    return audit_document


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply source-aligned Mahyawansi Table-1 total-pressure profiles "
            "to an ASCII OpenFOAM-v2512 p_rgh field."
        )
    )
    parser.add_argument("--case", required=True, help="OpenFOAM case directory")
    parser.add_argument(
        "--field", default="0/p_rgh", help="p_rgh path relative to --case (default: 0/p_rgh)"
    )
    parser.add_argument(
        "--audit",
        default="total_pressure_profile_audit.json",
        help="JSON audit path relative to --case",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        audit = apply_total_pressure_profiles(args.case, field=args.field, audit=args.audit)
    except (ConversionError, OSError) as exc:
        raise SystemExit(f"TOTAL_PRESSURE_PROFILE_FAILED: {exc}") from exc
    print(
        json.dumps(
            {
                "status": audit["status"],
                "field": audit["field"],
                "field_sha256": audit["field_sha256"],
                "audit": args.audit,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
PCA-based ellipsoid fit + volume estimate from a PLY point cloud.

This script is intentionally lightweight (no Open3D / sklearn). It supports:
  - PLY binary_little_endian (common for Open3D exports)
  - PLY ascii (basic support)

Method:
  1) Read XYZ points
  2) PCA via covariance eigen-decomposition
  3) Project points into PCA frame
  4) Semi-axes (a,b,c) := 0.5 * (max - min) along each PCA axis
  5) Volume V = 4/3*pi*a*b*c

Notes:
  - This is not a least-squares ellipsoid fit. It's an "ellipsoid around the PCA OBB".
  - Units are whatever the PLY uses (your kmelon point clouds appear to be meters).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


try:
    import numpy as np
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: numpy.\n"
        "Install with: python3 -m pip install --user numpy\n"
        f"Original error: {type(e).__name__}: {e}"
    )


_PLY_SCALAR_DTYPES: Dict[str, np.dtype] = {
    "char": np.dtype("i1"),
    "int8": np.dtype("i1"),
    "uchar": np.dtype("u1"),
    "uint8": np.dtype("u1"),
    "short": np.dtype("<i2"),
    "int16": np.dtype("<i2"),
    "ushort": np.dtype("<u2"),
    "uint16": np.dtype("<u2"),
    "int": np.dtype("<i4"),
    "int32": np.dtype("<i4"),
    "uint": np.dtype("<u4"),
    "uint32": np.dtype("<u4"),
    "float": np.dtype("<f4"),
    "float32": np.dtype("<f4"),
    "double": np.dtype("<f8"),
    "float64": np.dtype("<f8"),
}


@dataclass
class PlyHeader:
    fmt: str  # "ascii" or "binary_little_endian"
    vertex_count: int
    vertex_properties: List[Tuple[str, str]]  # (name, ply_type)
    header_nbytes: int


def _read_ply_header(path: str) -> PlyHeader:
    """
    Parse a PLY header. Supports vertex properties only (enough for point clouds).
    Returns header byte length so binary payload can be sliced reliably.
    """
    with open(path, "rb") as f:
        data = f.read(1024 * 1024)  # up to 1MB header guard (way more than needed)

    if not data.startswith(b"ply\n"):
        raise ValueError(f"Not a PLY file (missing 'ply' magic): {path}")

    end_idx = data.find(b"end_header\n")
    if end_idx < 0:
        # Sometimes end_header has \r\n
        end_idx = data.find(b"end_header\r\n")
        if end_idx < 0:
            raise ValueError(f"PLY header missing end_header: {path}")
        end_token_len = len(b"end_header\r\n")
    else:
        end_token_len = len(b"end_header\n")

    header_bytes = data[: end_idx + end_token_len]
    header_text = header_bytes.decode("ascii", errors="replace")
    lines = [ln.strip() for ln in header_text.splitlines() if ln.strip()]

    fmt: Optional[str] = None
    vertex_count: Optional[int] = None
    in_vertex = False
    vertex_properties: List[Tuple[str, str]] = []

    for ln in lines:
        if ln.startswith("format "):
            parts = ln.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed format line: {ln}")
            fmt = parts[1]
        elif ln.startswith("element "):
            parts = ln.split()
            if len(parts) != 3:
                continue
            elem = parts[1]
            count = int(parts[2])
            in_vertex = elem == "vertex"
            if in_vertex:
                vertex_count = count
        elif ln.startswith("property ") and in_vertex:
            parts = ln.split()
            # We only support scalar properties: "property <type> <name>"
            if len(parts) != 3:
                # list properties are not supported here
                continue
            ply_type, name = parts[1], parts[2]
            vertex_properties.append((name, ply_type))

    if fmt is None:
        raise ValueError(f"PLY header missing format: {path}")
    if fmt not in ("ascii", "binary_little_endian"):
        raise ValueError(f"Unsupported PLY format '{fmt}' in {path}")
    if vertex_count is None:
        raise ValueError(f"PLY header missing vertex element count: {path}")
    if not vertex_properties:
        raise ValueError(f"PLY header has no vertex properties: {path}")

    return PlyHeader(
        fmt=fmt,
        vertex_count=vertex_count,
        vertex_properties=vertex_properties,
        header_nbytes=len(header_bytes),
    )


def _load_ply_xyz(path: str) -> np.ndarray:
    """
    Load Nx3 float64 XYZ from PLY (binary_little_endian or ascii).
    """
    hdr = _read_ply_header(path)
    names = [n for n, _t in hdr.vertex_properties]
    if not all(k in names for k in ("x", "y", "z")):
        raise ValueError(
            f"PLY vertex properties must include x,y,z. Found: {names}"
        )

    if hdr.fmt == "ascii":
        # Basic ASCII loader: read lines after end_header, parse according to property order.
        # This assumes each vertex is on one line.
        xyz_idx = [names.index("x"), names.index("y"), names.index("z")]
        pts = np.zeros((hdr.vertex_count, 3), dtype=np.float64)
        with open(path, "rb") as f:
            f.seek(hdr.header_nbytes)
            for i in range(hdr.vertex_count):
                line = f.readline()
                if not line:
                    raise ValueError(f"Unexpected EOF while reading vertices: {path}")
                parts = line.decode("ascii", errors="replace").strip().split()
                if len(parts) < len(names):
                    raise ValueError(f"Malformed vertex line {i}: {parts}")
                pts[i, 0] = float(parts[xyz_idx[0]])
                pts[i, 1] = float(parts[xyz_idx[1]])
                pts[i, 2] = float(parts[xyz_idx[2]])
        return pts

    # binary_little_endian: build a structured dtype for the vertex record
    dtype_fields = []
    for name, ply_type in hdr.vertex_properties:
        if ply_type not in _PLY_SCALAR_DTYPES:
            raise ValueError(f"Unsupported PLY scalar type '{ply_type}' for '{name}'")
        dt = _PLY_SCALAR_DTYPES[ply_type]
        # ensure little-endian for multi-byte scalars
        if dt.itemsize > 1:
            dt = dt.newbyteorder("<")
        dtype_fields.append((name, dt))
    vtx_dtype = np.dtype(dtype_fields, align=False)

    with open(path, "rb") as f:
        f.seek(hdr.header_nbytes)
        raw = f.read(hdr.vertex_count * vtx_dtype.itemsize)
        if len(raw) < hdr.vertex_count * vtx_dtype.itemsize:
            raise ValueError(
                f"Unexpected EOF while reading binary vertices. "
                f"Expected {hdr.vertex_count * vtx_dtype.itemsize} bytes, got {len(raw)}."
            )
    verts = np.frombuffer(raw, dtype=vtx_dtype, count=hdr.vertex_count)
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64, copy=False)
    return xyz


def _write_ascii_ply_mesh(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """
    Write a triangle mesh as ASCII PLY with vertex xyz and face indices.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (M,3), got {faces.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Generated by corepp/size_evaluation/fit_pca_ellipsoid.py\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property double x\n")
        f.write("property double y\n")
        f.write("property double z\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def _uv_sphere_mesh(n_lon: int = 64, n_lat: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a unit UV sphere triangle mesh.

    Returns:
      vertices: (N,3) float64 on unit sphere
      faces:    (M,3) int64
    """
    n_lon = int(n_lon)
    n_lat = int(n_lat)
    if n_lon < 3 or n_lat < 2:
        raise ValueError("Need n_lon>=3 and n_lat>=2")

    # latitude from 0..pi (inclusive poles), longitude from 0..2pi (seam)
    lat = np.linspace(0.0, math.pi, n_lat + 1, dtype=np.float64)
    lon = np.linspace(0.0, 2.0 * math.pi, n_lon, endpoint=False, dtype=np.float64)

    # vertices:
    # - one north pole
    # - rings for lat=1..n_lat-1
    # - one south pole
    verts: List[List[float]] = []
    verts.append([0.0, 0.0, 1.0])  # north

    for i in range(1, n_lat):
        theta = lat[i]
        z = math.cos(theta)
        r = math.sin(theta)
        for phi in lon:
            x = r * math.cos(phi)
            y = r * math.sin(phi)
            verts.append([x, y, z])

    verts.append([0.0, 0.0, -1.0])  # south
    V = np.asarray(verts, dtype=np.float64)

    def ring_index(i_lat: int, i_lon: int) -> int:
        # i_lat in 1..n_lat-1
        return 1 + (i_lat - 1) * n_lon + (i_lon % n_lon)

    faces: List[Tuple[int, int, int]] = []
    north = 0
    south = V.shape[0] - 1

    # top cap
    if n_lat > 1:
        for j in range(n_lon):
            a = ring_index(1, j)
            b = ring_index(1, j + 1)
            faces.append((north, a, b))

    # middle quads (two triangles each)
    for i in range(1, n_lat - 1):
        for j in range(n_lon):
            a = ring_index(i, j)
            b = ring_index(i, j + 1)
            c = ring_index(i + 1, j)
            d = ring_index(i + 1, j + 1)
            faces.append((a, c, b))
            faces.append((b, c, d))

    # bottom cap
    if n_lat > 1:
        for j in range(n_lon):
            a = ring_index(n_lat - 1, j)
            b = ring_index(n_lat - 1, j + 1)
            faces.append((a, south, b))

    F = np.asarray(faces, dtype=np.int64)
    return V, F


def _pca_basis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (center, R, evals) where:
      - center: (3,)
      - R: (3,3) columns are principal axes (right-handed)
      - evals: (3,) eigenvalues sorted desc
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape}")
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points for PCA")

    center = points.mean(axis=0)
    X = points - center
    # covariance (3x3)
    C = (X.T @ X) / max(1, (X.shape[0] - 1))
    evals, evecs = np.linalg.eigh(C)  # ascending
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # enforce right-handed basis: det(R)=+1
    R = evecs.copy()
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return center, R, evals


def fit_pca_ellipsoid(points: np.ndarray) -> Dict[str, object]:
    """
    Fit PCA ellipsoid (via PCA OBB extents). Returns a dict with:
      - center (3,)
      - rotation (3x3, columns are axes)
      - extents (3,) full lengths along PCA axes
      - radii (3,) half-lengths (a,b,c) in same order
      - volume (float) in point units^3
    """
    center, R, evals = _pca_basis(points)
    # project points into PCA frame (coordinates along principal axes)
    local = (points - center) @ R
    mins = local.min(axis=0)
    maxs = local.max(axis=0)
    extents = (maxs - mins).astype(np.float64)
    radii = 0.5 * extents
    volume = (4.0 / 3.0) * math.pi * float(radii[0] * radii[1] * radii[2])

    return {
        "center": center.tolist(),
        "rotation": R.tolist(),
        "eigenvalues": evals.tolist(),
        "extents": extents.tolist(),
        "radii": radii.tolist(),
        "volume": volume,
    }


def fit_pca_ellipsoid_sym_z(
    points: np.ndarray,
    *,
    quantile: float = 0.995,
    t_min: float = 0.10,
    search_steps: int = 201,
    refine_rounds: int = 3,
) -> Dict[str, object]:
    """
    Fit an ellipsoid while explicitly accounting for missing bottom points (+Z is up).

    Problem:
      PCA-OBB extents use (max-min) which underestimates when one side is missing (e.g., bottom occlusion).

    Approach (numpy-only, robust-ish):
      1) Use 3D PCA for orientation (R) and a reference origin (mean).
      2) In PCA frame, estimate cx,cy and a,b using symmetric quantiles (assumes no missing in +/-X, +/-Y).
      3) Estimate cz and c by using the ellipsoid equation on side points:
           (z-cz)^2 = c^2 * (1 - ((x-cx)/a)^2 - ((y-cy)/b)^2)
         Solve for cz by 1D search; for each cz compute best c^2 via least squares (no intercept).

    Returns same keys as fit_pca_ellipsoid, plus:
      - fit_mode = "pca_sym_z"
      - pca_mean (world)
      - sym_fit_debug (mask stats, search range)
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape}")
    if points.shape[0] < 10:
        raise ValueError("Need at least 10 points for robust fit")
    if not (0.5 < float(quantile) < 1.0):
        raise ValueError("quantile must be in (0.5, 1.0)")

    # PCA orientation + reference origin
    pca_mean, R, evals = _pca_basis(points)
    local = (points - pca_mean) @ R  # (N,3)
    x, y, z = local[:, 0], local[:, 1], local[:, 2]

    q_lo = 1.0 - float(quantile)
    q_hi = float(quantile)

    # robust symmetric center in x,y
    x_lo, x_hi = float(np.quantile(x, q_lo)), float(np.quantile(x, q_hi))
    y_lo, y_hi = float(np.quantile(y, q_lo)), float(np.quantile(y, q_hi))
    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)

    # robust radii in x,y using absolute quantile about cx,cy
    ax = float(np.quantile(np.abs(x - cx), q_hi))
    by = float(np.quantile(np.abs(y - cy), q_hi))
    ax = max(ax, 1e-9)
    by = max(by, 1e-9)

    u = (x - cx) / ax
    v = (y - cy) / by
    t = 1.0 - (u * u + v * v)  # expected >=0 for points inside the XY cross-section
    mask = np.isfinite(t) & np.isfinite(z) & (t > float(t_min))
    if int(mask.sum()) < 50:
        # if too strict, relax a bit
        mask = np.isfinite(t) & np.isfinite(z) & (t > 0.02)
    if int(mask.sum()) < 20:
        raise ValueError(
            "Not enough side points to infer z radius (try lowering --t_min or increasing --max_points)."
        )

    tz = t[mask].astype(np.float64, copy=False)
    zz = z[mask].astype(np.float64, copy=False)

    # Search cz in a range that allows missing bottom: extend below observed z_min
    z_min = float(np.min(zz))
    z_max = float(np.max(zz))
    z_span = max(1e-6, z_max - z_min)
    lo = z_min - 1.0 * z_span
    hi = z_max + 0.2 * z_span

    def best_c2_and_sse(cz_val: float) -> Tuple[float, float]:
        y2 = (zz - cz_val) ** 2
        denom = float(np.sum(tz * tz))
        if denom <= 0:
            return 0.0, float("inf")
        c2 = float(np.sum(tz * y2) / denom)
        c2 = max(c2, 1e-12)
        pred = c2 * tz
        sse = float(np.mean((y2 - pred) ** 2))
        return c2, sse

    best_cz = None
    best_c2 = None
    best_sse = float("inf")
    cur_lo, cur_hi = lo, hi

    for _ in range(int(refine_rounds)):
        grid = np.linspace(cur_lo, cur_hi, int(search_steps), dtype=np.float64)
        sses = []
        c2s = []
        for cz_val in grid:
            c2, sse = best_c2_and_sse(float(cz_val))
            c2s.append(c2)
            sses.append(sse)
        sses_arr = np.asarray(sses, dtype=np.float64)
        k = int(np.argmin(sses_arr))
        best_cz = float(grid[k])
        best_c2 = float(c2s[k])
        best_sse = float(sses_arr[k])
        # refine range around best (keep some width)
        width = (cur_hi - cur_lo) * 0.25
        cur_lo = best_cz - width
        cur_hi = best_cz + width

    assert best_cz is not None and best_c2 is not None
    cz = best_cz
    cz_c = float(math.sqrt(best_c2))

    # center in local PCA frame
    center_local = np.array([cx, cy, cz], dtype=np.float64)
    # convert center to world
    center_world = pca_mean + (center_local @ R.T)

    radii = np.array([ax, by, cz_c], dtype=np.float64)
    extents = 2.0 * radii
    volume = (4.0 / 3.0) * math.pi * float(radii[0] * radii[1] * radii[2])

    return {
        "fit_mode": "pca_sym_z",
        "pca_mean": pca_mean.tolist(),
        "center": center_world.tolist(),
        "rotation": R.tolist(),
        "eigenvalues": evals.tolist(),
        "extents": extents.tolist(),
        "radii": radii.tolist(),
        "volume": volume,
        "sym_fit_debug": {
            "quantile": float(quantile),
            "t_min": float(t_min),
            "masked_points": float(int(mask.sum())),
            "masked_ratio": float(int(mask.sum()) / max(1, points.shape[0])),
            "cz_search_lo": float(lo),
            "cz_search_hi": float(hi),
            "cz_best_sse": float(best_sse),
        },
    }


def apply_z_axis_compensation(
    res: Dict[str, object],
    *,
    z_compensation: float,
) -> Dict[str, object]:
    """
    Apply an additional multiplicative compensation to the ellipsoid radius along the axis
    most aligned with world +Z.

    This is useful when point clouds are systematically missing/cropped on the bottom.
    We increase the selected semi-axis by (1 + z_compensation) and shift the center along
    that axis to keep the "top" roughly fixed (i.e., extend mostly downward).
    """
    z_comp = float(z_compensation)
    if z_comp == 0.0:
        return res
    if z_comp < -0.5:
        raise ValueError("z_compensation is too negative; would collapse the ellipsoid.")

    center = np.asarray(res["center"], dtype=np.float64).reshape(3)
    R = np.asarray(res["rotation"], dtype=np.float64).reshape(3, 3)
    radii = np.asarray(res["radii"], dtype=np.float64).reshape(3)

    world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # pick axis whose direction is most aligned with world +Z (by absolute dot)
    dots = np.abs(R.T @ world_z)  # (3,)
    k = int(np.argmax(dots))
    axis = R[:, k].copy()
    sign = 1.0 if float(np.dot(axis, world_z)) >= 0.0 else -1.0
    axis_up = sign * axis  # points roughly towards +Z

    old_r = float(radii[k])
    new_r = old_r * (1.0 + z_comp)
    delta_r = new_r - old_r

    # Shift center so the top point along +Z stays (approximately) fixed:
    # old_top = c + old_r*axis_up
    # new_top = (c + shift) + new_r*axis_up
    # => shift = (old_r - new_r) * axis_up = -delta_r * axis_up
    center = center - delta_r * axis_up
    radii[k] = new_r

    # Update derived fields
    extents = 2.0 * radii
    volume = (4.0 / 3.0) * math.pi * float(radii[0] * radii[1] * radii[2])

    res2 = dict(res)
    res2["center"] = center.tolist()
    res2["radii"] = radii.tolist()
    res2["extents"] = extents.tolist()
    res2["volume"] = volume
    res2["z_compensation"] = z_comp
    res2["z_comp_axis_index"] = k
    res2["z_comp_axis_world"] = axis_up.tolist()
    return res2


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fit PCA ellipsoid from point clouds and estimate volume (supports missing-bottom via pca_sym_z)."
    )
    ap.add_argument(
        "--ply",
        default="",
        help="Path to a point cloud PLY (supports ascii or binary_little_endian). If provided, overrides --sample/--all.",
    )
    ap.add_argument(
        "--root",
        default="data/kmelon_centered",
        help="Root directory containing samples as <root>/S*/laser/fruit.ply (used with --sample/--samples/--all).",
    )
    ap.add_argument(
        "--sample",
        default="",
        help="Single sample ID like 'S7' (used with --root).",
    )
    ap.add_argument(
        "--samples",
        default="",
        help="Comma-separated sample IDs like 'S7,S58,S69' (used with --root).",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Process all samples under --root that have laser/fruit.ply.",
    )
    ap.add_argument(
        "--max_points",
        type=int,
        default=20000,
        help="If >0, uniformly subsample to at most this many points (for speed). Default: 20000",
    )
    ap.add_argument(
        "--fit_mode",
        choices=["pca3d", "pca_sym_z"],
        default="pca_sym_z",
        help="Fitting mode. 'pca3d' uses PCA extents (fast, biased if one side is missing). "
        "'pca_sym_z' assumes +Z is up and infers the missing bottom in Z using an ellipsoid constraint on side points.",
    )
    ap.add_argument(
        "--quantile",
        type=float,
        default=0.999,
        help="Robust quantile used to estimate x/y radii in pca_sym_z mode.",
    )
    ap.add_argument(
        "--t_min",
        type=float,
        default=0.10,
        help="In pca_sym_z mode, only use points with t=1-u^2-v^2 > t_min to estimate z radius (focus on side points).",
    )
    ap.add_argument(
        "--z_compensation",
        type=float,
        default=0.05,
        help="Extra multiplicative compensation for the ellipsoid radius along the axis most aligned with world +Z. "
        "Default 0.05 means +5%% on that radius (and we shift center downward to keep the top roughly fixed). "
        "Set to 0 to disable.",
    )
    ap.add_argument(
        "--out_dir",
        default="size_evaluation/ellipsoid_fitting",
        help="Output directory for auto-named results (JSON + optional ellipsoid mesh).",
    )
    ap.add_argument(
        "--name",
        default="",
        help="Optional explicit base name for outputs (without extension). If omitted, derived from sample/ply filename.",
    )
    ap.add_argument(
        "--suffix",
        default="",
        help="Optional suffix appended to auto output names (default derives from fit_mode/quantile).",
    )
    ap.add_argument(
        "--out_ellipsoid_ply",
        default="",
        help="Optional: write fitted ellipsoid as an ASCII mesh PLY to this path. If omitted, may be auto-generated unless --no_ellipsoid.",
    )
    ap.add_argument(
        "--no_ellipsoid",
        action="store_true",
        help="Disable writing ellipsoid mesh when using auto output naming (useful for --all).",
    )
    ap.add_argument(
        "--ellipsoid_lon",
        type=int,
        default=96,
        help="Ellipsoid mesh longitude resolution (only used with --out_ellipsoid_ply).",
    )
    ap.add_argument(
        "--ellipsoid_lat",
        type=int,
        default=48,
        help="Ellipsoid mesh latitude resolution (only used with --out_ellipsoid_ply).",
    )
    ap.add_argument(
        "--assume_units",
        choices=["m", "mm", "cm"],
        default="m",
        help="Only affects derived unit conversions printed to stdout.",
    )
    args = ap.parse_args(argv)

    def _derive_name_from_ply(p: str) -> str:
        base = os.path.splitext(os.path.basename(p))[0]
        # if path contains ".../<Sxx>/laser/fruit.ply", prefer Sxx
        parts = os.path.normpath(p).split(os.sep)
        for i in range(len(parts) - 1):
            if parts[i].startswith("S") and parts[i + 1] == "laser":
                return parts[i]
        return base

    def _default_suffix() -> str:
        # Keep filenames vanilla by default (no fit hyperparams in names).
        # Users can still override with --suffix if they want to disambiguate runs.
        return ""

    def _fit_one(ply_path: str, *, name_hint: str = "") -> Tuple[Dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ply_path = os.path.abspath(ply_path)
        pts = _load_ply_xyz(ply_path)
        if args.max_points and pts.shape[0] > args.max_points:
            idx = np.linspace(0, pts.shape[0] - 1, num=args.max_points, dtype=np.int64)
            pts = pts[idx]

        if args.fit_mode == "pca3d":
            res = fit_pca_ellipsoid(pts)
            res["fit_mode"] = "pca3d"
        else:
            res = fit_pca_ellipsoid_sym_z(
                pts,
                quantile=float(args.quantile),
                t_min=float(args.t_min),
            )
        # Optional post-compensation in the world +Z direction
        if float(args.z_compensation) != 0.0:
            res = apply_z_axis_compensation(res, z_compensation=float(args.z_compensation))

        radii = np.asarray(res["radii"], dtype=np.float64)
        extents = np.asarray(res["extents"], dtype=np.float64)

        # derived conversions for convenience
        unit_scale_to_m = {"m": 1.0, "cm": 0.01, "mm": 0.001}[args.assume_units]
        vol_m3 = float(res["volume"] * (unit_scale_to_m ** 3))
        vol_ml = vol_m3 * 1e6
        extents_mm = extents * (1000.0 * unit_scale_to_m)
        radii_mm = radii * (1000.0 * unit_scale_to_m)

        display_name = name_hint or _derive_name_from_ply(ply_path)
        print("\n=== PCA Ellipsoid ===")
        print(f"name: {display_name}")
        print(f"PLY: {ply_path}")
        print(f"points: {pts.shape[0]}")
        print(f"units (assumed): {args.assume_units}")
        print(f"fit_mode: {res.get('fit_mode','')}")
        if float(res.get("z_compensation", 0.0) or 0.0) != 0.0:
            print(
                f"z_compensation: {float(res.get('z_compensation')):.3f} "
                f"(axis_index={int(res.get('z_comp_axis_index', -1))})"
            )
        print("")
        print("Extents along PCA axes (full lengths):")
        print(f"  extents: {extents} ({extents_mm} mm)")
        print("Semi-axes (radii a,b,c):")
        print(f"  radii:   {radii} ({radii_mm} mm)")
        print("")
        print(f"Volume: {res['volume']:.6g} {args.assume_units}^3  (~{vol_ml:.2f} mL if units=m)")

        return res, pts, radii, extents_mm, radii_mm

    def _write_outputs(
        *,
        res: Dict[str, object],
        pts_used: int,
        ply_path: str,
        radii: np.ndarray,
        extents_mm: np.ndarray,
        radii_mm: np.ndarray,
        base_name: str,
    ) -> None:
        out_dir = os.path.abspath(args.out_dir)
        # Per-sample structure:
        #   <out_dir>/<sample>/<stem>.json
        #   <out_dir>/<sample>/<stem>_ellipsoid.ply
        sample_dir = os.path.join(out_dir, base_name)
        os.makedirs(sample_dir, exist_ok=True)
        suf = args.suffix.strip() or _default_suffix()
        stem = f"{base_name}_{suf}" if suf else base_name

        # JSON
        unit_scale_to_m = {"m": 1.0, "cm": 0.01, "mm": 0.001}[args.assume_units]
        vol_m3 = float(res["volume"] * (unit_scale_to_m ** 3))
        vol_ml = vol_m3 * 1e6
        out_json = os.path.join(sample_dir, f"{stem}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **res,
                    "ply": os.path.abspath(ply_path),
                    "points_used": int(pts_used),
                    "assume_units": args.assume_units,
                    "volume_m3": vol_m3,
                    "volume_ml": vol_ml,
                    "extents_mm": extents_mm.tolist(),
                    "radii_mm": radii_mm.tolist(),
                },
                f,
                indent=2,
            )
        print(f"Wrote: {out_json}")

        # Ellipsoid mesh PLY (optional)
        if args.no_ellipsoid:
            return

        out_ply = args.out_ellipsoid_ply.strip()
        if not out_ply:
            out_ply = os.path.join(sample_dir, f"{stem}_ellipsoid.ply")
        out_ply = os.path.abspath(out_ply)

        center = np.asarray(res["center"], dtype=np.float64)
        R = np.asarray(res["rotation"], dtype=np.float64)
        V0, F0 = _uv_sphere_mesh(n_lon=args.ellipsoid_lon, n_lat=args.ellipsoid_lat)
        V_local = V0 * radii.reshape(1, 3)
        V_world = center.reshape(1, 3) + (V_local @ R.T)
        _write_ascii_ply_mesh(out_ply, V_world, F0)
        print(f"Wrote: {out_ply}")

    # Determine which files to process
    ply_paths: List[Tuple[str, str]] = []  # (path, name_hint)
    if args.ply:
        ply_paths = [(args.ply, args.name.strip() or _derive_name_from_ply(args.ply))]
    else:
        root = os.path.abspath(args.root)
        if args.all:
            # scan directories under root
            if not os.path.isdir(root):
                raise SystemExit(f"--root does not exist or is not a directory: {root}")
            for d in sorted(os.listdir(root)):
                if not d.startswith("S"):
                    continue
                p = os.path.join(root, d, "laser", "fruit.ply")
                if os.path.isfile(p):
                    ply_paths.append((p, d))
        else:
            samples: List[str] = []
            if args.sample.strip():
                samples.append(args.sample.strip())
            if args.samples.strip():
                samples.extend([s.strip() for s in args.samples.split(",") if s.strip()])
            if not samples:
                raise SystemExit("Provide --ply, or --sample/--samples, or --all.")
            for s in samples:
                if not s.startswith("S"):
                    s = "S" + s
                p = os.path.join(root, s, "laser", "fruit.ply")
                if not os.path.isfile(p):
                    raise SystemExit(f"Missing point cloud: {p}")
                ply_paths.append((p, s))

    # Process
    for ply_path, name_hint in ply_paths:
        # for batch mode we should not reuse a single --out_ellipsoid_ply path
        if len(ply_paths) > 1:
            args.out_ellipsoid_ply = ""
        base_name = args.name.strip() or name_hint
        res, pts_used, radii, extents_mm, radii_mm = _fit_one(ply_path, name_hint=base_name)
        _write_outputs(
            res=res,
            pts_used=int(pts_used.shape[0]),
            ply_path=ply_path,
            radii=radii,
            extents_mm=extents_mm,
            radii_mm=radii_mm,
            base_name=base_name,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



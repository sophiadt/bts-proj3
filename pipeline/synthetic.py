"""Synthetic phone captures from MediaPipe's canonical face, for tests.

Writes the same files the app writes (landmarks_3d_mm.json with nulls,
depth_<W>x<H>_float32_m.bin, meta.json), by rendering a z-buffer of the
canonical mesh placed in the camera frame and sampling it at the landmarks'
pixels the way the app does (median of a 3x3 patch).

`mode` reproduces each way the capture can be wrong:
    ok                 everything consistent
    labels_swapped     cloud correct, MediaPipe's left/right indices swapped
    geometry_mirrored  depth image (and the landmarks' pixels) mirrored:
                       wrong-handed cloud, labels on the right physical points
    twin               both: a consistent mirror image of the subject
    reg_mismatch       landmarks' pixels mirrored but not the depth: depth is
                       sampled at the wrong places
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import canonical as mp                   # noqa: E402
from pipeline import capture as cap                    # noqa: E402

W, H = 640, 480
REF_W, REF_H = 4032, 3024
FX_REF = 2900.0
BACKGROUND_M = 1.2


def _yaw(deg):
    """Rotation about template Y. +deg turns the head to the subject's own left
    (the nose, at -Z, swings toward -X: the subject's left)."""
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def head_template(scale=1.0, yaw_deg=0.0, warp=None):
    """Canonical vertices in the template frame (mm), turned by yaw_deg."""
    from pipeline.fixture import R_CANON_TO_TEMPLATE
    c = mp.load()
    v = c["vertices"] @ R_CANON_TO_TEMPLATE.T * scale
    if warp is not None:
        v = warp(v)
    ctr = v.mean(axis=0)
    return (v - ctr) @ _yaw(yaw_deg).T + ctr, c["faces"]


def dense_points(v, f, spacing_mm=0.3):
    """Barycentric samples on every triangle, finer than a depth pixel, so the
    z-buffer has no holes (MediaPipe's forehead triangles are ~30 mm)."""
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    longest = np.max(np.stack([np.linalg.norm(p1 - p0, axis=1), np.linalg.norm(p2 - p1, axis=1),
                               np.linalg.norm(p0 - p2, axis=1)]), axis=0)
    ns = np.clip(np.ceil(longest / spacing_mm).astype(int) + 1, 2, 400)
    out = [v]
    for n in np.unique(ns):
        sel = ns == n
        a, b = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        keep = a + b <= 1
        a, b = a[keep], b[keep]
        q0, q1, q2 = p0[sel], p1[sel], p2[sel]
        pts = q0[:, None] + a[None, :, None] * (q1 - q0)[:, None] + b[None, :, None] * (q2 - q0)[:, None]
        out.append(pts.reshape(-1, 3))
    return np.concatenate(out)


def make_view(folder, prefix="", yaw_deg=0.0, mode="ok", distance_mm=350.0, offset_mm=(25.0, 10.0),
              scale=1.0, warp=None, null_idx=(), noise_mm=0.0, seed=0):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    v_t, f = head_template(scale, yaw_deg, warp)
    # Template -> camera: inverse of CAM_TO_TEMPLATE, then push away from the lens.
    to_cam = lambda p: p @ cap.R_CAM_TO_TEMPLATE + np.array([offset_mm[0], offset_mm[1], distance_mm]) \
        - np.array([0, 0, v_t[:, 2].min()])
    v_c = to_cam(v_t)
    s = W / REF_W
    fx = fy = FX_REF * s
    cx, cy = W / 2.0, H / 2.0
    pts = to_cam(dense_points(v_t, f))
    u = np.round(fx * pts[:, 0] / pts[:, 2] + cx).astype(int)
    w = np.round(fy * pts[:, 1] / pts[:, 2] + cy).astype(int)
    ok = (u >= 0) & (u < W) & (w >= 0) & (w < H)
    depth = np.full((H, W), BACKGROUND_M, dtype=np.float64)
    order = np.argsort(-pts[ok, 2])                       # far first, near overwrites
    depth[w[ok][order], u[ok][order]] = pts[ok, 2][order] / 1000.0
    if noise_mm:
        depth += rng.normal(scale=noise_mm / 1000.0, size=depth.shape)

    lm_c = np.vstack([v_c, np.repeat(v_c[[468 - 1]], 10, axis=0)])        # irises: placeholder
    lm_c[468:473] = v_c[mp.RIGHT_EYE_UPPER + mp.RIGHT_EYE_LOWER].mean(axis=0)
    lm_c[473:478] = v_c[mp.LEFT_EYE_UPPER + mp.LEFT_EYE_LOWER].mean(axis=0)
    lu = fx * lm_c[:, 0] / lm_c[:, 2] + cx
    lv = fy * lm_c[:, 1] / lm_c[:, 2] + cy

    if mode in ("geometry_mirrored", "twin"):
        depth = depth[:, ::-1].copy()
        lu = (W - 1) - lu
    if mode == "reg_mismatch":
        lu = (W - 1) - lu

    def sample(uu, vv):
        ui, vi = int(round(uu)), int(round(vv))
        if not (0 <= ui < W and 0 <= vi < H):
            return None
        win = depth[max(vi - 1, 0):vi + 2, max(ui - 1, 0):ui + 2]
        win = win[np.isfinite(win) & (win > 0)]
        return float(np.median(win)) if win.size else None

    out = []
    for i in range(mp.N_LANDMARKS):
        z = sample(lu[i], lv[i])
        if z is None or i in null_idx:
            out.append(None)
            continue
        out.append([(lu[i] - cx) * z / fx * 1000.0, (lv[i] - cy) * z / fy * 1000.0, z * 1000.0])

    if mode in ("labels_swapped", "twin"):
        from pipeline.fixture import mirror_permutation
        perm = list(mirror_permutation(mp.load()["vertices"])) + [473, 474, 475, 476, 477, 468, 469, 470, 471, 472]
        out = [out[perm[i]] for i in range(mp.N_LANDMARKS)]

    with open(folder / (prefix + "landmarks_3d_mm.json"), "w") as fh:
        json.dump(out, fh)
    depth.astype("<f4").tofile(str(folder / "{0}depth_{1}x{2}_float32_m.bin".format(prefix, W, H)))
    meta = {"pose": prefix.rstrip("_") or "single", "depth_width": W, "depth_height": H,
            "fx": FX_REF, "fy": FX_REF, "cx": cx / s, "cy": cy / s,
            "intrinsics_ref_width": REF_W, "intrinsics_ref_height": REF_H, "synthetic": mode}
    with open(folder / (prefix + "meta.json"), "w") as fh:
        json.dump(meta, fh)
    return folder


def make_guided(folder, mode="ok", turn_deg=30.0, **kw):
    """front_ / left_ / right_ like the app's guided capture ("turn to YOUR LEFT")."""
    make_view(folder, "front_", 0.0, mode, **kw)
    make_view(folder, "left_", +turn_deg, mode, **kw)
    make_view(folder, "right_", -turn_deg, mode, **kw)
    return Path(folder)

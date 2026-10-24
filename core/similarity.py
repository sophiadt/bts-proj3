"""Least-squares similarity alignment (Umeyama 1991), reflections suppressed.

Vendored from bts-proj1/experimental/retarget.py at commit 7ed50fb (MIT,
Sophia Don Tranho). Copied rather than imported so this repo does not depend on
bts-proj1's layout.
"""

import numpy as np


def similarity_transform(src_pts, dst_pts):
    """Least-squares similarity mapping src onto dst (Umeyama 1991).

    Returns `(scale, R, t)` with `dst ~= scale * (R @ src.T).T + t`.

    Needs at least 3 non-collinear correspondences; 4+ well-spread points are
    much better conditioned. Reflections are suppressed -- a mirrored fit would
    silently swap left and right on the face.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(
            "Need matching (N,3) arrays, got {0} and {1}".format(src.shape, dst.shape))
    if len(src) < 3:
        raise ValueError("Need at least 3 correspondences, got {0}".format(len(src)))

    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s_c, d_c = src - mu_s, dst - mu_d

    cov = (s_c.T @ d_c) / len(src)
    u, sig, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    corr = np.diag([1.0, 1.0, d])
    rot = vt.T @ corr @ u.T

    var_s = (s_c ** 2).sum() / len(src)
    if var_s < 1e-12:
        raise ValueError("Source correspondences are coincident")
    scale = float(np.trace(np.diag(sig) @ corr) / var_s)
    trans = mu_d - scale * (rot @ mu_s)
    return scale, rot, trans


def apply_similarity(pts, scale, rot, trans):
    """Apply `(scale, R, t)` to (N,3) or (T,N,3) points."""
    p = np.asarray(pts, dtype=np.float64)
    return scale * (p @ rot.T) + trans

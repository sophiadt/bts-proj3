"""Bring the scan's features back after the stiff skin fit, and measure them.

The skin fit (pipeline/nonrigid.py) keeps the start head's local shape at a
stiff setting, so it follows the scan only to about its noise level. From the
human base head that keeps the averaged MediaPipe face's detail instead of the
person's: the nose, brow, cheekbones and chin come out softer than the scan.

detail_pass moves each seen skin vertex along its normal by a smoothed,
clamped share of its distance to the (already denoised) scan. Moving only
along the normals of a surface that is already smooth brings the shape back
without the crumpling a softer full fit gave (stiffness 20: 1,047 folds).

feature_metric reports the scan-to-skin distance per facial region, so a
softer or sharper result shows up as a number.
"""

import numpy as np
from scipy.spatial import cKDTree

from core import geometry as geo
from pipeline import nonrigid

GAIN = 0.8
CLAMP_MM = 4.0
SMOOTH_ITERS = 4
SEARCH_MM = 6.0
HOLE_FADE_MM = (3.0, 8.0)          # no detail within 3 mm of a hole, full beyond 8 mm
SEEN_SMOOTH_ITERS = 6              # how far detail fades in from unseen skin (in rings of vertices)
REGIONS = ("nose", "brow", "cheeks", "chin", "lips")


def _neighbour_mean(n, faces):
    """Uniform vertex-averaging operator (sparse, rows sum to 1)."""
    import scipy.sparse as sp
    e = geo.edges(faces)
    i, j = np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]]
    A = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    return sp.diags(1.0 / np.maximum(deg, 1)) @ A


def detail_pass(X, faces, cloud, cloud_n, seen, rim_dist=None, gain=GAIN, clamp_mm=CLAMP_MM,
                smooth_iters=SMOOTH_ITERS, search_mm=SEARCH_MM):
    """X moved along its outward normals toward the scan.

    cloud, cloud_n: denoised scan and its normals (any sign); seen: (N,) bool,
    skin the scan covers (the fit's data mask and coverage); rim_dist: geodesic
    mm to the nearest hole, for the fade near openings. -> (X', info)."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    vn = geo.vertex_normals(X, faces) * nonrigid.outward_sign(X, faces)
    d = np.zeros(n)
    have = np.zeros(n, dtype=bool)
    if len(cloud):
        dist, j = cKDTree(cloud).query(X, distance_upper_bound=search_mm)
        ok = np.isfinite(dist) & np.asarray(seen, dtype=bool)
        cn = np.asarray(cloud_n)[j[ok]]
        cn = cn * np.sign((cn * vn[ok]).sum(axis=1))[:, None]          # orient like the skin
        cos = (cn * vn[ok]).sum(axis=1)
        good = cos > 0.5
        idx = np.flatnonzero(ok)[good]
        # distance to the scan's tangent plane, measured along the skin normal
        d[idx] = ((cloud[j[idx]] - X[idx]) * cn[good]).sum(axis=1) / cos[good]
        have[idx] = True
    # weights: seen skin, faded in from unseen skin and away from holes
    W = _neighbour_mean(n, faces)
    w = have.astype(np.float64)
    for _ in range(SEEN_SMOOTH_ITERS):
        w = np.minimum(w, W @ w)                    # erode: low near any unseen vertex
    if rim_dist is not None:
        a, b = HOLE_FADE_MM
        w = w * np.clip((np.asarray(rim_dist) - a) / (b - a), 0.0, 1.0)
    # smooth the distance over the mesh (weighted), then clamp
    num, den = d * have, have.astype(np.float64)
    for _ in range(smooth_iters):
        num, den = W @ num, W @ den
    ds = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)
    ds = np.clip(ds, -clamp_mm, clamp_mm)
    step = gain * w * ds
    Xn = X + step[:, None] * vn
    moved = np.abs(step) > 1e-6
    return Xn, {"gain": gain, "clamp_mm": clamp_mm, "n_moved": int(moved.sum()),
                "move_median_mm": float(np.median(np.abs(step[moved]))) if moved.any() else 0.0,
                "move_max_mm": float(np.abs(step).max())}


def feature_regions(V, regions):
    """Facial regions on the skin (start head positions, template frame:
    +Y up, face toward -Z, +X the subject's right), from fixture.skin_regions'
    labels and anchors. -> {name: vertex indices}."""
    label, _rim, info, an = regions
    V = np.asarray(V, dtype=np.float64)
    out = {"lips": np.flatnonzero(np.isin(label, ["lip_upper", "lip_lower"]))}
    loops = info["loops"]
    eye_c = [V[loops[k]].mean(axis=0) for k in ("eye_r", "eye_l")]
    eye_top = max(V[loops[k]][:, 1].max() for k in ("eye_r", "eye_l"))
    eye_bot = min(V[loops[k]][:, 1].min() for k in ("eye_r", "eye_l"))
    eye_dx = abs(eye_c[0][0] - eye_c[1][0])
    x0 = info["x0"]
    mouth_top = V[loops["mouth"]][:, 1].max()
    front = V[:, 2] < np.percentile(V[:, 2], 60)                 # the face, not the sides of the head
    out["nose"] = np.flatnonzero((np.linalg.norm(V - an["prn"], axis=1) < 0.45 * eye_dx) & front)
    out["chin"] = np.flatnonzero((np.linalg.norm(V - an["gn"], axis=1) < 0.4 * eye_dx) & front)
    out["brow"] = np.flatnonzero((V[:, 1] > eye_top + 3.0) & (V[:, 1] < eye_top + 0.5 * eye_dx) &
                                 (np.abs(V[:, 0] - x0) < 1.0 * eye_dx) & front & (label == "face"))
    out["cheeks"] = np.flatnonzero((V[:, 1] < eye_bot - 3.0) & (V[:, 1] > mouth_top) &
                                   (np.abs(V[:, 0] - x0) > 0.45 * eye_dx) & (np.abs(V[:, 0] - x0) < 1.1 * eye_dx) &
                                   front & (label == "face"))
    return out


def feature_metric(X, faces, cloud, fregions, max_mm=10.0):
    """Scan-to-skin distance per region: for each region, the scan points whose
    nearest skin vertex is in it. -> {region: {"median_mm", "p90_mm", "n"}}."""
    out = {}
    if not len(cloud):
        return out
    tree = cKDTree(X)
    d, j = tree.query(cloud, distance_upper_bound=max_mm)
    ok = np.isfinite(d)
    vn = geo.vertex_normals(X, faces)
    dist = np.full(len(cloud), np.inf)
    dist[ok] = np.abs(((cloud[ok] - X[j[ok]]) * vn[j[ok]]).sum(axis=1))
    owner = np.full(len(X), "", dtype=object)
    for name, idx in fregions.items():
        owner[idx] = name
    for name in fregions:
        sel = ok & (owner[np.where(ok, j, 0)] == name)
        if sel.sum() < 10:
            continue
        out[name] = {"median_mm": float(np.median(dist[sel])), "p90_mm": float(np.percentile(dist[sel], 90)),
                     "n": int(sel.sum())}
    return out

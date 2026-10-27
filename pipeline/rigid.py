"""Rigid alignment: template skin -> captured landmarks, uniform scale.

Closed-form weighted Umeyama on the fixture's landmark -> vertex pairs
(core/similarity.py; reflections suppressed), then one robust pass that drops
pairs beyond median + 3 * 1.4826 * MAD (with a 3 mm floor) and refits.

Weights: the fixture's (high on nose, chin, jaw, temples and MediaPipe's
Procrustes basis; low elsewhere; zero on lip and eyelid contours, hole rims,
irises), times zero for any landmark that is null or whose depth disagrees
with its neighbourhood.

Output: (scale, R, t) mapping template -> capture, and per-landmark residuals
in mm at subject scale. Residuals are the project's first real number.
"""

import numpy as np

from core.similarity import apply_similarity, similarity_transform

MAD_K = 3.0
FLOOR_MM = 3.0


def pairs(fixture, lm, valid):
    """-> (landmark idx, vertex idx, weight) for every usable pair."""
    out = []
    for i, (v, w) in enumerate(zip(fixture["vertex"], fixture["weight"])):
        if v is None or w <= 0 or not valid[i] or not np.isfinite(lm[i]).all():
            continue
        out.append((i, v, w))
    if len(out) < 6:
        raise ValueError("only {0} usable landmark pairs".format(len(out)))
    a = np.array(out)
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2].astype(float)


def rigid_transform(src, dst, weights=None):
    """Least-squares rotation + translation (no scale), reflections suppressed."""
    _s, R, _t = similarity_transform(src, dst, weights)
    w = np.full(len(src), 1.0 / len(src)) if weights is None else np.asarray(weights) / np.sum(weights)
    t = w @ dst - (w @ src) @ R.T
    return R, t


def align(template_skin, fixture, lm, valid):
    """-> dict(scale, R, t, residual_mm (478,) NaN where unused, used, rejected)."""
    li, vi, w = pairs(fixture, lm, valid)
    src, dst = template_skin[vi], lm[li]
    s, R, t = similarity_transform(src, dst, weights=w)
    res = np.linalg.norm(apply_similarity(src, s, R, t) - dst, axis=1)
    med = np.median(res)
    mad = np.median(np.abs(res - med)) * 1.4826
    keep = res <= max(med + MAD_K * mad, FLOOR_MM)
    if keep.sum() >= 6 and not keep.all():
        s, R, t = similarity_transform(src[keep], dst[keep], weights=w[keep])
    full = np.full(len(lm), np.nan)
    ok = np.isfinite(lm).all(axis=1) & np.array([v is not None for v in fixture["vertex"]])
    vv = np.array([v if v is not None else 0 for v in fixture["vertex"]])
    full[ok] = np.linalg.norm(apply_similarity(template_skin[vv[ok]], s, R, t) - lm[ok], axis=1)
    res_used = full[li[keep]]
    return {"scale": float(s), "R": R, "t": t, "residual_mm": full,
            "used": li[keep].tolist(), "rejected": li[~keep].tolist(),
            "rms_used_mm": float(np.sqrt(np.mean(res_used ** 2))),
            "median_used_mm": float(np.median(res_used)),
            "max_used_mm": float(res_used.max()),
            "det_R": float(np.linalg.det(R))}


def to_fit_frame(pts, R, t):
    """Capture (template axes) -> fit frame: the rigid part undone, scale kept on
    the template side, so the template scaled by s about its origin lands on it."""
    return (np.asarray(pts) - t) @ R

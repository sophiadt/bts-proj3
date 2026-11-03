"""Align the template's eye and mouth openings with the person's eyelids and lips.

Each opening of the skin (eye_r, eye_l, mouth: boundary loops found by
pipeline/fixture.skin_regions) is split at its corners into an upper and a
lower edge. Each edge is resampled by arc length, inner corner to outer, and
its vertices are pulled toward the matching MediaPipe contour resampled the
same way:

  eye_r   RIGHT_EYE_UPPER / RIGHT_EYE_LOWER   (33 outer, 133 inner)
  eye_l   LEFT_EYE_UPPER  / LEFT_EYE_LOWER    (263 outer, 362 inner)
  mouth   LIPS_INNER_UPPER / LIPS_INNER_LOWER (78, 308 corners)

The template's subject-right eye is at +X and MediaPipe's RIGHT_EYE lists are
the subject's right, so this holds only for correctly labelled captures (fit
after the mirror check, with --relabel if it said LABELS_SWAPPED).

The hole edges are not the visible margins (the lip lining wraps 7-9 mm into
the mouth, the lids 4.5-5 mm toward the eye), so with `ref_skin` and the
fixture each target is the contour point plus the template's own
edge-behind-margin offset; without them the lining is pulled out (the first
real fit hung its mouth open that way).

Depth at an eyelid margin or the lip line is the least reliable on the face
(lashes, the wet eye, the dark mouth), so each target keeps the landmark's
position across the face and takes its depth from the smoothed scan where a
scan point lies within SNAP_MM of it.
"""

import numpy as np
from scipy.spatial import cKDTree

from pipeline import canonical as mp
from pipeline import fixture as fx

N_PER_EDGE = 12
SNAP_MM = 6.0

CONTOURS = {
    "eye_r": (mp.RIGHT_EYE_UPPER, mp.RIGHT_EYE_LOWER, mp.EYE_R_INNER),
    "eye_l": (mp.LEFT_EYE_UPPER, mp.LEFT_EYE_LOWER, mp.EYE_L_INNER),
    "mouth": (mp.LIPS_INNER_UPPER, mp.LIPS_INNER_LOWER, None),
}


def _resample(pts, n):
    """n points evenly spaced by arc length along a polyline."""
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, pts[:, k]) for k in range(3)])


def _resample_idx(V, chain, n):
    """n vertex indices along a loop chain, evenly spaced by arc length."""
    p = V[chain]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.linspace(0.0, s[-1], n)
    return np.array([chain[int(np.argmin(np.abs(s - ti)))] for ti in t])


def template_edges(V, F):
    """{opening: (upper vertex chain, lower chain)}, each ordered from the
    corner nearest the face midline (inner) to the outer one."""
    _r, _rim, info, an = fx.skin_regions(V, F)
    loops = info["loops"]
    x0 = info["x0"]
    out = {}
    corners = {"eye_r": (an["en_r"], an["ex_r"]), "eye_l": (an["en_l"], an["ex_l"]),
               "mouth": (an["ch_r"], an["ch_l"])}
    for name, (c_in, c_out) in corners.items():
        loop = loops[name]
        a, b = fx.nearest_in(V, loop, c_in), fx.nearest_in(V, loop, c_out)
        up, lo = fx.split_loop(V, loop, a, b)
        out[name] = tuple(ch if ch[0] == a else ch[::-1] for ch in (up, lo))
    return out, x0


def constraints(V, F, landmarks, valid, cloud=None, n=N_PER_EDGE, ref_skin=None, fixture=None, only=None):
    """-> (vertex indices, target points (k,3)) pulling each opening's edges
    onto the person's contours, in the fit frame. Edges whose contour has
    missing landmarks are skipped."""
    edges, _x0 = template_edges(V, F)
    tree = cKDTree(cloud) if cloud is not None and len(cloud) else None
    idx, tgt = [], []
    for name, (up_chain, lo_chain) in edges.items():
        if only is not None and name not in only:
            continue
        upper, lower, inner = CONTOURS[name]
        for chain, contour in ((up_chain, upper), (lo_chain, lower)):
            ids = list(contour)
            # Inner corner first, like the template chain. The inner-lip lists already
            # run 78 -> 308 (subject's right -> left), and the template mouth chain
            # starts at its +X corner, the subject's right.
            if inner is not None and ids[0] != inner:
                ids = ids[::-1]
            if not valid[ids].all():
                continue
            target = _resample(landmarks[ids], n)
            if tree is not None:
                d, j = tree.query(target)
                near = d < SNAP_MM
                target[near, 2] = cloud[j[near], 2]       # depth from the scan, position from MediaPipe
            edge = _resample_idx(V, chain, n)
            if ref_skin is not None and fixture is not None:
                # The hole edge is not the visible margin: the lip lining wraps
                # 7-9 mm into the mouth, the lids 4.5-5 mm toward the eye. Keep the
                # template's own edge-behind-margin offset (at subject scale), so
                # the edge follows the person's outline without rolling out.
                verts = [fixture["vertex"][i] for i in ids]
                if any(v is None for v in verts):
                    continue
                margin = _resample(ref_skin[verts], n)
                target = target + (ref_skin[edge] - margin)
            idx.extend(edge)
            tgt.extend(target)
    return np.array(idx, dtype=np.int64), np.array(tgt, dtype=np.float64)


def forward_shift(X0, X, idx):
    """Mean movement (mm) of the hole edges toward the camera (-Z): the lip
    lining rolling out shows as a large value."""
    if not len(idx):
        return None
    return float(-(X[idx, 2] - X0[idx, 2]).mean())


def alignment_error(X, idx, tgt):
    """Mean distance (mm) of the openings' edges from the person's contours."""
    if not len(idx):
        return None
    return float(np.linalg.norm(X[idx] - tgt, axis=1).mean())

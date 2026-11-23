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

The mouth is the exception by default (mouth_gap_constraints): the rig rests
with its lips apart and a neutral capture has them closed, so the hole keeps
the lab head's resting opening, centred on the person's lip line, instead of
being pulled shut onto the coinciding inner-lip contours.
"""

import numpy as np
from scipy.spatial import cKDTree

from pipeline import canonical as mp
from pipeline import fixture as fx

N_PER_EDGE = 12
SNAP_MM = 6.0

# Optional, off by default (fit_skin(smooth_mm=..., ring_depth=...); --smooth-contours / --ring-depth).
SMOOTH_SIGMA_MM = 0.8      # per-axis landmark scatter measured on the first captures (0.75-1.1 mm)
RING_IN_MM = 2.0           # skin closer than this to an opening's outline is not used (lashes, lip contact)
RING_OUT_MM = 9.0          # ... and nothing farther than this
RING_MIN_POINTS = 20
RING_ZWIN_SCAN_MM = 8.0    # ring points within this depth of the landmark (hair, neck, wet-eye returns out)
RING_ZWIN_REF_MM = 3.0     # same on the lab head, whose lining wraps 7-9 mm behind the margin
RING_DEADBAND_MM = 0.5     # a ring/scan disagreement below this is left alone (continuous soft threshold)
RING_CLAMP_MM = 3.0        # largest depth change one target can take
RING_STATS = {}            # filled by ring_adjust: what the last fit_skin changed

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


def smooth_contour(pts, sigma_mm):
    """A smoothing spline through a landmark polyline (its vertices re-evaluated on the curve).

    The curve may miss a landmark by about sigma_mm in 3-D (the measured scatter of one
    landmark). On an 11-point lip contour that cuts the jitter by roughly a fifth and keeps
    about 80% of a 2 mm cupid's bow; a looser spline removes no more jitter and flattens the
    bow (tests/test_contours.py). Returns the points unchanged for sigma_mm <= 0 or < 5 points."""
    pts = np.asarray(pts, dtype=np.float64)
    m = len(pts)
    if sigma_mm <= 0 or m < 5:
        return pts
    chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if not chord[-1] > 0 or np.any(np.diff(chord) <= 0):
        return pts
    from scipy.interpolate import splev, splprep
    u = chord / chord[-1]
    tck, _ = splprep(pts.T, u=u, w=np.full(m, 1.0 / sigma_mm), s=float(m), k=3)
    return np.array(splev(u, tck)).T


def _quad_at(P, q, iters=4):
    """Robust (Tukey-weighted) local quadratic z(x, y) of points P (k,3), evaluated at xy = q."""
    x, y = P[:, 0] - q[0], P[:, 1] - q[1]
    A = np.c_[np.ones(len(P)), x, y, x * x, x * y, y * y]
    w = np.ones(len(P))
    c = np.zeros(6)
    for _ in range(iters):
        c, *_ = np.linalg.lstsq(A * w[:, None], P[:, 2] * w, rcond=None)
        r = P[:, 2] - A @ c
        scale = max(1.4826 * np.median(np.abs(r)), 0.2)
        u = np.clip(r / (4.685 * scale), -1.0, 1.0)
        w = 1.0 - u * u
    return float(c[0])


def _outline_dense(xy, closed, per_seg=12):
    p = np.asarray(xy, dtype=np.float64)
    if closed:
        p = np.vstack([p, p[:1]])
    segs = [np.linspace(p[i], p[i + 1], per_seg, endpoint=False) for i in range(len(p) - 1)]
    return np.vstack(segs + [p[-1:]])


def ring_depth(points, q_xy, z_hint, outline_xy, closed, zwin, r_in=RING_IN_MM, r_out=RING_OUT_MM):
    """Depth (z) at xy positions q_xy of the skin AROUND an opening, carried across it.

    points: (N,3) skin samples (scan cloud or template vertices). outline_xy: the opening's
    outline in x, y (a closed polygon, or an open polyline for the lip line). Skin inside the
    outline or closer than r_in to it is ignored (that is where the sensor reads the wet eye
    or the dark mouth); a robust local quadratic through the skin r_in..r_out outside it gives
    the depth at each q. Only points within zwin of z_hint count. NaN where fewer than
    RING_MIN_POINTS remain."""
    pts = np.asarray(points, dtype=np.float64)
    q_xy = np.asarray(q_xy, dtype=np.float64)
    out = np.full(len(q_xy), np.nan)
    if len(pts) < RING_MIN_POINTS:
        return out
    d_out, _ = cKDTree(_outline_dense(outline_xy, closed)).query(pts[:, :2])
    inside = np.zeros(len(pts), dtype=bool)
    if closed:
        from matplotlib.path import Path as Polygon
        inside = Polygon(np.asarray(outline_xy, dtype=np.float64)).contains_points(pts[:, :2])
    ring = pts[(~inside) & (d_out >= r_in) & (d_out <= r_out)]
    if len(ring) < RING_MIN_POINTS:
        return out
    tree = cKDTree(ring[:, :2])
    for k in range(len(q_xy)):
        idx = tree.query_ball_point(q_xy[k], r_out)
        if len(idx) < RING_MIN_POINTS:
            continue
        P = ring[idx]
        P = P[np.abs(P[:, 2] - z_hint[k]) <= zwin]
        if len(P) >= RING_MIN_POINTS:
            out[k] = _quad_at(P, q_xy[k])
    return out


def ring_adjust(target, cloud, outline_xy, closed, ref_points, ref_outline_xy, ref_ref, ref_closed=True,
                r_in=RING_IN_MM):
    """Move each target's depth toward what the skin around its opening says it should be.

    target (k,3): the landmark contour, depth already taken from the scan. The person's ring
    surface gives a depth at each target; the lab head's own margin-minus-ring offset
    (ref_ref: its margin points, (k,3), against ref_points, its skin vertices) is added so a
    lid's or lip's real curve survives. The change is the ring/scan disagreement past a soft
    RING_DEADBAND_MM, clamped to RING_CLAMP_MM; a target whose ring or reference ring cannot
    be fitted is left as it is. -> (new targets, number changed)."""
    z_old = target[:, 2].copy()
    z_ring = ring_depth(cloud, target[:, :2], z_old, outline_xy, closed, RING_ZWIN_SCAN_MM, r_in)
    z_refring = ring_depth(ref_points, ref_ref[:, :2], ref_ref[:, 2], ref_outline_xy, ref_closed,
                           RING_ZWIN_REF_MM, r_in)
    delta = (z_ring + (ref_ref[:, 2] - z_refring)) - z_old
    ok = np.isfinite(delta)
    shrunk = np.sign(delta) * np.maximum(np.abs(delta) - RING_DEADBAND_MM, 0.0)
    shrunk = np.clip(shrunk, -RING_CLAMP_MM, RING_CLAMP_MM)
    out = np.asarray(target, dtype=np.float64).copy()
    out[ok, 2] += shrunk[ok]
    RING_STATS["targets"] = RING_STATS.get("targets", 0) + int(len(target))
    RING_STATS["fitted"] = RING_STATS.get("fitted", 0) + int(ok.sum())
    RING_STATS["changed"] = RING_STATS.get("changed", 0) + int((np.abs(shrunk[ok]) > 0).sum())
    RING_STATS["abs_shift_sum_mm"] = RING_STATS.get("abs_shift_sum_mm", 0.0) + float(np.abs(shrunk[ok]).sum())
    return out, int((np.abs(shrunk[ok]) > 0).sum())


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


def constraints(V, F, landmarks, valid, cloud=None, n=N_PER_EDGE, ref_skin=None, fixture=None, only=None,
                smooth_mm=0.0, ring=False):
    """-> (vertex indices, target points (k,3)) pulling each opening's edges
    onto the person's contours, in the fit frame. Edges whose contour has
    missing landmarks are skipped.

    smooth_mm > 0: the landmark contours go through smooth_contour() first.
    ring: eyelid depth also comes from the skin around each eye (ring_adjust); needs
    cloud, ref_skin and fixture, and applies to the eyes only (the mouth gap has its own)."""
    edges, _x0 = template_edges(V, F)
    tree = cKDTree(cloud) if cloud is not None and len(cloud) else None
    idx, tgt = [], []
    for name, (up_chain, lo_chain) in edges.items():
        if only is not None and name not in only:
            continue
        upper, lower, inner = CONTOURS[name]
        outlines = None
        if ring and name != "mouth" and cloud is not None and ref_skin is not None and fixture is not None \
                and valid[list(upper) + list(lower)].all():
            vu = [fixture["vertex"][i] for i in upper]
            vl = [fixture["vertex"][i] for i in lower]
            if not any(v is None for v in vu + vl):
                outlines = (np.r_[landmarks[list(upper)][:, :2], landmarks[list(lower)][::-1][:, :2]],
                            np.r_[ref_skin[vu][:, :2], ref_skin[vl][::-1][:, :2]])
        for chain, contour in ((up_chain, upper), (lo_chain, lower)):
            ids = list(contour)
            # Inner corner first, like the template chain. The inner-lip lists already
            # run 78 -> 308 (subject's right -> left), and the template mouth chain
            # starts at its +X corner, the subject's right.
            if inner is not None and ids[0] != inner:
                ids = ids[::-1]
            if not valid[ids].all():
                continue
            target = _resample(smooth_contour(landmarks[ids], smooth_mm), n)
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
                if outlines is not None:
                    target, _ = ring_adjust(target, cloud, outlines[0], True, ref_skin, outlines[1], margin)
                target = target + (ref_skin[edge] - margin)
            idx.extend(edge)
            tgt.extend(target)
    return np.array(idx, dtype=np.int64), np.array(tgt, dtype=np.float64)


def gap_targets(seam, ref_seam, ref_edge_up, ref_edge_lo):
    """Targets for the mouth hole's upper and lower edges: the reference head's
    edges relative to its own lip line (ref edge - ref seam), laid on the
    person's lip line. All (n,3), sampled the same way. -> (upper, lower)."""
    return seam + (ref_edge_up - ref_seam), seam + (ref_edge_lo - ref_seam)


def mouth_gap_constraints(V, F, landmarks, valid, gap_ref, cloud=None, fixture=None, n=N_PER_EDGE, edges=None,
                          smooth_mm=0.0, ring=False):
    """The mouth hole keeps the rig's resting opening, centred on the person's lips.

    The lab rig rests with the mouth slightly open (hole edges 8.3 mm apart at
    the middle) and its phoneme motion starts from there. A neutral capture has
    closed lips, so its inner-lip contours coincide, and pulling the edges onto
    them (constraints()) shut the hole and fused the lip muscle. Instead the
    person's lip line (midway between MediaPipe's inner-lip contours, depth from
    the scan) takes the reference head's edges as they sit around its own lip
    line (the fixture's vertices for the same contours).

    gap_ref: the reference skin (the lab head) at the subject's size, same
    vertex order as V. -> (vertex indices, targets (k,3), person's lip line
    (n,3) or None if the contours or their fixture vertices are missing)."""
    empty = (np.zeros(0, dtype=np.int64), np.zeros((0, 3)), None)
    if fixture is None:
        return empty
    if edges is None:
        edges, _x0 = template_edges(V, F)
    up_chain, lo_chain = edges["mouth"]
    ids_u, ids_l = list(mp.LIPS_INNER_UPPER), list(mp.LIPS_INNER_LOWER)   # both run 78 -> 308, subject's right first
    if not (valid[ids_u].all() and valid[ids_l].all()):
        return empty
    vu = [fixture["vertex"][i] for i in ids_u]
    vl = [fixture["vertex"][i] for i in ids_l]
    if any(v is None for v in vu + vl):
        return empty
    seam = 0.5 * (_resample(smooth_contour(landmarks[ids_u], smooth_mm), n)
                  + _resample(smooth_contour(landmarks[ids_l], smooth_mm), n))
    if cloud is not None and len(cloud):
        d, j = cKDTree(cloud).query(seam)
        near = d < SNAP_MM
        seam[near, 2] = cloud[j[near], 2]
    ref_seam = 0.5 * (_resample(gap_ref[vu], n) + _resample(gap_ref[vl], n))
    if ring and cloud is not None and len(cloud):
        # the lip line's depth from the lips on either side of it, plus the lab head's own offset
        ref_outline = np.r_[gap_ref[vu][:, :2], gap_ref[vl][::-1][:, :2]]
        seam, _ = ring_adjust(seam, cloud, seam[:, :2], False, gap_ref, ref_outline, ref_seam, True, r_in=2.5)
    eu, el = _resample_idx(V, up_chain, n), _resample_idx(V, lo_chain, n)
    tu, tl = gap_targets(seam, ref_seam, gap_ref[eu], gap_ref[el])
    return np.r_[eu, el], np.r_[tu, tl], seam


def drop_seam_band(cloud, seam, band_mm=2.5, depth_mm=12.0):
    """Keep mask for scan points: False within band_mm (across the face, x-y)
    of the person's lip line and within depth_mm of it in z. That strip is
    where closed lips touch; matched to the rig's parted lips it would pull
    them shut."""
    cloud = np.asarray(cloud, dtype=np.float64)
    best = np.full(len(cloud), np.inf)
    if seam is None or not len(cloud):
        return np.ones(len(cloud), dtype=bool)
    for a, b in zip(seam[:-1], seam[1:]):
        ab = b - a
        t = np.clip(((cloud - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        off = cloud - (a + t[:, None] * ab)
        dxy = np.linalg.norm(off[:, :2], axis=1)
        best = np.minimum(best, np.where(np.abs(off[:, 2]) < depth_mm, dxy, np.inf))
    return best >= band_mm


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

"""Carry the skin's displacement into the head: tissues, bones, joints.

The skin fit gives a displacement d(x) at the skin. A thin-plate-spline RBF
(scipy RBFInterpolator, with its affine term, so it extrapolates affinely
away from the face) turns it into a field over all of space, evaluated at
every rest vertex and joint.

  tissue, cloth, joints  full field
  bones                  the best-fit similarity of the field over the bone's
                         vertices, so skull, jaw and teeth stay rigid-shaped;
                         how far the field departs from that similarity is
                         reported as the bone's conformity

The template's own skin-to-skull distances are not real tissue depths (the
skin was artist-fitted onto a store-bought anatomy), so everything this
produces is plausible relative to the template, not to the subject's anatomy.
"""

import numpy as np
from scipy.interpolate import RBFInterpolator

from core import geometry as geo
from core.similarity import apply_similarity, similarity_transform

N_CENTRES = 2500
SMOOTHING = 1.0


def build_field(skin_pts, skin_disp, n_centres=N_CENTRES, smoothing=SMOOTHING, seed=0):
    """Displacement field from skin positions (mm) and their displacements.

    Smoothed on purpose. An exact field (all skin vertices, smoothing 0, local
    150-neighbour solves) was tried on the synthetic fit: it folded thousands
    of muscle triangles, made frames 0-50 of the scene 5.6x faster than the
    template's, and left the skin's 0.66 mm rest gap in Maya unchanged -- so
    that gap does not come from this field.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(skin_pts))
    if len(idx) > n_centres:
        idx = np.sort(rng.choice(idx, n_centres, replace=False))
    return RBFInterpolator(skin_pts[idx], skin_disp[idx], kernel="thin_plate_spline",
                           smoothing=smoothing, degree=1)


def inward_normals(skin, faces, centre):
    """Unit skin normals pointing into the head (toward `centre`)."""
    n = geo.vertex_normals(skin, faces)
    return n * -np.sign(((skin - centre) * n).sum(axis=1))[:, None]


def bone_from_skin(skin0, skin1, faces, sample_idx, bone0, bone_faces, max_mm=40.0, seed=0):
    """Similarity for one bone that keeps the skin-to-bone depth under the scan.

    The phone sees skin, never bone, so the bone is placed by what the scan
    does constrain: at skin vertices the camera saw, the template's depth to
    this bone straight inward (already at subject scale) is laid off again
    under the fitted skin, and one similarity (rotation, translation, uniform
    scale) is fitted to those target points, with one robust pass. The back
    of the head, which no camera saw, plays no part.

    skin0/skin1: scaled-template and fitted skin (V,3); sample_idx: vertices to
    cast from. -> (s, R, t), info  or  (None, info) if too few rays hit.
    """
    n0 = geo.vertex_normals(skin0, faces)
    centre = bone0.mean(axis=0)
    sgn = -np.sign(((skin0 - centre) * n0).sum(axis=1))
    in0 = n0 * sgn[:, None]
    lo = skin0[sample_idx].min(axis=0) - max_mm
    hi = skin0[sample_idx].max(axis=0) + max_mm
    cen = bone0[bone_faces].mean(axis=1)
    near = ((cen > lo) & (cen < hi)).all(axis=1)
    t0 = geo.ray_hit_distance(skin0[sample_idx], in0[sample_idx], bone0, bone_faces[near], max_dist=max_mm)
    hit = np.isfinite(t0)
    info = {"n_rays": int(len(sample_idx)), "n_hits": int(hit.sum())}
    if hit.sum() < 30:
        return None, info
    idx, t0 = np.asarray(sample_idx)[hit], t0[hit]
    n1 = geo.vertex_normals(skin1, faces)[idx]
    in1 = n1 * np.sign((n1 * in0[idx]).sum(axis=1))[:, None]
    h0 = skin0[idx] + in0[idx] * t0[:, None]
    h1 = skin1[idx] + in1 * t0[:, None]
    s, R, t = similarity_transform(h0, h1)
    res = np.linalg.norm(apply_similarity(h0, s, R, t) - h1, axis=1)
    med = np.median(res)
    keep = res <= max(med + 3.0 * 1.4826 * np.median(np.abs(res - med)), 2.0)
    if keep.sum() >= 30 and not keep.all():
        s, R, t = similarity_transform(h0[keep], h1[keep])
        res = np.linalg.norm(apply_similarity(h0, s, R, t) - h1, axis=1)
    info.update({"extra_scale": float(s),
                 "rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
                 "depth_template_median_mm": float(np.median(t0)),
                 "target_residual_median_mm": float(np.median(res[keep])),
                 "target_residual_p95_mm": float(np.percentile(res[keep], 95)),
                 "n_used": int(keep.sum())})
    info["_pairs"] = (h0[keep], h1[keep])            # for a joint fit over several bones
    return (s, R, t), info


def similarity_from_pairs(h0, h1):
    """One similarity for several bones' depth targets together, with the same
    robust pass as bone_from_skin. -> ((s, R, t), info)"""
    s, R, t = similarity_transform(h0, h1)
    res = np.linalg.norm(apply_similarity(h0, s, R, t) - h1, axis=1)
    med = np.median(res)
    keep = res <= max(med + 3.0 * 1.4826 * np.median(np.abs(res - med)), 2.0)
    if keep.sum() >= 30 and not keep.all():
        s, R, t = similarity_transform(h0[keep], h1[keep])
        res = np.linalg.norm(apply_similarity(h0, s, R, t) - h1, axis=1)
    return (s, R, t), {"extra_scale": float(s),
                       "rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
                       "target_residual_median_mm": float(np.median(res[keep])), "n_used": int(keep.sum())}


EYEBALL_DIAMETER_MM = 24.0


def place_eyeball(eye0, ring0, ring1, diameter=EYEBALL_DIAMETER_MM):
    """A rigid eyeball of `diameter`, seated behind the fitted eye opening.

    The template's eyeball is scaled about the centre of its own eye-opening
    ring to `diameter`, then moved and turned with the ring (rigid fit of the
    template ring onto the fitted one), so it keeps the template's
    eyeball-behind-the-lids geometry at the new size. The lab head's eyeballs
    are 34 mm across even at subject scale (stylised); a real adult eye is
    about 24 mm. -> (new vertices, info)"""
    from pipeline.rigid import rigid_transform
    k = diameter / float(np.ptp(eye0, axis=0).max())
    c0 = ring0.mean(axis=0)
    R, t = rigid_transform(ring0 - c0, ring1 - ring1.mean(axis=0))
    new = ring1.mean(axis=0) + (k * (eye0 - c0)) @ R.T
    return new, {"diameter_mm": float(np.ptp(new, axis=0).max()), "resize": float(k),
                 "ring_rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))}


def keep_behind(verts, skin, faces, margin_mm=0.5, max_mm=30.0):
    """Push a rigid body back (+Z, away from the camera) until none of it pokes
    in front of the skin: a vertex is in front if, looking away from the
    camera, the skin lies behind it. -> (verts, pushed mm)"""
    dirs = np.tile([0.0, 0.0, 1.0], (len(verts), 1))
    behind = geo.ray_hit_distance(verts, dirs, skin, faces, max_dist=max_mm)
    poke = behind[np.isfinite(behind)]
    if not len(poke):
        return verts, 0.0
    push = float(poke.max() + margin_mm)
    return verts + np.array([0.0, 0.0, push]), push


def clearance_along_view(front_pts, skin, faces, max_mm=40.0, poke_mm=15.0):
    """Signed distance (mm) from each point to the skin along the view: positive
    when the skin is in front of it (toward the camera, -Z), negative when the
    point has come through and the skin lies behind it; inf if neither."""
    ahead = geo.ray_hit_distance(front_pts, np.tile([0.0, 0.0, -1.0], (len(front_pts), 1)), skin, faces,
                                 max_dist=max_mm)
    # Only a point with no skin in front has come through; looking back, a
    # tooth that is still inside can meet the mouth lining before the lip.
    behind = geo.ray_hit_distance(front_pts, np.tile([0.0, 0.0, 1.0], (len(front_pts), 1)), skin, faces,
                                  max_dist=poke_mm)
    return np.where(np.isfinite(ahead), ahead, np.where(np.isfinite(behind), -behind, np.inf))


LIP_TISSUES = ("middleres_OrbicularisOris", "middleres_DepressorLabiiInferioris_l",
               "middleres_DepressorLabiiInferioris_r", "middleres_Mentalis")
SLIT_BAND_MM = 1.5          # across the slit, upper/lower blend over +-1.5 mm
SLIT_CORNER_MM = 4.0        # past the mouth corners, fade back to one shared field
SLIT_DEPTH_MM = 30.0        # the slit reaches this far back into the mouth


def slit_line(skin, parts):
    """The mouth slit's centre line on the skin: (xs, ys) sorted by x, the
    midpoint of the upper and lower hole edges, plus the corners' x and the
    hole's front z."""
    up, lo = skin[parts["upper"]], skin[parts["lower"]]
    xs = np.linspace(max(up[:, 0].min(), lo[:, 0].min()), min(up[:, 0].max(), lo[:, 0].max()), 40)
    yu = np.interp(xs, *zip(*sorted(zip(up[:, 0], up[:, 1]))))
    yl = np.interp(xs, *zip(*sorted(zip(lo[:, 0], lo[:, 1]))))
    loop = skin[parts["loop"]]
    return {"xs": xs, "ys": 0.5 * (yu + yl), "x_lo": float(loop[:, 0].min()), "x_hi": float(loop[:, 0].max()),
            "z_front": float(loop[:, 2].min())}


def lip_side(P, slit, upper=None):
    """Weight of the upper lip's field for each point (1 above the slit, 0
    below), sharp across the slit and 0.5 (shared) past the corners and deep
    behind the mouth, so the two fields stay joined there. upper: the point's
    half, if known (the lip muscle's, mouth.muscle_upper_mask)."""
    P = np.asarray(P, dtype=np.float64)
    if upper is not None:
        t = np.where(upper, 1.0, 0.0)
    else:
        yc = np.interp(P[:, 0], slit["xs"], slit["ys"])
        t = np.clip(0.5 + (P[:, 1] - yc) / (2.0 * SLIT_BAND_MM), 0.0, 1.0)
    out_x = np.maximum(slit["x_lo"] - P[:, 0], P[:, 0] - slit["x_hi"])
    a = np.clip(1.0 - out_x / SLIT_CORNER_MM, 0.0, 1.0)
    a *= np.clip(1.0 - (P[:, 2] - slit["z_front"]) / SLIT_DEPTH_MM, 0.0, 1.0)
    return 0.5 + a * (t - 0.5)


def split_lip_warp(rest, field_up, field_lo, slit, upper=None):
    """Move lip tissue with the upper lip's field above the slit and the lower
    lip's below. One smooth field gives both halves of the lip muscle (about
    2.5 mm apart across the slit) nearly the same motion, so any squeeze
    closes them, and Ziva then builds its tets across the closed gap."""
    t = lip_side(rest, slit, upper)[:, None]
    d = t * eval_field(field_up, rest) + (1.0 - t) * eval_field(field_lo, rest)
    return rest + d, {"disp_max_mm": float(np.linalg.norm(d, axis=1).max()), "split_field": True}


def open_lip_muscle(P, skin, parts, ref_P, ref_skin, upper=None, min_ratio=0.7, falloff_mm=3.0):
    """Lip-muscle guard: where the gap between the muscle's upper and lower
    halves (pipeline/mouth.muscle_gap, per 1 mm slice) is under min_ratio of
    the lab head's at the same place, the halves move apart by the shortfall
    (upper up, lower down, half each), fading over falloff_mm from where they
    are closest. upper: the halves' labels (default mouth.muscle_upper_mask).
    -> (P', info)."""
    from pipeline import mouth
    P = np.asarray(P, dtype=np.float64).copy()
    if upper is None:
        upper = mouth.muscle_upper_mask()
    if upper is None or len(upper) != len(P):
        return P, {"moved": False, "reason": "no half labels"}
    g = mouth.muscle_gap(P, skin, parts, upper=upper)
    g0 = mouth.muscle_gap(ref_P, ref_skin, parts, upper=upper)
    if not len(g["gaps"]) or not len(g0["gaps"]):
        return P, {"moved": False}

    def rel(xs, sk):
        Q = np.asarray(sk)[parts["loop"], 0]
        return (xs - 0.5 * (Q.min() + Q.max())) / np.ptp(Q)

    ref_at = np.interp(rel(g["xs"], skin), rel(g0["xs"], ref_skin), g0["gaps"])
    short = np.clip(min_ratio * ref_at - g["gaps"], 0.0, 2.0)      # more would fold the lower lip over itself
    info = {"min_ratio": min_ratio, "slices_short": int((short > 0).sum()), "slices": int(len(short)),
            "gap_median_before_mm": g["median"]}
    if not (short > 0).any():
        return P, dict(info, moved=False, gap_median_after_mm=g["median"])
    # where the halves meet, per slice: midway between the closest pair
    from scipy.spatial import cKDTree
    ym = []
    for x in g["xs"]:
        s = np.abs(P[:, 0] - x) < 0.75
        a, b = P[s & upper], P[s & ~upper]
        d, j = cKDTree(b).query(a)
        k = int(np.argmin(d))
        ym.append(0.5 * (a[k, 1] + b[j[k], 1]))
    ym = np.interp(P[:, 0], g["xs"], ym)
    push = np.interp(P[:, 0], g["xs"], short, left=0.0, right=0.0)
    w = np.exp(-((P[:, 1] - ym) / falloff_mm) ** 2)
    P[:, 1] += 0.5 * push * w * np.where(upper, 1.0, -1.0)
    g1 = mouth.muscle_gap(P, skin, parts, upper=upper)
    return P, dict(info, moved=True, push_max_mm=float(short.max()), gap_median_after_mm=g1["median"])


AFFINE_FALLBACK_FLIPS = 0.05      # a tissue with more of its triangles inverted gets its warp's best affine map


def affine_fallback(rest, warped):
    """The best affine map of a tissue's warp (least squares over its
    vertices), or its best similarity if that affine map would mirror it. An
    affine map with positive determinant inverts no triangle, so a small
    muscle the field folded keeps a valid shape while following the motion
    around it. (On s01 the chin muscle had 22% of its triangles inverted, and
    the morph's in-between steps passed through near-zero volume.)
    -> (new rest, info)."""
    A = np.c_[rest, np.ones(len(rest))]
    M, *_ = np.linalg.lstsq(A, warped, rcond=None)
    det = float(np.linalg.det(M[:3]))
    if det > 0:
        new, kind = A @ M, "affine"
    else:
        # the field turned it inside out (on s01: the chin muscle, det -0.65);
        # a similarity keeps it the right way round
        s, R, t = similarity_transform(rest, warped)
        new, kind = apply_similarity(rest, s, R, t), "similarity"
    return new, {"fallback": kind, "affine_det": det,
                 "fallback_residual_median_mm": float(np.median(np.linalg.norm(new - warped, axis=1)))}


def push_out_of_bones(P, faces, bones, was_inside, margin_mm=0.3, spread_iters=3):
    """Move tissue vertices that the warp pushed into a bone back to its
    surface (plus margin), and spread the move to their neighbours so the
    surface stays smooth. bones: [(V, F)] already moved; was_inside: (N,) bool,
    vertices inside a bone on the lab head (the artist's meshes overlap a
    little; those are left alone). -> (P', info)."""
    from pipeline import nonrigid
    from pipeline.verify import inside
    import scipy.sparse as sp
    from scipy.spatial import cKDTree
    P = np.asarray(P, dtype=np.float64).copy()
    prepared = []
    for bv, bf in bones:
        sign = nonrigid.outward_sign(bv, bf)
        prepared.append((bv, bf, sign, geo.vertex_normals(bv, bf) * sign, cKDTree(bv)))

    def pushes(Q):
        move = np.zeros_like(Q)
        hit = np.zeros(len(Q), dtype=bool)
        for bv, bf, sign, vn, tree in prepared:
            ins = inside(Q, bv, bf, sign) & ~was_inside
            if not ins.any():
                continue
            _d, j = tree.query(Q[ins])
            depth = -((Q[ins] - bv[j]) * vn[j]).sum(axis=1)           # how far behind the surface
            step = (depth + margin_mm)[:, None] * vn[j]
            cur = move[ins]
            move[ins] = np.where((np.linalg.norm(step, axis=1) > np.linalg.norm(cur, axis=1))[:, None], step, cur)
            hit |= ins
        return move, hit

    move, hit = pushes(P)
    if not hit.any():
        return P, {"n_pushed": 0}
    info = {"n_inside": int(hit.sum())}
    # A tissue largely inside a bone (the bone moved through it) comes out as
    # a whole first; pushing each vertex to the surface flattened the chin
    # muscle to 18% of its volume.
    if hit.mean() > 0.1:
        shift = np.median(move[hit], axis=0)
        P += shift
        info["shifted_mm"] = float(np.linalg.norm(shift))
        move, hit = pushes(P)
        if not hit.any():
            return P, dict(info, n_pushed=0)
    e = geo.edges(faces)
    i, k = np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]]
    A = sp.csr_matrix((np.ones(len(i)), (i, k)), shape=(len(P), len(P)))
    W = sp.diags(1.0 / np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1)) @ A
    spread = move.copy()
    for _ in range(spread_iters):                       # neighbours follow part of the way
        nb = W @ spread
        spread = np.where(hit[:, None], move, 0.6 * nb)
    P += spread
    return P, dict(info, n_pushed=int(hit.sum()), push_max_mm=float(np.linalg.norm(move, axis=1).max()))


def eval_field(field, pts, chunk=20000):
    pts = np.asarray(pts, dtype=np.float64)
    return np.concatenate([field(pts[s:s + chunk]) for s in range(0, len(pts), chunk)]) if len(pts) else pts


def warp_body(role, rest, field):
    """-> (new rest (V,3), info)."""
    d = eval_field(field, rest)
    if role != "bone":
        return rest + d, {"disp_max_mm": float(np.linalg.norm(d, axis=1).max())}
    s, R, t = similarity_transform(rest, rest + d)
    new = apply_similarity(rest, s, R, t)
    dev = np.linalg.norm(new - (rest + d), axis=1)
    return new, {"bone_scale": float(s), "bone_rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
                 "conformity_median_mm": float(np.median(dev)), "conformity_max_mm": float(dev.max()),
                 "similarity": (s, R, t)}


def flip_report(before, after, faces):
    f = geo.flipped_faces(before, after, faces, min_area_ratio=0.05)
    return {"n_flipped_or_collapsed": int(f.sum()), "frac": float(f.mean())}


def thickness_check(skin_before, skin_after, skin_faces, skull_before, skull_faces, skull_sim,
                    sample=2000, max_mm=40.0, seed=0):
    """Skin-to-skull thickness along the template's inward skin normals, before
    and after. The skull hit point moves with the skull's similarity, so a
    negative new thickness means the skin went through the skull."""
    rng = np.random.default_rng(seed)
    n = geo.vertex_normals(skin_before, skin_faces)
    centroid = skull_before.mean(axis=0)
    inward = -n * np.sign(((skin_before - centroid) * n).sum(axis=1))[:, None]
    idx = rng.choice(len(skin_before), min(sample, len(skin_before)), replace=False)
    t0 = geo.ray_hit_distance(skin_before[idx], inward[idx], skull_before, skull_faces, max_dist=max_mm)
    hit = np.isfinite(t0)
    idx, t0 = idx[hit], t0[hit]
    h0 = skin_before[idx] + inward[idx] * t0[:, None]
    s, R, t = skull_sim
    h1 = apply_similarity(h0, s, R, t)
    n1 = geo.vertex_normals(skin_after, skin_faces)[idx]
    in1 = n1 * np.sign((n1 * inward[idx]).sum(axis=1))[:, None]       # oriented like the template's
    t1 = ((h1 - skin_after[idx]) * in1).sum(axis=1)
    ratio = t1 / t0
    return {"n_rays": int(len(idx)), "template_mm_median": float(np.median(t0)),
            "new_mm_median": float(np.median(t1)), "ratio_median": float(np.median(ratio)),
            "n_negative": int((t1 < 0).sum()), "n_below_30pct": int((ratio < 0.3).sum()),
            "ratio_p05": float(np.percentile(ratio, 5)), "ratio_p95": float(np.percentile(ratio, 95))}

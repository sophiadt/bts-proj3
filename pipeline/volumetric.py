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
    of muscle triangles, made the speech animation 5.6x faster than the
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
    return (s, R, t), info


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

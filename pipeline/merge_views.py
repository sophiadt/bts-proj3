"""Bring every view of a capture into one fit frame.

Each view is rotated into template axes (capture.to_template) and rigidly
aligned to the template on its own landmarks (rigid.align); the template's
uniform scale comes from the primary (front) view. Turned views are then
refined against the front's dense cloud by trimmed point-to-point ICP, since
the subject cannot be trusted to hold still between shots.

Landmark targets come from the primary view; slots it lacks (null or
unreliable depth) are filled from turned views when both agree.
"""

import numpy as np
from scipy.spatial import cKDTree

from pipeline import capture as cap
from pipeline import nonrigid, rigid

ICP_ITERS = 30
ICP_TRIM_MM = 4.0
FILL_AGREE_MM = 4.0


def icp(src, dst, iters=ICP_ITERS, trim_mm=ICP_TRIM_MM):
    """Rigid ICP of src onto dst. -> (R, t, rms_mm, overlap_fraction)."""
    tree = cKDTree(dst)
    R, t = np.eye(3), np.zeros(3)
    cur = src.copy()
    rms, frac = np.inf, 0.0
    for k in range(iters):
        d, j = tree.query(cur)
        thr = max(trim_mm, np.percentile(d, 50)) if k < 3 else trim_mm
        ok = d < thr
        if ok.sum() < 100:
            break
        dR, dt = rigid.rigid_transform(cur[ok], dst[j[ok]])
        cur = cur @ dR.T + dt
        R, t = dR @ R, dR @ t + dt
        rms, frac = float(np.sqrt(np.mean(d[ok] ** 2))), float(ok.mean())
    return R, t, rms, frac


def merge(views, fixture, template_skin, stride=2):
    """-> dict(scale, landmarks (478,3) fit frame, valid (478,), cloud (M,3),
    cloud_view (M,), per_view {...})."""
    prim = cap.primary(views)
    per = {}
    clouds, labels = [], []
    lm_fit = {}
    for name, v in views.items():
        lm_t = cap.to_template(v.landmarks)
        cap.check_upright(lm_t)
        al = rigid.align(template_skin, fixture, lm_t, v.valid)
        lm_fit[name] = rigid.to_fit_frame(lm_t, al["R"], al["t"])
        cl = rigid.to_fit_frame(cap.to_template(v.cloud(stride=stride)), al["R"], al["t"])
        per[name] = {"scale": al["scale"], "rms_used_mm": al["rms_used_mm"], "median_used_mm": al["median_used_mm"],
                     "max_used_mm": al["max_used_mm"], "n_used": len(al["used"]), "rejected": al["rejected"],
                     "residual_mm": al["residual_mm"], "n_cloud": len(cl),
                     # the camera (origin of its own frame) in the fit frame
                     "camera_centre": rigid.to_fit_frame(np.zeros((1, 3)), al["R"], al["t"])[0]}
        clouds.append(cl)
        labels.append(name)

    base = clouds[labels.index(prim.name)]
    for k, name in enumerate(labels):
        if name == prim.name or not len(clouds[k]) or not len(base):
            continue
        R, t, rms, frac = icp(clouds[k], base)
        clouds[k] = clouds[k] @ R.T + t
        lm_fit[name] = lm_fit[name] @ R.T + t
        per[name]["camera_centre"] = per[name]["camera_centre"] @ R.T + t
        per[name]["icp_rms_mm"], per[name]["icp_overlap"] = rms, frac

    lm = lm_fit[prim.name].copy()
    valid = prim.valid.copy()
    filled = []
    for name, v in views.items():
        if name == prim.name:
            continue
        cand = ~valid & v.valid
        for i in np.flatnonzero(cand):
            others = [lm_fit[o][i] for o, ov in views.items() if o != prim.name and ov.valid[i]]
            if len(others) >= 2 and np.linalg.norm(others[0] - others[1]) > FILL_AGREE_MM:
                continue
            lm[i] = np.mean(others, axis=0)
            valid[i] = True
            filled.append(int(i))
    cloud = np.concatenate(clouds) if clouds else np.zeros((0, 3))
    view_of = np.concatenate([np.full(len(c), k) for k, c in enumerate(clouds)]) if clouds else np.zeros(0, int)
    # Normals per view (a view's points only), each oriented toward its camera.
    normals = np.zeros_like(cloud)
    for k, name in enumerate(labels):
        sel = view_of == k
        if sel.sum() >= 3:
            normals[sel] = nonrigid.orient_toward(cloud[sel], nonrigid.cloud_normals(cloud[sel]),
                                                  per[name]["camera_centre"])
    return {"scale": per[prim.name]["scale"], "landmarks": lm, "valid": valid, "cloud": cloud,
            "cloud_normals": normals, "cloud_view": view_of, "view_names": labels, "per_view": per,
            "filled": sorted(set(filled)), "primary": prim.name}

"""
Morph a COPY of the lab skin toward the phone scan, in adjustable steps.   (Maya Script Editor > Python tab)

What it does
  The registration (scan_to_maya_v4_2) put 468 scan landmarks onto the lab model. For each one it knows
  where the lab skin is and where the scan's face surface is. This script pushes the lab skin in or out
  along its own surface normal to meet the scan at those points, and blends the movement smoothly over the
  nearby skin. Only a copy is changed ("scan_fitted_face"); your original skin is hidden, not touched.

  It is a landmark-only morph (no dense depth), so it shows how the lab face would change shape toward
  yours. It is not the Ziva-workstation fit, and it does not move muscles or bones.

How to use
  1. Run scan_to_maya_v4_2.py on the scan first (it saves registration.json + landmarks_on_model.json).
  2. Run this file. It asks for those two files and makes the morphed copy at STEP 0.5.
  3. Change the step any time (0 = lab skin, 1 = full morph) by typing, in the Python tab:
         scan_fit_set(0.25)      scan_fit_set(0.75)      scan_fit_set(1.0)
  4. Back to the original skin:
         scan_fit_off()

Dense depth (uses the TrueDepth point cloud too):
         scan_fit_dense(1.0, 1.0)     # (landmark step, depth step)
  After the landmark morph it pulls the skin onto the phone's dense depth surface (front_face_crop.ply in the
  scan folder), inside the face outline only, and prints how far the skin is from the scan before and after.

Two modes (needs scan_to_maya_v4_3.py for "proportions"):
  "proportions"  aims at the scan's REAL proportions (bigger change, like the Ziva-workstation fit; can tear
                 around the eyes and lips at high steps, so go up in steps)
  "gentle"       only pushes along the skin normal toward the scan after it was stretched to the lab's shape
Switch with:  scan_fit_mode("gentle")   or   scan_fit_mode("proportions")
"""
import json
import os

import numpy as np

SKIN_MESH = "skin_cloth_copy_v5_pull_back_RENDER"
ALPHA = 0.5            # how far toward the scan to go: 0 = lab skin, 1 = all the way
MODE = "proportions"   # or "gentle"
SIGMA_PROP = 20.0      # spread for the proportions mode (Maya units, about 12 mm)
MAX_FULL = 70.0        # ignore pushes bigger than this in proportions mode (about 40 mm): bad depth
SIGMA = 14.0           # Maya units each landmark's push spreads (about 8 mm)
FAR = 0.5              # skin far from every landmark fades to no movement
MAX_MOVE = 25.0        # ignore pushes bigger than this (Maya units, about 15 mm): bad depth at the edge
LIP_SPLIT = True
SPLIT_BAND = 80.0
CORNER_FADE = 20.0
IRIS_FROM = 468
MOUTH_UP, MOUTH_LOW, CORNER_L, CORNER_R = 13, 14, 61, 291
MM_PER_UNIT = 1.0 / 1.685


def build_weights(base, lm_xyz, mouth_y, x_lo, x_hi, sigma=SIGMA, far=FAR, split=LIP_SPLIT,
                  band=SPLIT_BAND, fade=CORNER_FADE):
    """(V,3) skin vertices, (K,3) landmark positions -> (V,K) float32 weights (upper/lower lip kept apart)."""
    base = np.asarray(base, float)
    lm = np.asarray(lm_xyz, float)
    d2 = (base ** 2).sum(1)[:, None] + (lm ** 2).sum(1)[None, :] - 2.0 * base @ lm.T
    np.maximum(d2, 0.0, out=d2)
    G = np.exp(-d2 / (2.0 * sigma * sigma))
    if split:
        sv = base[:, 1] - mouth_y
        sl = lm[:, 1] - mouth_y
        same = (sv[:, None] * sl[None, :]) > 0
        same |= (np.abs(sl) < 3.0)[None, :]
        out_x = np.maximum(np.maximum(x_lo - base[:, 0], base[:, 0] - x_hi), 0.0)
        blend = np.clip(1.0 - out_x / fade, 0.0, 1.0)
        blend = np.where(np.abs(sv) < band, blend, 0.0)
        G = G * (blend[:, None] * same + (1.0 - blend)[:, None])
    return (G / (G.sum(1, keepdims=True) + far)).astype(np.float32)


def normal_pushes(src, dst, base, normals, max_move=MAX_MOVE):
    """For each landmark: how far (and which way) to push the lab skin so it meets the scan point.
    src = point on the lab skin, dst = the scan's point. Only the part along the skin normal is kept,
    so sliding along the surface (snapping noise, lids, lips) never drags the skin sideways.
    Returns (K,3) push vectors and a (K,) bool of the ones used."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    d2 = (src ** 2).sum(1)[:, None] + (base ** 2).sum(1)[None, :] - 2.0 * src @ base.T
    nearest = np.argmin(d2, axis=1)
    n = normals[nearest]
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    along = ((dst - src) * n).sum(1)                       # signed distance along the normal
    push = n * along[:, None]
    ok = np.abs(along) <= max_move
    push[~ok] = 0.0
    return push, ok


def full_pushes(src, dst, max_move=MAX_FULL):
    """Proportions mode: move each lab-skin point all the way to the scan's point (any direction).
    Pushes bigger than max_move are treated as bad depth and ignored."""
    push = np.asarray(dst, float) - np.asarray(src, float)
    ok = np.linalg.norm(push, axis=1) <= max_move
    push[~ok] = 0.0
    return push, ok


OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]     # MediaPipe face outline
DENSE_RADIUS = 20.0     # Maya units (about 12 mm): only scan points this close to the skin can pull on it
DENSE_MAX = 16.0        # biggest pull per skin point (about 9 mm)
SMOOTH_ITERS = 20       # how much the pull is smoothed over the neighbouring skin


def read_ply(path):
    """ASCII ply from the app (x y z per line) -> (N,3) float array in mm."""
    with open(path) as f:
        n = 0
        for line in f:
            if line.startswith("element vertex"):
                n = int(line.split()[2])
            if line.strip() == "end_header":
                break
        return np.loadtxt(f, max_rows=n, dtype=np.float64).reshape(-1, 3)


def inside_polygon(xy, poly):
    """Even-odd test of (N,2) points against an (M,2) polygon."""
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi, xj, yj = poly[i][0], poly[i][1], poly[j][0], poly[j][1]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


def face_cloud(cloud, lm_cam, shrink=0.06):
    """Keep only scan points inside the face outline (drops hair, neck, hoodie)."""
    zs = [p[2] for p in lm_cam if p is not None]
    zmed = float(np.median(zs))
    pts = [lm_cam[i] for i in OVAL if i < len(lm_cam) and lm_cam[i] is not None and abs(lm_cam[i][2] - zmed) < 60]
    poly = np.array([[p[0], p[1]] for p in pts], float)
    c = poly.mean(0)
    poly = c + (poly - c) * (1.0 - shrink)
    keep = inside_polygon(cloud[:, :2], poly) & (np.abs(cloud[:, 2] - zmed) < 90)
    return cloud[keep]


def nearest(P, C, chunk=700):
    """For each row of P the nearest row of C: (index, distance). Plain numpy, in chunks."""
    P = np.asarray(P, np.float32)
    C = np.asarray(C, np.float32)
    c2 = (C ** 2).sum(1)
    idx = np.empty(len(P), np.int64)
    dist = np.empty(len(P), np.float32)
    for a in range(0, len(P), chunk):
        b = P[a:a + chunk]
        d2 = (b ** 2).sum(1)[:, None] + c2[None, :] - 2.0 * b @ C.T
        k = np.argmin(d2, axis=1)
        idx[a:a + chunk] = k
        dist[a:a + chunk] = np.sqrt(np.maximum(d2[np.arange(len(b)), k], 0.0))
    return idx, dist


def smooth_field(x, e0, e1, n, iters=SMOOTH_ITERS):
    """Average each skin vertex with its mesh neighbours, iters times."""
    x = np.asarray(x, np.float64).copy()
    deg = np.maximum(np.bincount(e0, minlength=n) + np.bincount(e1, minlength=n), 1).astype(np.float64)
    for _ in range(iters):
        acc = np.zeros_like(x)
        np.add.at(acc, e0, x[e1])
        np.add.at(acc, e1, x[e0])
        x = 0.5 * x + 0.5 * acc / deg[:, None]
    return x


def dense_push(P, normals, C, e0, e1, radius=DENSE_RADIUS, max_push=DENSE_MAX, iters=SMOOTH_ITERS):
    """Pull the skin points P onto the scan cloud C along the skin normal, smoothly.
    Returns (V,3) offsets and the (V,) bool of skin points that had a scan point nearby."""
    idx, dist = nearest(P, C)
    q = C[idx]
    along = ((q - P) * normals).sum(1)
    ok = (dist <= radius) & (np.abs(along) <= max_push)
    push = normals * np.where(ok, along, 0.0)[:, None]
    w = ok.astype(np.float64)[:, None]
    both = smooth_field(np.hstack([push, w]), e0, e1, len(P), iters)
    return both[:, :3].astype(np.float32), ok


def morph_delta(W, push, alpha):
    """(V,K) weights, (K,3) pushes -> (V,3) vertex offsets."""
    return (alpha * (W @ push.astype(np.float32))).astype(np.float32)


# ---------------------------------------------------------------- Maya side
_STATE = {}


def scan_fit_set(alpha):
    import maya.api.OpenMaya as om
    s = _STATE
    if "fn" not in s:
        print("Run the script first (no morphed copy yet).")
        return
    s["alpha"] = float(alpha)
    delta = morph_delta(s["W"], s["push"], float(alpha))
    pts = s["base"] + delta
    s["fn"].setPoints(om.MPointArray([om.MPoint(float(a), float(b), float(c)) for a, b, c in pts]), om.MSpace.kWorld)
    s["fn"].updateSurface()
    mv = np.linalg.norm(delta, axis=1) * MM_PER_UNIT
    print("step %.2f: skin moved %.1f mm on average, %.1f mm at most (%.1f mm at the 95th percentile)"
          % (float(alpha), mv.mean(), mv.max(), np.percentile(mv, 95)))


def scan_fit_mode(mode):
    """Switch between "proportions" (bigger change) and "gentle", then redraw at the current step."""
    s = _STATE
    if "fn" not in s:
        print("Run the script first.")
        return
    if mode == "proportions":
        if s.get("dst_rigid") is None:
            print("No real-proportions data: run scan_to_maya_v4_3.py on the scan, then run this script again.")
            return
        push, ok = full_pushes(s["src"], s["dst_rigid"])
        W = s["W_prop"]
    else:
        push, ok = normal_pushes(s["src"], s["dst_warped"], s["base"].astype(float), s["normals"])
        W = s["W_gentle"]
    s.update(push=push, W=W, mode=mode)
    mm = np.linalg.norm(push[ok], axis=1) * MM_PER_UNIT
    print("mode %s: %d landmarks used (%d ignored); typical push %.1f mm, largest %.1f mm"
          % (mode, int(ok.sum()), int((~ok).sum()), float(np.median(mm)), float(mm.max())))
    scan_fit_set(s.get("alpha", ALPHA))


def scan_fit_dense(step1=1.0, step2=1.0):
    """Landmark morph at step1, then pull the skin onto the dense TrueDepth surface at step2."""
    import maya.api.OpenMaya as om
    s = _STATE
    if "fn" not in s:
        print("Run the script first.")
        return
    if "cloud_model" not in s:
        folder = s["folder"]
        ply = os.path.join(folder, "front_face_crop.ply")
        lmj = os.path.join(folder, "front_landmarks_3d_mm.json")
        if not (os.path.exists(ply) and os.path.exists(lmj)):
            print("Need front_face_crop.ply and front_landmarks_3d_mm.json next to registration.json (%s)." % folder)
            return
        cloud = face_cloud(read_ply(ply), json.load(open(lmj)))
        reg = s["reg"]
        R = np.array(reg["R"], float)
        t = np.array(reg["t"], float)
        sc = float(reg["scale"])
        s["cloud_model"] = (sc * (R @ cloud.T)).T + t
        print("dense scan: %d points inside the face outline" % len(cloud))
    C = s["cloud_model"]
    if s.get("dst_rigid") is not None and s.get("mode") != "proportions":
        scan_fit_mode("proportions")
    delta1 = morph_delta(s["W"], s["push"], float(step1))
    P1 = s["base"].astype(np.float64) + delta1
    push, ok = dense_push(P1, s["normals"], C, s["e0"], s["e1"])
    near = nearest(P1, C)[1]
    sm = ok & (near <= 10.0)                       # skin points that have the scan right next to them
    d_before = near[sm] * MM_PER_UNIT
    P2 = P1 + float(step2) * push
    d_after = nearest(P2[sm], C)[1] * MM_PER_UNIT
    s["fn"].setPoints(om.MPointArray([om.MPoint(float(a), float(b), float(c)) for a, b, c in P2]), om.MSpace.kWorld)
    s["fn"].updateSurface()
    mv = np.linalg.norm(P2 - s["base"], axis=1) * MM_PER_UNIT
    print("dense fit (landmarks %.2f, depth %.2f): skin-to-scan distance on %d skin points: %.2f mm before the depth step, %.2f mm after "
          "(95%% under %.2f mm). Skin moved %.1f mm on average, %.1f mm at most."
          % (step1, step2, int(sm.sum()), float(d_before.mean()), float(d_after.mean()), float(np.percentile(d_after, 95)),
             float(mv.mean()), float(mv.max())))


def scan_fit_off():
    import maya.cmds as cmds
    if cmds.objExists("scan_fitted_face"):
        cmds.delete("scan_fitted_face")
    if cmds.objExists(SKIN_MESH):
        cmds.setAttr(SKIN_MESH + ".visibility", 1)
    _STATE.clear()
    print("Morph off. Original skin is visible again.")


def _find(cmds, name, hints, caption):
    for folder in hints:
        p = os.path.join(folder, name)
        if folder and os.path.exists(p):
            return p
    pick = cmds.fileDialog2(fileMode=1, caption=caption, fileFilter="JSON (*.json)")
    return pick[0] if pick else None


def run():
    import maya.cmds as cmds
    import maya.api.OpenMaya as om

    home = os.path.expanduser("~")
    hints = [os.path.join(home, "Downloads", "Burst Mode Captures"), os.path.join(home, "Downloads")]
    reg_path = _find(cmds, "registration.json", hints, "Choose registration.json (from scan_to_maya_v4_2)")
    if not reg_path:
        return
    pos_path = _find(cmds, "landmarks_on_model.json", [os.path.dirname(reg_path)] + hints,
                     "Choose landmarks_on_model.json")
    if not pos_path:
        return
    reg = json.load(open(reg_path))
    warped = (reg.get("stats") or {}).get("warped")
    if not warped:
        print("registration.json has no scan shape in it. Run scan_to_maya_v4_2.py on the scan first, then this.")
        return
    snapped = json.load(open(pos_path))
    ids = [i for i in sorted(int(k) for k in snapped) if str(i) in warped and i < IRIS_FROM]
    for need in (MOUTH_UP, MOUTH_LOW, CORNER_L, CORNER_R):
        if need not in ids:
            print("landmark %d is missing; cannot find the lip line" % need)
            return
    rigid = (reg.get("stats") or {}).get("rigid")
    src = np.array([snapped[str(i)] for i in ids], float)
    dst = np.array([warped[str(i)] for i in ids], float)
    dst_rigid = np.array([rigid[str(i)] for i in ids], float) if rigid and all(str(i) in rigid for i in ids) else None
    at = {i: k for k, i in enumerate(ids)}
    mouth_y = 0.5 * (src[at[MOUTH_UP]][1] + src[at[MOUTH_LOW]][1])
    xs = [src[at[CORNER_L]][0], src[at[CORNER_R]][0]]

    scan_fit_off()
    dup = cmds.duplicate(SKIN_MESH, name="scan_fitted_face", returnRootsOnly=True)[0]
    if cmds.listRelatives(dup, parent=True):
        dup = cmds.parent(dup, world=True)[0]
    cmds.setAttr(SKIN_MESH + ".visibility", 0)
    sel = om.MSelectionList()
    sel.add(dup)
    dag = sel.getDagPath(0)
    dag.extendToShape()
    fn = om.MFnMesh(dag)
    base = np.array([[p.x, p.y, p.z] for p in fn.getPoints(om.MSpace.kWorld)], float)
    normals = np.array([[v.x, v.y, v.z] for v in fn.getVertexNormals(False, om.MSpace.kWorld)], float)

    W_gentle = build_weights(base, src, mouth_y, min(xs), max(xs))
    W_prop = build_weights(base, src, mouth_y, min(xs), max(xs), sigma=SIGMA_PROP)
    counts, tri = fn.getTriangles()
    tri = np.array(tri, np.int64).reshape(-1, 3)
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    _STATE.update(fn=fn, base=base.astype(np.float32), normals=normals, src=src, dst_warped=dst,
                  dst_rigid=dst_rigid, W_gentle=W_gentle, W_prop=W_prop, alpha=ALPHA,
                  e0=e[:, 0], e1=e[:, 1], reg=reg, folder=os.path.dirname(reg_path))
    mode = MODE if (MODE == "gentle" or dst_rigid is not None) else "gentle"
    if mode != MODE:
        print("(no real-proportions data in registration.json: starting in gentle mode. Run scan_to_maya_v4_3.py for more.)")
    scan_fit_mode(mode)
    print("Done. 'scan_fitted_face' is the morphed copy. Step: scan_fit_set(0.25) ... scan_fit_set(1.0). "
          "Mode: scan_fit_mode('gentle') / scan_fit_mode('proportions'). Back to the original: scan_fit_off().")


if __name__ == "__main__":
    try:
        run()
    except ImportError:
        print("Run this inside Maya.")

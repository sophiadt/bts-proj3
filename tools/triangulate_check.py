"""Does multi-view triangulation agree with TrueDepth at the eyelids and lips?  (numpy + scipy)

    python tools/triangulate_check.py s01_20261110T212937Z_burst
    python tools/triangulate_check.py s01_20261030T170049Z_guided

The front view's landmarks get their 3D position from the depth map. The same
landmarks can also be triangulated from two views, using only their 2D pixels
and the camera poses, so the result does not read depth at the landmark at all.
For each turned view this tool

  1. registers it onto the front view (landmark Procrustes, then trimmed ICP on the
     face clouds), which gives that view's camera pose in the front camera frame;
  2. keeps the landmarks that face the turned camera (surface normal from the front
     cloud) and lie inside its image;
  3. intersects each front ray with the turned view's ray (midpoint of closest approach).

Per region it prints how far the triangulated point is from the depth-based one
along the viewing axis. Ordinary skin (cheeks, nose, brow) shows the triangulation's
own noise, because depth is reliable there; the eyes and lips only mean something
when they differ from that baseline.

It is a cross-check, not a replacement: both estimates carry about a millimetre of
noise, and it cannot say which one is right without a ground-truth scan.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import capture as cap                   # noqa: E402
from pipeline import merge_views, rigid               # noqa: E402

REGIONS = {
    "eye (L+R)": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                  362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    "inner lips": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],
    "outer lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
    "control skin": [50, 101, 118, 117, 123, 205, 280, 330, 347, 346, 352, 425, 1, 2, 4, 5, 6, 19, 94, 98,
                     97, 168, 195, 197, 9, 8, 107, 66, 105, 63, 70, 336, 296, 334, 293, 300],
}
FACING_MIN = 0.35        # cosine between the surface normal and the direction to the turned camera
GAP_MAX_MM = 3.0         # rays that miss each other by more than this are inconsistent
MIN_ANGLE_DEG = 12.0     # weaker convergence than this triangulates depth badly


def register(front, turned):
    """-> (R, t, icp_rms_mm): the turned camera frame onto the front camera frame."""
    both = front.valid & turned.valid
    src, dst = turned.landmarks[both], front.landmarks[both]
    keep = np.ones(len(src), bool)
    for _ in range(4):                                   # robust Procrustes: drop what the fit rejects
        R, t = rigid.rigid_transform(src[keep], dst[keep])
        res = np.linalg.norm(src @ R.T + t - dst, axis=1)
        keep = res < max(8.0, 2.5 * np.median(res))
        if keep.sum() < 12:
            break
    cl_f, cl_t = front.cloud(stride=2), turned.cloud(stride=2)
    cur = cl_t @ R.T + t
    dR, dt, rms, _frac = merge_views.icp(cur, cl_f)
    return dR @ R, dR @ t + dt, rms


def normals_at(points, cloud_tree, cloud, k=40):
    """Unit normals at `points` from the front cloud, oriented toward the front camera (origin)."""
    out = np.full((len(points), 3), np.nan)
    for i, p in enumerate(points):
        if not np.isfinite(p).all():
            continue
        _d, j = cloud_tree.query(p, k=k)
        q = cloud[j] - cloud[j].mean(0)
        n = np.linalg.svd(q, full_matrices=False)[2][-1]
        out[i] = n if np.dot(n, -p) > 0 else -n
    return out


def ray(view, idx):
    """Unit ray directions (camera frame) through the landmarks' pixels, from their 3D points."""
    uv = view.project(view.landmarks[idx])
    k = view.K
    d = np.stack([(uv[:, 0] - k["cx"]) / k["fx"], (uv[:, 1] - k["cy"]) / k["fy"], np.ones(len(uv))], axis=1)
    return d / np.linalg.norm(d, axis=1, keepdims=True), uv


def triangulate(o1, d1, o2, d2):
    """Midpoint of closest approach of two rays. -> (point, gap_mm, angle_deg)."""
    w0 = o1 - o2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w0, d2 @ w0
    den = a * c - b * b
    if abs(den) < 1e-9:
        return None, np.inf, 0.0
    s, u = (b * e - c * d) / den, (a * e - b * d) / den
    p1, p2 = o1 + s * d1, o2 + u * d2
    return 0.5 * (p1 + p2), float(np.linalg.norm(p1 - p2)), float(np.degrees(np.arccos(np.clip(b, -1, 1))))


def run(arg):
    folder = cap.resolve_capture(arg)
    views = cap.load_capture(folder)
    front = views["front"]
    cl_f = front.cloud(stride=2)
    tree = cKDTree(cl_f)
    n_front = normals_at(front.landmarks, tree, cl_f)
    h, w = front.depth.shape
    est = {}                                            # landmark -> list of (point, gap, angle)
    info = []
    for name in ("left", "right"):
        if name not in views:
            continue
        tv = views[name]
        R, t, rms = register(front, tv)
        cam = t.copy()                                  # turned camera centre in the front frame
        ok_idx = np.flatnonzero(np.isfinite(front.landmarks).all(1) & np.isfinite(tv.landmarks).all(1))
        d_f, uv_f = ray(front, ok_idx)
        d_t, uv_t = ray(tv, ok_idx)
        d_t = d_t @ R.T                                 # into the front frame
        th, tw = tv.depth.shape
        n_used = 0
        for k, i in enumerate(ok_idx):
            if not (0 <= uv_t[k, 0] < tw and 0 <= uv_t[k, 1] < th):
                continue
            p = front.landmarks[i]
            if not np.isfinite(n_front[i]).all():
                continue
            to_cam = (cam - p) / np.linalg.norm(cam - p)
            if n_front[i] @ to_cam < FACING_MIN:
                continue
            pt, gap, ang = triangulate(np.zeros(3), d_f[k], cam, d_t[k])
            if pt is None or gap > GAP_MAX_MM or ang < MIN_ANGLE_DEG:
                continue
            est.setdefault(i, []).append((pt, gap, ang, name))
            n_used += 1
        info.append("{0}: ICP rms {1:.2f} mm, camera at ({2:.0f}, {3:.0f}, {4:.0f}) mm, {5} landmarks triangulated"
                    .format(name, rms, *cam, n_used))
    print("\n" + cap.capture_id(folder))
    for line in info:
        print("  " + line)
    print("  %-14s %5s %22s %22s %9s" % ("region", "n", "tri - depth, along z", "tri - depth, 3D dist", "ray gap"))
    print("  %-14s %5s %22s %22s %9s" % ("", "", "median (MAD) mm", "median mm", "median mm"))
    for name, ids in REGIONS.items():
        dz, d3, gp = [], [], []
        for i in ids:
            for pt, gap, ang, _v in est.get(i, []):
                diff = pt - front.landmarks[i]
                dz.append(diff[2]); d3.append(np.linalg.norm(diff)); gp.append(gap)
        if len(dz) < 3:
            print("  %-14s %5d   (too few visible in both views)" % (name, len(dz)))
            continue
        dz = np.array(dz)
        mad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
        print("  %-14s %5d %12.2f (%5.2f) %22.2f %9.2f" % (name, len(dz), np.median(dz), mad, np.median(d3), np.median(gp)))
    return est


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for a in sys.argv[1:]:
        run(a)

"""Non-rigid ICP: deform a mesh onto landmarks and a point cloud.

Unknown: new positions X (N,3) of the source mesh, same vertex order. Each
stage solves one sparse linear least-squares problem

    alpha * ||L (X - X0)||^2                     keep the template's shape
  + sum_l  w_l ||X[idx_l] - q_l||^2              landmarks
  + beta  * sum_i ||X_i - c_i||^2                closest cloud point, if accepted

with L the graph Laplacian, so the regulariser penalises change of local
shape, not position: a uniform scale or translation costs nothing locally,
and regions the camera never saw are carried along smoothly. alpha runs a
stiff-to-soft schedule; correspondences are re-found at every iteration.

A data correspondence is accepted only if the source vertex is not masked
(hole rims, eyelid and lip margins), the cloud point is within `max_dist`,
and the local normals agree to within ~55 degrees. With oriented cloud
normals (`cloud_n` pointing toward the camera that saw each point) the test
is signed: skin that faces away -- the lining that wraps 24 mm into the
mouth, the inside of the nostrils -- is never pulled onto the visible surface.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree

from core import geometry as geo

DEFAULT_SCHEDULE = (100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0)


def cloud_normals(pts, k=12):
    """Unoriented PCA normals of a point cloud (sign is arbitrary)."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return np.zeros_like(pts)
    tree = cKDTree(pts)
    _, nb = tree.query(pts, k=min(k, len(pts)))
    q = pts[nb] - pts[nb].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", q, q)
    _w, v = np.linalg.eigh(cov)
    return v[:, :, 0]


def boundary_points(pts, normals, k=16, ratio=0.35):
    """Points on the edge of a cloud: their neighbours' centroid sits off to one
    side in the tangent plane. Correspondences to these are rejected -- skin the
    camera never saw (ears, under the jaw) would otherwise be dragged onto the
    cloud's rim."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < k:
        return np.zeros(len(pts), dtype=bool)
    d, nb = cKDTree(pts).query(pts, k=k)
    off = pts[nb[:, 1:]].mean(axis=1) - pts
    off -= (off * normals).sum(axis=1, keepdims=True) * normals          # tangent part
    return np.linalg.norm(off, axis=1) > ratio * d[:, 1:].mean(axis=1)


def orient_toward(pts, normals, centres):
    """Flip each normal to face its viewpoint (centres: (3,) or (M,3))."""
    s = np.sign(((np.asarray(centres) - pts) * normals).sum(axis=1))
    s[s == 0] = 1.0
    return normals * s[:, None]


def outward_sign(verts, faces):
    """+1 if the mesh's face winding gives outward normals (away from the
    vertex centroid on average), else -1."""
    vn = geo.vertex_normals(verts, faces)
    return 1.0 if ((verts - verts.mean(axis=0)) * vn).sum(axis=1).mean() >= 0 else -1.0


def fit(src_verts, faces, lm_idx=None, lm_pts=None, lm_w=None, cloud=None, cloud_n=None,
        data_mask=None, schedule=DEFAULT_SCHEDULE, iters_per_stage=3, beta=1.0,
        max_dist=10.0, normal_cos=0.55, oriented=False, eps=1e-6, verbose=False):
    """-> (X (N,3), info dict). All inputs in mm, one frame.

    oriented=True: `cloud_n` must face outward (toward the camera) and the
    source normals are made outward with `outward_sign`; the test is signed."""
    x0 = np.asarray(src_verts, dtype=np.float64)
    n = len(x0)
    faces = np.asarray(faces, dtype=np.int64)
    L = geo.uniform_laplacian(n, faces)
    LtL = (L.T @ L).tocsr()
    rhs_reg = LtL @ x0

    lm_idx = np.asarray([] if lm_idx is None else lm_idx, dtype=np.int64)
    lm_pts = np.asarray(np.zeros((0, 3)) if lm_pts is None else lm_pts, dtype=np.float64)
    lm_w = np.ones(len(lm_idx)) if lm_w is None else np.asarray(lm_w, dtype=np.float64)
    keep = np.isfinite(lm_pts).all(axis=1) & (lm_w > 0)
    lm_idx, lm_pts, lm_w = lm_idx[keep], lm_pts[keep], lm_w[keep]
    d_lm = np.bincount(lm_idx, weights=lm_w, minlength=n)
    rhs_lm = np.zeros((n, 3))
    np.add.at(rhs_lm, lm_idx, lm_w[:, None] * lm_pts)

    has_cloud = cloud is not None and len(cloud) > 0
    if has_cloud:
        cloud = np.asarray(cloud, dtype=np.float64)
        tree = cKDTree(cloud)
        cn = cloud_normals(cloud) if cloud_n is None else np.asarray(cloud_n)
        on_rim = boundary_points(cloud, cn)
    mask = np.ones(n, dtype=bool) if data_mask is None else np.asarray(data_mask, dtype=bool)
    nsign = outward_sign(x0, faces) if oriented else 1.0

    x = x0.copy()
    history = []
    for alpha in schedule:
        for _ in range(iters_per_stage if has_cloud else 1):
            d_data = np.zeros(n)
            rhs_data = np.zeros((n, 3))
            n_acc = 0
            if has_cloud:
                dist, j = tree.query(x, distance_upper_bound=max_dist)
                ok = mask & np.isfinite(dist)
                ok[ok] &= ~on_rim[j[ok]]
                if normal_cos > 0 and ok.any():
                    vn = geo.vertex_normals(x, faces) * nsign
                    jj = np.where(ok, j, 0)
                    dots = (vn * cn[jj]).sum(axis=1)
                    ok &= (dots if oriented else np.abs(dots)) >= normal_cos
                n_acc = int(ok.sum())
                d_data[ok] = beta
                rhs_data[ok] = beta * cloud[j[ok]]
            A = (alpha * LtL + sp.diags(d_lm + d_data + eps)).tocsc()
            solve = factorized(A)
            b = alpha * rhs_reg + rhs_lm + rhs_data + eps * x0
            x = np.column_stack([solve(b[:, k]) for k in range(3)])
            rec = {"alpha": alpha, "accepted": n_acc}
            if len(lm_idx):
                rec["lm_rms_mm"] = float(np.sqrt(np.mean(np.sum((x[lm_idx] - lm_pts) ** 2, axis=1))))
            if has_cloud and n_acc:
                rec["data_rms_mm"] = float(np.sqrt(np.mean(np.sum((x[ok] - cloud[j[ok]]) ** 2, axis=1))))
            history.append(rec)
            if verbose:
                print("   nonrigid", rec)
    return x, {"history": history}


def cloud_to_mesh_distance(cloud, verts, faces, max_dist=20.0):
    """One-sided distance (mm) from each cloud point to the mesh, approximated
    by point-to-plane distance at the nearest vertex (exact enough at ~2 mm
    vertex spacing). Points farther than max_dist return inf (not covered)."""
    tree = cKDTree(verts)
    d, j = tree.query(cloud, distance_upper_bound=max_dist)
    vn = geo.vertex_normals(verts, faces)
    ok = np.isfinite(d)
    out = np.full(len(cloud), np.inf)
    out[ok] = np.abs(((cloud[ok] - verts[j[ok]]) * vn[j[ok]]).sum(axis=1))
    out[ok] = np.minimum(out[ok], d[ok])
    return out

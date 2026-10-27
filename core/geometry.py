"""Mesh geometry the fit needs, numpy/scipy only (runs under mayapy).

Nothing here reorders or welds vertices: every function takes (V,3) vertices
and (F,3) faces and returns per-vertex or per-face arrays in the same order.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra


def edges(faces):
    """Unique undirected edges (E,2), sorted per row."""
    f = np.asarray(faces, dtype=np.int64)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0)


def boundary_vertices(faces):
    f = np.asarray(faces, dtype=np.int64)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    return np.unique(u[c == 1])


def edge_graph(verts, faces):
    """Sparse symmetric graph weighted by edge length (mm), for geodesics."""
    e = edges(faces)
    w = np.linalg.norm(verts[e[:, 0]] - verts[e[:, 1]], axis=1)
    n = len(verts)
    g = sp.coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    return g.tocsr()


def geodesic_from(verts, faces, sources, limit=np.inf, graph=None):
    """Edge-path distance (mm) from the nearest of `sources` to every vertex.

    Paths run along mesh edges, so they never cross a hole: a point on the
    upper lip is far from the lower lip even when they almost touch.
    """
    g = edge_graph(verts, faces) if graph is None else graph
    d = dijkstra(g, directed=False, indices=np.atleast_1d(sources), limit=limit, min_only=True)
    return d


def uniform_laplacian(n_verts, faces):
    """Graph Laplacian L = I - D^-1 A (rows sum to zero)."""
    e = edges(faces)
    a = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n_verts, n_verts)).tocsr()
    deg = np.asarray(a.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    return (sp.identity(n_verts, format="csr") - sp.diags(1.0 / deg) @ a).tocsr()


def face_normals(verts, faces, unit=True):
    v = np.asarray(verts)
    f = np.asarray(faces)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    if unit:
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.where(ln == 0, 1.0, ln)
    return n


def vertex_normals(verts, faces):
    fn = face_normals(verts, faces, unit=False)          # area-weighted
    vn = np.zeros_like(np.asarray(verts, dtype=np.float64))
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.where(ln == 0, 1.0, ln)


def flipped_faces(verts_before, verts_after, faces, min_area_ratio=0.0):
    """Faces whose normal reversed (dot < 0) or collapsed below a fraction of
    their original area. -> boolean (F,)."""
    n0 = face_normals(verts_before, faces, unit=False)
    n1 = face_normals(verts_after, faces, unit=False)
    flip = (n0 * n1).sum(axis=1) < 0
    if min_area_ratio > 0:
        a0 = np.linalg.norm(n0, axis=1)
        a1 = np.linalg.norm(n1, axis=1)
        flip |= a1 < min_area_ratio * np.where(a0 == 0, 1.0, a0)
    return flip


def ray_hit_distance(origins, dirs, verts, faces, max_dist=200.0, chunk=16):
    """Distance along each ray to the first triangle hit (Moller-Trumbore), inf
    if none within max_dist. Brute force over a bounding-box prefilter; fine for
    a few thousand rays against a skull."""
    o = np.asarray(origins, dtype=np.float64)
    d = np.asarray(dirs, dtype=np.float64)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    out = np.full(len(o), np.inf)
    for s in range(0, len(o), chunk):
        oo, dd = o[s:s + chunk, None, :], d[s:s + chunk, None, :]
        p = np.cross(dd, e2[None])
        det = (e1[None] * p).sum(-1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = oo - v0[None]
        u = (tv * p).sum(-1) * inv
        q = np.cross(tv, e1[None])
        vv = (dd * q).sum(-1) * inv
        t = (e2[None] * q).sum(-1) * inv
        hit = ok & (u >= 0) & (vv >= 0) & (u + vv <= 1) & (t > 1e-6) & (t < max_dist)
        t = np.where(hit, t, np.inf)
        out[s:s + chunk] = t.min(axis=1)
    return out

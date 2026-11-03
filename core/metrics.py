"""Geometry metrics shared by the checks and the fit.

`boundary_loops` and `enclosed_volume` are vendored from
bts-proj1/analysis/metrics.py at commit 7ed50fb (MIT, Sophia Don Tranho), where
they are tested against analytic answers. Copied rather than imported so this
repo does not depend on bts-proj1's layout.
"""

import numpy as np


def boundary_loops(faces):
    """Ordered vertex loops bounding an open surface. -> [[vtx, ...], ...]

    An edge used by exactly one face is a boundary edge; those edges are walked
    into cycles. A closed surface returns []. Loops come back longest-first so
    callers can pick the dominant hole deterministically.
    """
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    bnd = uniq[cnt == 1]
    if len(bnd) == 0:
        return []

    adj = {}
    for a, b in bnd:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))

    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        cur = start
        while True:
            nxt = [n for n in adj[cur] if n not in seen]
            if not nxt:
                break
            cur = nxt[0]
            seen.add(cur)
            loop.append(cur)
        loops.append(loop)
    return sorted(loops, key=len, reverse=True)


def enclosed_volume(verts, faces):
    """Volume enclosed by a closed triangulated surface, in mm^3.

    Divergence theorem: sum(dot(v0, cross(v1, v2))) / 6.

    Signed on purpose. A negative result means inward-facing winding, which is a
    diagnostic worth seeing rather than hiding behind abs(). Valid for any closed
    surface regardless of genus or component count -- which matters here, since
    the lips are a torus and each teeth mesh is 16 separate shells.
    """
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(v) == 0 or len(f) == 0:
        return 0.0
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)

"""Where the skin fit needs more control: an importance map and Poisson-disk
anchors on the template skin.

Importance per skin vertex, both normalised to their 99th percentile:
  motion  how far the vertex moves when the lab head speaks: the generic
          head playing "not even you" through the lab's keyboard UI
          (rig/ui_smoke_test.py saves it), measured from the neutral frame
          14 so the first frames' settling does not count
  crease  surface curvature, |L x| with the graph Laplacian: nose, the
          nasolabial folds, the lip line
  importance = MOTION_W * motion + CREASE_W * crease

Anchors follow the lab's own joint placement (a3-stochastic-V1/b98-sampling-
joint/d2-possion-sampling.py: deformation-weighted Poisson-disk sampling, with
the radius found by binary search): vertices are drawn with probability
proportional to importance and kept only if no anchor is within the radius,
so anchors are dense where the skin moves and creases, sparse elsewhere.
Hole rims are excluded (pipeline/holes.py aligns the openings).

    mayapy pipeline/importance.py [--motion PATH.npz] [--n 600]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import geometry as geo                       # noqa: E402
from core import io as cio                             # noqa: E402
from pipeline import fixture as fx                     # noqa: E402

MOTION_W, CREASE_W, JOINT_W = 0.4, 0.3, 0.3
UI_NEUTRAL_FRAME = 14
N_ANCHORS = 600
R_MIN_MM, R_MAX_MM = 3.0, 12.0     # anchor spacing at the most / least important skin
OUTER_RIM_MM = 15.0                # no anchors this close to the skin's outer edge (never seen)
UI_JOINT_PREFIXES = ("oor_", "ooc_l_", "ooc_r_", "frontalis_")
IMPORTANCE_NPZ = "skin_importance.npz"


def _norm(x):
    p = np.percentile(x, 99)
    return np.clip(x / (p if p > 0 else 1.0), 0, 1)


def crease_map(V, F):
    """Curvature-like crease strength: the normal part of the graph
    Laplacian over the squared local edge length. The tangential part only
    reflects uneven triangle spacing (it lit up mesh edge-loops), so it is
    dropped."""
    L = geo.uniform_laplacian(len(V), F)
    n = geo.vertex_normals(V, F)
    e = geo.edges(F)
    el = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    mean_el = np.bincount(e[:, 0], el, len(V)) + np.bincount(e[:, 1], el, len(V))
    mean_el /= np.maximum(np.bincount(e[:, 0], minlength=len(V)) + np.bincount(e[:, 1], minlength=len(V)), 1)
    c = np.abs(((L @ V) * n).sum(axis=1)) / np.maximum(mean_el, 1e-6) ** 2
    return c - 0.5 * (L @ c)                    # one smoothing pass: half toward the neighbours' mean


def joint_density(V, joints, radius=8.0):
    """How many of the lab's UI control joints (oor/ooc/frontalis, placed by
    deformation-weighted Poisson-disk sampling) lie within `radius` of each
    skin vertex: where the lab already decided motion matters (mouth, eyes,
    brows)."""
    if joints is None or not len(joints):
        return np.zeros(len(V))
    return np.array([len(x) for x in cKDTree(joints).query_ball_point(V, radius)], dtype=np.float64)


def importance_map(V, F, motion_frames, ui_joints=None):
    """-> (importance, motion_mm, crease, joint density) per skin vertex."""
    a = np.asarray(motion_frames, dtype=np.float64)
    base = a[min(UI_NEUTRAL_FRAME, len(a) - 1)]
    motion = np.linalg.norm(a[UI_NEUTRAL_FRAME:] - base, axis=2).max(axis=0)
    crease = crease_map(V, F)
    dens = joint_density(V, ui_joints)
    imp = MOTION_W * _norm(motion) + CREASE_W * _norm(crease) + JOINT_W * _norm(dens)
    return imp / imp.max(), motion, crease, dens


def _greedy(points, order, radius):
    """Accept points in `order` unless an accepted one is closer than the
    mean of the two points' radii (radius: per point)."""
    cell = float(radius.max())
    grid = {}
    keep = []
    for i in order:
        p = points[i]
        key = tuple((p // cell).astype(int))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        r = 0.5 * (radius[i] + radius[j])
                        if np.dot(points[j] - p, points[j] - p) < r * r:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault(key, []).append(i)
            keep.append(i)
    return keep


def poisson_disk(points, weight, n, seed=0, iters=20):
    """Adaptive Poisson-disk sampling in the spirit of the lab's joint
    placement: candidates are tried in a random order biased by weight, and
    each point's exclusion radius shrinks with its weight (R_MIN_MM at the
    most important skin, R_MAX_MM at the least), so anchors are dense where
    the skin moves and creases. A global scale on the radii is found by
    binary search so about n anchors are kept. -> (indices, spacing scale)"""
    rng = np.random.default_rng(seed)
    w = np.asarray(weight, dtype=np.float64)
    cand = np.flatnonzero(w > 0)
    key = rng.random(len(cand)) ** (1.0 / w[cand])        # weighted random order (Efraimidis-Spirakis)
    order = cand[np.argsort(-key)]
    wn = np.clip(w / w[cand].max(), 0, 1)
    base = R_MAX_MM - (R_MAX_MM - R_MIN_MM) * wn
    lo, hi = 0.05, 8.0
    best = (order[:n], 1.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        keep = _greedy(points, order, base * mid)
        if len(keep) >= n:
            best = (np.array(keep[:n]), mid)
            lo = mid
        else:
            hi = mid
    return np.asarray(best[0], dtype=np.int64), float(best[1])


def default_motion_path():
    p = config.template_export_dir() / "anim_ui_generic.npz"
    if p.exists():
        return p
    found = sorted(config.template_export_dir().glob("subjects/*/anim_ui_generic.npz"))
    if not found:
        raise FileNotFoundError("no anim_ui_generic.npz: run rig/ui_smoke_test.py once")
    return found[-1]


def build(motion_path=None, n=N_ANCHORS, seed=0):
    V, F, _ = fx.load_skin()
    motion_path = Path(motion_path or default_motion_path())
    frames = np.load(motion_path)["skin"]
    if frames.shape[1] != len(V):
        raise ValueError("motion has {0} vertices, skin {1}".format(frames.shape[1], len(V)))
    with np.load(config.template_export_dir() / "joints.npz", allow_pickle=False) as z:
        names = [str(s).split("|")[-1] for s in z["names"]]
        ui = np.array([p for nm, p in zip(names, z["positions"]) if nm.startswith(UI_JOINT_PREFIXES)])
    imp, motion, crease, dens = importance_map(V, F, frames, ui)
    _r, rim_dist, loops, _a = fx.skin_regions(V, F)
    outer = geo.geodesic_from(V, F, loops["loops"]["rim"], limit=OUTER_RIM_MM)
    w = imp.copy()
    w[(rim_dist <= fx.HOLE_ADJ_MM) | (outer < OUTER_RIM_MM)] = 0.0
    idx, scale = poisson_disk(V, w, n, seed)
    return {"importance": imp, "motion_mm": motion, "crease": crease, "joint_density": dens,
            "anchors": idx, "radius_mm": scale * np.array([R_MIN_MM, R_MAX_MM]),
            "source": str(motion_path)}, V, F


def load(path=None):
    path = Path(path or config.template_export_dir() / IMPORTANCE_NPZ)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion", default=None)
    ap.add_argument("--n", type=int, default=N_ANCHORS)
    args = ap.parse_args()
    with cio.RunDir("template", "importance", "pipeline/importance.py") as run:
        res, V, F = build(args.motion, args.n)
        out = config.template_export_dir() / IMPORTANCE_NPZ
        np.savez(out, importance=res["importance"], motion_mm=res["motion_mm"], crease=res["crease"],
                 joint_density=res["joint_density"],
                 anchors=res["anchors"], radius_mm=res["radius_mm"], source=res["source"])
        regions, _rd, _l, _a = fx.skin_regions(V, F)
        x = V[res["anchors"]]
        mouth_zone = np.isin(regions[res["anchors"]], ["lip_upper", "lip_lower"]).mean()
        run.metrics.update({"n_anchors": int(len(res["anchors"])), "spacing_mm": res["radius_mm"],
                            "anchor_frac_in_lip_regions": float(mouth_zone),
                            "skin_frac_in_lip_regions": float(np.isin(regions, ["lip_upper", "lip_lower"]).mean()),
                            "motion_max_mm": float(res["motion_mm"].max()), "source": res["source"]})
        print("anchors {0}, spacing {1[0]:.1f}-{1[1]:.1f} mm; {2:.0%} of anchors on the lips vs {3:.0%} of the skin".format(
            len(res["anchors"]), res["radius_mm"], mouth_zone, run.metrics["skin_frac_in_lip_regions"]))
        print("wrote", out)


if __name__ == "__main__":
    main()

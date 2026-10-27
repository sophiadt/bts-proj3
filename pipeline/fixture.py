"""Correspondence fixture: MediaPipe landmark index -> template skin vertex.

Computed once, on the neutral template, and committed as
config/landmark_to_vertex.json. Never regenerated per subject.

1. Skin boundary loops -> rim, eyes, mouth (checks/template_features.py),
   other small loops = nostrils. Eye and mouth loops split at their corners
   into upper/lower chains.
2. Skin regions by geodesic distance (edge paths never cross a hole):
   lip_upper / lip_lower near the mouth, lid_upper_r/l / lid_lower_r/l near
   the eyes, face elsewhere. Hole rims (within RIM_MM) are masked out of every
   search: the holes exclude, they never anchor.
3. Canonical mesh (MediaPipe, human-scale) rotated into the template frame and
   similarity-aligned on anchors (canthi, nose tip, chin; mouth corners light).
4. Non-rigid fit of the canonical mesh onto the skin surface, with lip and
   eyelid margins masked (they ride the regulariser).
5. Each canonical vertex snaps to the nearest unmasked skin vertex in its own
   region (upper lip never lands on the lower lip). Irises (468-477) -> null.

Weights for rigid alignment: high on the nose, chin, jawline, temples and
MediaPipe's own Procrustes basis; low elsewhere; zero on lip and eyelid
contours, hole-adjacent snaps, poor snaps and irises.

    mayapy pipeline/fixture.py [--skin PATH.npz]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from checks.template_features import canthi, classify_loops, loop_diameter, midplane_x  # noqa: E402
from core import geometry as geo                       # noqa: E402
from core import io as cio                             # noqa: E402
from core.metrics import boundary_loops                # noqa: E402
from core.similarity import apply_similarity, similarity_transform  # noqa: E402
from pipeline import canonical as mp                   # noqa: E402
from pipeline import nonrigid                          # noqa: E402

SKIN = "skin_cloth_copy_v5_pull_back"
RIM_MM = 1.5            # skin vertices this close (geodesic) to a hole are never snapped to
HOLE_ADJ_MM = 3.0       # snapped vertex this close to a hole -> weight zero
LIP_ZONE_MM = 18.0
LID_ZONE_MM = 12.0
POOR_SNAP_MM = 6.0

# Canonical frame (Y-up, face +Z, subject's right -X) -> template frame
# (Y-up, face -Z, subject's right +X): 180 degrees about Y.
R_CANON_TO_TEMPLATE = np.diag([-1.0, 1.0, -1.0])

NOSE = [1, 2, 4, 5, 6, 19, 94, 168, 195, 197, 98, 327, 48, 278]
CHIN = [152, 148, 377, 175, 199, 200, 18, 171, 396, 176, 400]
JAW_TEMPLE = [234, 454, 93, 323, 132, 361, 58, 288, 172, 397, 136, 365, 150, 379, 127, 356, 162, 389,
              21, 251, 54, 284, 103, 332]
FOREHEAD = [10, 151, 9, 108, 337, 67, 297, 109, 338]
W_HIGH, W_LOW, W_EYE = 1.0, 0.3, 0.1


def mirror_permutation(verts, tol_mm=3.0):
    """perm[i] = index of i's mirror image across x = 0 (canonical frame)."""
    v = np.asarray(verts)
    d, j = cKDTree(v).query(v * [-1.0, 1.0, 1.0])
    if d.max() > tol_mm:
        raise ValueError("canonical mesh is not mirror-symmetric within {0} mm".format(tol_mm))
    return j


def split_loop(verts, loop, a, b):
    """Split a closed loop at vertices a and b into two chains -> (upper, lower) by mean y."""
    loop = list(loop)
    ia, ib = loop.index(a), loop.index(b)
    if ia > ib:
        ia, ib = ib, ia
    c1 = loop[ia:ib + 1]
    c2 = loop[ib:] + loop[:ia + 1]
    return (c1, c2) if verts[c1][:, 1].mean() > verts[c2][:, 1].mean() else (c2, c1)


def nearest_in(verts, loop, p):
    loop = np.asarray(loop)
    return int(loop[np.argmin(np.linalg.norm(verts[loop] - p, axis=1))])


def skin_regions(V, F):
    """-> (region labels (N,), rim_dist (N,) geodesic mm to nearest hole, loops, anchors)."""
    loops_all = boundary_loops(F)
    loops = classify_loops(V, loops_all)
    named = {id(loops[k]) for k in ("rim", "eye_r", "eye_l", "mouth")}
    nostrils = [L for L in loops_all if id(L) not in named]
    x0 = midplane_x(V, [loops["eye_r"], loops["eye_l"]])
    g = geo.edge_graph(V, F)
    holes = loops["eye_r"] + loops["eye_l"] + loops["mouth"] + [v for L in nostrils for v in L]
    rim_dist = geo.geodesic_from(V, F, holes, graph=g)

    region = np.array(["face"] * len(V), dtype=object)
    anchors = {}
    ch_a, ch_b = loop_diameter(V[loops["mouth"]])
    ch_r, ch_l = (ch_a, ch_b) if ch_a[0] > ch_b[0] else (ch_b, ch_a)
    a, b = nearest_in(V, loops["mouth"], ch_r), nearest_in(V, loops["mouth"], ch_l)
    up, lo = split_loop(V, loops["mouth"], a, b)
    du = geo.geodesic_from(V, F, up, limit=LIP_ZONE_MM, graph=g)
    dl = geo.geodesic_from(V, F, lo, limit=LIP_ZONE_MM, graph=g)
    region[(du < LIP_ZONE_MM) & (du <= dl)] = "lip_upper"
    region[(dl < LIP_ZONE_MM) & (dl < du)] = "lip_lower"
    anchors["ch_r"], anchors["ch_l"] = ch_r, ch_l
    for side in ("r", "l"):
        loop = loops["eye_" + side]
        en, ex = canthi(V[loop], x0)
        up, lo = split_loop(V, loop, nearest_in(V, loop, en), nearest_in(V, loop, ex))
        du = geo.geodesic_from(V, F, up, limit=LID_ZONE_MM, graph=g)
        dl = geo.geodesic_from(V, F, lo, limit=LID_ZONE_MM, graph=g)
        region[(du < LID_ZONE_MM) & (du <= dl)] = "lid_upper_" + side
        region[(dl < LID_ZONE_MM) & (dl < du)] = "lid_lower_" + side
        anchors["en_" + side], anchors["ex_" + side] = en, ex
    mid = np.flatnonzero(np.abs(V[:, 0] - x0) < 3.0)
    anchors["prn"] = V[mid[np.argmin(V[mid, 2])]]
    below = mid[V[mid, 1] < anchors["ch_r"][1] - 15.0]
    anchors["gn"] = V[below[np.argmax(-(V[below, 1] + V[below, 2]))]]
    return region, rim_dist, {"loops": loops, "nostrils": nostrils, "x0": x0}, anchors


def canonical_regions(C):
    """Region per canonical vertex (template frame) from MediaPipe's contours:
    near the mouth, above/below the seam between inner lip contours; same for
    each eye between its lid contours."""
    region = np.array(["face"] * len(C), dtype=object)

    def seam_split(upper, lower, zone_up, zone_lo, x_pad, name_up, name_lo):
        u, l = C[upper], C[lower]
        xs = np.r_[u[:, 0], l[:, 0]]
        xlo, xhi = xs.min() - x_pad, xs.max() + x_pad
        order_u, order_l = np.argsort(u[:, 0]), np.argsort(l[:, 0])
        for i, p in enumerate(C):
            if not (xlo <= p[0] <= xhi):
                continue
            seam = 0.5 * (np.interp(p[0], u[order_u, 0], u[order_u, 1]) +
                          np.interp(p[0], l[order_l, 0], l[order_l, 1]))
            dy = p[1] - seam
            if 0 <= dy < zone_up:
                region[i] = name_up
            elif -zone_lo < dy < 0:
                region[i] = name_lo

    seam_split(mp.LIPS_INNER_UPPER, mp.LIPS_INNER_LOWER, 20.0, 25.0, 2.0, "lip_upper", "lip_lower")
    # The template's subject-right eye is +X; MediaPipe's RIGHT_EYE lists are the subject's right.
    seam_split(mp.RIGHT_EYE_UPPER, mp.RIGHT_EYE_LOWER, 10.0, 8.0, 3.0, "lid_upper_r", "lid_lower_r")
    seam_split(mp.LEFT_EYE_UPPER, mp.LEFT_EYE_LOWER, 10.0, 8.0, 3.0, "lid_upper_l", "lid_lower_l")
    for i in mp.LIPS_INNER_UPPER + mp.LIPS_OUTER_UPPER:
        region[i] = "lip_upper"
    for i in mp.LIPS_INNER_LOWER + mp.LIPS_OUTER_LOWER:
        region[i] = "lip_lower"
    for i in (61, 291, 78, 308):                      # corners belong to neither lip
        region[i] = "face"
    return region


def load_skin(path=None):
    path = Path(path or config.template_export_dir() / (SKIN + ".npz"))
    with np.load(path, allow_pickle=False) as z:
        V = np.asarray(z["vertices"][0], dtype=np.float64)
        F = np.asarray(z["faces"], dtype=np.int64).reshape(-1, 3)
    return V, F, path


def build(V, F, verbose=True):
    canon = mp.load()
    C0 = canon["vertices"] @ R_CANON_TO_TEMPLATE.T
    CF = canon["faces"]
    perm = mirror_permutation(canon["vertices"])

    region_s, rim_dist, loops, an = skin_regions(V, F)
    pairs = [(mp.EYE_R_OUTER, an["ex_r"], 1.0), (mp.EYE_R_INNER, an["en_r"], 1.0),
             (mp.EYE_L_INNER, an["en_l"], 1.0), (mp.EYE_L_OUTER, an["ex_l"], 1.0),
             (mp.NOSE_TIP, an["prn"], 1.0), (mp.CHIN, an["gn"], 1.0),
             (mp.MOUTH_R, an["ch_r"], 0.3), (mp.MOUTH_L, an["ch_l"], 0.3)]
    idx = np.array([p[0] for p in pairs])
    tgt = np.array([p[1] for p in pairs])
    w = np.array([p[2] for p in pairs])
    s, R, t = similarity_transform(C0[idx], tgt, weights=w)
    C1 = apply_similarity(C0, s, R, t)
    if verbose:
        print("canonical->template similarity scale {0:.3f}".format(s))

    region_c = canonical_regions(C1)
    margins = set(mp.LIPS_INNER_UPPER + mp.LIPS_INNER_LOWER + mp.LIPS_OUTER_UPPER + mp.LIPS_OUTER_LOWER +
                  mp.RIGHT_EYE_UPPER + mp.RIGHT_EYE_LOWER + mp.LEFT_EYE_UPPER + mp.LEFT_EYE_LOWER)
    data_mask = np.array([i not in margins for i in range(len(C1))])
    unmasked = rim_dist > RIM_MM
    C2, info = nonrigid.fit(C1, CF, lm_idx=idx, lm_pts=tgt, lm_w=w * 10.0,
                            cloud=V[unmasked], data_mask=data_mask, max_dist=15.0,
                            schedule=(50.0, 20.0, 10.0, 5.0, 2.0), verbose=False)

    vertex = [None] * mp.N_LANDMARKS
    snap = [None] * mp.N_LANDMARKS
    for reg in sorted(set(region_c)):
        cand = np.flatnonzero(unmasked & ((region_s == reg) if reg != "face" else np.ones(len(V), bool)))
        if not len(cand):
            raise ValueError("no skin vertices in region {0}".format(reg))
        tree = cKDTree(V[cand])
        members = np.flatnonzero(region_c == reg)
        d, j = tree.query(C2[members])
        for i, dd, jj in zip(members, d, j):
            vertex[i], snap[i] = int(cand[jj]), float(dd)

    basis = set(canon["basis"])
    weight, wclass, hole_adj, out_region = [], [], [], []
    eye_zone = set(i for i in range(mp.N_CANONICAL) if region_c[i].startswith("lid"))
    for i in range(mp.N_LANDMARKS):
        if i >= mp.N_CANONICAL:
            weight.append(0.0); wclass.append("zero"); hole_adj.append(None); out_region.append("iris")
            continue
        adj = bool(rim_dist[vertex[i]] < HOLE_ADJ_MM)
        hole_adj.append(adj)
        out_region.append(str(region_c[i]))
        if i in margins or adj or snap[i] > POOR_SNAP_MM:
            weight.append(0.0); wclass.append("zero")
        elif i in eye_zone:
            weight.append(W_EYE); wclass.append("low")
        elif i in NOSE or i in CHIN or i in JAW_TEMPLE or i in FOREHEAD or i in basis:
            weight.append(W_HIGH); wclass.append("high")
        else:
            weight.append(W_LOW); wclass.append("low")

    checks = verify(V, vertex, out_region, perm, loops)
    fixture = {
        "format": 1,
        "description": "MediaPipe landmark index -> template skin vertex (" + SKIN + ", rest frame)",
        "skin": SKIN, "n_landmarks": mp.N_LANDMARKS,
        "vertex": vertex, "region": out_region, "weight": weight, "weight_class": wclass,
        "hole_adjacent": hole_adj, "snap_mm": snap,
        "mirror_permutation": perm.tolist(),
        "canonical_to_template_scale": s,
        "checks": checks,
    }
    return fixture, {"C2": C2, "region_s": region_s, "rim_dist": rim_dist, "nonrigid": info, "anchors": an}


def verify(V, vertex, region, perm, loops):
    """Invariants the fixture must satisfy; returned and asserted."""
    out = {}
    lr_bad = []
    for i in range(mp.N_CANONICAL):
        j = int(perm[i])
        if i == j or vertex[i] is None:
            continue
        xi, xj = V[vertex[i], 0] - loops["x0"], V[vertex[j], 0] - loops["x0"]
        if abs(xi) > 5 and abs(xj) > 5 and np.sign(xi) == np.sign(xj):
            lr_bad.append(i)
    out["left_right_same_side"] = lr_bad
    # MediaPipe's subject-right (33) must land at +X (template: subject's right is +X).
    out["landmark33_x_mm"] = float(V[vertex[mp.EYE_R_OUTER], 0] - loops["x0"])
    mouth_y = V[loops["loops"]["mouth"]][:, 1].mean()
    out["upper_lip_below_mouth"] = [i for i in range(mp.N_CANONICAL)
                                    if region[i] == "lip_upper" and V[vertex[i], 1] < mouth_y - 8]
    out["lower_lip_above_mouth"] = [i for i in range(mp.N_CANONICAL)
                                    if region[i] == "lip_lower" and V[vertex[i], 1] > mouth_y + 8]
    out["n_mapped"] = sum(v is not None for v in vertex)
    out["n_distinct_vertices"] = len({v for v in vertex if v is not None})
    out["pass"] = (not lr_bad and out["landmark33_x_mm"] > 0 and not out["upper_lip_below_mouth"]
                   and not out["lower_lip_above_mouth"] and out["n_mapped"] == mp.N_CANONICAL)
    return out


REGION_COLOURS = {"face": (200, 200, 200), "lip_upper": (230, 60, 60), "lip_lower": (60, 60, 230),
                  "lid_upper_r": (240, 170, 30), "lid_lower_r": (30, 170, 90),
                  "lid_upper_l": (250, 220, 80), "lid_lower_l": (80, 220, 160)}


def write_ply(path, V, F, region_s, fixture):
    """Skin coloured by region, snapped vertices black (mapped) / magenta (weight 0)."""
    col = np.array([REGION_COLOURS.get(r, (200, 200, 200)) for r in region_s], dtype=np.int64)
    for i, v in enumerate(fixture["vertex"]):
        if v is not None:
            col[v] = (255, 0, 255) if fixture["weight"][i] == 0 else (0, 0, 0)
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\nelement vertex {0}\n".format(len(V)))
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("element face {0}\nproperty list uchar int vertex_indices\nend_header\n".format(len(F)))
        for p, c in zip(V, col):
            fh.write("{0:.4f} {1:.4f} {2:.4f} {3} {4} {5}\n".format(p[0], p[1], p[2], *c))
        for f in F:
            fh.write("3 {0} {1} {2}\n".format(*f))


def load_fixture(path=None):
    import json
    with open(path or config.LANDMARK_TO_VERTEX_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skin", default=None)
    ap.add_argument("--out", default=str(config.LANDMARK_TO_VERTEX_JSON))
    args = ap.parse_args()
    with cio.RunDir("template", "fixture", "pipeline/fixture.py") as run:
        V, F, path = load_skin(args.skin)
        fixture, aux = build(V, F)
        fixture["skin_source"] = {"path": str(path), "md5": cio.md5_file(path)}
        cio.write_json_atomic(run.path / "outputs" / "landmark_to_vertex.json", fixture)
        write_ply(run.path / "views" / "fixture_regions.ply", V, F, aux["region_s"], fixture)
        wc = fixture["weight_class"]
        run.metrics.update({"checks": fixture["checks"], "scale": fixture["canonical_to_template_scale"],
                            "weight_classes": {k: wc.count(k) for k in ("high", "low", "zero")},
                            "snap_mm_median": float(np.median([s for s in fixture["snap_mm"] if s is not None])),
                            "snap_mm_p95": float(np.percentile([s for s in fixture["snap_mm"] if s is not None], 95)),
                            "nonrigid_last": aux["nonrigid"]["history"][-1]})
        print("checks:", fixture["checks"])
        print("weights:", run.metrics["weight_classes"], "snap median/p95 mm: {0:.2f}/{1:.2f}".format(
            run.metrics["snap_mm_median"], run.metrics["snap_mm_p95"]))
        print("view:", run.path / "views" / "fixture_regions.ply")
        if not fixture["checks"]["pass"]:
            print("FIXTURE CHECKS FAILED -- not written to", args.out)
            sys.exit(2)
        cio.write_json_atomic(args.out, fixture)
        print("wrote", args.out)


if __name__ == "__main__":
    main()

"""
Show a phone scan's landmarks on the lab's skin mesh in Maya (registration check).

How to use (in Maya on the school PC):
  1. Script Editor > Python tab > paste this whole file > Ctrl+Enter.
  2. A file window opens: choose the phone's front_landmarks_3d_mm.json.
  3. Wait about a minute. It prints the fit and creates a group "scan_landmarks_GRP"
     with one locator per landmark. If the locators sit on the face, the registration works.

It also saves the fit to registration.json next to the scan, for the next step.
"""
import json, math, os

# ---------- settings ----------
SCAN_JSON = r"C:\Users\YOUR_NAME\Documents\front_landmarks_3d_mm.json"
SKIN_MESH = "skin_cloth_copy_v5_pull_back_RENDER"
SCALE_GUESSES = [1.0, 1.6, 2.2]              # Maya units per mm to try
KEY_LANDMARKS = {1: "nose", 33: "eyeA", 263: "eyeB", 61: "mouthA", 291: "mouthB",
                 13: "lipTop", 14: "lipBottom", 10: "forehead", 152: "chin"}
# ------------------------------


# ===== small linear algebra (pure Python, no numpy needed) =====

def _jacobi_eigen(A, iters=60):
    """Eigen-decomposition of a small symmetric matrix. Returns (values, vectors as columns)."""
    n = len(A)
    A = [row[:] for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(iters):
        p = q = 0
        big = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > big:
                    big, p, q = abs(A[i][j]), i, j
        if big < 1e-12:
            break
        theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q])
        t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
        c = 1.0 / math.sqrt(t * t + 1.0)
        s = t * c
        for k in range(n):
            akp, akq = A[k][p], A[k][q]
            A[k][p], A[k][q] = c * akp - s * akq, s * akp + c * akq
        for k in range(n):
            apk, aqk = A[p][k], A[q][k]
            A[p][k], A[q][k] = c * apk - s * aqk, s * apk + c * aqk
        for k in range(n):
            vkp, vkq = V[k][p], V[k][q]
            V[k][p], V[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [A[i][i] for i in range(n)], V


def _quat_to_matrix(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def _apply(s, R, t, p):
    return [s * (R[i][0] * p[0] + R[i][1] * p[1] + R[i][2] * p[2]) + t[i] for i in range(3)]


def fit_similarity(src, dst):
    """Scale s, rotation R, translation t that move src points onto dst points (least squares)."""
    n = float(len(src))
    ca = [sum(p[i] for p in src) / n for i in range(3)]
    cb = [sum(p[i] for p in dst) / n for i in range(3)]
    a = [[p[i] - ca[i] for i in range(3)] for p in src]
    b = [[p[i] - cb[i] for i in range(3)] for p in dst]
    S = [[sum(a[k][i] * b[k][j] for k in range(len(a))) for j in range(3)] for i in range(3)]
    N = [[S[0][0] + S[1][1] + S[2][2], S[1][2] - S[2][1], S[2][0] - S[0][2], S[0][1] - S[1][0]],
         [S[1][2] - S[2][1], S[0][0] - S[1][1] - S[2][2], S[0][1] + S[1][0], S[2][0] + S[0][2]],
         [S[2][0] - S[0][2], S[0][1] + S[1][0], -S[0][0] + S[1][1] - S[2][2], S[1][2] + S[2][1]],
         [S[0][1] - S[1][0], S[2][0] + S[0][2], S[1][2] + S[2][1], -S[0][0] - S[1][1] + S[2][2]]]
    vals, V = _jacobi_eigen(N)
    k = max(range(4), key=lambda i: vals[i])
    q = [V[r][k] for r in range(4)]
    norm = math.sqrt(sum(x * x for x in q))
    R = _quat_to_matrix([x / norm for x in q])
    num = sum(sum(b[m][i] * sum(R[i][j] * a[m][j] for j in range(3)) for i in range(3)) for m in range(len(a)))
    den = sum(sum(x * x for x in p) for p in a)
    s = num / den if den > 0 else 1.0
    t = [cb[i] - s * sum(R[i][j] * ca[j] for j in range(3)) for i in range(3)]
    return s, R, t


# ===== registration =====

# The phone saves points in the sensor's own frame (landscape), where the face lies on its side:
#   X runs along the face's up/down direction, Y along its left/right direction, Z is depth away from the phone.
# Maya wants x right, y up, z toward the viewer. Several axis arrangements are possible, so we try them all.
ORIENTATIONS = [
    ("rotated A", lambda p: [-p[1], -p[0], -p[2]]),
    ("rotated B", lambda p: [p[1], p[0], -p[2]]),
    ("mirrored A", lambda p: [p[1], -p[0], -p[2]]),
    ("mirrored B", lambda p: [-p[1], p[0], -p[2]]),
]


def to_model_axes(p, orient):
    return ORIENTATIONS[orient][1](p)


def _nearest(points, verts):
    out = []
    for p in points:
        best, bd = None, 1e30
        for v in verts:
            d = (p[0] - v[0]) ** 2 + (p[1] - v[1]) ** 2 + (p[2] - v[2]) ** 2
            if d < bd:
                bd, best = d, v
        out.append((best, bd))
    return out


def icp(points, verts, s0, iters=20, keep=0.8, scale_range=(0.5, 4.0), fix_scale=None, init=None):
    """Fit scan points onto mesh vertices. Starts with scale s0 and the points' centroid on the front of the mesh."""
    n = float(len(points))
    cp = [sum(p[i] for p in points) / n for i in range(3)]
    zs = sorted(v[2] for v in verts)
    zmid = zs[len(zs) // 2]
    front = [v for v in verts if v[2] >= zmid]
    cf = [sum(v[i] for v in front) / len(front) for i in range(3)]
    s, R, t = s0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [cf[i] - s0 * cp[i] for i in range(3)]
    if init:
        R, t = init
    err = 1e30
    for _ in range(iters):
        moved = [_apply(s, R, t, p) for p in points]
        near = _nearest(moved, verts)
        order = sorted(range(len(points)), key=lambda i: near[i][1])
        use = order[: max(10, int(len(order) * keep))]
        s2, R2, t2 = fit_similarity([points[i] for i in use], [near[i][0] for i in use])
        s2 = fix_scale if fix_scale else min(max(s2, scale_range[0]), scale_range[1])
        # recompute translation for the clamped scale
        ca = [sum(points[i][k] for i in use) / len(use) for k in range(3)]
        cb = [sum(near[i][0][k] for i in use) / len(use) for k in range(3)]
        t2 = [cb[k] - s2 * sum(R2[k][j] * ca[j] for j in range(3)) for k in range(3)]
        new_err = sum(near[i][1] for i in use) / len(use)
        s, R, t = s2, R2, t2
        if abs(err - new_err) < 1e-4:
            err = new_err
            break
        err = new_err
    return s, R, t, math.sqrt(err)


def register(scan_points, verts, scale_guesses=SCALE_GUESSES, fix_scale=None, log=print):
    """Try each axis arrangement and several starting scales; return the best fit."""
    step = max(1, len(verts) // 3000)
    sub = verts[::step]
    sparse = scan_points[::3]
    best = None
    for k in range(len(ORIENTATIONS)):
        pts = [to_model_axes(p, k) for p in sparse]
        for g in ([fix_scale] if fix_scale else scale_guesses):
            s, R, t, rms = icp(pts, sub, g, fix_scale=fix_scale)
            log("  %-10s start scale %.1f -> scale %.2f, rms distance %.2f" % (ORIENTATIONS[k][0], g, s, rms))
            if best is None or rms < best["rms"]:
                best = {"orient": k, "s": s, "R": R, "t": t, "rms": rms}
    # refine on all points
    pts = [to_model_axes(p, best["orient"]) for p in scan_points]
    s, R, t, rms = icp(pts, verts[::max(1, len(verts) // 6000)], best["s"], iters=15, fix_scale=fix_scale)
    best.update({"s": s, "R": R, "t": t, "rms": rms})
    return best


# ===== finding the model's scale from its eye holes =====

def loops_from_edges(edge_vertex_pairs):
    """Group border edges into connected loops. Returns a list of vertex-index lists."""
    adj = {}
    for a, b in edge_vertex_pairs:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        loops.append(comp)
    return loops


def _box(loop, verts):
    xs = [verts[i][0] for i in loop]
    ys = [verts[i][1] for i in loop]
    return min(xs), max(xs), min(ys), max(ys)


def pick_eye_loops(loops, verts, log=print):
    """Pick the two border loops that are the eye openings. Returns (loop_left, loop_right) or None."""
    info = []
    for lp in loops:
        x0, x1, y0, y1 = _box(lp, verts)
        info.append({"loop": lp, "w": x1 - x0, "h": y1 - y0, "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2})
    info.sort(key=lambda d: -(d["w"] * d["h"]))
    for d in info:
        log("  border loop: %4d vertices, width %.1f, height %.1f, center x %.1f y %.1f"
            % (len(d["loop"]), d["w"], d["h"], d["cx"], d["cy"]))
    rest = info[1:7]            # skip the biggest (the outer edge of the mesh)
    best, best_score = None, 0.0
    for i in range(len(rest)):
        for j in range(i + 1, len(rest)):
            a, b = rest[i], rest[j]
            size = min(a["w"], b["w"])
            if size <= 0 or max(a["w"], b["w"]) / size > 1.5:
                continue                        # eyes are about the same size
            if abs(a["cy"] - b["cy"]) > 0.5 * max(a["h"], b["h"]):
                continue                        # and at the same height
            if abs(a["cx"] - b["cx"]) < 2.0 * size:
                continue                        # and well apart sideways
            score = size
            if score > best_score:
                best, best_score = (a, b), score
    if not best:
        return None
    a, b = sorted(best, key=lambda d: d["cx"])
    return a["loop"], b["loop"]


def eye_corner_distance(loop_left, loop_right, verts):
    """Distance between the two OUTER eye corners of the model."""
    pa = min((verts[i] for i in loop_left), key=lambda v: v[0])
    pb = max((verts[i] for i in loop_right), key=lambda v: v[0])
    return math.sqrt(sum((pa[k] - pb[k]) ** 2 for k in range(3)))


def pick_mouth_loop(loops, verts, eyes):
    """The border loop that is the mouth opening: centered between the eyes and below them."""
    la, lb = eyes
    ea, eb = _box(la, verts), _box(lb, verts)
    mid_x = (ea[0] + ea[1] + eb[0] + eb[1]) / 4.0
    eye_y = (ea[2] + ea[3] + eb[2] + eb[3]) / 4.0
    sep = abs((ea[0] + ea[1]) / 2.0 - (eb[0] + eb[1]) / 2.0)
    skip = set(id(l) for l in (la, lb))
    best, best_w = None, 0.0
    for lp in loops:
        if id(lp) in skip:
            continue
        x0, x1, y0, y1 = _box(lp, verts)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if abs(cx - mid_x) < 0.35 * sep and cy < eye_y - 0.2 * sep and (x1 - x0) < 1.5 * sep:
            if x1 - x0 > best_w:
                best, best_w = lp, x1 - x0
    return best


def _mean(points):
    n = float(len(points))
    return [sum(p[i] for p in points) / n for i in range(3)]


def _sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _unit(a):
    n = math.sqrt(_dot(a, a)) or 1.0
    return [c / n for c in a]


def model_facing(verts, pa, pb, pm):
    """Which way the model's face points, from its own shape: the nose is the part of the mesh that
    sticks out of the eye/mouth plane near the middle of the face (the back of the head is much farther away)."""
    n = _unit(_cross(_sub(pb, pa), _sub(pm, pa)))
    fc = _mean([pa, pb, pm])
    sep = math.sqrt(_dot(_sub(pa, pb), _sub(pa, pb)))
    best, best_d = None, 0.0
    for v in verts:
        w = _sub(v, fc)
        d = _dot(w, n)
        lateral = math.sqrt(max(_dot(w, w) - d * d, 0.0))
        if lateral < 0.2 * sep and abs(d) < 0.7 * sep and abs(d) > abs(best_d):
            best, best_d = v, d
    if best is None:
        return None
    return [c if best_d > 0 else -c for c in n]


def anchor_registration(raw, verts, loop_left, loop_right, loop_mouth, log=print):
    """Fit the scan to the mesh from three landmarks: the two outer eye corners and the mouth center.
    The scan's depth direction and left/right are settled by which way the nose points."""
    pa = min((verts[i] for i in loop_left), key=lambda v: v[0])      # model outer eye corner, left
    pb = max((verts[i] for i in loop_right), key=lambda v: v[0])     # model outer eye corner, right
    pm = _mean([verts[i] for i in loop_mouth])                       # model mouth center
    model_pts = [pa, pb, pm]
    d_model = math.sqrt(_dot(_sub(pa, pb), _sub(pa, pb)))
    s33, s263 = raw[33], raw[263]
    smouth = [(raw[61][i] + raw[291][i]) / 2.0 for i in range(3)]
    d_scan = math.sqrt(_dot(_sub(s33, s263), _sub(s33, s263)))
    scale = d_model / d_scan
    face_c = _mean(model_pts)
    facing = model_facing(verts, pa, pb, pm)
    if facing is None:
        log("  could not tell which way the model's face points")
        return None
    log("  model faces toward (%.2f, %.2f, %.2f)" % tuple(facing))
    best = None
    for a, b in ((s33, s263), (s263, s33)):
        _, R, _ = fit_similarity([a, b, smouth], model_pts)
        ca, cb = _mean([a, b, smouth]), _mean(model_pts)
        t = [cb[i] - scale * sum(R[i][j] * ca[j] for j in range(3)) for i in range(3)]
        nose = _apply(scale, R, t, raw[1])
        ahead = _dot(_sub(nose, face_c), facing)            # the nose should stick out toward the face direction
        resid = sum(math.sqrt(_dot(_sub(_apply(scale, R, t, p), q), _sub(_apply(scale, R, t, p), q)))
                    for p, q in zip([a, b, smouth], model_pts))
        log("  anchors fit: nose sticks out %.1f units, anchor error %.1f" % (ahead, resid))
        if ahead > 0 and (best is None or resid < best["resid"]):
            best = {"R": R, "t": t, "scale": scale, "resid": resid}
    return best


def refine_from_anchors(raw, idx, pts, verts, anchored):
    """Start from the anchor fit and polish it with a few ICP steps (scale stays fixed)."""
    R0, t0, sc = anchored["R"], anchored["t"], anchored["scale"]
    sub = verts[::max(1, len(verts) // 4000)]
    s, R, t, rms = icp(pts[::2], sub, sc, iters=12, fix_scale=sc, init=(R0, t0))
    # rms of the anchor-only fit, for comparison
    near = _nearest([_apply(sc, R0, t0, p) for p in pts[::3]], sub)
    rms0 = math.sqrt(sum(d for _, d in near) / len(near))
    print("  distance to mesh: anchors only %.2f, after refining %.2f (units)" % (rms0, rms))
    if rms > rms0 * 1.1:
        R, t, rms = R0, t0, rms0
        print("  using the anchor-only fit")
    return {"orient": None, "s": sc, "R": R, "t": t, "rms": rms}



# ===== Precision step: warp the scan onto this particular face =====
# The scan is one person's face and the model is another's, so no rotation/scale/shift can make them agree.
# We pair up ~14 facial features (eye corners + lids, mouth corners + lips, nose tip) between scan and model,
# then bend the scan smoothly so those pairs coincide, and finally rest every landmark on the skin.

IRIS = set(range(468, 478))          # iris centres float behind the eye opening, not on skin
EYE_SETS = (
    {"outer": 33, "inner": 133, "top": 159, "bottom": 145},
    {"outer": 263, "inner": 362, "top": 386, "bottom": 374},
)


def _solve(A, B):
    """Solve A X = B (A n x n, B n x m) by Gaussian elimination with pivoting."""
    n, m = len(A), len(B[0])
    M = [list(A[i]) + list(B[i]) for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        piv = M[c][c] or 1e-12
        for j in range(c, n + m):
            M[c][j] /= piv
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                for j in range(c, n + m):
                    M[r][j] -= f * M[c][j]
    return [row[n:] for row in M]


def _extreme(loop, verts, axis, sign):
    return max((verts[i] for i in loop), key=lambda v: sign * v[axis])


def model_nose_tip(verts, pa, pb, pm, facing):
    fc = _mean([pa, pb, pm])
    sep = math.sqrt(_dot(_sub(pa, pb), _sub(pa, pb)))
    best, best_d = None, -1e30
    for v in verts:
        w = _sub(v, fc)
        d = _dot(w, facing)
        lateral = math.sqrt(max(_dot(w, w) - d * d, 0.0))
        if lateral < 0.2 * sep and 0 < d < 0.7 * sep and d > best_d:
            best, best_d = v, d
    return best


def feature_pairs(raw, fit, verts, eyes, mouth, facing, outer=None):
    """(landmark index, model point) pairs, matched left/right using the rigid fit."""
    def tx(i):
        return _apply(fit["s"], fit["R"], fit["t"], raw[i])
    pairs = []
    if raw[33] is not None and raw[263] is not None:
        order = (0, 1) if tx(33)[0] < tx(263)[0] else (1, 0)      # which scan eye lies on the low-x side
    else:
        order = (0, 1)
    for k, loop in enumerate(eyes):
        lm = EYE_SETS[order[k]]
        lo = (k == 0)                                              # eyes[0] is the low-x loop
        cand = {"outer": _extreme(loop, verts, 0, -1 if lo else 1),
                "inner": _extreme(loop, verts, 0, 1 if lo else -1),
                "top": _extreme(loop, verts, 1, 1),
                "bottom": _extreme(loop, verts, 1, -1)}
        for name, i in lm.items():
            if raw[i] is not None:
                pairs.append((i, cand[name]))
    if all(raw[i] is not None for i in (61, 291)):
        a, b = (61, 291) if tx(61)[0] < tx(291)[0] else (291, 61)
        pairs.append((a, _extreme(mouth, verts, 0, -1)))
        pairs.append((b, _extreme(mouth, verts, 0, 1)))
    xs = [verts[i][0] for i in mouth]
    cx, w = (min(xs) + max(xs)) / 2.0, max(xs) - min(xs)
    mid = [verts[i] for i in mouth if abs(verts[i][0] - cx) < 0.15 * w]
    if mid and raw[13] is not None and raw[14] is not None:
        pairs.append((13, max(mid, key=lambda v: v[1])))
        pairs.append((14, min(mid, key=lambda v: v[1])))
    pa = _extreme(eyes[0], verts, 0, -1)
    pb = _extreme(eyes[1], verts, 0, 1)
    pm = _mean([verts[i] for i in mouth])
    nt = model_nose_tip(verts, pa, pb, pm, facing)
    if nt is not None and raw[1] is not None:
        pairs.append((1, nt))
    if outer and raw[152] is not None:                 # chin = lowest point of the mesh's outer edge, below the mouth
        cx2 = (pa[0] + pb[0]) / 2.0
        sep2 = abs(pa[0] - pb[0])
        low = [verts[i] for i in outer if abs(verts[i][0] - cx2) < 0.15 * sep2 and verts[i][1] < pm[1]]
        if low:
            pairs.append((152, min(low, key=lambda v: v[1])))
    return pairs


def gaussian_warp(src, dst, sigma, ridge=1e-3):
    """Smooth map that moves each src point onto its dst point and fades to nothing far away.
    Returns a function f(p) -> displacement."""
    n = len(src)
    def k(a, b):
        return math.exp(-_dot(_sub(a, b), _sub(a, b)) / (2.0 * sigma * sigma))
    K = [[k(src[i], src[j]) + (ridge if i == j else 0.0) for j in range(n)] for i in range(n)]
    R = [_sub(dst[i], src[i]) for i in range(n)]
    W = _solve(K, R)
    def f(p):
        d = [0.0, 0.0, 0.0]
        for i in range(n):
            w = k(p, src[i])
            for c in range(3):
                d[c] += w * W[i][c]
        return d
    return f


def fit_affine(src, dst, ridge=1e-6):
    """Least-squares affine map (3x3 matrix + shift) taking src to dst. Returns a function p -> q."""
    A = [[p[0], p[1], p[2], 1.0] for p in src]
    AtA = [[sum(r[i] * r[j] for r in A) + (ridge if i == j else 0.0) for j in range(4)] for i in range(4)]
    Atb = [[sum(A[n][i] * dst[n][c] for n in range(len(A))) for c in range(3)] for i in range(4)]
    X = _solve(AtA, Atb)
    def f(p):
        v = [p[0], p[1], p[2], 1.0]
        return [sum(v[i] * X[i][c] for i in range(4)) for c in range(3)]
    return f


def refine_nonrigid(raw, idx, fit, verts, eyes, mouth, facing, log=print, outer=None):
    """Returns {landmark index: model position resting on the skin} plus error numbers."""
    s = fit["s"]
    pairs = feature_pairs(raw, fit, verts, eyes, mouth, facing, outer)
    src = [_apply(s, fit["R"], fit["t"], raw[i]) for i, _ in pairs]
    dst = [q for _, q in pairs]
    before = [math.sqrt(_dot(_sub(a, b), _sub(a, b))) / s for a, b in zip(src, dst)]
    log("  matched %d features between scan and model; they were off by %.1f mm on average (worst %.1f)"
        % (len(pairs), sum(before) / len(before), max(before)))
    pa0, pb0 = _extreme(eyes[0], verts, 0, -1), _extreme(eyes[1], verts, 0, 1)
    sep = math.sqrt(_dot(_sub(pa0, pb0), _sub(pa0, pb0)))
    # step 1: stretch the whole scan to the model's overall proportions (affine, keeps it smooth)
    aff = fit_affine(src, dst) if len(src) >= 6 else (lambda p: list(p))
    mid = [aff(p) for p in src]
    # step 2: nudge locally so each matched feature lands exactly on its partner
    g = gaussian_warp(mid, dst, sigma=max(sep * 0.45, 1.0))
    def f(p):
        q = aff(p)
        d = g(q)
        return [q[c] + d[c] - p[c] for c in range(3)]
    rigid = {i: _apply(s, fit["R"], fit["t"], raw[i]) for i in idx if i not in IRIS}
    warped = {i: [p[c] + d for c, d in enumerate(f(p))] for i, p in rigid.items()}
    after = []
    for a, b in zip(src, dst):
        w = [a[c] + d for c, d in enumerate(f(a))]
        after.append(math.sqrt(_dot(_sub(w, b), _sub(w, b))) / s)
    ids = sorted(warped)
    sub = verts
    near = _nearest([warped[i] for i in ids], sub)
    snapped = {i: list(v) for i, (v, _) in zip(ids, near)}
    gap_warp = [math.sqrt(d) / s for _, d in near]
    near0 = _nearest([rigid[i] for i in ids], sub)
    gap_rigid = [math.sqrt(d) / s for _, d in near0]
    log("  matched features now off by %.2f mm" % (sum(after) / len(after)))
    log("  landmarks vs skin surface: before warp %.2f mm average, after warp %.2f mm average (worst %.1f mm)"
        % (sum(gap_rigid) / len(gap_rigid), sum(gap_warp) / len(gap_warp), max(gap_warp)))
    worst = sorted(zip(gap_warp, ids), reverse=True)[:5]
    log("  five landmarks farthest from skin: " + ", ".join("#%d (%.1f mm)" % (i, g) for g, i in worst))
    # how a movement of 1 mm in the scan turns into movement in Maya units (rotation + scale + the stretch)
    o = aff([0.0, 0.0, 0.0])
    cols = []
    for k in range(3):
        e = [0.0, 0.0, 0.0]
        e[k] = 1.0
        v = _apply(s, fit["R"], [0.0, 0.0, 0.0], e)
        a = aff(v)
        cols.append([a[c] - o[c] for c in range(3)])
    motion = [[cols[k][r] for k in range(3)] for r in range(3)]      # 3x3, rows = Maya x,y,z
    return snapped, {"motion_matrix": motion, "features": len(pairs), "skin_gap_mm": sum(gap_warp) / len(gap_warp),
                     "skin_gap_before_mm": sum(gap_rigid) / len(gap_rigid),
                     # where each scan landmark sits in the model's space (before snapping onto the lab skin):
                     # the face shape the lab skin can be morphed toward
                     "warped": {str(i): [round(c, 4) for c in warped[i]] for i in ids},
                     # the same landmarks with only rotation + scale + shift (no stretch): the scan's REAL proportions
                     "rigid": {str(i): [round(c, 4) for c in rigid[i]] for i in ids}}


# ===== Maya part =====

def run():
    import maya.cmds as cmds
    scan_path = SCAN_JSON
    if not os.path.exists(scan_path):
        pick = cmds.fileDialog2(fileMode=1, caption="Choose front_landmarks_3d_mm.json",
                                fileFilter="JSON (*.json)")
        if not pick:
            print("No file chosen.")
            return
        scan_path = pick[0]
    raw = json.load(open(scan_path))
    # drop landmarks whose depth is far from the rest of the face (edge points that picked up the background)
    zs = sorted(p[2] for p in raw if p is not None)
    zmed = zs[len(zs) // 2]
    dropped = [i for i, p in enumerate(raw) if p is not None and abs(p[2] - zmed) > 100]
    for i in dropped:
        raw[i] = None
    if dropped:
        print("ignoring %d landmarks with bad depth: %s" % (len(dropped), dropped))
    idx = [i for i, p in enumerate(raw) if p is not None]
    pts = [raw[i] for i in idx]
    print("scan: %d of %d landmarks usable" % (len(pts), len(raw)))
    flat = cmds.xform(SKIN_MESH + ".vtx[*]", query=True, worldSpace=True, translation=True)
    verts = [flat[i:i + 3] for i in range(0, len(flat), 3)]
    print("skin mesh: %d vertices" % len(verts))
    fix_scale = None
    anchored = None
    eyes = mouth = None
    try:
        cmds.select(SKIN_MESH, replace=True)
        cmds.polySelectConstraint(mode=3, type=0x8000, where=1)
        border = cmds.ls(selection=True, flatten=True)
        cmds.polySelectConstraint(mode=0, where=0)
        cmds.select(clear=True)
        pairs = []
        for line in cmds.polyInfo(border, edgeToVertex=True):
            nums = [int(x) for x in line.split(":")[1].split() if x.lstrip("-").isdigit()]
            if len(nums) >= 2:
                pairs.append((nums[0], nums[1]))
        loops = loops_from_edges(pairs)
        print("found %d border loops (eye holes, mouth, mesh edge...)" % len(loops))
        eyes = pick_eye_loops(loops, verts)
        if eyes and raw[33] is not None and raw[263] is not None:
            d_model = eye_corner_distance(eyes[0], eyes[1], verts)
            d_scan = math.sqrt(sum((raw[33][k] - raw[263][k]) ** 2 for k in range(3)))
            fix_scale = d_model / d_scan
            print("eye corners: model %.1f units, scan %.1f mm -> scale %.3f units per mm" % (d_model, d_scan, fix_scale))
            mouth = pick_mouth_loop(loops, verts, eyes)
            if mouth and all(raw[i] is not None for i in (1, 61, 291)):
                anchored = anchor_registration(raw, verts, eyes[0], eyes[1], mouth)
        else:
            print("could not identify the two eye holes; falling back to a scale search")
    except Exception as e:
        print("eye-hole scale step failed (%s); falling back to a scale search" % e)
    if anchored:
        print("fitting from eyes + mouth, then refining...")
        fit = refine_from_anchors(raw, idx, pts, verts, anchored)
    else:
        print("could not use eyes + mouth; trying axis arrangements (less reliable)...")
        fit = register(pts, verts, fix_scale=fix_scale)
    print("BEST: %s  scale=%.3f Maya units per mm  rms=%.2f units" % ("eyes+mouth anchors" if fit["orient"] is None else ORIENTATIONS[fit["orient"]][0], fit["s"], fit["rms"]))

    positions = {}
    stats = {}
    if anchored and eyes and mouth:
        try:
            pa = _extreme(eyes[0], verts, 0, -1)
            pb = _extreme(eyes[1], verts, 0, 1)
            pm = _mean([verts[i] for i in mouth])
            facing = model_facing(verts, pa, pb, pm)
            print("precision step: matching facial features and bending the scan onto this face...")
            outer = max(loops, key=lambda l: (lambda b: (b[1] - b[0]) * (b[3] - b[2]))(_box(l, verts)))
            positions, stats = refine_nonrigid(raw, idx, fit, verts, eyes, mouth, facing, outer=outer)
        except Exception as e:
            print("precision step failed (%s); using the plain fit" % e)
            positions = {}
    if not positions:
        positions = {i: _apply(fit["s"], fit["R"], fit["t"], p if fit["orient"] is None else to_model_axes(p, fit["orient"]))
                     for i, p in zip(idx, pts) if i not in IRIS}

    grp = "scan_landmarks_GRP"
    if cmds.objExists(grp):
        cmds.delete(grp)
    grp = cmds.group(empty=True, name=grp)
    for i in sorted(positions):
        q = positions[i]
        name = "lm_%03d" % i + ("_" + KEY_LANDMARKS[i] if i in KEY_LANDMARKS else "")
        loc = cmds.spaceLocator(name=name)[0]
        cmds.xform(loc, worldSpace=True, translation=q)
        sc = 4.0 if i in KEY_LANDMARKS else 1.5
        for ax in "XYZ":
            cmds.setAttr("%s.localScale%s" % (loc, ax), sc)
        cmds.parent(loc, grp)

    out = os.path.join(os.path.dirname(scan_path), "registration.json")
    json.dump({"orientation": "anchors" if fit["orient"] is None else ORIENTATIONS[fit["orient"]][0], "orient_index": fit["orient"], "scale": fit["s"], "R": fit["R"], "t": fit["t"],
               "rms": fit["rms"], "skin_mesh": SKIN_MESH, "stats": stats}, open(out, "w"))
    json.dump({str(i): positions[i] for i in positions}, open(os.path.join(os.path.dirname(scan_path), "landmarks_on_model.json"), "w"))
    print("saved", out)
    print("Look at the viewport: the locators should sit on the face.")


if __name__ == "__main__":
    try:
        run()
    except ImportError:
        print("Run this inside Maya.")

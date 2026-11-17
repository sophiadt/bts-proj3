"""Does the fitted head's mouth still open? Measures on the skin and lip muscle.

The lab rig rests with the mouth slightly open: the skin's mouth-hole edges sit
8.3 mm apart at the middle, about 272 mm^2 of the hole shows from the front,
and the lip muscle's (OrbicularisOris) upper and lower halves are about 2.5 mm
apart. Its phoneme motion was authored from that open rest. A neutral capture
has closed lips, and pulling both hole edges onto MediaPipe's inner-lip
contours (which coincide on a closed mouth) shut the hole and squeezed the lip
muscle's halves together; the head then animated with a sealed mouth. These
measures catch that:

  front opening   hole area visible from the front (mm^2), at rest and with
                  speech motion added
  edge gap        distance between the hole's upper and lower edges at the middle
  muscle gap      per 1 mm slice across the mouth, the gap between the lip
                  muscle's upper and lower halves

Each is compared with the lab head at the subject's size. numpy/scipy only.
"""

import numpy as np
from scipy.spatial import cKDTree

import config
from pipeline import fixture as fx

SKIN = fx.SKIN
LIP_MUSCLE = "middleres_OrbicularisOris"
# Verdict thresholds, from the first two captures (skin-only fits on the
# laptop): mouth pulled shut, speech opened 32-48% of the lab head's; with the
# rig's resting gap kept, 88-96%.
SPEECH_OK = 0.7
SPEECH_SEALED = 0.4
# Lip muscle: a slice whose gap is under a third of the lab head's at the same
# place across the mouth counts as closing. From the base head, mouth pulled
# shut: 3 and 5 of 30 slices (lowest ratios 0.23, 0.24); gap kept: none
# (lowest 0.43, 0.50). Lips thinner than the lab head's squeeze the whole lip,
# gap included, by up to about half; that is not closing.
MUSCLE_CLOSING_RATIO = 1.0 / 3.0
MUSCLE_CLOSING_MAX = 0.05


def mouth_parts(V, F, regions=None):
    """Vertex indices of the mouth hole: loop, upper and lower edge chains
    (subject's right corner first), and the two edge vertices at the midline.
    Topology only, so they hold for any pose of the same mesh.
    regions: fixture.skin_regions(V, F) if already computed."""
    _region, _rim, info, an = regions if regions is not None else fx.skin_regions(V, F)
    loop = info["loops"]["mouth"]
    a, b = fx.nearest_in(V, loop, an["ch_r"]), fx.nearest_in(V, loop, an["ch_l"])
    up, lo = fx.split_loop(V, loop, a, b)
    up = up if up[0] == a else up[::-1]
    lo = lo if lo[0] == a else lo[::-1]
    x0 = info["x0"]
    mid_u = int(up[int(np.argmin(np.abs(V[up, 0] - x0)))])
    mid_l = int(lo[int(np.argmin(np.abs(V[lo, 0] - x0)))])
    return {"loop": np.asarray(loop), "upper": np.asarray(up), "lower": np.asarray(lo),
            "mid_upper": mid_u, "mid_lower": mid_l}


def edge_gap_mm(X, parts):
    return float(np.linalg.norm(X[parts["mid_upper"]] - X[parts["mid_lower"]]))


def front_opening_mm2(X, F, parts, step=0.25):
    """Area of the mouth hole seen straight from the front (template frame:
    face toward -Z), in mm^2: cells of a grid over the mouth that no skin
    triangle near the mouth covers. Two-sided, like a ray cast."""
    X = np.asarray(X, dtype=np.float64)
    Q = X[parts["loop"]]
    w = float(np.ptp(Q[:, 0]))
    c = Q.mean(axis=0)
    x0, x1 = Q[:, 0].min() - 2.0, Q[:, 0].max() + 2.0
    y0, y1 = Q[:, 1].min() - 0.2 * w, Q[:, 1].max() + 0.2 * w
    nx, ny = int(np.ceil((x1 - x0) / step)), int(np.ceil((y1 - y0) / step))
    cen = X[F].mean(axis=1)
    near = (np.abs(cen[:, 0] - c[0]) < 0.75 * w) & (np.abs(cen[:, 1] - c[1]) < 0.75 * w) & \
           (np.abs(cen[:, 2] - c[2]) < w)
    T = X[F[near]][:, :, :2]
    lo, hi = T.min(axis=1), T.max(axis=1)
    i0 = np.maximum(np.ceil((lo[:, 0] - x0) / step - 0.5), 0).astype(int)
    i1 = np.minimum(np.floor((hi[:, 0] - x0) / step - 0.5), nx - 1).astype(int)
    j0 = np.maximum(np.ceil((lo[:, 1] - y0) / step - 0.5), 0).astype(int)
    j1 = np.minimum(np.floor((hi[:, 1] - y0) / step - 0.5), ny - 1).astype(int)
    covered = np.zeros((ny, nx), dtype=bool)
    for k in np.flatnonzero((i1 >= i0) & (j1 >= j0)):
        a, b, cc = T[k]
        e1, e2 = b - a, cc - a
        det = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(det) < 1e-12:
            continue
        gx = x0 + (np.arange(i0[k], i1[k] + 1) + 0.5) * step
        gy = y0 + (np.arange(j0[k], j1[k] + 1) + 0.5) * step
        px, py = np.meshgrid(gx - a[0], gy - a[1])
        u = (px * e2[1] - py * e2[0]) / det
        v = (py * e1[0] - px * e1[1]) / det
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        covered[j0[k]:j1[k] + 1, i0[k]:i1[k] + 1] |= inside
    return float((~covered).sum() * step * step)


def muscle_gap(P, X, parts, slice_mm=1.0):
    """Gap (mm) between the lip muscle's upper and lower halves, per x-slice
    across the middle 70% of the mouth. P: lip muscle vertices, X: the skin in
    the same pose (its mouth hole defines the slit's centre line). The gap is
    the closest pair across the centre line, so halves that touch or cross give
    ~0. -> dict(min, median, gaps, xs)."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(X, dtype=np.float64)[parts["loop"]]
    xa, xb = Q[:, 0].min(), Q[:, 0].max()
    w = xb - xa
    cx, half = 0.5 * (xa + xb), 0.35 * w
    zmax = Q[:, 2].max() + 0.25 * w
    xs, gaps = [], []
    for x in np.arange(cx - half, cx + half, slice_mm):
        q = Q[np.abs(Q[:, 0] - x) < 1.5 * slice_mm]
        if len(q) < 2:
            continue
        yc = 0.5 * (q[:, 1].max() + q[:, 1].min())
        sl = P[(np.abs(P[:, 0] - x) < 0.75 * slice_mm) & (np.abs(P[:, 1] - yc) < 0.3 * w) & (P[:, 2] < zmax)]
        up, lo = sl[sl[:, 1] >= yc], sl[sl[:, 1] < yc]
        if not len(up) or not len(lo):
            continue
        xs.append(float(x))
        gaps.append(float(cKDTree(lo).query(up)[0].min()))
    gaps = np.array(gaps)
    return {"min": float(gaps.min()) if len(gaps) else None,
            "median": float(np.median(gaps)) if len(gaps) else None,
            "gaps": gaps, "xs": np.array(xs)}


def muscle_compare(P, X, ref_P, ref_X, parts):
    """muscle_gap of P against the reference's, slice by slice at the same
    relative place across the mouth (each head's own mouth width)."""
    g, g0 = muscle_gap(P, X, parts), muscle_gap(ref_P, ref_X, parts)
    out = {"muscle_gap_min_mm": g["min"], "muscle_gap_median_mm": g["median"],
           "ref_muscle_gap_min_mm": g0["min"], "ref_muscle_gap_median_mm": g0["median"],
           "muscle_slices": int(len(g["gaps"])), "muscle_slices_under_1mm": int((g["gaps"] < 1.0).sum()),
           "muscle_ratio": None, "muscle_ratio_min": None, "muscle_slices_closing": None,
           "muscle_closing_frac": None}
    if not len(g["gaps"]) or not len(g0["gaps"]):
        return out

    def rel(xs, skin):
        Q = np.asarray(skin)[parts["loop"], 0]
        return (xs - 0.5 * (Q.min() + Q.max())) / np.ptp(Q)

    ref_at = np.interp(rel(g["xs"], X), rel(g0["xs"], ref_X), g0["gaps"])
    r = g["gaps"] / np.maximum(ref_at, 1e-9)
    closing = r < MUSCLE_CLOSING_RATIO
    out.update({"muscle_ratio": float(np.median(r)), "muscle_ratio_min": float(r.min()),
                "muscle_slices_closing": int(closing.sum()), "muscle_closing_frac": float(closing.mean())})
    return out


def openings(frames, F, parts, step=0.25):
    """front_opening_mm2 for each frame (K, V, 3)."""
    return np.array([front_opening_mm2(f, F, parts, step) for f in frames])


def load_lab_mesh(name, all_frames=False):
    """The lab head's mesh from config.lab_data_dir() (this machine's
    template_export, else ../bts-proj1/data/lab): vertices of frame 0 (or all
    frames) in mm, and faces."""
    with np.load(config.lab_data_dir() / (name + ".npz"), allow_pickle=False) as z:
        v = np.asarray(z["vertices"], dtype=np.float64)
        f = np.asarray(z["faces"], dtype=np.int64).reshape(-1, 3)
    return (v if all_frames else v[0]), f


def lab_speech_motion(step=4):
    """The lab head's own speech motion as displacements from its neutral
    frame, (K, V, 3) mm, and where it came from; (None, None) if none here.
    1. template_export/anim_ui_generic.npz: "not even you" from the keyboard UI
       on the generic head (rig/ui_smoke_test.py), neutral frame 14;
    2. the lab skin's own animated sequence (bts-proj1 export), rest frame 15."""
    f = config.template_export_dir() / "anim_ui_generic.npz"
    if f.exists():
        with np.load(f) as z:
            frames = np.asarray(z["skin"], dtype=np.float64)
        neutral, src = 14, "anim_ui_generic.npz (keyboard UI, generic head)"
    else:
        p = config.lab_data_dir() / (SKIN + ".npz")
        if not p.exists():
            return None, None
        with np.load(p) as z:
            if z["vertices"].shape[0] <= 16:
                return None, None
            frames = np.asarray(z["vertices"], dtype=np.float64)
        neutral, src = 15, "lab skin sequence ({0})".format(p.parent)
    return frames[neutral + 1::step] - frames[neutral], src


def verdict(speech_ratio, muscle_closing_frac=None):
    """OK / LOW / SEALED from the speech-opening ratio (fitted / lab head at the
    same size) and, if known, the share of lip-muscle slices that are closing
    (gap under a third of the lab head's there)."""
    if speech_ratio is None:
        return "UNKNOWN"
    if speech_ratio < SPEECH_SEALED:
        return "SEALED"
    if speech_ratio >= SPEECH_OK and (muscle_closing_frac is None or muscle_closing_frac <= MUSCLE_CLOSING_MAX):
        return "OK"
    return "LOW"


def _ratio(a, b):
    return None if a is None or b is None or b <= 0 else float(a / b)


def assess(X, F, parts, ref_skin, muscle=None, ref_muscle=None, motion=None, motion_scale=1.0):
    """Mouth of a fitted rest skin X against the lab head at the same size
    (ref_skin, ref_muscle: already scaled). motion: the lab head's speech
    displacements (K, V, 3) at lab size, added to both after scaling by
    motion_scale. -> metrics dict with a verdict."""
    out = {"edge_gap_mm": edge_gap_mm(X, parts), "ref_edge_gap_mm": edge_gap_mm(ref_skin, parts),
           "rest_opening_mm2": front_opening_mm2(X, F, parts),
           "ref_rest_opening_mm2": front_opening_mm2(ref_skin, F, parts)}
    sr = None
    if motion is not None:
        fit_o = openings(X[None] + motion_scale * motion, F, parts)
        ref_o = openings(ref_skin[None] + motion_scale * motion, F, parts)
        out.update({"speech_opening_mm2": float(fit_o.max()), "ref_speech_opening_mm2": float(ref_o.max()),
                    "speech_frames": int(len(motion)), "widest_motion_index": int(np.argmax(ref_o))})
        sr = _ratio(fit_o.max(), ref_o.max())
    out["speech_ratio"] = sr
    if muscle is not None and ref_muscle is not None:
        out.update(muscle_compare(muscle, X, ref_muscle, ref_skin, parts))
    out["verdict"] = verdict(sr, out.get("muscle_closing_frac"))
    return out


def assess_frames(fit_frames, ref_frames, F, parts, scale, step=2):
    """Simulated animations: the fitted head's skin frames against the generic
    head's (same sentence, lab size), compared at the subject's size (areas x
    scale^2). -> metrics dict with a verdict."""
    fo = openings(fit_frames[::step], F, parts)
    ro = openings(ref_frames[::step], F, parts) * scale ** 2
    n = min(len(fo), len(ro))
    sr = _ratio(fo[:n].max(), ro[:n].max())
    return {"speech_opening_mm2": float(fo[:n].max()), "ref_speech_opening_mm2": float(ro[:n].max()),
            "widest_frame": int(np.argmax(fo[:n]) * step), "ref_widest_frame": int(np.argmax(ro[:n]) * step),
            "speech_ratio": sr, "per_frame_mm2": [round(float(a), 1) for a in fo[:n]],
            "ref_per_frame_mm2": [round(float(a), 1) for a in ro[:n]], "frame_step": step,
            "verdict": verdict(sr)}

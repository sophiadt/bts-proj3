"""Gate: are the capture's left/right labels and its geometry the right way round?

Diagnoses only. Any fix belongs in the app and waits for a person to look at
the verdict on a real capture: `.leftMirrored` and the depth lookup's
transpose are both reflections and may currently cancel, so changing one
without the other can break a mapping that works.

Three measurements per view:
  handedness  sign of det[p263-p33, p10-p152, p1-mid] against the canonical
              mesh: do the labels agree with the cloud's handedness?
  turn        in a turned view, is the side MediaPipe calls "right" closer to
              the lens when the subject turned to their left (and vice versa)?
              Depth is physical, so this checks labels against anatomy.
  registration  do landmarks sit on a consistent surface? (z vs their mesh
              neighbours, off-face hits) -- a mirrored RGB->depth lookup on a
              turned, off-centre face puts landmarks on the wrong surface.

Verdicts (needs turned views to separate the mirror cases):
  RGB_DEPTH_MISMATCH  registration fails: depth sampled at the wrong pixels
  OK                  handedness ok, turn ok
  LABELS_SWAPPED      handedness flipped, turn wrong   (cloud fine, indices mirrored)
  GEOMETRY_MIRRORED   handedness flipped, turn right   (wrong-handed cloud: alignment
                                                        would need a reflection)
  MIRROR_TWIN         handedness ok,     turn wrong   (consistent mirror image: every
                                                        residual looks fine)
  INCONCLUSIVE_OK / INCONCLUSIVE_FLIPPED   no usable turned view; handedness only

    mayapy pipeline/mirror_check.py CAPTURE_DIR
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import geometry as geo                       # noqa: E402
from core import io as cio                             # noqa: E402
from pipeline import canonical as mp                   # noqa: E402
from pipeline import capture as cap                    # noqa: E402
from pipeline.fixture import mirror_permutation        # noqa: E402

TURN_MIN_MM = 4.0           # mean right-left depth difference that counts as a turn
REG_JUMP_MM = 15.0          # landmark z vs median of its mesh neighbours
REG_BAD_FRAC = 0.10
SIDE_X_MM = 15.0            # canonical |x| beyond which a vertex is clearly on one side


def handedness(p):
    need = [mp.EYE_R_OUTER, mp.EYE_L_OUTER, mp.FOREHEAD, mp.CHIN, mp.NOSE_TIP]
    if not np.isfinite(p[need]).all():
        return None
    mid = 0.5 * (p[mp.EYE_R_OUTER] + p[mp.EYE_L_OUTER])
    m = np.stack([p[mp.EYE_L_OUTER] - p[mp.EYE_R_OUTER], p[mp.FOREHEAD] - p[mp.CHIN], p[mp.NOSE_TIP] - mid])
    return float(np.sign(np.linalg.det(m)))


def neighbour_lists(faces, n):
    e = geo.edges(faces)
    nb = [[] for _ in range(n)]
    for a, b in e:
        nb[a].append(b)
        nb[b].append(a)
    return nb


def registration_bad_fraction(lm, valid, nb):
    oval = set(mp.FACE_OVAL)
    z = lm[:, 2]
    zmed = np.nanmedian(z[valid])
    bad, n = 0, 0
    for i in range(mp.N_CANONICAL):
        if not valid[i] or i in oval:
            continue
        n += 1
        if abs(z[i] - zmed) > 150.0:
            bad += 1
            continue
        nz = [z[j] for j in nb[i] if valid[j]]
        if nz and abs(z[i] - np.median(nz)) > REG_JUMP_MM:
            bad += 1
    return bad / max(n, 1)


def turn_delta(lm, valid, right_idx, left_idx):
    """mean z (camera, mm) of right-labelled minus left-labelled landmarks;
    pairs used only when both are valid."""
    pairs = [(r, l) for r, l in zip(right_idx, left_idx) if valid[r] and valid[l]]
    if len(pairs) < 10:
        return None
    r, l = np.array(pairs).T
    return float(np.mean(lm[r, 2] - lm[l, 2]))


def check(views):
    canon = mp.load()
    cv = canon["vertices"]
    ref = handedness(cv)
    perm = mirror_permutation(cv)
    right_idx = [i for i in range(mp.N_CANONICAL) if cv[i, 0] < -SIDE_X_MM]   # canonical: subject's right is -X
    left_idx = [int(perm[i]) for i in right_idx]
    nb = neighbour_lists(canon["faces"], mp.N_CANONICAL)

    per = {}
    for name, v in views.items():
        lm = v.landmarks
        valid = np.isfinite(lm).all(axis=1)
        h = handedness(lm)
        rec = {"n_valid": int(valid.sum()), "n_null": int((~valid).sum()),
               "n_depth_outliers": int(v.unreliable.sum()),
               "handedness_ok": None if h is None else bool(h == ref),
               "registration_bad_frac": registration_bad_fraction(lm, valid, nb),
               "nose_is_nearest": bool(valid[mp.NOSE_TIP] and lm[mp.NOSE_TIP, 2] <= np.nanpercentile(lm[valid, 2], 2))}
        prefix = v.prefix
        expect = config.TURN_FOR_PREFIX.get(prefix)
        dz = turn_delta(lm, valid, right_idx, left_idx)
        rec["turn_instruction"] = expect
        rec["right_minus_left_z_mm"] = dz
        if expect and dz is not None and abs(dz) >= TURN_MIN_MM:
            # Turning to the subject's left brings their right cheek toward the lens.
            want_neg = expect == "subject_left"
            rec["turn_ok"] = bool((dz < 0) == want_neg)
        else:
            rec["turn_ok"] = None
        per[name] = rec

    hands = [r["handedness_ok"] for r in per.values() if r["handedness_ok"] is not None]
    turns = [r["turn_ok"] for r in per.values() if r["turn_ok"] is not None]
    turned = [r for r in per.values() if r["turn_instruction"]]
    reg_bad = any(r["registration_bad_frac"] > REG_BAD_FRAC for r in (turned or per.values()))
    hand_ok = bool(hands) and all(hands)
    hand_flip = bool(hands) and not any(hands)

    if reg_bad:
        verdict = "RGB_DEPTH_MISMATCH"
    elif not hands or not (hand_ok or hand_flip):
        verdict = "INCONSISTENT"
    elif not turns:
        verdict = "INCONCLUSIVE_OK" if hand_ok else "INCONCLUSIVE_FLIPPED"
    elif not (all(turns) or not any(turns)):
        verdict = "INCONSISTENT"
    else:
        turn_ok = all(turns)
        verdict = {(True, True): "OK", (False, False): "LABELS_SWAPPED",
                   (False, True): "GEOMETRY_MIRRORED", (True, False): "MIRROR_TWIN"}[(hand_ok, turn_ok)]
    return {"verdict": verdict, "views": per, "canonical_handedness": ref}


ADVICE = {
    "OK": "Labels and geometry agree with anatomy. Proceed.",
    "LABELS_SWAPPED": "The cloud is correct but MediaPipe's left/right indices are mirrored. Fit with "
                      "--relabel, or fix the app (un-mirror MediaPipe's input AND remap depth to match). "
                      "Confirm on the RGB before changing the app.",
    "GEOMETRY_MIRRORED": "The 3D cloud is wrong-handed. Aligning it needs a reflection, which inverts "
                         "every tet. Do not fit. Fix the depth unprojection in the app.",
    "MIRROR_TWIN": "Labels and cloud are a consistent mirror image of the subject: the fit would "
                   "succeed and build the subject's mirror twin. Do not fit.",
    "RGB_DEPTH_MISMATCH": "Landmarks sample depth at the wrong pixels in turned views: the RGB->depth "
                          "remap does not match MediaPipe's input. Do not fit.",
    "INCONCLUSIVE_OK": "Handedness agrees but a mirror twin cannot be ruled out without turned views. "
                       "Capture with guided mode (front/left/right).",
    "INCONCLUSIVE_FLIPPED": "Handedness is flipped; turned views are needed to tell labels from geometry.",
    "INCONSISTENT": "Views disagree with each other; check whether the subject followed the turn prompts.",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    args = ap.parse_args()
    folder = Path(args.capture)
    with cio.RunDir(folder.name, "mirror_check", "pipeline/mirror_check.py") as run:
        views = cap.load_capture(folder)
        res = check(views)
        res["advice"] = ADVICE.get(res["verdict"])
        run.metrics.update(res)
        for name, r in res["views"].items():
            print("  {0:6s} valid {1:3d} handedness_ok {2} turn {3} dz {4} reg_bad {5:.2f}".format(
                name, r["n_valid"], r["handedness_ok"], r["turn_ok"],
                None if r["right_minus_left_z_mm"] is None else round(r["right_minus_left_z_mm"], 1),
                r["registration_bad_frac"]))
        print("VERDICT:", res["verdict"])
        print(res["advice"])
        if res["verdict"] != "OK":
            sys.exit(2)


if __name__ == "__main__":
    main()

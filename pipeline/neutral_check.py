"""Is the captured face neutral? Measured from 3D landmark geometry, in mm.

MediaPipe's blendshape scores (saved in meta.json) are not used: they read
monolid and hooded eyes as squinting and drift with camera angle (a neutral
face scored eyeSquint ~0.5 and mouthPucker 0.71 on the first capture). The
fit never uses them either.

Only landmarks whose depth passed capture.py's checks are measured (the first
capture's chin point read 80 mm behind the nose tip: neck depth, not chin).

What is checked, and what may warn:
  mouth open   inner lip gap (13-14) > MOUTH_OPEN_MM                 warns
  jaw open     nose-to-chin length (1-152) differs from the person's   warns
               own other views by > JAW_SPREAD_FRAC of their
               outer-eye-corner width (33-263) -- each person is
               their own baseline, no population norms
  eyes         eyelid openings (159-145, 386-374) in mm               never warns:
                                                                       natural openings
                                                                       vary too much
  phone height where the camera sat as seen from the face, from the    warns
               rigid fit (hundreds of weighted points, not two)
               -- more than PHONE_TILT_WARN_DEG above or below
"""

import numpy as np

from pipeline import canonical as mp

MOUTH_OPEN_MM = 3.0
JAW_SPREAD_FRAC = 0.06
PHONE_TILT_WARN_DEG = 15.0
LIP_UPPER_INNER, LIP_LOWER_INNER = 13, 14
R_LID = (159, 145)
L_LID = (386, 374)


def _d(p, a, b):
    if not (np.isfinite(p[a]).all() and np.isfinite(p[b]).all()):
        return None
    return float(np.linalg.norm(p[a] - p[b]))


def camera_angles(direction):
    """(elevation, side) in degrees for a unit vector from the face to the
    camera in the template frame (Y up, subject's right +X, face toward -Z).
    Elevation < 0: phone below the face. Side > 0: toward the subject's right."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    return (float(np.degrees(np.arcsin(np.clip(d[1], -1, 1)))),
            float(np.degrees(np.arcsin(np.clip(d[0], -1, 1)))))


def check(views, camera_dirs=None, primary_name="front"):
    """views: {name: capture.View}; camera_dirs: optional {name: face->camera
    unit vector in the template frame}, from the rigid fit."""
    per, warnings = {}, []
    for name, v in views.items():
        p = np.where(v.valid[:, None], v.landmarks, np.nan)
        rec = {"lip_gap_mm": _d(p, LIP_UPPER_INNER, LIP_LOWER_INNER),
               "mouth_width_mm": _d(p, mp.MOUTH_R, mp.MOUTH_L),
               "nose_chin_mm": _d(p, mp.NOSE_TIP, mp.CHIN),
               "eye_width_mm": _d(p, mp.EYE_R_OUTER, mp.EYE_L_OUTER),
               "eyelid_opening_r_mm": _d(p, *R_LID),
               "eyelid_opening_l_mm": _d(p, *L_LID)}
        if camera_dirs and name in camera_dirs:
            rec["camera_elevation_deg"], rec["camera_side_deg"] = camera_angles(camera_dirs[name])
        per[name] = rec
        if rec["lip_gap_mm"] is not None and rec["lip_gap_mm"] > MOUTH_OPEN_MM:
            warnings.append("{0}: lips apart {1:.1f} mm (relax the mouth and close the lips)".format(
                name, rec["lip_gap_mm"]))

    jaw = [(n, r["nose_chin_mm"], r["eye_width_mm"]) for n, r in per.items()
           if r["nose_chin_mm"] and r["eye_width_mm"]]
    if len(jaw) >= 2:
        ref = float(np.median([j[1] for j in jaw]))
        for n, length, ew in jaw:
            if abs(length - ref) > JAW_SPREAD_FRAC * ew:
                warnings.append("{0}: nose-to-chin {1:.1f} mm vs {2:.1f} mm in the other views "
                                "(jaw moved between photos?)".format(n, length, ref))

    prim = per.get(primary_name) or next(iter(per.values()))
    elev = prim.get("camera_elevation_deg")
    if elev is not None and abs(elev) > PHONE_TILT_WARN_DEG:
        warnings.append("{0}: phone {1:.0f} deg {2} the face (hold the phone at eye level)".format(
            primary_name, abs(elev), "below" if elev < 0 else "above"))
    return {"views": per, "warnings": warnings, "neutral": not warnings,
            "blendshapes": "saved by the app, not used (unreliable across eye shapes and camera angles)"}

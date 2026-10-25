"""Measure the lab template's face against adult norms: stylised, or a unit error?

The template's eye opening measures about 44 mm corner to corner against about
29 mm on real adults. If one global scale factor brings every feature into the
normal range, the 0.962 mm-per-unit figure is wrong. If features stay out of
range after the best global scale, the model is stylised and the fit must
expect large non-uniform warps.

Landmarks are found automatically on the skin mesh (frame 0, mm, template
frame: Y-up, face toward -Z, so the subject's right is +X):
  - eyes and mouth from the skin's boundary loops (the openings);
  - a midline profile from slicing the skin with the midplane (halfway
    between the eye openings; x = 0.2 mm on the template).
Norms: 3D Facial Norms Database, adults 18-40 (fixtures/anthropometry_norms.json).

    .venv\\Scripts\\python checks\\template_features.py [--sex male|female]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import io as cio                             # noqa: E402
from core.metrics import boundary_loops                # noqa: E402

NORMS_JSON = config.REPO_ROOT / "fixtures" / "anthropometry_norms.json"
SKIN = "skin_cloth_copy_v5_pull_back"

# Decision rule, fixed before the first run on the template.
RULE = {
    "outlier_sd": 2.0,          # |z| above this, after removing the best global scale, is out of range
    "scale_error_frac": 0.10,   # |k - 1| above this counts as a material scale error
}

# Measures whose template landmark is ambiguous. They are reported but kept out
# of the global-scale fit and the verdict.
AMBIGUOUS = {
    "labfiswidth": "The skin's mouth opening wraps about 24 mm into the mouth, so its "
                   "corners lie behind the visible commissures; the loop width over-states ch-ch.",
}


# --- landmark finding -----------------------------------------------------------

def classify_loops(verts, loops):
    """Assign the skin's boundary loops: rim, eye_r, eye_l, mouth.

    The rim is the longest loop. Of the rest, the two highest are the eyes,
    split by x (subject's right is +X); the lowest near the midline is the mouth.
    """
    if len(loops) < 4:
        raise ValueError("expected rim + two eyes + mouth, got {0} loops".format(len(loops)))
    inner = sorted(loops[1:], key=lambda L: -verts[L][:, 1].mean())
    eyes, rest = inner[:2], inner[2:]
    x0 = midplane_x(verts, eyes)
    mouth = min(rest, key=lambda L: abs(verts[L][:, 0].mean() - x0))
    eye_r, eye_l = sorted(eyes, key=lambda L: -verts[L][:, 0].mean())
    return {"rim": loops[0], "eye_r": eye_r, "eye_l": eye_l, "mouth": mouth}


def midplane_x(verts, eye_loops):
    """x of the face's midplane: halfway between the two eye openings' centres."""
    return float(np.mean([verts[L][:, 0].mean() for L in eye_loops]))


def loop_diameter(points):
    """The two points of a loop farthest apart."""
    d = np.linalg.norm(points[:, None] - points[None], axis=2)
    i, j = np.unravel_index(int(d.argmax()), d.shape)
    return points[i], points[j]


def canthi(eye_points, x0=0.0):
    """(endocanthion, exocanthion): ends of the opening's diameter, inner one nearer the midplane."""
    a, b = loop_diameter(eye_points)
    return (a, b) if abs(a[0] - x0) < abs(b[0] - x0) else (b, a)


def midline_profile(verts, faces, x0=0.0):
    """Points of the skin cut by the plane x = 0, ordered from the top of the head down.

    The mouth opening breaks the cut into two curves (above and below the
    mouth); each is ordered top-down and they are joined upper piece first.
    """
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    segs = trimesh.intersections.mesh_plane(mesh, plane_normal=[1.0, 0.0, 0.0],
                                            plane_origin=[x0, 0.0, 0.0])
    if len(segs) == 0:
        raise ValueError("the midplane does not cut the skin")
    pts = segs.reshape(-1, 3)
    key = np.round(pts, 6)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1, 2)
    adj = {}
    for a, b in inv:
        if a != b:
            adj.setdefault(int(a), set()).add(int(b))
            adj.setdefault(int(b), set()).add(int(a))
    pieces, seen = [], set()
    while len(seen) < len(adj):
        left = [n for n in adj if n not in seen]
        ends = [n for n in left if len(adj[n]) == 1]
        start = max(ends or left, key=lambda n: uniq[n][1])
        order, cur = [start], start
        seen.add(start)
        while True:
            nxt = [n for n in adj[cur] if n not in seen]
            if not nxt:
                break
            cur = nxt[0]
            seen.add(cur)
            order.append(cur)
        piece = uniq[order]
        if piece[0][1] < piece[-1][1]:
            piece = piece[::-1]
        pieces.append(piece)
    pieces = [p for p in pieces if len(p) >= 10]          # drop slivers
    pieces.sort(key=lambda p: -p[:, 1].max())
    return np.concatenate(pieces)


def _deepest_behind_chord(profile, i0, i1):
    """Index between i0 and i1 lying farthest behind (+Z) the chord joining them.

    The soft-tissue nasion and subnasale are the deepest points of a concavity;
    measuring depth from the chord still finds them when the profile has no
    local maximum in z.
    """
    p0, p1 = profile[i0][1:], profile[i1][1:]
    lo, hi = min(i0, i1), max(i0, i1)
    seg = profile[lo:hi + 1, 1:]
    d = p1 - p0
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    if n[1] < 0:                                     # point the normal backwards, +Z
        n = -n
    depth = (seg - p0) @ n
    return lo + int(depth.argmax())


def find_landmarks(verts, faces):
    """Template skin landmarks in mm, keyed by the 3D Facial Norms landmark names."""
    loops = classify_loops(verts, boundary_loops(faces))
    x0 = midplane_x(verts, [loops["eye_r"], loops["eye_l"]])
    en_r, ex_r = canthi(verts[loops["eye_r"]], x0)
    en_l, ex_l = canthi(verts[loops["eye_l"]], x0)
    ch_a, ch_b = loop_diameter(verts[loops["mouth"]])
    ch_r, ch_l = (ch_a, ch_b) if ch_a[0] > ch_b[0] else (ch_b, ch_a)

    mouth = verts[loops["mouth"]]
    # Stomion: where the mouth opening crosses the midplane (window = 5% of its width).
    near_mid = mouth[np.abs(mouth[:, 0] - x0) < 0.05 * np.linalg.norm(ch_r - ch_l)]
    sto = near_mid.mean(axis=0) if len(near_mid) else mouth.mean(axis=0)

    prof = midline_profile(verts, faces, x0)
    i_prn = int(prof[:, 2].argmin())                                  # pronasale: most anterior
    i_n = _deepest_behind_chord(prof, 0, i_prn)                       # nasion: forehead top .. prn
    below = np.arange(i_prn, len(prof))
    # Labrale superius: most anterior point of the upper lip, within 20 mm above
    # the stomion (adult philtrum + upper vermilion is about 20 mm).
    lip = below[(prof[below, 1] > sto[1]) & (prof[below, 1] < sto[1] + 20.0)]
    i_ls = int(lip[prof[lip, 2].argmin()]) if len(lip) else i_prn     # labrale superius
    i_sn = _deepest_behind_chord(prof, i_prn, i_ls)                   # subnasale
    chin = below[prof[below, 1] < sto[1]]
    i_gn = int(chin[(-(prof[chin, 1] + prof[chin, 2])).argmax()])     # gnathion: most anterior-inferior

    return {
        "en_r": en_r, "ex_r": ex_r, "en_l": en_l, "ex_l": ex_l,
        "ch_r": ch_r, "ch_l": ch_l, "sto": sto,
        "n": prof[i_n], "prn": prof[i_prn], "sn": prof[i_sn], "ls": prof[i_ls], "gn": prof[i_gn],
    }, prof


def measure(landmarks, norms):
    """Template value (mm) for every norms measure, from its landmark pair."""
    return {key: float(np.linalg.norm(landmarks[m["landmarks"][0]] - landmarks[m["landmarks"][1]]))
            for key, m in norms["measures"].items()}


# --- comparison -----------------------------------------------------------------

def compare(values, norms, sex="male", rule=RULE, ambiguous=AMBIGUOUS):
    """Ratios, z-scores before and after the best global scale, and the verdict."""
    rows = {}
    for key, value in values.items():
        ref = norms["measures"][key][sex]
        rows[key] = {"template_mm": value, "norm_mean_mm": ref["mean_mm"], "norm_sd_mm": ref["sd_mm"],
                     "ratio": value / ref["mean_mm"], "z": (value - ref["mean_mm"]) / ref["sd_mm"],
                     "used_in_fit": key not in ambiguous}
    fit = [r for r in rows.values() if r["used_in_fit"]]
    k = math.exp(sum(math.log(r["ratio"]) for r in fit) / len(fit))
    for r in rows.values():
        r["z_after_k"] = (r["template_mm"] / k - r["norm_mean_mm"]) / r["norm_sd_mm"]
    outliers = sorted(key for key, r in rows.items()
                      if r["used_in_fit"] and abs(r["z_after_k"]) > rule["outlier_sd"])
    if outliers:
        verdict = "stylised"
    elif abs(k - 1.0) > rule["scale_error_frac"]:
        verdict = "scale_error"
    else:
        verdict = "consistent_with_0.962"
    # Post-hoc diagnostic, added after the first run and NOT part of the rule:
    # the global scale implied by the non-eye features alone, and how far the
    # eye region sits from the norms once that scale is removed.
    eye = {"palpfislength_r", "palpfislength_l", "incanthwidth", "outcanthwidth"}
    rest = [r for key, r in rows.items() if r["used_in_fit"] and key not in eye]
    k_rest = math.exp(sum(math.log(r["ratio"]) for r in rest) / len(rest)) if rest else None
    diagnostic = None
    if k_rest:
        diagnostic = {"k_without_eye_region": k_rest,
                      "mm_per_unit_implied": config.MAYA_UNIT_MM / k_rest,
                      "eye_z_after_k_without_eye_region": {
                          key: (rows[key]["template_mm"] / k_rest - rows[key]["norm_mean_mm"])
                          / rows[key]["norm_sd_mm"] for key in sorted(eye & set(rows))}}
    return {"sex": sex, "k": k, "corrected_mm_per_unit_if_scale_error": config.MAYA_UNIT_MM / k,
            "post_hoc_diagnostic": diagnostic,
            "verdict": verdict, "outliers_after_k": outliers, "rule": rule,
            "ambiguous": ambiguous, "rows": rows}


# --- views ----------------------------------------------------------------------

def save_views(verts, landmarks, profile, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax_f, ax_s) = plt.subplots(1, 2, figsize=(12, 7))
    step = max(1, len(verts) // 6000)
    ax_f.scatter(verts[::step, 0], verts[::step, 1], s=0.3, c="0.75")
    ax_s.scatter(verts[::step, 2], verts[::step, 1], s=0.3, c="0.75")
    ax_s.plot(profile[:, 2], profile[:, 1], c="tab:blue", lw=1)
    for name, p in landmarks.items():
        ax_f.scatter(p[0], p[1], s=14, c="tab:red")
        ax_f.annotate(name, (p[0], p[1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        if name in ("n", "prn", "sn", "ls", "sto", "gn"):
            ax_s.scatter(p[2], p[1], s=14, c="tab:red")
            ax_s.annotate(name, (p[2], p[1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax_f.set_title("Front (x right = subject's right, mm)")
    ax_s.set_title("Side (z: face toward -Z, mm)")
    for ax in (ax_f, ax_s):
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sex", choices=["male", "female"], default="male")
    args = ap.parse_args(argv)

    with cio.RunDir("template", "features", "checks/template_features.py") as run:
        norms = json.loads(NORMS_JSON.read_text(encoding="utf-8"))
        registry = cio.load_registry()
        skin = cio.load_lab_mesh(SKIN, registry=registry)
        run.metrics["provenance"] = registry["npz_source"]["provenance"]
        run.metrics["norms_source"] = norms["source"]["citation"]

        landmarks, profile = find_landmarks(skin.vertices, skin.faces)
        values = measure(landmarks, norms)
        result = compare(values, norms, sex=args.sex)
        result["female_check"] = compare(values, norms, sex="female")["verdict"]
        run.metrics.update(result)
        run.metrics["landmarks_mm"] = {k: v.round(3).tolist() for k, v in landmarks.items()}
        run.metrics["status_note"] = "provisional: npz exported from a different copy of the scene"
        save_views(skin.vertices, landmarks, profile, run.path / "views" / "template_landmarks.png")

    print("verdict: {0} (k = {1:.3f}, {2} norms; female norms: {3})".format(
        result["verdict"], result["k"], args.sex, result["female_check"]))
    for key, r in result["rows"].items():
        print("  {0:16} template {1:6.1f}  norm {2:6.1f} +/- {3:4.1f}  ratio {4:4.2f}  "
              "z {5:6.1f}  z/k {6:6.1f}{7}".format(
                  key, r["template_mm"], r["norm_mean_mm"], r["norm_sd_mm"], r["ratio"],
                  r["z"], r["z_after_k"], "" if r["used_in_fit"] else "  (ambiguous, not fitted)"))
    d = result["post_hoc_diagnostic"]
    if d:
        print("post-hoc (not the rule): non-eye features imply k = {0:.3f} ({1:.3f} mm/unit); "
              "eye-region z after that scale: {2}".format(
                  d["k_without_eye_region"], d["mm_per_unit_implied"],
                  ", ".join("{0} {1:.1f}".format(k, z) for k, z in d["eye_z_after_k_without_eye_region"].items())))
    print("run folder:", run.path)


if __name__ == "__main__":
    main()

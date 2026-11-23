"""Gate: does the fitted head's mouth open like the lab head's? (no Maya, no Ziva)

    python tools/mouth_check.py SUBJECT
        The simulated animation. Reads template_export/subjects/<id>/:
        anim_ui_fit.npz and anim_ui_generic.npz (rig/ui_smoke_test.py, the
        sentence on the fitted and on the generic head), report.json (scale),
        skin_fit_output.npz and the lip muscle's rest. Needs template_export/
        (on the workstation's disk) but not the Ziva licence.

    python tools/mouth_check.py --preview CAPTURE [--start base|lab] [--close-mouth | --compare]
        Before any Maya time, on any machine with the lab skin
        (template_export/ or ../bts-proj1/data/lab): the skin-only fit of
        pipeline/fit_subject.fit_skin, the lip muscle carried by the same kind
        of smooth field (from the skin only; the full fit also pins it at the
        bones), and the lab head's speech motion added to the fitted rest.
        --start base (default) builds the human base head's skin first, as
        pipeline/build_base.py does; --compare runs the old mouth pull too.

Measures (pipeline/mouth.py), each against the lab head at the subject's size:
mouth-hole area seen from the front at rest and at the widest speech frame,
the hole edges' gap at the middle, and the gap between the lip muscle's upper
and lower halves. Verdict OK (speech opening >= 70% of the lab head's, and at
most 5% of lip-muscle slices closing: gap under a third of the lab head's
there), LOW, or SEALED (speech opening < 40%). Exit 2 unless OK.

Why: the lab rig rests with the lips apart, and the first fits pulled the
mouth hole shut on closed-lip captures; the heads then animated with a sealed
mouth that sim_check and ui_smoke_test both passed.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import geometry as geo                       # noqa: E402
from core import io as cio                             # noqa: E402
from pipeline import capture as cap                    # noqa: E402
from pipeline import fit_subject as fs                 # noqa: E402
from pipeline import fixture as fx                     # noqa: E402
from pipeline import merge_views, mirror_check, mouth, synthetic, volumetric  # noqa: E402

UI_NEUTRAL_FRAME = 14


# --- pictures -------------------------------------------------------------------------

def _front(ax, X, F, parts, title):
    from matplotlib.collections import PolyCollection
    Q = X[parts["loop"]]
    w, c = float(np.ptp(Q[:, 0])), Q.mean(axis=0)
    cen = X[F].mean(axis=1)
    near = (np.abs(cen[:, 0] - c[0]) < 0.75 * w) & (np.abs(cen[:, 1] - c[1]) < 0.75 * w) & \
           (np.abs(cen[:, 2] - c[2]) < w)
    Fn = F[near]
    n = geo.face_normals(X, Fn)
    light = np.array([-0.3, 0.4, -1.0]) / np.linalg.norm([-0.3, 0.4, -1.0])
    shade = 0.35 + 0.65 * np.abs(n @ light)
    order = np.argsort(-X[Fn][:, :, 2].mean(axis=1))
    cols = np.clip(np.array([0.80, 0.66, 0.58]) * shade[:, None], 0, 1)
    ax.add_collection(PolyCollection(np.c_[-X[:, 0], X[:, 1]][Fn[order]], facecolors=cols[order],
                                     edgecolors="none", antialiased=False))
    ax.set_xlim(-c[0] - 0.65 * w, -c[0] + 0.65 * w)
    ax.set_ylim(c[1] - 0.38 * w, c[1] + 0.38 * w)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#1d2733")
    ax.set_title(title, fontsize=9)


def _section(ax, X, F, P, PF, parts, title):
    from matplotlib.collections import LineCollection

    def cut(V, faces, x0):
        d = V[:, 0] - x0
        segs = []
        for a, b, c in faces:
            pts = []
            for i, j in ((a, b), (b, c), (c, a)):
                if (d[i] < 0) != (d[j] < 0):
                    q = V[i] + d[i] / (d[i] - d[j]) * (V[j] - V[i])
                    pts.append((-q[2], q[1]))
            if len(pts) == 2:
                segs.append(pts)
        return segs

    Q = X[parts["loop"]]
    w, c = float(np.ptp(Q[:, 0])), Q.mean(axis=0)
    x0 = 0.5 * (X[parts["mid_upper"], 0] + X[parts["mid_lower"], 0])
    ax.add_collection(LineCollection(cut(X, F, x0), colors="#c98b62", linewidths=1.6))
    if P is not None:
        ax.add_collection(LineCollection(cut(P, PF, x0), colors="#b03a3a", linewidths=1.1))
    ax.set_xlim(-c[2] - 0.1 * w, -c[2] + 0.55 * w)
    ax.set_ylim(c[1] - 0.32 * w, c[1] + 0.32 * w)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def picture(path, rows, F, parts, PF, suptitle):
    """rows: [(label, rest skin, speaking skin, lip muscle or None)]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.4 * len(rows)), facecolor="white", squeeze=False)
    for r, (label, rest, speak, P) in enumerate(rows):
        _front(axes[r, 0], rest, F, parts, label + ": at rest")
        _front(axes[r, 1], speak, F, parts, label + ": widest speech frame")
        _section(axes[r, 2], rest, F, P, PF, parts, label + ": midline at rest, skin (tan) and lip muscle (red)")
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=70)
    plt.close(fig)
    return path


def _fmt(rep):
    f = lambda v, p=1: "n/a" if v is None else "{0:.{1}f}".format(v, p)
    pct = lambda v: "n/a" if v is None else "{0:.0%}".format(v)
    lines = ["  hole edges at the middle  {0} mm   (lab head {1} mm)".format(
                 f(rep.get("edge_gap_mm")), f(rep.get("ref_edge_gap_mm"))),
             "  opening at rest           {0} mm2  (lab head {1} mm2)".format(
                 f(rep.get("rest_opening_mm2"), 0), f(rep.get("ref_rest_opening_mm2"), 0)),
             "  opening when speaking     {0} mm2  (lab head {1} mm2) = {2}".format(
                 f(rep.get("speech_opening_mm2"), 0), f(rep.get("ref_speech_opening_mm2"), 0),
                 pct(rep.get("speech_ratio")))]
    if rep.get("muscle_gap_median_mm") is not None:
        lines.append("  lip muscle gap            median {0} mm, min {1} mm  (lab head {2} / {3}); narrowest "
                     "{4} of the lab head's there; {5} of {6} slices closing (under a third)".format(
                         f(rep["muscle_gap_median_mm"], 2), f(rep["muscle_gap_min_mm"], 2),
                         f(rep["ref_muscle_gap_median_mm"], 2), f(rep["ref_muscle_gap_min_mm"], 2),
                         pct(rep.get("muscle_ratio_min")), rep.get("muscle_slices_closing"), rep["muscle_slices"]))
    lines.append("  -> " + rep["verdict"])
    return "\n".join(lines)


# --- the simulated animation (workstation outputs) ----------------------------------------

def check_subject(subject):
    tpl = config.template_export_dir()
    sub = tpl / "subjects" / subject
    with open(sub / "report.json", encoding="utf-8") as fh:
        M = json.load(fh)["metrics"]
    s = float(M.get("scale_vs_lab_head") or M["scale"])
    if M.get("start_head", "lab head") != "lab head" and "scale_vs_lab_head" not in M:
        raise RuntimeError("report.json has no scale_vs_lab_head for a base-head fit; re-run fit_subject.py")
    lab_skin, F = mouth.load_lab_mesh(fx.SKIN)
    parts = mouth.mouth_parts(lab_skin, F)
    with np.load(sub / "anim_ui_fit.npz") as z:
        fit_frames = np.asarray(z["skin"], dtype=np.float64)
    gen_file = sub / "anim_ui_generic.npz"
    if not gen_file.exists():
        gen_file = tpl / "anim_ui_generic.npz"
    with np.load(gen_file) as z:
        gen_frames = np.asarray(z["skin"], dtype=np.float64)
    with cio.RunDir(subject, "mouth_check", "tools/mouth_check.py") as run:
        rep = mouth.assess_frames(fit_frames, gen_frames, F, parts, s)
        k0 = min(UI_NEUTRAL_FRAME, len(fit_frames) - 1, len(gen_frames) - 1)
        rest, ref_rest = fit_frames[k0], s * gen_frames[k0]
        rep.update({"rest_frame": k0, "edge_gap_mm": mouth.edge_gap_mm(rest, parts),
                    "ref_edge_gap_mm": mouth.edge_gap_mm(ref_rest, parts),
                    "rest_opening_mm2": mouth.front_opening_mm2(rest, F, parts),
                    "ref_rest_opening_mm2": mouth.front_opening_mm2(ref_rest, F, parts), "scale_vs_lab_head": s})
        P, PF = None, None
        oo_file = sub / (mouth.LIP_MUSCLE + ".npz")
        if oo_file.exists():
            with np.load(oo_file) as z:
                P = np.asarray(z["rest_vertices"], dtype=np.float64)
            lab_oo, PF = mouth.load_lab_mesh(mouth.LIP_MUSCLE)
            with np.load(sub / "skin_fit_output.npz") as z:
                X = np.asarray(z["vertices"], dtype=np.float64)
            rep.update(mouth.muscle_compare(P, X, s * lab_oo, s * lab_skin, parts))
        rep["verdict"] = mouth.verdict(rep["speech_ratio"], rep.get("muscle_closing_frac"))
        run.metrics.update(rep)
        k1, k2 = rep["widest_frame"], rep["ref_widest_frame"]
        png = picture(sub / "views" / "mouth_check.png",
                      [(subject, rest, fit_frames[k1], P), ("lab head at this size", ref_rest, s * gen_frames[k2],
                                                            None if P is None else s * lab_oo)],
                      F, parts, PF, "Mouth check: {0} (keyboard UI, simulated)".format(subject))
        print("mouth check:", subject, "(simulated keyboard-UI sentence, frames {0} and {1})".format(k1, k2))
        print(_fmt(rep))
        print("  picture:", png)
    return rep


# --- preview: skin-only fit, any machine ------------------------------------------------

def preview(capture_arg, start="base", modes=(fs.MOUTH_GAP,), force=False, smooth_mm=0.0, ring_depth=False, smooth_skin=0):
    capture = cap.resolve_capture(capture_arg)
    cid = cap.capture_id(capture)
    manifest = cap.read_manifest(capture) or {}
    lab_skin, F = mouth.load_lab_mesh(fx.SKIN)
    lab_oo, PF = mouth.load_lab_mesh(mouth.LIP_MUSCLE)
    parts = mouth.mouth_parts(lab_skin, F)
    fixture = fx.load_fixture()
    motion, src = mouth.lab_speech_motion()
    views = cap.load_capture(capture)
    if manifest.get("labels") == "swapped":
        fs.relabel(views)
    gate = mirror_check.check(views)["verdict"]
    if gate != "OK" and not force:
        raise RuntimeError("mirror gate: {0} ({1})".format(gate, mirror_check.ADVICE.get(gate)))
    results, rows = {}, []
    with cio.RunDir(cid, "mouth_preview", "tools/mouth_check.py --preview") as run:
        run.metrics.update({"capture": cid, "start": start, "mirror_verdict": gate, "motion_source": src,
                            "approximate": "skin-only fit; lip muscle moved by a field from the skin only"})
        for mode in modes:
            # the start head, as fit_subject would see it
            if start == "base":
                with tempfile.TemporaryDirectory() as tmp:
                    avg = synthetic.make_guided(Path(tmp) / "average_face")
                    mvb = merge_views.merge(cap.load_capture(avg), fixture, lab_skin)
                sb = mvb["scale"]
                Xb, X0b, _ = fs.fit_skin(lab_skin, F, mvb, fixture, sb, gap_ref=sb * lab_skin, base_mode=True,
                                         mouth=mode, verbose=False)
                fb = volumetric.build_field(X0b, Xb - X0b)
                V, s_base = Xb, sb
                oo_start = sb * lab_oo + volumetric.eval_field(fb, sb * lab_oo)
            else:
                V, s_base, oo_start = lab_skin, 1.0, lab_oo
            mv = merge_views.merge(views, fixture, V)
            s = mv["scale"]
            s_lab = s * s_base
            X, X0, ctx = fs.fit_skin(V, F, mv, fixture, s, gap_ref=s_lab * lab_skin, mouth=mode, verbose=False,
                                     smooth_mm=smooth_mm, ring_depth=ring_depth, smooth_skin=smooth_skin)
            field = volumetric.build_field(X0, X - X0)
            P = s * oo_start + volumetric.eval_field(field, s * oo_start)
            rep = mouth.assess(X, F, parts, s_lab * lab_skin, muscle=P, ref_muscle=s_lab * lab_oo,
                               motion=motion, motion_scale=s_lab)
            n = ctx["metrics"]["nonrigid"]
            rep.update({"scale_vs_lab_head": s_lab, "landmark_residual_median_mm": n["landmark_residual_median_mm"],
                        "cloud_to_skin_median_mm": n["cloud_to_skin_median_mm"],
                        "skin_folds": int(geo.flipped_faces(X0, X, F, min_area_ratio=0.05).sum()),
                        "seam_points_dropped": ctx["metrics"].get("mouth_seam_points_dropped", 0)})
            results[mode] = rep
            label = {"gap": "mouth keeps the rig's gap", "contours": "old: mouth pulled onto the lips"}[mode]
            print("{0} from the {1} head, mouth mode '{2}' ({3}):".format(cid, start, mode, label))
            print(_fmt(rep))
            print("  skin: landmarks median {0:.2f} mm, scan->skin median {1:.2f} mm, {2} folded triangles".format(
                rep["landmark_residual_median_mm"], rep["cloud_to_skin_median_mm"], rep["skin_folds"]))
            k = rep.get("widest_motion_index", 0)
            speak = (lambda Y: Y + s_lab * motion[k]) if motion is not None else (lambda Y: Y)
            rows.append(("{0} ({1})".format(cid.split("_")[-1], mode), X, speak(X), P))
        ref = s_lab * lab_skin
        rows.append(("lab head at this size", ref, ref + s_lab * motion[k] if motion is not None else ref,
                     s_lab * lab_oo))
        run.metrics["modes"] = results
        png = picture(run.path / "views" / "mouth_preview.png", rows, F, parts, PF,
                      "Mouth preview: {0} from the {1} head (skin-only fit; lab speech motion added)".format(cid, start))
        print("  picture:", png)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="subject id (template_export/subjects/<id>), or with --preview a capture id/folder")
    ap.add_argument("--preview", action="store_true", help="skin-only fit of a capture on this machine")
    ap.add_argument("--start", choices=("base", "lab"), default="base", help="preview: start head (default base)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--close-mouth", action="store_true", help="preview: the old mouth pull only")
    g.add_argument("--compare", action="store_true", help="preview: the old pull and the kept gap")
    ap.add_argument("--force", action="store_true", help="preview: past a failing mirror gate")
    ap.add_argument("--smooth-contours", nargs="?", type=float, const=fs.holes.SMOOTH_SIGMA_MM, default=0.0,
                    metavar="MM", help="preview: smooth the eyelid/lip landmark contours (bare flag: {0} mm)"
                                       .format(fs.holes.SMOOTH_SIGMA_MM))
    ap.add_argument("--ring-depth", action="store_true",
                    help="preview: eyelid and lip-line depth from the skin around each opening")
    ap.add_argument("--smooth-skin", nargs="?", type=int, const=10, default=0, metavar="N",
                    help="preview: N Taubin smoothing iterations on the skin after the fit (bare flag: 10)")
    args = ap.parse_args(argv)
    if args.preview:
        modes = ((fs.MOUTH_CONTOURS, fs.MOUTH_GAP) if args.compare else
                 (fs.MOUTH_CONTOURS,) if args.close_mouth else (fs.MOUTH_GAP,))
        res = preview(args.target, start=args.start, modes=modes, force=args.force,
                      smooth_mm=args.smooth_contours, ring_depth=args.ring_depth, smooth_skin=args.smooth_skin)
        final = res[modes[-1]]["verdict"]
    else:
        final = check_subject(args.target)["verdict"]
    return 0 if final == "OK" else 2


if __name__ == "__main__":
    sys.exit(main())

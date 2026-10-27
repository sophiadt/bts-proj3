"""Gate: does a written rig show the fitted head, keep the lab rig's mesh, and simulate?

  shape     animation muted, frame 0: the skin vs the fitted skin the fit
            produced (skin_fit_output.npz, every vertex), every other
            registered body vs the rest written for it (p99, mm)
  mesh      every body's triangles and UVs identical to the untouched lab
            scene (the rig, its weights and textures carry over only if so)
  animated  the speech animation, frames 0..END: everything finite, and the
            worst per-frame vertex speed per body compared with the untouched
            template's on the same frames (a blow-up shows as a large ratio)

The skin is compared with what should be seen, not with its rest: the rig
adds the template's fixed output-minus-rest offset (the forehead
"pull_back", up to 16.6 mm), which fit_subject.py already subtracts from the
written rest. Legacy bodies' vertices with > 1 mm template offset stay
excluded, as in roundtrip_test.

    mayapy rig/sim_check.py SCENE.mb --subject DIR [--end 50] [--no-baseline]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import io as cio                             # noqa: E402
from rig import _maya as M                             # noqa: E402
from rig.roundtrip_test import FREE_MM, registered     # noqa: E402
from rig.write_rest import load_rest_dir               # noqa: E402

REST_P99_MM = 0.5
SPEED_RATIO_MAX = 3.0


SKIN = "skin_cloth_copy_v5_pull_back"


def mesh_signature(shape, rest):
    """Polygon connectivity of the rest mesh, UVs of the visible mesh.

    The rest mesh is what the rig is built on (attachments, weights), so its
    polygons are what must not change. Not the visible mesh's: Maya splits
    quads along whichever diagonal suits their current shape, in
    getTriangles() and in polyTriangulate nodes (OrbicularisOris has one
    downstream of Ziva), so reshaping changes that split while no vertex or
    face changes. Measured: OrbicularisOris rest polygons 0 of 40,960 entries
    different; its triangulated output 5,708 of 61,440.
    """
    fn = M.mesh_fn(shape)
    u, v = fn.getUVs()
    counts, ids = fn.getAssignedUVs()
    pcounts, pverts = M.mesh_fn(rest).getVertices()
    return {"poly_counts": np.asarray(pcounts), "poly_verts": np.asarray(pverts),
            "n_verts": np.asarray([fn.numVertices]),
            "u": np.asarray(u), "v": np.asarray(v),
            "uv_counts": np.asarray(counts), "uv_ids": np.asarray(ids)}


def speeds(scene, end, mute, check_md5):
    """-> ({body: (frames, V, 3)}, {body: mesh signature})"""
    M.open_scene(scene, check_md5=check_md5)
    M.cmds.currentTime(config.REST_FRAME)
    if mute:
        M.mute_animation(True)
    bodies = M.ziva_meshes()
    keys = sorted(bodies)
    sig = {k: mesh_signature(bodies[k]["shape"], M.rest_target(bodies[k]["shape"], bodies[k]["node"])[0])
           for k in keys}
    rec = M.simulate(list(range(config.REST_FRAME, end + 1)), [bodies[k]["shape"] for k in keys],
                     start=config.REST_FRAME)
    return {k: rec[bodies[k]["shape"]] for k in keys}, sig


def compare_signatures(a, b):
    """Bodies whose triangles or UVs differ between two scenes."""
    bad = {}
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            bad[k] = "missing"
            continue
        diffs = [f for f in ("poly_counts", "poly_verts", "n_verts", "uv_counts", "uv_ids") if not np.array_equal(a[k][f], b[k][f])]
        diffs += [f for f in ("u", "v") if a[k][f].shape != b[k][f].shape or not np.allclose(a[k][f], b[k][f], atol=1e-7)]
        if diffs:
            bad[k] = diffs
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--subject", required=True, help="directory written by fit_subject.py")
    ap.add_argument("--end", type=int, default=50)
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()

    M.boot()
    sub = Path(args.subject)
    with cio.RunDir(sub.name, "sim_check", "rig/sim_check.py") as run:
        rest, _j = load_rest_dir(sub)
        tmpl, _ = load_rest_dir(config.template_export_dir())
        free = {}
        for k in rest:
            with np.load(config.template_export_dir() / (k + ".npz"), allow_pickle=False) as z:
                free[k] = np.linalg.norm(z["vertices"][0] - z["rest_vertices"], axis=1) > FREE_MM

        with np.load(sub / "skin_fit_output.npz", allow_pickle=False) as z:
            wanted_skin = np.asarray(z["vertices"], dtype=np.float64)

        print("== shape (muted)")
        muted, sig_fit = speeds(args.scene, 10, True, False)
        rest_dev = {}
        for k, r in rest.items():
            if k not in muted:
                continue
            if k == SKIN:
                d = np.linalg.norm(muted[k][0] - wanted_skin, axis=1)
            else:
                d = np.linalg.norm(muted[k][0] - r, axis=1)[~free[k]]
            rest_dev[k] = {"p99_mm": float(np.percentile(d, 99)), "max_mm": float(d.max()),
                           "settle_max_mm": float(np.linalg.norm(muted[k][-1] - muted[k][0], axis=1).max())}
        reg = {k: v for k, v in rest_dev.items() if registered(k)}
        worst = max(reg.items(), key=lambda kv: kv[1]["p99_mm"])
        run.metrics["rest"] = rest_dev
        run.metrics["rest_p99_worst"] = {"body": worst[0], **worst[1]}
        rest_ok = worst[1]["p99_mm"] <= REST_P99_MM and all(np.isfinite(v).all() for v in muted.values())
        print("   worst registered body {0}: p99 {1:.3f} mm -> {2}".format(worst[0], worst[1]["p99_mm"],
                                                                         "PASS" if rest_ok else "FAIL"))

        print("== animated 0..{0}".format(args.end))
        anim, _sig = speeds(args.scene, args.end, False, False)
        finite = all(np.isfinite(v).all() for v in anim.values())
        vmax = {k: float(np.linalg.norm(np.diff(v, axis=0), axis=2).max()) for k, v in anim.items()}
        run.metrics["animated_finite"] = bool(finite)
        run.metrics["max_speed_mm_per_frame"] = vmax
        anim_ok = finite
        if not args.no_baseline:
            print("== template baseline, animated")
            base, sig_lab = speeds(None, args.end, False, True)
            mesh_bad = compare_signatures(sig_lab, sig_fit)
            run.metrics["mesh_changed"] = mesh_bad
            run.metrics["mesh_identical_bodies"] = len(sig_lab) - len(mesh_bad)
            print("   mesh: polygons and UVs identical to the lab scene in {0}/{1} bodies{2}".format(
                len(sig_lab) - len(mesh_bad), len(sig_lab), "" if not mesh_bad else ", CHANGED: {0}".format(mesh_bad)))
            anim_ok = anim_ok and not mesh_bad
            vb = {k: float(np.linalg.norm(np.diff(v, axis=0), axis=2).max()) for k, v in base.items()}
            ratio = {k: vmax[k] / max(vb[k], 1e-6) for k in vmax if k in vb and registered(k)}
            run.metrics["speed_ratio_vs_template"] = ratio
            bad = {k: r for k, r in ratio.items() if r > SPEED_RATIO_MAX}
            run.metrics["speed_ratio_over_max"] = bad
            anim_ok = anim_ok and not bad
            top = sorted(ratio.items(), key=lambda kv: -kv[1])[:5]
            print("   finite {0}; worst speed ratios vs template: {1}".format(
                finite, [(k, round(r, 2)) for k, r in top]))
        print("   -> {0}".format("PASS" if anim_ok else "FAIL"))
        run.metrics["pass"] = bool(rest_ok and anim_ok)
        if not run.metrics["pass"]:
            sys.exit(2)


if __name__ == "__main__":
    main()

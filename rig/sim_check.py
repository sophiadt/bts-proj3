"""Gate: does a written rig sit at its new rest and simulate stably?

  rest      animation muted: frame-0 output of every registered body vs the
            rest written for it (p99, mm), then settle to frame 10
  animated  the speech animation, frames 0..END: everything finite, and the
            worst per-frame vertex speed per body compared with the untouched
            template's on the same frames (a blow-up shows as a large ratio)

Vertices whose template output differs from their rest by > 1 mm (the skin's
forehead strip) are excluded from the rest comparison, as in roundtrip_test.

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


def speeds(scene, end, mute, check_md5):
    M.open_scene(scene, check_md5=check_md5)
    M.cmds.currentTime(config.REST_FRAME)
    if mute:
        M.mute_animation(True)
    bodies = M.ziva_meshes()
    keys = sorted(bodies)
    rec = M.simulate(list(range(config.REST_FRAME, end + 1)), [bodies[k]["shape"] for k in keys],
                     start=config.REST_FRAME)
    return {k: rec[bodies[k]["shape"]] for k in keys}


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

        print("== rest (muted)")
        muted = speeds(args.scene, 10, True, False)
        rest_dev = {}
        for k, r in rest.items():
            if k not in muted:
                continue
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
        anim = speeds(args.scene, args.end, False, False)
        finite = all(np.isfinite(v).all() for v in anim.values())
        vmax = {k: float(np.linalg.norm(np.diff(v, axis=0), axis=2).max()) for k, v in anim.items()}
        run.metrics["animated_finite"] = bool(finite)
        run.metrics["max_speed_mm_per_frame"] = vmax
        anim_ok = finite
        if not args.no_baseline:
            print("== template baseline, animated")
            base = speeds(None, args.end, False, True)
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

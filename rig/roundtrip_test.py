"""Gate: does the rig take new rest geometry, and keep it?

Cases, each on a fresh open of the scene, animation muted after any joint
move (a muted channel holds the value it had when muted):

  baseline   untouched scene
  identity   exported rest written straight back      pass: max |sim - baseline| <= 0.02 mm
  scale s    every rest shape and skin joint scaled    pass, per registered body:
             by s about the skull's centroid             takes it:  frame-0 p99 |sim - s*baseline| <= 0.5 mm
                                                         keeps it:  median kept fraction >= 0.95 at the last frame

"Kept fraction" per vertex = 1 - |sim - expected| / |expected - baseline|:
1 means the rig settled at the new shape, 0 means it snapped back to the
generic face (the maintainOffset failure an identity test cannot see).
Vertices that move less than 0.5 mm under the scale are not scored.

The last-frame deviation is reported, not gated: the solver's first step
settles the tissue under gravity and attachments (skin p99 1.1 mm in the
baseline), and a resized model settles by a different amount -- measured at
0.75: skin p99 0.17 mm at frame 0, 0.54 mm from frame 1 on, constant after.
Snap-back would instead grow with time and show in the kept fraction.

Identity tolerance: writing the rest connects each tissue's rest mesh to its
zGeo.iNeutralMesh (rig/write_rest.py), so Ziva re-reads the neutral from the
mesh; the neutral it had stored differs from that mesh by <= 0.011 mm (skin,
eyes). Without the connection the write-back is exact (1.7e-5 mm), but then
the solver ignores any new rest.

Registered bodies are the lab model's: middleres_*, polySurface* and the
registered skin. The legacy jack01_* meshes and the older 47k-vertex
skin_cloth are simulated too, reported, and not gated.

Vertices whose output already differs from their rest by > 1 mm in the
untouched scene (the skin's "pull_back" forehead strip, up to 16.6 mm; parts
of the legacy meshes, up to 78 mm) are not determined by their rest, so
"settles at the scaled rest" is not their expectation: they are reported
separately ("free" vertices) and not gated.

Kept threshold 0.95, not 0.99: gravity sag does not scale linearly with size,
so a 5% scale leaves a few tenths of a mm of honest difference on a move of a
few mm (measured: cheek muscles 0.987-0.989 at 1.05 with p99 <= 0.38 mm, and
>= 0.994 at 0.75). Snap-back reads ~0, a partial maintainOffset pull ~0.5.

    mayapy rig/roundtrip_test.py [--scales 1.05 0.75] [--frames 0 1 5 10 20 30]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import io as cio                             # noqa: E402
from rig import _maya as M                             # noqa: E402
from rig import write_rest as W                        # noqa: E402

IDENTITY_TOL_MM = 0.02
KEPT_MIN = 0.95
FREE_MM = 1.0
SCALE_DEV_P99_MM = 0.5
MIN_MOVE_MM = 0.5
PIVOT_BODY = "middleres_skull"
REGISTERED_SKIN = "skin_cloth_copy_v5_pull_back"


def registered(key):
    return key.startswith("middleres_") or key.startswith("polySurface") or key == REGISTERED_SKIN


def run_case(frames, rest=None, joints=None):
    M.open_scene()
    M.cmds.currentTime(config.REST_FRAME)
    info = {}
    if rest is not None:
        info["readback_max_mm"] = max(W.apply_rest_shapes(rest).values())
    if joints is not None:
        info["joints"] = W.move_joints(joints)
    info["muted_plugs"] = len(M.mute_animation(True))
    bodies = sorted(M.ziva_meshes())
    shapes = {k: v["shape"] for k, v in M.ziva_meshes().items()}
    rec = M.simulate(frames, [shapes[k] for k in bodies], start=config.REST_FRAME)
    out = {k: rec[shapes[k]] for k in bodies}
    finite = all(np.isfinite(v).all() for v in out.values())
    info["finite"] = bool(finite)
    return out, info


def compare_identity(base, sim):
    return {k: float(np.abs(sim[k] - base[k]).max()) for k in base}


def free_vertices(base, rest):
    """Vertices whose untouched frame-0 output is not their rest (> FREE_MM)."""
    return {k: np.linalg.norm(base[k][0] - rest[k], axis=1) > FREE_MM for k in base if k in rest}


def compare_scale(base, sim, s, pivot, free=None):
    per = {}
    for k in base:
        b = base[k]
        fr = free.get(k) if free else None
        fr = np.zeros(b.shape[1], dtype=bool) if fr is None else fr
        expected = pivot + s * (b - pivot)
        dev_all = np.linalg.norm(sim[k] - expected, axis=-1)      # (F, V)
        dev = dev_all[:, ~fr]
        move = np.linalg.norm(expected - b, axis=-1)[:, ~fr]
        scored = move > MIN_MOVE_MM
        kept = np.where(scored, 1.0 - dev / np.where(scored, move, 1.0), np.nan)
        last = kept[-1]
        per[k] = {"registered": registered(k), "n_free": int(fr.sum()),
                  "free_max_dev_mm": float(dev_all[:, fr].max()) if fr.any() else None,
                  "max_dev_mm": float(dev.max()), "max_dev_last_frame_mm": float(dev[-1].max()),
                  "p99_dev_frame0_mm": float(np.percentile(dev[0], 99)),
                  "p99_dev_last_frame_mm": float(np.percentile(dev[-1], 99)),
                  "n_dev_over_0p5_last": int((dev[-1] > 0.5).sum()),
                  "kept_median_last": float(np.nanmedian(last)) if np.isfinite(last).any() else None,
                  "kept_p05_last": float(np.nanpercentile(last, 5)) if np.isfinite(last).any() else None,
                  "frame0_max_dev_mm": float(dev[0].max())}
    return per


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scales", type=float, nargs="*", default=[1.05, 0.75])
    ap.add_argument("--frames", type=int, nargs="*", default=[0, 1, 5, 10, 20, 30])
    ap.add_argument("--src", default=None, help="exported rest dir (default template_export)")
    args = ap.parse_args()
    src = Path(args.src) if args.src else config.template_export_dir()

    M.boot()
    with cio.RunDir("template", "roundtrip", "rig/roundtrip_test.py") as run:
        rest, joints = W.load_rest_dir(src)
        pivot = rest[PIVOT_BODY].mean(axis=0)
        run.metrics.update({"frames": args.frames, "pivot_mm": pivot, "n_bodies": len(rest),
                            "n_joints": len(joints or {})})

        print("== baseline")
        base, run.metrics["baseline"] = run_case(args.frames)
        free = free_vertices(base, rest)
        run.metrics["free_vertices"] = {k: int(v.sum()) for k, v in free.items() if v.any()}
        np.savez_compressed(run.path / "outputs" / "baseline.npz", **base)

        print("== identity")
        sim, info = run_case(args.frames, rest=rest)
        diff = compare_identity(base, sim)
        info["max_diff_mm"] = max(diff.values())
        info["per_body_max_diff_mm"] = diff
        info["pass"] = info["finite"] and info["max_diff_mm"] <= IDENTITY_TOL_MM
        run.metrics["identity"] = info
        print("   max diff {0:.3g} mm -> {1}".format(info["max_diff_mm"], "PASS" if info["pass"] else "FAIL"))
        ok = info["pass"]

        for s in args.scales:
            print("== scale", s)
            srest = {k: pivot + s * (v - pivot) for k, v in rest.items()}
            sj = {k: pivot + s * (v - pivot) for k, v in (joints or {}).items()} or None
            sim, info = run_case(args.frames, rest=srest, joints=sj)
            np.savez_compressed(run.path / "outputs" / "scale_{0}.npz".format(s), **sim)
            per = compare_scale(base, sim, s, pivot, free)
            reg = {k: v for k, v in per.items() if v["registered"]}
            kept = [v["kept_median_last"] for v in reg.values() if v["kept_median_last"] is not None]
            info["per_body"] = per
            info["kept_median_worst_body"] = float(min(kept)) if kept else None
            info["p99_dev_frame0_worst_mm"] = max(v["p99_dev_frame0_mm"] for v in reg.values())
            info["p99_dev_last_worst_mm"] = max(v["p99_dev_last_frame_mm"] for v in reg.values())
            info["max_dev_registered_mm"] = max(v["max_dev_mm"] for v in reg.values())
            info["max_dev_legacy_mm"] = max([v["max_dev_mm"] for v in per.values() if not v["registered"]] or [0.0])
            info["failing_bodies"] = sorted(k for k, v in reg.items()
                                            if (v["kept_median_last"] is not None and v["kept_median_last"] < KEPT_MIN)
                                            or v["p99_dev_frame0_mm"] > SCALE_DEV_P99_MM)
            info["pass"] = bool(info["finite"] and kept and not info["failing_bodies"])
            run.metrics["scale_{0}".format(s)] = info
            print("   registered: frame-0 p99 {0:.3f} mm (takes it), worst kept {1:.4f} (keeps it); "
                  "settled p99 {2:.3f} max {3:.2f} mm; legacy max {4:.1f} mm -> {5}".format(
                      info["p99_dev_frame0_worst_mm"], info["kept_median_worst_body"], info["p99_dev_last_worst_mm"],
                      info["max_dev_registered_mm"], info["max_dev_legacy_mm"], "PASS" if info["pass"] else "FAIL"))
            for k in info["failing_bodies"]:
                v = per[k]
                print("     FAIL {0:42s} kept {1:.4f} frame-0 p99 {2:.3f} mm".format(
                    k, v["kept_median_last"], v["p99_dev_frame0_mm"]))
            ok = ok and info["pass"]

        run.metrics["pass"] = bool(ok)
        if not ok:
            sys.exit(2)


if __name__ == "__main__":
    main()

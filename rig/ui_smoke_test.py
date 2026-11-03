"""Gate: can the lab's keyboard UI animate a fitted head?

Headless (mayapy), through rig/keyboard_ui.py's patched in-memory copy of
the UI, the same chain its Go button runs for a sentence of 3+ phonemes:
parse the sentence, then run_single_transition() for each phoneme pair from
the neutral frame 14. Done twice:

  fitted    template_export/scenes/<id>_ui.mb, motion scaled by the fit
  generic   the lab scene with the same keys removed (made once, cached as
            template_export/scenes/generic_ui.mb), motion unscaled

Then each is simulated from frame 0 to the end. Pass: the sentence keyed the
UI joints, every body stays finite, and no registered body moves faster per
frame than SPEED_RATIO_MAX x the generic head does.

Runs the lab's UI code and reads its data on the lab share.

    mayapy rig/ui_smoke_test.py capture_01 [--sentence "not even you"]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import io as cio                             # noqa: E402
from rig import _maya as M                             # noqa: E402
from rig import keyboard_ui as K                       # noqa: E402
from rig.roundtrip_test import registered              # noqa: E402
from rig.write_rest import strip_ui_keys               # noqa: E402

SPEED_RATIO_MAX = 3.0
SKIN = "skin_cloth_copy_v5_pull_back"


def generic_ui_scene():
    path = config.template_export_dir() / "scenes" / "generic_ui.mb"
    if not path.exists():
        M.open_scene()
        strip_ui_keys()
        M.cmds.fileInfo("bts_fit_scale", "1.0")
        M.cmds.fileInfo("bts_subject", "generic")
        M.save_copy(path)
    return path


def play(scene, scale, sentence):
    """Key the sentence onto `scene` through the UI, then simulate it.
    -> (frames {body: (F, V, 3)}, info)"""
    M.open_scene(scene, check_md5=False)
    mod = K.load_ui(scene, scale, open_window=False)
    phonemes, _ = mod.parse_sequence_string(sentence)
    if len(phonemes) < 3:
        raise ValueError("use a sentence of 3+ phonemes; got {0}".format(phonemes))
    mod.ACCUMULATED_ERROR = None
    mod.DEBUG_DATA.clear()
    mod.ensure_proxy_render_meshes()
    start, end = K.UI_NEUTRAL_FRAME, K.UI_NEUTRAL_FRAME
    for i in range(len(phonemes) - 1):
        end = mod.run_single_transition(phonemes[i], phonemes[i + 1], start, i > 0, is_first_transition=(i == 0))
        if phonemes[i + 1] == "dataV22_AU0_neutral":
            snap = end + 1
            mod.insert_ziva_hidden_snap(snap)
            mod.insert_smooth_snap_blendshape(snap, buffer_frames=15)
            end = snap + 15
        start = end
    keyed = M.cmds.keyframe("oor_01", q=True, timeChange=True) or []
    bodies = M.ziva_meshes()
    keys = sorted(k for k in bodies)
    rec = M.simulate(list(range(0, int(end) + 1)), [bodies[k]["shape"] for k in keys], start=0)
    frames = {k: rec[bodies[k]["shape"]] for k in keys}
    info = {"phonemes": phonemes, "end_frame": int(end), "oor_01_keys_after_14": sum(1 for t in keyed if t > 14),
            "scaled_checkpoint_files": len(mod.pickle.scaled_files), "scale": scale}
    return frames, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject")
    ap.add_argument("--sentence", default="not even you")
    args = ap.parse_args()
    M.boot()
    sub = config.template_export_dir() / "subjects" / args.subject
    fit_scene = config.template_export_dir() / "scenes" / (args.subject + "_ui.mb")
    with cio.RunDir(args.subject, "ui_smoke", "rig/ui_smoke_test.py") as run:
        M.open_scene(fit_scene, check_md5=False)
        scale = K.fit_scale_from_scene(M.cmds)
        print("== fitted ({0}, motion x{1:.4f})".format(fit_scene.name, scale))
        fit, run.metrics["fitted"] = play(fit_scene, scale, args.sentence)
        np.savez_compressed(sub / "anim_ui_fit.npz", skin=fit[SKIN].astype(np.float32))
        print("== generic")
        gen_scene = generic_ui_scene()
        gen, run.metrics["generic"] = play(gen_scene, 1.0, args.sentence)
        np.savez_compressed(sub / "anim_ui_generic.npz", skin=gen[SKIN].astype(np.float32))

        n = min(len(next(iter(fit.values()))), len(next(iter(gen.values()))))
        finite = all(np.isfinite(v).all() for v in fit.values())
        vmax = lambda d: {k: float(np.linalg.norm(np.diff(v[:n], axis=0), axis=2).max()) for k, v in d.items()}
        vf, vg = vmax(fit), vmax(gen)
        ratio = {k: vf[k] / max(vg[k], 1e-6) for k in vf if k in vg and registered(k)}
        bad = {k: r for k, r in ratio.items() if r > SPEED_RATIO_MAX}
        keyed = run.metrics["fitted"]["oor_01_keys_after_14"] > 0
        run.metrics.update({"finite": bool(finite), "speed_ratio_vs_generic": ratio, "over_max": bad,
                            "skin_motion_max_mm": float(np.linalg.norm(fit[SKIN] - fit[SKIN][:1], axis=2).max()),
                            "generic_skin_motion_max_mm": float(np.linalg.norm(gen[SKIN] - gen[SKIN][:1], axis=2).max()),
                            "pass": bool(finite and keyed and not bad)})
        top = sorted(ratio.items(), key=lambda kv: -kv[1])[:5]
        print("   keyed: {0} keys on oor_01 after frame 14; end frame {1}".format(
            run.metrics["fitted"]["oor_01_keys_after_14"], run.metrics["fitted"]["end_frame"]))
        print("   finite {0}; skin moves up to {1:.1f} mm (generic {2:.1f} mm); worst speed ratios {3}".format(
            finite, run.metrics["skin_motion_max_mm"], run.metrics["generic_skin_motion_max_mm"],
            [(k, round(r, 2)) for k, r in top]))
        print("   ->", "PASS" if run.metrics["pass"] else "FAIL")
        if not run.metrics["pass"]:
            sys.exit(2)


if __name__ == "__main__":
    main()

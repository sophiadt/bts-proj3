"""Gate: can the lab's keyboard UI animate a fitted head?

Headless (mayapy), through rig/keyboard_ui.py's patched in-memory copy of
the UI, the same chain its Go button runs for a sentence of 3+ phonemes:
parse the sentence, then run_single_transition() for each phoneme pair from
the neutral frame 14. Done twice:

  fitted    template_export/scenes/<id>_ui.mb, motion scaled by the fit
  generic   the lab scene with the same keys removed (made once, cached as
            template_export/scenes/generic_ui.mb), motion unscaled

Then each is simulated from frame 0 to the end. Pass: the sentence keyed the
UI joints, every body stays finite, no registered body moves faster per
frame than SPEED_RATIO_MAX x the generic head does, and the mouth opens at
least 70% as wide as the generic head's at this size (pipeline/mouth.py;
tools/mouth_check.py repeats this later from the saved frames, with pictures),
and the lip muscle's halves part while speaking: their widest gap reaches at
least half the generic head's (the "sandman mouth" -- lips stuck together when
they move -- opened the skin a little but left the muscle closed).

--reuse-generic takes the generic head's frames from the last full run
(template_export/anim_ui_generic.npz + anim_ui_generic_speed.json) instead of
playing it again; rig/morph_search.py uses it for every step.

Runs the lab's UI code and reads its data on the lab share.

    mayapy rig/ui_smoke_test.py SUBJECT [--sentence "not even you"] [--reuse-generic]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import io as cio                             # noqa: E402
from pipeline import mouth                             # noqa: E402
from rig import _maya as M                             # noqa: E402
from rig import keyboard_ui as K                       # noqa: E402
from rig.roundtrip_test import registered              # noqa: E402
from rig.write_rest import strip_ui_keys               # noqa: E402

SPEED_RATIO_MAX = 3.0
MUSCLE_SPEECH_MIN = 0.5            # widest lip-muscle gap while speaking, against the generic head's
SKIN = "skin_cloth_copy_v5_pull_back"
GENERIC_SPEED = "anim_ui_generic_speed.json"


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
    ap.add_argument("--reuse-generic", action="store_true",
                    help="use the generic head's frames from the last full run instead of playing it again")
    args = ap.parse_args()
    M.boot()
    sub = config.template_export_dir() / "subjects" / args.subject
    fit_scene = config.template_export_dir() / "scenes" / (args.subject + "_ui.mb")
    with cio.RunDir(args.subject, "ui_smoke", "rig/ui_smoke_test.py") as run:
        M.open_scene(fit_scene, check_md5=False)
        scale = K.fit_scale_from_scene(M.cmds)
        print("== fitted ({0}, motion x{1:.4f})".format(fit_scene.name, scale))
        fit, run.metrics["fitted"] = play(fit_scene, scale, args.sentence)
        np.savez_compressed(sub / "anim_ui_fit.npz", skin=fit[SKIN].astype(np.float32),
                            oo=fit[mouth.LIP_MUSCLE].astype(np.float32))
        cache = config.template_export_dir() / "anim_ui_generic.npz"
        speed_file = config.template_export_dir() / GENERIC_SPEED
        vmax = lambda d, n: {k: float(np.linalg.norm(np.diff(v[:n], axis=0), axis=2).max()) for k, v in d.items()}
        reuse = args.reuse_generic and cache.exists() and speed_file.exists()
        if reuse:
            with np.load(cache) as z:
                if "oo" not in z.files:
                    reuse = False
                else:
                    gen = {SKIN: np.asarray(z["skin"], dtype=np.float64), mouth.LIP_MUSCLE: np.asarray(z["oo"], dtype=np.float64)}
            spd = json.load(open(speed_file, encoding="utf-8")) if reuse else None
            if reuse and spd.get("sentence") != args.sentence:
                reuse = False
        if reuse:
            print("== generic: frames from the last full run ({0})".format(cache.name))
            run.metrics["generic"] = dict(spd.get("info", {}), reused=True)
            vg, n_gen = spd["vmax"], spd["n_frames"]
        else:
            print("== generic")
            gen_scene = generic_ui_scene()
            gen, run.metrics["generic"] = play(gen_scene, 1.0, args.sentence)
            np.savez_compressed(sub / "anim_ui_generic.npz", skin=gen[SKIN].astype(np.float32))
            # The lab head's own speech motion: pipeline/importance.py builds the skin
            # importance map from it (where the face moves when it talks).
            np.savez_compressed(cache, skin=gen[SKIN].astype(np.float32), oo=gen[mouth.LIP_MUSCLE].astype(np.float32))
            n_gen = len(next(iter(gen.values())))
            vg = vmax(gen, n_gen)
            cio.write_json_atomic(speed_file, {"sentence": args.sentence, "n_frames": n_gen, "vmax": vg,
                                               "info": run.metrics["generic"]})

        n = min(len(next(iter(fit.values()))), n_gen)
        finite = all(np.isfinite(v).all() for v in fit.values())
        vf = vmax(fit, n)
        ratio = {k: vf[k] / max(vg[k], 1e-6) for k in vf if k in vg and registered(k)}
        bad = {k: r for k, r in ratio.items() if r > SPEED_RATIO_MAX}
        keyed = run.metrics["fitted"]["oor_01_keys_after_14"] > 0
        # The mouth opens like the generic head's at this size (pipeline/mouth.py):
        # a fit that pulled the mouth shut animated sealed and still passed the
        # speed and finiteness checks.
        lab_skin, faces = mouth.load_lab_mesh(SKIN)
        mouth_rep = mouth.assess_frames(np.asarray(fit[SKIN], dtype=np.float64),
                                        np.asarray(gen[SKIN], dtype=np.float64), faces,
                                        mouth.mouth_parts(lab_skin, faces), scale)
        # The lip muscle's halves part while speaking (at the subject's size).
        parts = mouth.mouth_parts(lab_skin, faces)
        widest = lambda sk, oo, k: max((mouth.muscle_gap(oo[i], sk[i], parts)["median"] or 0.0)
                                       for i in range(0, min(len(sk), n), k))
        m_fit = widest(np.asarray(fit[SKIN], float), np.asarray(fit[mouth.LIP_MUSCLE], float), 4)
        m_gen = scale * widest(np.asarray(gen[SKIN], float), np.asarray(gen[mouth.LIP_MUSCLE], float), 4)
        mouth_rep.update({"muscle_widest_gap_mm": m_fit, "ref_muscle_widest_gap_mm": m_gen,
                          "muscle_speech_ratio": m_fit / max(m_gen, 1e-9)})
        mouth_ok = mouth_rep["verdict"] == "OK" and mouth_rep["muscle_speech_ratio"] >= MUSCLE_SPEECH_MIN
        run.metrics.update({"finite": bool(finite), "speed_ratio_vs_generic": ratio, "over_max": bad,
                            "skin_motion_max_mm": float(np.linalg.norm(fit[SKIN] - fit[SKIN][:1], axis=2).max()),
                            "generic_skin_motion_max_mm": float(np.linalg.norm(gen[SKIN] - gen[SKIN][:1], axis=2).max()),
                            "mouth": mouth_rep,
                            "pass": bool(finite and keyed and not bad and mouth_ok)})
        top = sorted(ratio.items(), key=lambda kv: -kv[1])[:5]
        print("   keyed: {0} keys on oor_01 after frame 14; end frame {1}".format(
            run.metrics["fitted"]["oor_01_keys_after_14"], run.metrics["fitted"]["end_frame"]))
        print("   finite {0}; skin moves up to {1:.1f} mm (generic {2:.1f} mm); worst speed ratios {3}".format(
            finite, run.metrics["skin_motion_max_mm"], run.metrics["generic_skin_motion_max_mm"],
            [(k, round(r, 2)) for k, r in top]))
        print("   mouth: widest opening {0:.0f} mm2, generic head at this size {1:.0f} mm2 ({2}) -> {3}".format(
            mouth_rep["speech_opening_mm2"], mouth_rep["ref_speech_opening_mm2"],
            "n/a" if mouth_rep["speech_ratio"] is None else "{0:.0%}".format(mouth_rep["speech_ratio"]),
            mouth_rep["verdict"]))
        print("   lip muscle: widest gap while speaking {0:.2f} mm, generic head at this size {1:.2f} mm ({2:.0%}; "
              "needs {3:.0%})".format(m_fit, m_gen, mouth_rep["muscle_speech_ratio"], MUSCLE_SPEECH_MIN))
        print("   ->", "PASS" if run.metrics["pass"] else "FAIL")
        if not run.metrics["pass"]:
            sys.exit(2)


if __name__ == "__main__":
    main()

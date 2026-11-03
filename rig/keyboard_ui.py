"""Run the lab's keyboard/phoneme UI on a fitted head. Their file is not edited.

The UI (h99-UI-mayascipt-V1_comsciV1.py, in the lab's a3-stochastic-V1 repo)
plays phonemes by adding stored joint displacements (delta_b, from
combined_checkpoints_base10.pkl on the lab share) to the open scene's
oor/ooc_l/ooc_r/frontalis joints at frame 14, and rotating jaw01. On a fitted
head that needs four things, applied to an in-memory copy of its source:

  size       the displacements were recorded on the generic head (~1.45x a
             real face): positions are scaled by the fit's uniform scale
             (fileInfo bts_fit_scale), the jaw-rotation slice [900:903] is not
  neutral    feedback control measures error against the generic head's
             neutral (joint_position_* modules); it is given the fitted
             scene's own frame-14 joint positions instead
  reset      its reset reopens a hard-coded generic scene; it reopens the
             fitted UI scene instead
  writes     combined sentences (hard-coded to the lab share's hub) and debug
             output go to template_export/ui_results instead

Its last line opens the window (`create_automation_ui()`); that line is left
out of the in-memory copy so the window is built after the patches, with its
buttons bound to the patched functions.
"""

import importlib
import os
import pickle as _pickle
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_UI_SCRIPT = Path(os.environ.get(
    "BTS_LAB_UI_SCRIPT",
    r"C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3\d4 interactive interface of animation"
    r"\h99-UI-mayascipt-V1_comsciV1.py"))
SHARE_HUB = (r"\\bbysvm-nfs1.its.sfu.ca\GrUViLab\Data\richardPeng\SFU_PhD_cloud\2-Projects"
             r"\1-physically based facial animation\3-Experiments\a3-stochastic-V1-backup-and-results"
             r"\results\sequence_combined_deformation_parameters_hub")
GROUPS = ("oor", "ooc_l", "ooc_r", "frontalis")
N_PER_GROUP = 300
JAW_ROT = slice(900, 903)
UI_NEUTRAL_FRAME = 14


class ScaledPickle(object):
    """Stands in for the UI module's `pickle`: combined_checkpoints_*.pkl
    displacement vectors come back scaled by `scale` (jaw rotation untouched);
    every other file loads unchanged."""

    def __init__(self, scale):
        self.scale = float(scale)
        self.scaled_files = []

    def __getattr__(self, name):
        return getattr(_pickle, name)

    def load(self, f, *a, **kw):
        data = _pickle.load(f, *a, **kw)
        name = os.path.basename(getattr(f, "name", "") or "")
        if self.scale == 1.0 or not name.startswith("combined_checkpoints"):
            return data
        out = []
        for vec in data:
            v = np.asarray(vec, dtype=np.float64) * self.scale
            if len(v) >= JAW_ROT.stop:
                v[JAW_ROT] /= self.scale
            out.append(v.tolist() if isinstance(vec, list) else v)
        self.scaled_files.append(getattr(f, "name", name))
        return out


def install_paths():
    """Register rig/ui_paths.py as the UI's `paths` module."""
    sys.path.insert(0, str(REPO))
    from rig import ui_paths
    ui_paths.RESULTS.mkdir(parents=True, exist_ok=True)
    sys.modules["paths"] = ui_paths
    return ui_paths


def fitted_neutral(cmds, frame=UI_NEUTRAL_FRAME):
    """World positions of each group's 300 joints at the UI's neutral frame,
    in joint-index order (oor_01 .. oor_300)."""
    cmds.currentTime(frame)
    out = {}
    for g in GROUPS:
        out[g] = [cmds.xform("{0}_{1:02d}".format(g, i + 1), q=True, translation=True, worldSpace=True)
                  for i in range(N_PER_GROUP)]
    return out


def _wave(points):
    return types.SimpleNamespace(x_joint_position_1=points)


def patch_neutral(mod, neutral):
    """Point the UI's neutral references (joint_position_* 'waves', which it
    concatenates) at the fitted head: all 300 points in the first wave."""
    mod.joint_position_first_wave = _wave(neutral["oor"])
    mod.joint_position = _wave([])
    mod.joint_position_third_wave = _wave([])
    for side in ("l", "r"):
        g = neutral["ooc_" + side]
        setattr(mod, "joint_position_ooc_{0}_base_0".format(side), _wave(g))
        for suffix in ("base_pose11_0", "base_pose11_AU1_0", "base_pose11_AU1_AU2_0"):
            setattr(mod, "joint_position_ooc_{0}_{1}".format(side, suffix), _wave([]))
    mod.joint_position_frontalis_base_0 = _wave(neutral["frontalis"])
    mod.joint_position_frontalis_base_AU1_0 = _wave([])
    mod.joint_position_frontalis_base_AU1_AU2_0 = _wave([])


def load_ui(scene, scale, ui_script=DEFAULT_UI_SCRIPT, open_window=False):
    """Load the UI into a fresh module against the open fitted `scene`, with
    the patches above. -> the module (window opened if open_window)."""
    from maya import cmds
    paths = install_paths()
    src = Path(ui_script).read_text(encoding="utf-8")
    lines = src.rstrip().splitlines()
    if lines[-1].strip() != "create_automation_ui()":
        raise RuntimeError("unexpected UI script: last line is not create_automation_ui()")
    src = "\n".join(lines[:-1]) + "\n"
    hub_literal = 'r"' + SHARE_HUB + '"'
    if hub_literal not in src:
        raise RuntimeError("unexpected UI script: the sequence hub path has changed")
    hub = paths.RESULTS / "sequence_combined_deformation_parameters_hub"
    hub.mkdir(parents=True, exist_ok=True)
    src = src.replace(hub_literal, "r" + repr(str(hub)))

    mod = types.ModuleType("h99_ui_fitted")
    mod.__file__ = str(ui_script)          # the UI finds its repo from __file__
    sys.modules[mod.__name__] = mod
    exec(compile(src, str(ui_script), "exec"), mod.__dict__)

    mod.pickle = ScaledPickle(scale)
    mod.DEBUG_OUTPUT_DIR = str(paths.RESULTS / "debug")
    patch_neutral(mod, fitted_neutral(cmds))
    scene = str(scene)

    def reset_animation_chain(*args):
        """Reset chaining to neutral and reopen the fitted UI scene."""
        mod.LAST_ANIMATION_END_FRAME = UI_NEUTRAL_FRAME
        mod.LAST_ANIMATION_PHONEME = "dataV22_AU0_neutral"
        mod.CONTINUE_FROM_LAST = False
        mod.ACCUMULATED_ERROR = None
        mod.DEBUG_DATA.clear()
        if cmds.checkBox("continue_checkbox", exists=True):
            cmds.checkBox("continue_checkbox", edit=True, value=False)
        cmds.file(scene, open=True, force=True)
        patch_neutral(mod, fitted_neutral(cmds))
        print("[fitted UI] reset; reopened", scene)

    mod.reset_animation_chain = reset_animation_chain
    mod.BTS_FIT = {"scene": scene, "scale": float(scale), "pickle": mod.pickle}
    if open_window:
        mod.create_automation_ui()
    return mod


def fit_scale_from_scene(cmds):
    v = cmds.fileInfo("bts_fit_scale", query=True)
    return float(v[0]) if v else None


def launch(subject="capture_01", scene=None, ui_script=DEFAULT_UI_SCRIPT):
    """Open the subject's fitted UI scene and the keyboard UI on it (Maya GUI)."""
    from maya import cmds
    scene = Path(scene) if scene else REPO / "template_export" / "scenes" / (subject + "_ui.mb")
    if not scene.exists():
        raise RuntimeError("no fitted UI scene at {0}: run rig/write_rest.py ... --ui-ready".format(scene))
    cmds.file(str(scene), open=True, force=True)
    scale = fit_scale_from_scene(cmds)
    if scale is None:
        raise RuntimeError("{0} has no bts_fit_scale; re-run rig/write_rest.py".format(scene))
    print("[fitted UI] {0}: phoneme motion scaled by {1:.4f}".format(scene.name, scale))
    return load_ui(scene, scale, ui_script, open_window=True)

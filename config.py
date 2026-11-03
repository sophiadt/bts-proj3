"""Units, frames and paths. Declared once here; everything else imports them.

Pipeline convention: millimetres, Y-up, right-handed. Convert at the boundaries
(lift.py for the phone, build_rig.py for Maya), never in between.

Sources of each convention:
  - 1 Maya unit = 0.962 mm: paper Appendix C (one author's mandible, 155.92
    units against 15 cm); config/units.txt. The scene header's `cm` is nominal.
  - Template frame: Y-up, face toward -Z (measured on the exported skin; the
    Ziva solver's gravity is -9.8 along Y).
  - Phone frame: camera frame in mm -- x along the sensor's long side, y down,
    z away from the lens. Right-handed, not Y-up.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

MAYA_UNIT_MM = 0.962

# Frame 0 is the rest geometry before gravity acts. In the exports frames 0 and
# 15 are identical (max difference 0.0003 mm); landmarks_lab.json records
# positions at frame 15.
REST_FRAME = 0
LANDMARKS_LAB_FRAME = 15

TEMPLATE_JSON = REPO_ROOT / "config" / "template.json"
LANDMARKS_LAB_JSON = REPO_ROOT / "config" / "landmarks_lab.json"
RUNS_DIR = REPO_ROOT / "runs"

# Maya side. The scene is opened read-only; modified copies go to
# TEMPLATE_EXPORT_DIR/scenes and never overwrite SCENE_PATH.
MAYAPY = Path(os.environ.get("MAYAPY", r"C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe"))
SCENE_PATH = REPO_ROOT / "v68 neutral v13.8 after housekeeping.mb"
SCENE_MD5 = "b174b0d59a89495c6f57f779b8063b27"
TEMPLATE_EXPORT_DIR = REPO_ROOT / "template_export"

# MediaPipe's own bundle: the canonical face mesh lives inside it.
MEDIAPIPE_TASK = REPO_ROOT / "Project 3 - TrueDepth Camera Data" / "face_landmarker.task"
MEDIAPIPE_CANONICAL_NPZ = REPO_ROOT / "config" / "mediapipe_canonical.npz"
LANDMARK_TO_VERTEX_JSON = REPO_ROOT / "config" / "landmark_to_vertex.json"

# Phone camera frame -> template frame, applied once in pipeline/capture.py.
# The app shows MediaPipe the sensor frame transposed (.leftMirrored), so the
# upright image's down axis is sensor +x and its horizontal axis is sensor y.
# Head-up is therefore -x_cam; the face looks along -z_cam, as the template's
# looks along -Z. Rows are the template axes in camera coordinates; det = +1.
CAM_TO_TEMPLATE = (
    (0.0, 1.0, 0.0),    # X =  y_cam
    (-1.0, 0.0, 0.0),   # Y = -x_cam
    (0.0, 0.0, 1.0),    # Z =  z_cam
)

# Guided capture: which way the subject was asked to turn for each prefix, from
# the subject's own point of view. From the app's prompts (CameraManager.swift,
# startGuidedCapture: "turn your head to YOUR LEFT"). Whether the subject
# complied is checked per capture by pipeline/mirror_check.py.
TURN_FOR_PREFIX = {"left_": "subject_left", "right_": "subject_right"}


def template_export_dir():
    """This machine's fresh exports (gitignored). $BTS_TEMPLATE_EXPORT overrides."""
    env = os.environ.get("BTS_TEMPLATE_EXPORT")
    return Path(env) if env else TEMPLATE_EXPORT_DIR


def lab_data_dir():
    """Directory holding the lab template .npz exports.

    The lab data is not licensed for redistribution, so it is never committed:
    $BTS_LAB_DATA, else this machine's fresh export (template_export/, from
    rig/export_rest.py) if present, else ../bts-proj1/data/lab.
    """
    env = os.environ.get("BTS_LAB_DATA")
    if env:
        return Path(env)
    fresh = template_export_dir()
    if (fresh / "skin_cloth_copy_v5_pull_back.npz").exists():
        return fresh
    return REPO_ROOT.parent / "bts-proj1" / "data" / "lab"

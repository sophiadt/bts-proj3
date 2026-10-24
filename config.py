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


def lab_data_dir():
    """Directory holding the lab template .npz exports.

    The lab data is not licensed for redistribution, so it stays in bts-proj1
    and is referenced by path: $BTS_LAB_DATA, else ../bts-proj1/data/lab.
    """
    env = os.environ.get("BTS_LAB_DATA")
    if env:
        return Path(env)
    return REPO_ROOT.parent / "bts-proj1" / "data" / "lab"

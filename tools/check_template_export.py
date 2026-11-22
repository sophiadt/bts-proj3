"""Is template_export/ complete enough to work from? (After a clone, on any machine.)

The lab head's exported bodies and the fitted subjects are committed, so a
partner can fit, verify and preview without Maya. The Maya scenes (.mb) are
not (430 MB each); rig/write_rest.py rebuilds one from its subject folder.

Checks, each with the command that rebuilds what is missing:
  - manifest.json and every body's .npz, with the vertex counts the rig needs
    (skin 15,905, jaw 7,426, lip muscle 10,240, and the rest from the manifest)
  - joints.npz, the importance map, the attachment maps, the scene probe
  - the keyboard UI's recorded generic-head speech (anim_ui_generic.npz)
  - each subject folder: report.json, every body, joints.npz

numpy only.

    python tools/check_template_export.py        # exit 1 if anything required is missing
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TE = ROOT / "template_export"
COUNTS = {"skin_cloth_copy_v5_pull_back": 15905, "middleres_Jaw": 7426, "middleres_OrbicularisOris": 10240}
EXTRAS = [  # (file, required, how to rebuild)
    ("joints.npz", True, "mayapy rig/export_rest.py (workstation)"),
    ("skin_importance.npz", False, "mayapy pipeline/importance.py"),
    ("attachment_maps.json", False, "mayapy rig/probe_scene.py --attachments (workstation)"),
    ("scene_probe.json", False, "mayapy rig/probe_scene.py (workstation)"),
    ("anim_ui_generic.npz", False, "mayapy rig/ui_smoke_test.py SUBJECT (workstation)"),
]


def check(te=TE):
    """-> (problems [str], notes [str])."""
    problems, notes = [], []
    man = te / "manifest.json"
    if not man.exists():
        return ["template_export/manifest.json missing -- mayapy rig/export_rest.py (workstation)"], notes
    with open(man, encoding="utf-8") as fh:
        bodies = json.load(fh)["bodies"]
    for key, info in bodies.items():
        p = te / (key + ".npz")
        if not p.exists():
            problems.append("{0}.npz missing -- mayapy rig/export_rest.py (workstation)".format(key))
            continue
        with np.load(p, allow_pickle=False) as z:
            n = z["rest_vertices"].shape[0]
        want = COUNTS.get(key, info["n_verts"])
        if n != want:
            problems.append("{0}: {1} vertices, expected {2} (vertex order is what the rig binds to)".format(key, n, want))
    for name, required, how in EXTRAS:
        if not (te / name).exists():
            (problems if required else notes).append("{0} missing -- {1}".format(name, how))
    subjects = te / "subjects"
    for d in sorted(subjects.iterdir()) if subjects.exists() else []:
        if not d.is_dir() or re.search(r"_t\d{3}$", d.name):      # morph search steps: machine-local
            continue
        if not (d / "report.json").exists():
            problems.append("subjects/{0}: no report.json".format(d.name))
            continue
        missing = [k for k in bodies if not (d / (k + ".npz")).exists()]
        if missing or not (d / "joints.npz").exists():
            problems.append("subjects/{0}: missing {1}".format(d.name, (missing + ["joints.npz"])[:3]))
        else:
            with open(d / "report.json", encoding="utf-8") as fh:
                v = json.load(fh).get("verify", {}).get("verdict")
            notes.append("subjects/{0}: complete (verify: {1}); scene: mayapy rig/write_rest.py "
                         "template_export/subjects/{0} --save {0} --ui-ready".format(d.name, v or "not run"))
    return problems, notes


def main():
    problems, notes = check()
    for n in notes:
        print("  " + n)
    for p in problems:
        print("MISSING " + p)
    print("template_export: " + ("OK" if not problems else "{0} problem(s)".format(len(problems))))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

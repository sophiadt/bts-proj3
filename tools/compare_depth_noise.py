"""How noisy is the depth in a guided capture? (numpy + scipy: mayapy or .venv)

    python tools/compare_depth_noise.py s01_20261030T170049Z_guided s01_20261110T212937Z_burst
    python tools/compare_depth_noise.py "<capture folder A>" "<capture folder B>" ...

For each pose it measures the fine roughness of the face surface in millimetres:
the depth minus its own 5x5 local average, over the face region only. A smooth
face with a quiet sensor gives a small number; lower is better. It also reports
"flying pixels", pixels more than 5 mm off their neighbourhood.

Written by the app's author to measure what burst depth (several frames merged
per pose) gains over a single frame. Adapted here to take capture ids, the
captures/<subject>/<capture_id>/ layout and Windows paths.
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import capture as cap                    # noqa: E402


def load_depth(folder, pose):
    """The pose's depth map in metres (flat layout or one view folder down), or None."""
    files = sorted(glob.glob(os.path.join(folder, pose + "_depth_*x*_float32_m.bin")) +
                   glob.glob(os.path.join(folder, "*", pose + "_depth_*x*_float32_m.bin")))
    if not files:
        return None
    w, h = [int(x) for x in os.path.basename(files[0]).split("_depth_")[1].split("_")[0].split("x")]
    return np.fromfile(files[0], dtype=np.float32).reshape(h, w)


def load_meta(folder, pose):
    files = sorted(glob.glob(os.path.join(folder, pose + "_meta.json")) +
                   glob.glob(os.path.join(folder, "*", pose + "_meta.json")))
    if not files:
        return {}
    with open(files[0], encoding="utf-8") as fh:
        return json.load(fh)


def roughness(d, k=5):
    """RMS and robust roughness (mm) of the face surface, plus the fraction of flying pixels."""
    valid = np.isfinite(d) & (d > 0.1) & (d < 1.5)               # burst holes are 0, single-frame holes NaN
    h, w = d.shape
    c = d[h // 2 - 60:h // 2 + 60, w // 2 - 60:w // 2 + 60]       # assumes the face is near the image centre
    cv = c[np.isfinite(c) & (c > 0.1) & (c < 1.5)]
    if cv.size < 500:
        return None
    zmed = float(np.median(cv))
    face = valid & (np.abs(d - zmed) < 0.10)                     # within 10 cm of the face's depth
    face = ndi.binary_erosion(face, iterations=k)                # whole neighbourhood on the face
    x = np.where(valid, d, 0.0).astype(np.float64)
    m = valid.astype(np.float64)
    num = ndi.uniform_filter(x, size=k)
    den = ndi.uniform_filter(m, size=k)
    local = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
    r = (d - local)[face] * 1000.0                                # mm
    if r.size < 1000:
        return None
    rms = float(np.sqrt(np.mean(r ** 2)))
    robust = float(1.4826 * np.median(np.abs(r - np.median(r))))
    fly = float(np.mean(np.abs(r) > 5.0))
    return rms, robust, fly, int(r.size)


def measure(captures):
    """[(capture id, pose, frames merged, (rms, robust, flying, n) or None)]."""
    rows = []
    for arg in captures:
        folder = cap.resolve_capture(arg)
        name = cap.capture_id(folder)
        for pose in ("front", "left", "right"):
            d = load_depth(str(folder), pose)
            if d is None:
                continue
            merged = load_meta(str(folder), pose).get("depth_frames_merged", 1)
            rows.append((name, pose, merged, roughness(d)))
    return rows


def main(captures):
    rows = measure(captures)
    width = max([len(r[0]) for r in rows] + [7])
    print("%-*s %-6s %-7s %12s %14s %14s" % (width, "capture", "pose", "frames", "rough (mm)", "robust (mm)",
                                             "flying px %"))
    for name, pose, merged, res in rows:
        if res is None:
            print("%-*s %-6s %-7s   (no usable face depth)" % (width, name, pose, merged))
        else:
            print("%-*s %-6s %-7s %12.3f %14.3f %14.2f" % (width, name, pose, merged, res[0], res[1], res[2] * 100))
    print("\nLower is better. 'robust' ignores the occasional jump; 'rough' includes them.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        main(sys.argv[1:])

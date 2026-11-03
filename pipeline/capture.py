"""Load a phone capture folder: landmarks, depth, intrinsics, dense cloud.

A folder holds one view (no prefix) or a guided set (front_, left_, right_).
Per view:
    <p>landmarks_3d_mm.json   478 x [x,y,z] mm or null, index slots stable
    <p>depth_<W>x<H>_float32_m.bin   metres, row-major, sensor landscape
    <p>meta.json              intrinsics at intrinsics_ref_width x _height
Never use landmarks.ply for correspondences: nulls are dropped, indices shift.

Coordinates stay in the camera frame (x along the sensor's long side, y down,
z away from the lens, mm) until `to_template`, the one explicit rotation into
the template's Y-up frame (config.CAM_TO_TEMPLATE).

Depth is not undistorted. TrueDepth depth is generally delivered rectified;
whether the saved distortion tables apply to it is unverified.
"""

import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from pipeline import canonical as mp                   # noqa: E402

R_CAM_TO_TEMPLATE = np.array(config.CAM_TO_TEMPLATE, dtype=np.float64)
DEPTH_RE = re.compile(r"depth_(\d+)x(\d+)_float32_m.*\.bin$")
OUTLIER_MM = 8.0           # landmark depth vs its 9x9 neighbourhood median
OUTLIER_WIN = 4
EDGE_MM = 10.0             # depth range in a 5x5 window: the landmark sits on a depth edge
EDGE_WIN = 2               # (silhouette landmarks 234/454/132/361 measured 13-30 mm off)


class View(object):
    def __init__(self, folder, prefix):
        self.folder, self.prefix = Path(folder), prefix
        self.name = prefix.rstrip("_") or "single"
        with open(self.folder / (prefix + "landmarks_3d_mm.json"), encoding="utf-8") as fh:
            raw = json.load(fh)
        if len(raw) != mp.N_LANDMARKS:
            raise ValueError("{0}: {1} landmarks, expected {2}".format(self.name, len(raw), mp.N_LANDMARKS))
        self.landmarks = np.array([p if p is not None else [np.nan] * 3 for p in raw], dtype=np.float64)
        with open(self.folder / (prefix + "meta.json"), encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.depth = self._load_depth()
        self.K = self._intrinsics()
        self.unreliable = self._depth_outliers()

    @property
    def valid(self):
        """Landmarks with depth that also agree with their neighbourhood."""
        return np.isfinite(self.landmarks).all(axis=1) & ~self.unreliable

    def _load_depth(self):
        files = sorted(f for f in glob.glob(str(self.folder / (glob.escape(self.prefix) + "depth_*.bin")))
                       if DEPTH_RE.search(os.path.basename(f)))
        if not files:
            return None
        frames = []
        for f in files:
            w, h = map(int, DEPTH_RE.search(os.path.basename(f)).groups())
            d = np.fromfile(f, dtype="<f4")
            if d.size != w * h:
                raise ValueError("{0}: {1} floats, expected {2}x{3}".format(f, d.size, w, h))
            d = d.reshape(h, w).astype(np.float64)
            d[~np.isfinite(d) | (d <= 0)] = np.nan
            frames.append(d)
        if len(frames) == 1:
            return frames[0]
        # A burst: per-pixel median over synchronised frames.
        with np.errstate(all="ignore"):
            return np.nanmedian(np.stack(frames), axis=0)

    def _intrinsics(self):
        m = self.meta
        if not all(k in m for k in ("fx", "fy", "cx", "cy", "intrinsics_ref_width")):
            return None
        w = m.get("depth_width") or (self.depth.shape[1] if self.depth is not None else None)
        if w is None:
            return None
        s = float(w) / float(m["intrinsics_ref_width"])
        return {"fx": m["fx"] * s, "fy": m["fy"] * s, "cx": m["cx"] * s, "cy": m["cy"] * s, "scale": s}

    def project(self, pts_mm):
        """Camera-frame mm -> depth pixel (u, v)."""
        k = self.K
        p = np.asarray(pts_mm, dtype=np.float64)
        return np.stack([k["fx"] * p[:, 0] / p[:, 2] + k["cx"], k["fy"] * p[:, 1] / p[:, 2] + k["cy"]], axis=1)

    def _depth_outliers(self):
        bad = np.zeros(len(self.landmarks), dtype=bool)
        if self.depth is None or self.K is None:
            return bad
        h, w = self.depth.shape
        ok = np.isfinite(self.landmarks).all(axis=1)
        uv = np.full((len(self.landmarks), 2), np.nan)
        uv[ok] = self.project(self.landmarks[ok])
        for i in np.flatnonzero(ok):
            u, v = int(round(uv[i, 0])), int(round(uv[i, 1]))
            if not (0 <= u < w and 0 <= v < h):
                bad[i] = True
                continue
            win = self.depth[max(v - OUTLIER_WIN, 0):v + OUTLIER_WIN + 1, max(u - OUTLIER_WIN, 0):u + OUTLIER_WIN + 1]
            win = win[np.isfinite(win)]
            if win.size == 0 or abs(np.median(win) * 1000.0 - self.landmarks[i, 2]) > OUTLIER_MM:
                bad[i] = True
                continue
            edge = self.depth[max(v - EDGE_WIN, 0):v + EDGE_WIN + 1, max(u - EDGE_WIN, 0):u + EDGE_WIN + 1]
            edge = edge[np.isfinite(edge)]
            if edge.size and (edge.max() - edge.min()) * 1000.0 > EDGE_MM:
                bad[i] = True
        return bad

    def cloud(self, margin_mm=15.0, z_margin_mm=40.0, stride=1):
        """Dense camera-frame cloud (mm) in a box around the valid landmarks."""
        if self.depth is None or self.K is None:
            return np.zeros((0, 3))
        k = self.K
        h, w = self.depth.shape
        vv, uu = np.mgrid[0:h:stride, 0:w:stride]
        z = self.depth[::stride, ::stride] * 1000.0
        ok = np.isfinite(z)
        x = (uu - k["cx"]) * z / k["fx"]
        y = (vv - k["cy"]) * z / k["fy"]
        pts = np.stack([x[ok], y[ok], z[ok]], axis=1)
        lm = self.landmarks[self.valid]
        if len(lm):
            lo = lm.min(0) - [margin_mm, margin_mm, z_margin_mm]
            hi = lm.max(0) + [margin_mm, margin_mm, z_margin_mm]
            pts = pts[((pts > lo) & (pts < hi)).all(axis=1)]
        return pts


def to_template(pts_cam):
    """Camera frame -> template axes (rotation only; translation and scale are
    left to the similarity fit)."""
    return np.asarray(pts_cam, dtype=np.float64) @ R_CAM_TO_TEMPLATE.T


def check_upright(lm_tpl):
    """After to_template: forehead above chin, nose tip the most forward point
    (face toward -Z). Raises with the numbers if the rotation is wrong."""
    p = lm_tpl
    probs = []
    if np.isfinite(p[[mp.FOREHEAD, mp.CHIN]]).all() and not p[mp.FOREHEAD, 1] > p[mp.CHIN, 1]:
        probs.append("forehead y {0:.1f} not above chin y {1:.1f}".format(p[mp.FOREHEAD, 1], p[mp.CHIN, 1]))
    core = [i for i in (mp.NOSE_TIP, mp.FOREHEAD, mp.CHIN, mp.FACE_EDGE_R, mp.FACE_EDGE_L) if np.isfinite(p[i]).all()]
    if mp.NOSE_TIP in core and p[mp.NOSE_TIP, 2] > min(p[i, 2] for i in core):
        probs.append("nose tip is not the most forward of {0}".format(core))
    if probs:
        raise ValueError("camera->template rotation looks wrong: " + "; ".join(probs))


def load_capture(folder):
    """-> {view name: View}, e.g. {"front": ..., "left": ..., "right": ...}."""
    folder = Path(folder)
    found = sorted(glob.glob(str(folder / "*landmarks_3d_mm.json")))
    if not found:
        raise FileNotFoundError("no *landmarks_3d_mm.json in {0}".format(folder))
    views = {}
    for f in found:
        prefix = os.path.basename(f)[:-len("landmarks_3d_mm.json")]
        v = View(folder, prefix)
        views[v.name] = v
    return views


def primary(views):
    return views.get("front") or views.get("single") or next(iter(views.values()))

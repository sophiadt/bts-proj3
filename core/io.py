"""Loading the lab template with its invariants asserted, and the run layout.

Vertex order is sacred: the Ziva rig binds attachments by vertex index, so a
load that reorders, welds or de-duplicates vertices breaks the rig with no
visible error. Every mesh load here checks counts against config/template.json,
and trimesh is never allowed to process a mesh.
"""

import datetime
import hashlib
import json
import math
import os
import platform
import secrets
import subprocess
import tempfile
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config

# Frames 0 and 15 of the exports differ by at most 0.0003 mm. Anything larger
# means the export or the scene changed, and frame-15 landmark positions no
# longer describe the frame-0 rest geometry.
REST_IDENTITY_TOL_MM = 0.01


class InvariantError(AssertionError):
    """A template, unit or vertex-order invariant does not hold."""


@dataclass
class Mesh:
    name: str
    role: str
    vertices: np.ndarray   # (V, 3) float64, mm, template frame
    faces: np.ndarray      # (F, 3) int64
    closed: bool


@dataclass
class Template:
    meshes: dict                                  # name -> Mesh
    missing: list = field(default_factory=list)   # optional meshes not found
    registry: dict = field(default_factory=dict)


def load_registry(path=None):
    with open(path or config.TEMPLATE_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def _registry_entry(registry, name):
    for entry in registry["meshes"]:
        if entry["name"] == name:
            return entry
    raise KeyError("{0} is not in the template registry".format(name))


def _check(cond, msg):
    if not cond:
        raise InvariantError(msg)


def load_lab_mesh(name, frame=None, registry=None, data_dir=None):
    """One template mesh at `frame` (default: the rest frame), invariants asserted."""
    registry = registry or load_registry()
    entry = _registry_entry(registry, name)
    src = registry["npz_source"]
    frame = config.REST_FRAME if frame is None else frame

    path = Path(data_dir or config.lab_data_dir()) / "{0}.npz".format(name)
    if not path.exists():
        raise FileNotFoundError(str(path))

    with np.load(path, allow_pickle=False) as z:
        verts = np.asarray(z["vertices"])
        faces = np.asarray(z["faces"])
        units = str(z["units"])
        unit_scale = float(z["unit_scale"])

    _check(verts.ndim == 3 and verts.shape[2] == 3,
           "{0}: vertices should be (frames, V, 3), got {1}".format(name, verts.shape))
    _check(units == src["units"],
           "{0}: units {1!r}, expected {2!r}".format(name, units, src["units"]))
    _check(math.isclose(unit_scale, src["unit_scale"], abs_tol=1e-9),
           "{0}: unit_scale {1}, expected {2}".format(name, unit_scale, src["unit_scale"]))
    if entry["n_verts"] is not None:
        _check(verts.shape[1] == entry["n_verts"],
               "{0}: {1} vertices, registry says {2}".format(name, verts.shape[1], entry["n_verts"]))
    if entry["n_faces"] is not None:
        _check(len(faces) == entry["n_faces"],
               "{0}: {1} faces, registry says {2}".format(name, len(faces), entry["n_faces"]))
    _check(faces.min() >= 0 and faces.max() < verts.shape[1],
           "{0}: face indices out of range".format(name))
    _check(np.isfinite(verts).all(), "{0}: non-finite vertices".format(name))
    if verts.shape[0] > config.LANDMARKS_LAB_FRAME:
        drift = float(np.abs(verts[0] - verts[config.LANDMARKS_LAB_FRAME]).max())
        _check(drift <= REST_IDENTITY_TOL_MM,
               "{0}: frames 0 and {1} differ by {2:.4f} mm".format(
                   name, config.LANDMARKS_LAB_FRAME, drift))

    return Mesh(name=name, role=entry["role"],
                vertices=verts[frame].astype(np.float64),
                faces=faces.astype(np.int64).reshape(-1, 3),
                closed=bool(entry["closed"]))


def load_template(registry=None, data_dir=None, frame=None):
    """Every registry mesh. Optional meshes that are absent are listed, not fatal."""
    registry = registry or load_registry()
    meshes, missing = {}, []
    for entry in registry["meshes"]:
        try:
            meshes[entry["name"]] = load_lab_mesh(entry["name"], frame, registry, data_dir)
        except FileNotFoundError:
            if not entry.get("optional"):
                raise
            warnings.warn("optional mesh {0} not found; continuing without it".format(entry["name"]))
            missing.append(entry["name"])
    return Template(meshes=meshes, missing=missing, registry=registry)


def spot_check(template, landmarks_json=None, tol_mm=0.01):
    """Named vertices must sit where landmarks_lab.json recorded them.

    Catches a reordered or re-exported mesh whose vertex count happens to match.
    Returns {landmark: error_mm}; raises InvariantError if any exceeds tol_mm.
    """
    with open(landmarks_json or config.LANDMARKS_LAB_JSON, encoding="utf-8") as fh:
        lm = json.load(fh)
    errors = {}
    for key, d in lm["landmarks"].items():
        mesh = template.meshes.get(d["mesh"])
        if mesh is None:
            raise InvariantError("{0}: mesh {1} not loaded".format(key, d["mesh"]))
        errors[key] = float(np.linalg.norm(mesh.vertices[d["vtx"]] - np.asarray(d["rest_pos_mm"])))
    bad = {k: e for k, e in errors.items() if e > tol_mm}
    _check(not bad, "spot check failed (mm): {0}".format(bad))
    return errors


def load_obj(path, expected_verts=None):
    """OBJ as (vertices, faces) without welding or reordering."""
    import trimesh
    mesh = trimesh.load(str(path), process=False, force="mesh")
    if expected_verts is not None:
        _check(len(mesh.vertices) == expected_verts,
               "{0}: {1} vertices, expected {2}".format(path, len(mesh.vertices), expected_verts))
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def save_obj(path, vertices, faces):
    """Write an OBJ, then read it back and check nothing was welded or moved."""
    import trimesh
    v = np.asarray(vertices, dtype=np.float64)
    trimesh.Trimesh(vertices=v, faces=np.asarray(faces, dtype=np.int64), process=False).export(str(path))
    back, _ = load_obj(path, expected_verts=len(v))
    _check(np.allclose(back, v, atol=1e-5), "{0}: vertices changed on round trip".format(path))


def md5_file(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --- run layout ----------------------------------------------------------------

def _jsonable(obj):
    """numpy -> python, NaN/inf -> None, so json.dump(allow_nan=False) succeeds."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _git_info():
    def git(*args):
        return subprocess.run(["git", *args], cwd=config.REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {"sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions(names=("numpy", "scipy", "trimesh")):
    from importlib import metadata
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def write_json_atomic(path, payload):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(_jsonable(payload), fh, indent=2, allow_nan=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


class RunDir:
    """runs/<subject>/<pathway>/<timestamp>_<id>/ with inputs/, outputs/, views/.

    metrics.json is written on every exit, including exceptions: failures are
    data. Exceptions are never swallowed.

        with RunDir("template", "features", "checks/template_features.py") as run:
            run.metrics["eye_width_mm"] = 43.1
    """

    def __init__(self, subject, pathway, script, root=None):
        self.subject, self.pathway, self.script = subject, pathway, script
        self.root = Path(root or config.RUNS_DIR)
        self.metrics = {}
        self.path = None

    def __enter__(self):
        stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")   # no ':' on Windows
        self.run_id = "{0}_{1}".format(stamp, secrets.token_hex(2))
        self.path = self.root / self.subject / self.pathway / self.run_id
        for sub in ("inputs", "outputs", "views"):
            (self.path / sub).mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.datetime.now().isoformat(timespec="seconds")
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            status = "ok"
        elif issubclass(exc_type, KeyboardInterrupt):
            status = "interrupted"
        else:
            status = "failed"
        record = {
            "subject": self.subject, "pathway": self.pathway, "script": self.script,
            "run_id": self.run_id, "status": status,
            "error": None if exc is None else repr(exc),
            "traceback": None if exc is None else "".join(traceback.format_exception(exc_type, exc, tb)),
            "started_at": self.started_at,
            "wall_clock_s": round(time.perf_counter() - self._t0, 3),
            "git": _git_info(), "python": platform.python_version(), "packages": _versions(),
            "metrics": self.metrics,
        }
        write_json_atomic(self.path / "metrics.json", record)
        return False

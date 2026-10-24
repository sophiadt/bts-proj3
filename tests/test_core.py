"""Invariant and analytic tests for core/: IO, run layout, vendored geometry.

Tests that need the lab .npz exports skip when they are not on this machine
($BTS_LAB_DATA, else ../bts-proj1/data/lab).

    .venv\\Scripts\\python -m pytest -v
"""

import copy
import json
import math

import numpy as np
import pytest
import trimesh

import config
from core import io as cio
from core.metrics import boundary_loops, enclosed_volume
from core.similarity import apply_similarity, similarity_transform

HAVE_LAB = (config.lab_data_dir() / "skin_cloth_copy_v5_pull_back.npz").exists()
needs_lab = pytest.mark.skipif(not HAVE_LAB, reason="lab .npz exports not available")


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


# --- similarity -------------------------------------------------------------

def test_similarity_recovers_known_transform():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3)) * 40
    R, s, t = random_rotation(rng), 1.07, np.array([3.0, -12.0, 40.0])
    dst = apply_similarity(src, s, R, t)
    s2, R2, t2 = similarity_transform(src, dst)
    assert math.isclose(s2, s, rel_tol=1e-12)
    assert np.allclose(R2, R, atol=1e-12) and np.allclose(t2, t, atol=1e-9)


def test_similarity_never_returns_a_reflection():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 3)) * 40
    mirrored = src * np.array([-1.0, 1.0, 1.0])
    _, R, _ = similarity_transform(src, mirrored)
    assert np.linalg.det(R) > 0


# --- vendored metrics -------------------------------------------------------

def test_boundary_loops_closed_surface_is_empty():
    assert boundary_loops(trimesh.creation.icosphere(subdivisions=2).faces) == []


def test_boundary_loops_finds_one_hole():
    m = trimesh.creation.icosphere(subdivisions=2)
    loops = boundary_loops(m.faces[1:])          # drop one triangle -> a 3-vertex hole
    assert len(loops) == 1 and sorted(loops[0]) == sorted(m.faces[0].tolist())


def test_enclosed_volume_unit_cube():
    cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    assert math.isclose(enclosed_volume(cube.vertices, cube.faces), 1.0, rel_tol=1e-12)


# --- template IO ------------------------------------------------------------

@needs_lab
def test_template_loads_with_expected_counts():
    tpl = cio.load_template()
    for entry in tpl.registry["meshes"]:
        if entry["name"] in tpl.missing:
            assert entry["optional"]
            continue
        mesh = tpl.meshes[entry["name"]]
        assert mesh.vertices.shape == (entry["n_verts"], 3)
        assert mesh.faces.shape == (entry["n_faces"], 3)


@needs_lab
def test_skull_is_optional_until_exported():
    tpl = cio.load_template()
    if "middleres_skull" not in tpl.meshes:
        assert "middleres_skull" in tpl.missing


@needs_lab
def test_spot_check_named_vertices():
    errors = cio.spot_check(cio.load_template())
    assert len(errors) == 7 and max(errors.values()) < 0.01


@needs_lab
def test_wrong_vertex_count_raises():
    reg = copy.deepcopy(cio.load_registry())
    cio._registry_entry(reg, "middleres_Jaw")["n_verts"] += 1
    with pytest.raises(cio.InvariantError):
        cio.load_lab_mesh("middleres_Jaw", registry=reg)


@needs_lab
def test_wrong_unit_scale_raises():
    reg = copy.deepcopy(cio.load_registry())
    reg["npz_source"]["unit_scale"] = 1.0
    with pytest.raises(cio.InvariantError):
        cio.load_lab_mesh("middleres_Jaw", registry=reg)


@needs_lab
def test_skin_has_four_boundary_loops():
    skin = cio.load_lab_mesh("skin_cloth_copy_v5_pull_back")
    assert [len(l) for l in boundary_loops(skin.faces)] == [250, 143, 99, 97]


def test_obj_round_trip_keeps_vertex_order(tmp_path):
    m = trimesh.creation.icosphere(subdivisions=2)
    v = np.asarray(m.vertices)[::-1].copy()      # an order trimesh would not choose itself
    f = (len(v) - 1) - np.asarray(m.faces)
    cio.save_obj(tmp_path / "s.obj", v, f)
    back, faces = cio.load_obj(tmp_path / "s.obj", expected_verts=len(v))
    assert np.allclose(back, v, atol=1e-5) and np.array_equal(faces, f)


# --- run layout -------------------------------------------------------------

def test_rundir_writes_metrics_on_success(tmp_path):
    with cio.RunDir("subj", "path", "test", root=tmp_path) as run:
        run.metrics["value"] = np.float64(1.5)
        run.metrics["missing"] = float("nan")
    rec = json.loads((run.path / "metrics.json").read_text())
    assert rec["status"] == "ok" and rec["metrics"] == {"value": 1.5, "missing": None}
    assert all((run.path / d).is_dir() for d in ("inputs", "outputs", "views"))
    assert ":" not in run.run_id


def test_rundir_writes_metrics_on_failure_and_reraises(tmp_path):
    with pytest.raises(ValueError):
        with cio.RunDir("subj", "path", "test", root=tmp_path) as run:
            run.metrics["stage"] = "before"
            raise ValueError("boom")
    rec = json.loads((run.path / "metrics.json").read_text())
    assert rec["status"] == "failed" and "boom" in rec["error"]
    assert rec["metrics"]["stage"] == "before" and "Traceback" in rec["traceback"]

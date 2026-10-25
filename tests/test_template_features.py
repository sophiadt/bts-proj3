"""Tests for checks/template_features.py: verdict logic, loop classification, invariance."""

import json

import numpy as np
import pytest
import trimesh

import config
from checks import template_features as tf
from core import io as cio
from core.metrics import boundary_loops

HAVE_LAB = (config.lab_data_dir() / "skin_cloth_copy_v5_pull_back.npz").exists()
needs_lab = pytest.mark.skipif(not HAVE_LAB, reason="lab .npz exports not available")
NORMS = json.loads(tf.NORMS_JSON.read_text(encoding="utf-8"))


def norm_values(scale=1.0, overrides=None):
    vals = {k: m["male"]["mean_mm"] * scale for k, m in NORMS["measures"].items()}
    vals.update(overrides or {})
    return vals


# --- verdict logic --------------------------------------------------------------

def test_normal_face_is_consistent():
    r = tf.compare(norm_values(), NORMS)
    assert r["verdict"] == "consistent_with_0.962" and abs(r["k"] - 1) < 1e-12


def test_uniformly_scaled_face_is_a_scale_error():
    r = tf.compare(norm_values(1.3), NORMS)
    assert r["verdict"] == "scale_error" and abs(r["k"] - 1.3) < 1e-12
    assert abs(r["corrected_mm_per_unit_if_scale_error"] - config.MAYA_UNIT_MM / 1.3) < 1e-12


def test_one_enlarged_feature_is_stylised():
    big = NORMS["measures"]["incanthwidth"]["male"]["mean_mm"] * 1.6
    r = tf.compare(norm_values(overrides={"incanthwidth": big}), NORMS)
    assert r["verdict"] == "stylised" and r["outliers_after_k"] == ["incanthwidth"]


def test_ambiguous_measures_do_not_drive_the_verdict():
    big = NORMS["measures"]["labfiswidth"]["male"]["mean_mm"] * 1.6
    r = tf.compare(norm_values(overrides={"labfiswidth": big}), NORMS)
    assert r["verdict"] == "consistent_with_0.962"


# --- landmark finding on synthetic geometry -------------------------------------

def holed_sphere(radius=100.0):
    """A face-like mask: sphere with the back removed, two eye holes and a mouth hole.

    Face toward -Z, Y up, subject's right = +X.
    """
    m = trimesh.creation.icosphere(subdivisions=5, radius=radius)
    c = m.triangles_center
    holes = [np.array([35.0, 25.0, -90.0]), np.array([-35.0, 25.0, -90.0]), np.array([0.0, -45.0, -88.0])]
    keep = c[:, 2] < 40.0                                   # open back -> the rim
    for h, r in zip(holes, (12.0, 12.0, 15.0)):
        keep &= np.linalg.norm(c - h, axis=1) > r
    sub = m.submesh([np.where(keep)[0]], append=True, repair=False)
    return np.asarray(sub.vertices), np.asarray(sub.faces)


def test_classify_loops_on_face_like_mask():
    v, f = holed_sphere()
    loops = tf.classify_loops(v, boundary_loops(f))
    assert v[loops["eye_r"]][:, 0].mean() > 0 > v[loops["eye_l"]][:, 0].mean()
    assert v[loops["mouth"]][:, 1].mean() < v[loops["eye_r"]][:, 1].mean()
    assert abs(v[loops["mouth"]][:, 0].mean()) < 2.0


def test_canthi_inner_point_is_nearer_the_midline():
    v, f = holed_sphere()
    loops = tf.classify_loops(v, boundary_loops(f))
    en, ex = tf.canthi(v[loops["eye_r"]])
    assert abs(en[0]) < abs(ex[0])


# --- the real template ----------------------------------------------------------

@needs_lab
def test_measurements_scale_and_translate_exactly():
    skin = cio.load_lab_mesh(tf.SKIN)
    lm, _ = tf.find_landmarks(skin.vertices, skin.faces)
    base = tf.measure(lm, NORMS)
    k, t = 1.25, np.array([10.0, -20.0, 5.0])
    lm2, _ = tf.find_landmarks(skin.vertices * k + t, skin.faces)
    moved = tf.measure(lm2, NORMS)
    for key in base:
        assert abs(moved[key] - k * base[key]) < 1e-6 * max(1.0, base[key]), key


@needs_lab
def test_midline_landmarks_are_in_anatomical_order():
    skin = cio.load_lab_mesh(tf.SKIN)
    lm, _ = tf.find_landmarks(skin.vertices, skin.faces)
    ys = [lm[k][1] for k in ("n", "prn", "sn", "ls", "sto", "gn")]
    assert ys == sorted(ys, reverse=True), ys

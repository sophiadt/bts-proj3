"""Tests for pipeline/ and the numpy-only geometry: run under mayapy.

No trimesh, no Maya. Synthetic captures come from pipeline/synthetic.py
(MediaPipe's canonical face rendered into a depth map the way the app samples
it). Tests needing this machine's template export skip without it.

    .venv-maya\\Scripts\\python -m pytest tests/test_pipeline.py -v
"""

import json

import numpy as np
import pytest

import config
from core import geometry as geo
from core.similarity import apply_similarity, similarity_transform
from pipeline import canonical as mp
from pipeline import capture as cap
from pipeline import mirror_check as mc
from pipeline import nonrigid, synthetic, volumetric
from pipeline.fixture import R_CANON_TO_TEMPLATE, mirror_permutation

HAVE_EXPORT = (config.template_export_dir() / "skin_cloth_copy_v5_pull_back.npz").exists()
needs_export = pytest.mark.skipif(not HAVE_EXPORT, reason="template_export/ not on this machine")


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


# --- canonical mesh -------------------------------------------------------------

def test_canonical_extract_matches_committed():
    fresh = mp.extract()
    committed = mp.load()
    assert fresh["vertices"].shape == (468, 3)
    assert len(fresh["faces"]) == 898
    assert len(fresh["basis"]) == 33
    np.testing.assert_allclose(fresh["vertices"], committed["vertices"])


def test_canonical_anchor_semantics():
    v = mp.load()["vertices"]
    assert v[mp.NOSE_TIP, 2] >= np.percentile(v[:, 2], 99)          # face toward +Z
    assert v[mp.FOREHEAD, 1] > v[mp.CHIN, 1]
    assert v[mp.EYE_R_OUTER, 0] < 0 < v[mp.EYE_L_OUTER, 0]          # subject's right at -X
    # Human scale: outer canthal width near the adult norm (88.9 mm).
    assert 80 < np.linalg.norm(v[mp.EYE_R_OUTER] - v[mp.EYE_L_OUTER]) < 98


def test_mirror_permutation_is_an_involution():
    perm = mirror_permutation(mp.load()["vertices"])
    assert (perm[perm] == np.arange(len(perm))).all()
    assert perm[mp.EYE_R_OUTER] == mp.EYE_L_OUTER and perm[mp.NOSE_TIP] == mp.NOSE_TIP


def test_canonical_to_template_is_a_rotation():
    assert np.isclose(np.linalg.det(R_CANON_TO_TEMPLATE), 1.0)
    t = mp.load()["vertices"] @ R_CANON_TO_TEMPLATE.T
    assert t[mp.EYE_R_OUTER, 0] > 0                                 # template: subject's right is +X
    assert t[mp.NOSE_TIP, 2] <= np.percentile(t[:, 2], 1)            # face toward -Z


# --- capture ----------------------------------------------------------------------

def test_cam_to_template_is_a_rotation():
    R = cap.R_CAM_TO_TEMPLATE
    np.testing.assert_allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_capture_load_upright_and_intrinsics(tmp_path):
    synthetic.make_view(tmp_path, "", 0.0, "ok", null_idx=(5, 200))
    v = cap.load_capture(tmp_path)["single"]
    assert v.landmarks.shape == (478, 3)
    assert np.isnan(v.landmarks[[5, 200]]).all()                    # nulls hold their slot
    assert np.isfinite(v.landmarks[[4, 6, 199, 201]]).all()
    assert np.isclose(v.K["fx"], synthetic.FX_REF * synthetic.W / synthetic.REF_W)
    cap.check_upright(cap.to_template(v.landmarks))
    with pytest.raises(ValueError):
        cap.check_upright(cap.to_template(v.landmarks) * [1, -1, 1])


def test_depth_edge_flags_silhouette_landmarks(tmp_path):
    synthetic.make_view(tmp_path, "", 0.0, "ok")
    v = cap.load_capture(tmp_path)["single"]
    v_t, _ = synthetic.head_template()
    truth = v_t @ cap.R_CAM_TO_TEMPLATE + np.array([25.0, 10.0, 350.0]) - [0, 0, v_t[:, 2].min()]
    err = np.linalg.norm(v.landmarks[:468] - truth, axis=1)
    bad = set(np.flatnonzero(err > 5.0).tolist())
    # Every badly sampled silhouette landmark is flagged (234/454 etc. are
    # 13-30 mm off). The only bad ones left unflagged are inner-lip contour
    # points (mouth concavity, 5-8 mm), which the fixture weights zero.
    oval_bad = bad & set(mp.FACE_OVAL)
    assert oval_bad and all(v.unreliable[i] for i in oval_bad)
    lips = set(mp.LIPS_INNER_UPPER + mp.LIPS_INNER_LOWER + mp.LIPS_OUTER_UPPER + mp.LIPS_OUTER_LOWER)
    assert {i for i in bad if not v.unreliable[i]} <= lips
    # Conservative (steep side walls flag too), but still a minority.
    assert v.unreliable.mean() < 0.25


# --- mirror gate --------------------------------------------------------------------

@pytest.mark.parametrize("mode,verdict", [
    ("ok", "OK"), ("labels_swapped", "LABELS_SWAPPED"), ("geometry_mirrored", "GEOMETRY_MIRRORED"),
    ("twin", "MIRROR_TWIN"), ("reg_mismatch", "RGB_DEPTH_MISMATCH")])
def test_mirror_check_verdicts(tmp_path, mode, verdict):
    synthetic.make_guided(tmp_path, mode=mode)
    assert mc.check(cap.load_capture(tmp_path))["verdict"] == verdict


def test_mirror_check_single_view_is_inconclusive(tmp_path):
    synthetic.make_view(tmp_path, "", 0.0, "ok")
    assert mc.check(cap.load_capture(tmp_path))["verdict"] == "INCONCLUSIVE_OK"


# --- similarity ---------------------------------------------------------------------

def test_weighted_similarity_none_matches_uniform():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(30, 3)) * 30
    dst = apply_similarity(src, 0.8, random_rotation(rng), np.array([1.0, 2.0, 3.0])) + rng.normal(size=(30, 3))
    a = similarity_transform(src, dst)
    b = similarity_transform(src, dst, weights=np.ones(30) * 7.0)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, atol=1e-10)


def test_weighted_similarity_ignores_zero_weight_outliers():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(40, 3)) * 40
    R, s, t = random_rotation(rng), 0.69, np.array([5.0, -3.0, 20.0])
    dst = apply_similarity(src, s, R, t)
    dst[:5] += 50.0
    w = np.ones(40)
    w[:5] = 0.0
    s2, R2, t2 = similarity_transform(src, dst, weights=w)
    assert np.isclose(s2, s) and np.allclose(R2, R) and np.allclose(t2, t)


# --- non-rigid and volumetric ---------------------------------------------------------

def _warp(v):
    return v + np.stack([3.0 * np.sin(v[:, 1] / 40.0), 2.0 * np.cos(v[:, 0] / 50.0), 0.02 * v[:, 0]], axis=1)


def test_nonrigid_recovers_smooth_warp_from_cloud():
    c = mp.load()
    v, f = c["vertices"], c["faces"]
    target = _warp(v)
    cloud = synthetic.dense_points(target, f, spacing_mm=1.0)
    lm = [mp.NOSE_TIP, mp.CHIN, mp.FOREHEAD, mp.EYE_R_OUTER, mp.EYE_L_OUTER, mp.MOUTH_R, mp.MOUTH_L]
    x, info = nonrigid.fit(v, f, lm_idx=lm, lm_pts=target[lm], lm_w=np.full(len(lm), 10.0), cloud=cloud,
                           max_dist=10.0)
    # Against a surface, vertices may slide tangentially: measure distance to
    # the surface, and position error only at the landmarks.
    from scipy.spatial import cKDTree
    surf = cKDTree(cloud).query(x)[0]
    assert np.median(surf) < 0.3 and np.percentile(surf, 95) < 1.0
    assert np.linalg.norm(x[lm] - target[lm], axis=1).max() < 0.5
    # With the target's own vertices as the cloud there is nothing to slide on.
    x2, _ = nonrigid.fit(v, f, lm_idx=lm, lm_pts=target[lm], lm_w=np.full(len(lm), 10.0), cloud=target)
    assert np.median(np.linalg.norm(x2 - target, axis=1)) < 0.1


def test_boundary_points_found_on_a_patch_edge():
    g = np.stack(np.meshgrid(np.arange(30.0), np.arange(30.0)), -1).reshape(-1, 2)
    pts = np.c_[g, np.zeros(len(g))]
    n = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    b = nonrigid.boundary_points(pts, n)
    edge = (g == 0).any(axis=1) | (g == 29).any(axis=1)
    assert b[edge].mean() > 0.9 and b[~edge].mean() < 0.05


def test_rbf_field_zero_and_affine():
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(500, 3)) * 50
    q = rng.normal(size=(200, 3)) * 40
    zero = volumetric.build_field(pts, np.zeros_like(pts), smoothing=0.0)
    assert np.abs(volumetric.eval_field(zero, q)).max() < 1e-8
    A = np.array([[0.1, 0.02, 0.0], [0.0, -0.05, 0.01], [0.03, 0.0, 0.07]])
    aff = volumetric.build_field(pts, pts @ A.T + [1, 2, 3], smoothing=0.0)
    np.testing.assert_allclose(volumetric.eval_field(aff, q), q @ A.T + [1, 2, 3], atol=1e-6)


def test_bone_warp_stays_rigid_shaped():
    rng = np.random.default_rng(4)
    bone = rng.normal(size=(300, 3)) * 30
    skin = rng.normal(size=(800, 3)) * 60
    field = volumetric.build_field(skin, _warp(skin) - skin)
    new, info = volumetric.warp_body("bone", bone, field)
    s, R, t = info["similarity"]
    np.testing.assert_allclose(new, apply_similarity(bone, s, R, t))


def test_flipped_faces_detects_inversion():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    f = np.array([[0, 1, 2]])
    assert not geo.flipped_faces(v, v * 2, f).any()
    assert geo.flipped_faces(v, v * [1, -1, 1], f).all()


# --- fixture (this machine's export) ----------------------------------------------------

@needs_export
def test_fixture_invariants():
    with open(config.LANDMARK_TO_VERTEX_JSON, encoding="utf-8") as fh:
        fx = json.load(fh)
    assert len(fx["vertex"]) == 478 and all(v is None for v in fx["vertex"][468:])
    assert fx["checks"]["pass"]
    assert fx["checks"]["landmark33_x_mm"] > 0
    assert not fx["checks"]["left_right_same_side"]
    assert set(fx["weight_class"]) <= {"high", "low", "zero"}
    lips = set(mp.LIPS_INNER_UPPER + mp.LIPS_INNER_LOWER + mp.LIPS_OUTER_UPPER + mp.LIPS_OUTER_LOWER)
    assert all(fx["weight"][i] == 0 for i in lips)


@needs_export
def test_fixture_is_reproducible():
    from pipeline import fixture as fxm
    V, F, _ = fxm.load_skin()
    rebuilt, _aux = fxm.build(V, F, verbose=False)
    committed = fxm.load_fixture()
    assert rebuilt["vertex"] == committed["vertex"]
    assert rebuilt["weight"] == committed["weight"]

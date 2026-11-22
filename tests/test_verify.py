"""Verification layer, step-by-step morph, lip separation, detail pass.

Synthetic shapes unless marked needs_export (this machine's lab head export).

    PYTHONPATH=".pytest-lib;." mayapy -m pytest tests/test_verify.py -v
"""

import numpy as np
import pytest

import config
from core import geometry as geo
from pipeline import detail, morph_steps, volumetric
from pipeline import verify as vf
from tests.test_mouth import _slit_sheet

HAVE_EXPORT = (config.template_export_dir() / "skin_cloth_copy_v5_pull_back.npz").exists()
needs_export = pytest.mark.skipif(not HAVE_EXPORT, reason="template_export/ not on this machine")


def _box(size=10.0, n=6):
    """A closed axis-aligned box surface, outward-wound triangles. -> (V, F)."""
    g = np.linspace(-size / 2, size / 2, n)
    verts, faces, index = [], [], {}

    def vid(p):
        k = tuple(np.round(p, 9))
        if k not in index:
            index[k] = len(verts)
            verts.append(p)
        return index[k]

    for axis in range(3):
        for sign in (-1.0, 1.0):
            u, v = [a for a in range(3) if a != axis]
            for i in range(n - 1):
                for j in range(n - 1):
                    q = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = np.zeros(3)
                        p[axis], p[u], p[v] = sign * size / 2, g[i + di], g[j + dj]
                        q.append(vid(p))
                    a, b, c, d = q
                    tri = [[a, b, c], [a, c, d]]
                    # outward normal: (b-a)x(c-a) must point along +sign on axis
                    nrm = np.cross(np.array(verts[b]) - verts[a], np.array(verts[c]) - verts[a])
                    if nrm[axis] * sign < 0:
                        tri = [[a, c, b], [a, d, c]]
                    faces += tri
    return np.array(verts), np.array(faces)


def _sheet(n=31, step=1.0, z=0.0):
    g = np.stack(np.meshgrid(np.arange(n) * step, np.arange(n) * step), -1).reshape(-1, 2)
    v = np.c_[g, np.full(len(g), z)] - [(n - 1) * step / 2, (n - 1) * step / 2, 0]
    i = np.arange(n * n).reshape(n, n)
    a, b, c, d = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, :-1].ravel(), i[1:, 1:].ravel()
    return v, np.r_[np.c_[a, c, b], np.c_[b, c, d]]


# --- verification checks -------------------------------------------------------------------

def test_box_volume_and_inside():
    V, F = _box(10.0)
    assert vf.signed_volume(V, F) == pytest.approx(1000.0, rel=1e-6)
    pts = np.array([[0, 0, 0], [4.5, 4.5, 4.5], [6, 0, 0], [0, 0, -5.5]])
    assert vf.inside(pts, V, F).tolist() == [True, True, False, False]


def test_volume_check_fails_a_squashed_tissue():
    V, F = _box(10.0)
    lab = {"m": {"rest": V, "faces": F, "role": "tissue"}}
    manifest = {"bodies": {"m": {"role": "tissue", "closed": True}}}
    assert vf.check_volume({"m": 0.8 * V}, lab, manifest, 0.8)["status"] == vf.PASS     # uniform scale: fine
    assert vf.check_volume({"m": V * [1, 1, 0.3]}, lab, manifest, 1.0)["status"] == vf.FAIL


def test_attachment_check_fails_a_torn_attachment():
    V, F = _box(10.0)
    src = np.arange(10)
    pairs = {"a": ("m", "bone", src, src, np.zeros(10))}
    rest = {"m": V.copy(), "bone": V.copy()}
    assert vf.check_attachments(rest, None, 1.0, pairs)["status"] == vf.PASS
    rest["m"] = V + [0, 0, 12.0]                                   # pulled 12 mm off its bone
    assert vf.check_attachments(rest, None, 1.0, pairs)["status"] == vf.FAIL


def test_mesh_check_catches_wrong_counts():
    V, F = _box()
    lab = {"m": {"rest": V, "faces": F, "role": "tissue"}}
    assert vf.check_mesh({"m": V}, lab)["status"] == vf.PASS
    assert vf.check_mesh({"m": V[:-1]}, lab)["status"] == vf.FAIL
    assert vf.check_mesh({"m": np.full_like(V, np.nan)}, lab)["status"] == vf.FAIL


@needs_export
def test_lab_head_passes_every_check():
    from pipeline import fit_subject as fs
    lab, _j, manifest = fs.load_template_bodies()
    rep = vf.verify_rest({k: b["rest"] for k, b in lab.items()}, 1.0, lab, manifest)
    assert rep["verdict"] == vf.PASS, vf.summary(rep)


# --- morph steps ---------------------------------------------------------------------------

def test_rigid_step_is_exact_at_the_ends_and_rigid_between():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(200, 3)) * 20
    ang = 0.7
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    B = 0.7 * A @ R.T + [5.0, -3.0, 8.0]
    assert np.allclose(morph_steps.rigid_step(A, B, 0.0), A)
    assert np.allclose(morph_steps.rigid_step(A, B, 1.0), B)
    M = morph_steps.rigid_step(A, B, 0.5)
    d = lambda X: np.linalg.norm(X[:40, None] - X[None, :40], axis=2)
    np.testing.assert_allclose(d(M), d(A) * 0.7 ** 0.5, rtol=1e-9)


def test_interpolate_moves_soft_bodies_linearly_and_bones_rigidly():
    rng = np.random.default_rng(2)
    A, B = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    out = morph_steps.interpolate({"t": A, "b": A}, {"t": B, "b": 2 * A + 1}, {"t": "tissue", "b": "bone"}, 0.25)
    assert np.allclose(out["t"], 0.75 * A + 0.25 * B)
    d = lambda X: np.linalg.norm(X[:, None] - X[None], axis=2)
    np.testing.assert_allclose(d(out["b"]), d(A) * 2 ** 0.25, rtol=1e-9)
    assert morph_steps.step_name("s01_x", 0.5) == "s01_x_t050"


# --- lips that separate --------------------------------------------------------------------

def _lip_slabs(gap):
    xs, ys, zs = np.arange(-8, 8.01, 0.5), np.arange(0, 3.01, 0.5), np.arange(0, 2.01, 1.0)
    slab = np.stack(np.meshgrid(xs, ys, zs), -1).reshape(-1, 3)
    P = np.r_[slab + [0, gap / 2, 0], slab * [1, -1, 1] - [0, gap / 2, 0]]
    upper = np.r_[np.ones(len(slab), bool), np.zeros(len(slab), bool)]
    return P, upper


def test_split_lip_warp_moves_each_half_with_its_own_lip():
    V, F, parts = _slit_sheet(half_w=10, half_h=1)
    slit = volumetric.slit_line(V, parts)
    P, upper = _lip_slabs(1.0)
    pts = np.random.default_rng(3).uniform(-30, 30, size=(300, 3))
    up = volumetric.build_field(pts, np.tile([0, 1.0, 0], (300, 1)), smoothing=0.0)       # upper lip rises
    lo = volumetric.build_field(pts, np.tile([0, -1.0, 0], (300, 1)), smoothing=0.0)      # lower lip drops
    new, _ = volumetric.split_lip_warp(P, up, lo, slit, upper)
    mid = np.abs(P[:, 0]) < 5
    # the split fades a little with depth into the mouth (2 mm deep: 0.93 mm)
    assert np.allclose(new[mid & upper] - P[mid & upper], [0, 1, 0], atol=0.1)
    assert np.allclose(new[mid & ~upper] - P[mid & ~upper], [0, -1, 0], atol=0.1)
    front = mid & (P[:, 2] == 0)
    assert np.allclose(new[front & upper] - P[front & upper], [0, 1, 0], atol=1e-6)
    far = np.array([[25.0, 0.5, 0.0]])                                                   # past the corner: shared
    assert np.allclose(volumetric.split_lip_warp(far, up, lo, slit)[0], far, atol=1e-6)


def test_lip_muscle_guard_reopens_a_squeezed_muscle():
    V, F, parts = _slit_sheet(half_w=10, half_h=1)
    ref, upper = _lip_slabs(2.0)
    squeezed, _ = _lip_slabs(0.3)
    out, info = volumetric.open_lip_muscle(squeezed, V, parts, ref, V, upper=upper)
    assert info["moved"] and info["gap_median_after_mm"] >= 0.65 * 2.0
    same, info2 = volumetric.open_lip_muscle(ref, V, parts, ref, V, upper=upper)
    assert not info2["moved"] and np.allclose(same, ref)


# --- tissue fallbacks ----------------------------------------------------------------------

def test_affine_fallback_keeps_an_affine_warp_and_unmirrors_a_reflection():
    V, F = _box()
    A = np.array([[1.1, 0.2, 0], [0, 0.9, 0.1], [0, 0, 1.2]])
    warped = V @ A.T + 2.0
    new, info = volumetric.affine_fallback(V, warped)
    assert info["fallback"] == "affine" and np.allclose(new, warped)
    mirrored = V * [1, 1, -1]
    new, info = volumetric.affine_fallback(V, mirrored)
    assert info["fallback"] == "similarity"
    assert vf.signed_volume(new, F) > 0                           # right side out (the mirror's is negative)
    assert vf.signed_volume(mirrored, F) < 0


def test_push_out_of_bones_clears_new_penetration_only():
    bone, bf = _box(10.0)
    tissue, tf = _sheet(11, 1.0, z=-4.0)                      # a sheet 1 mm inside the bone's -z face
    was = np.zeros(len(tissue), dtype=bool)
    new, info = volumetric.push_out_of_bones(tissue, tf, [(bone, bf)], was)
    assert info["n_inside"] > 0 and not vf.inside(new, bone, bf).any()
    kept, info2 = volumetric.push_out_of_bones(tissue, tf, [(bone, bf)], np.ones(len(tissue), bool))
    assert info2["n_pushed"] == 0 and np.allclose(kept, tissue)


# --- detail pass ---------------------------------------------------------------------------

def test_detail_pass_recovers_a_bump_and_respects_clamp_and_holes():
    V, F = _sheet(41, 1.0)
    r = np.linalg.norm(V[:, :2], axis=1)
    for h, expect in ((2.0, 2.0), (5.5, detail.CLAMP_MM)):        # within the 6 mm search; the second is clamped
        bump = V.copy()
        bump[:, 2] = -h * np.exp(-(r / 12.0) ** 2)                 # toward the camera (-Z), broad
        cn = np.tile([0, 0, -1.0], (len(bump), 1))
        X, info = detail.detail_pass(V, F, bump, cn, np.ones(len(V), bool), rim_dist=np.full(len(V), 50.0))
        centre = int(np.argmin(r))
        moved = V[centre, 2] - X[centre, 2]
        assert moved == pytest.approx(detail.GAIN * expect, rel=0.15)
    X, _ = detail.detail_pass(V, F, bump, cn, np.ones(len(V), bool), rim_dist=np.zeros(len(V)))
    assert np.allclose(X, V)                                      # within a hole's fade: untouched

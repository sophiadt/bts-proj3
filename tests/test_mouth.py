"""The mouth must still open: pipeline/mouth.py measures and the gap-keeping
mouth targets (pipeline/holes.py, pipeline/fit_subject.fit_skin).

Unit tests use synthetic sheets. Tests marked needs_lab use the lab skin from
config.lab_data_dir() (template_export/, else ../bts-proj1/data/lab) and the
committed captures; they skip where neither is present.

    .venv\\Scripts\\python -m pytest tests/test_mouth.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import config
from core.metrics import boundary_loops
from pipeline import holes, mouth

HAVE_LAB = (config.lab_data_dir() / (mouth.SKIN + ".npz")).exists()
needs_lab = pytest.mark.skipif(not HAVE_LAB, reason="lab .npz exports not available")


def _slit_sheet(half_w=10, half_h=2, size=30, open_slit=True):
    """A 1 mm grid sheet in the XY plane (facing -Z) with a 2*half_w x 2*half_h
    rectangular slit at the origin. -> (V, F, parts)."""
    n = 2 * size + 1
    g = np.stack(np.meshgrid(np.arange(n) - size, np.arange(n) - size), -1).reshape(-1, 2).astype(float)
    V = np.c_[g, np.zeros(len(g))]
    i = np.arange(n * n).reshape(n, n)
    a, b, c, d = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, :-1].ravel(), i[1:, 1:].ravel()
    F = np.r_[np.c_[a, c, b], np.c_[b, c, d]]
    cen = V[F].mean(axis=1)
    in_slit = (np.abs(cen[:, 0]) < half_w) & (np.abs(cen[:, 1]) < half_h)
    Fo = F[~in_slit]
    loop = min(boundary_loops(Fo), key=len)            # the slit, not the sheet's outer edge
    L = np.asarray(loop)
    up, lo = L[V[L, 1] > 0], L[V[L, 1] < 0]
    parts = {"loop": L, "upper": up, "lower": lo,
             "mid_upper": int(up[np.argmin(np.abs(V[up, 0]))]), "mid_lower": int(lo[np.argmin(np.abs(V[lo, 0]))])}
    return V, (Fo if open_slit else F), parts


# --- measures ----------------------------------------------------------------------------

def test_front_opening_is_the_slit_area():
    V, F, parts = _slit_sheet(half_w=10, half_h=2)
    assert mouth.front_opening_mm2(V, F, parts) == pytest.approx(80.0, rel=0.03)
    assert mouth.edge_gap_mm(V, parts) == pytest.approx(4.0)


def test_closed_sheet_has_no_opening():
    V, F, parts = _slit_sheet(open_slit=False)
    assert mouth.front_opening_mm2(V, F, parts) == 0.0


def test_something_behind_the_slit_closes_it_from_the_front():
    V, F, parts = _slit_sheet(half_w=10, half_h=2)
    # a patch 5 mm behind the slit (like the lip lining) blocks the view through it
    patch = np.array([[-12, -4, 5.0], [12, -4, 5.0], [12, 4, 5.0], [-12, 4, 5.0]])
    V2 = np.r_[V, patch]
    k = len(V)
    F2 = np.r_[F, [[k, k + 1, k + 2], [k, k + 2, k + 3]]]
    assert mouth.front_opening_mm2(V2, F2, parts) == 0.0


def test_muscle_gap_between_two_slabs():
    V, F, parts = _slit_sheet(half_w=10, half_h=2)
    xs, ys, zs = np.arange(-12, 12.01, 0.25), np.arange(0, 4.01, 0.5), np.arange(-2, 2.01, 1.0)
    slab = np.stack(np.meshgrid(xs, ys, zs), -1).reshape(-1, 3)
    for g in (0.4, 1.5, 3.0):
        P = np.r_[slab + [0, g / 2, 0], slab * [1, -1, 1] - [0, g / 2, 0]]
        r = mouth.muscle_gap(P, V, parts)
        assert r["min"] == pytest.approx(g, abs=1e-9) and r["median"] == pytest.approx(g, abs=1e-9)


def test_muscle_compare_counts_closing_slices():
    V, F, parts = _slit_sheet(half_w=10, half_h=2)
    xs, ys = np.arange(-12, 12.01, 0.25), np.arange(0, 4.01, 0.5)
    slab = np.stack(np.meshgrid(xs, ys, [0.0]), -1).reshape(-1, 3)
    ref = np.r_[slab + [0, 1.0, 0], slab * [1, -1, 1] - [0, 1.0, 0]]               # gap 2 mm everywhere
    squeezed = ref.copy()
    left = squeezed[:, 0] < 0
    squeezed[left, 1] -= 0.8 * np.sign(squeezed[left, 1])                          # left half: gap 0.4 mm
    r = mouth.muscle_compare(squeezed, V, ref, V, parts)
    assert r["muscle_gap_min_mm"] == pytest.approx(0.4)
    assert r["muscle_closing_frac"] == pytest.approx(0.5, abs=0.1)                  # 0.4 / 2 is under a third
    same = mouth.muscle_compare(ref, V, ref, V, parts)
    assert same["muscle_slices_closing"] == 0 and same["muscle_ratio"] == pytest.approx(1.0)
    thinner = ref.copy()
    thinner[:, 1] *= 0.6                                                            # whole lip squeezed: not closing
    assert mouth.muscle_compare(thinner, V, ref, V, parts)["muscle_slices_closing"] == 0


def test_verdict_thresholds():
    assert mouth.verdict(None) == "UNKNOWN"
    assert mouth.verdict(0.32) == "SEALED"
    assert mouth.verdict(0.55) == "LOW"
    assert mouth.verdict(0.9) == "OK"
    assert mouth.verdict(0.9, muscle_closing_frac=0.15) == "LOW"
    assert mouth.verdict(0.9, muscle_closing_frac=0.0) == "OK"


# --- the gap-keeping targets ---------------------------------------------------------------

def test_gap_targets_keep_the_reference_gap_on_the_persons_lip_line():
    n = 12
    seam = np.c_[np.linspace(20, -20, n), np.zeros(n), np.full(n, -30.0)]          # person's closed lips
    ref_seam = np.c_[np.linspace(25, -25, n), np.full(n, -60.0), np.full(n, -40.0)]
    ref_up, ref_lo = ref_seam + [0, 3.0, 2.0], ref_seam + [0, -3.0, 2.0]
    up, lo = holes.gap_targets(seam, ref_seam, ref_up, ref_lo)
    assert np.allclose(up - lo, [0, 6.0, 0])                                         # the reference's gap
    assert np.allclose(0.5 * (up + lo), seam + [0, 0, 2.0])                         # centred on the person
    shift = np.array([1.0, -4.0, 0.5])
    up2, lo2 = holes.gap_targets(seam + shift, ref_seam, ref_up, ref_lo)
    assert np.allclose(up2 - up, shift) and np.allclose(lo2 - lo, shift)


def test_drop_seam_band_removes_only_the_lip_contact_strip():
    seam = np.c_[np.linspace(-20, 20, 12), np.zeros(12), np.zeros(12)]
    pts = np.array([[0, 0.5, 0], [5, -2.0, 1], [0, 3.0, 0], [0, -3.0, 0], [0, 0.5, 20], [30, 0, 0]])
    keep = holes.drop_seam_band(pts, seam, band_mm=2.5, depth_mm=12.0)
    assert keep.tolist() == [False, False, True, True, True, True]
    assert holes.drop_seam_band(pts, None).all()


# --- on the lab head and the committed captures --------------------------------------------

@needs_lab
def test_lab_head_rests_open_and_opens_when_speaking():
    VV, F = mouth.load_lab_mesh(mouth.SKIN, all_frames=True)
    if len(VV) < 31:
        pytest.skip("single-frame lab export: no speech frames")
    parts = mouth.mouth_parts(VV[15], F)
    assert mouth.edge_gap_mm(VV[15], parts) == pytest.approx(8.27, abs=0.05)
    assert mouth.front_opening_mm2(VV[15], F, parts) == pytest.approx(272.0, rel=0.05)
    assert mouth.front_opening_mm2(VV[30], F, parts) == pytest.approx(640.0, rel=0.05)


def _skin_fits(capture_id, relabel):
    from pipeline import capture as cap, fit_subject as fs, fixture as fx, merge_views
    lab, F = mouth.load_lab_mesh(mouth.SKIN)
    fixture = fx.load_fixture()
    views = cap.load_capture(cap.resolve_capture(capture_id))
    if relabel:
        fs.relabel(views)
    mv = merge_views.merge(views, fixture, lab)
    s = mv["scale"]
    out = {}
    for mode in (fs.MOUTH_CONTOURS, fs.MOUTH_GAP):
        X, X0, _ctx = fs.fit_skin(lab, F, mv, fixture, s, gap_ref=s * lab, mouth=mode, verbose=False)
        out[mode] = (X, X0)
    return lab, F, s, out


@needs_lab
@pytest.mark.parametrize("capture_id,relabel", [("s01_20261110T212937Z_burst", False),
                                                ("s01_20261030T170049Z_guided", True)])
def test_closed_mouth_capture_keeps_the_rigs_mouth_gap(capture_id, relabel):
    """The bug and the fix on real captures: pulling the mouth onto a closed
    mouth's lip contours shuts it; keeping the rig's gap lets it open."""
    from pipeline import fit_subject as fs, volumetric
    motion, _src = mouth.lab_speech_motion(step=8)
    if motion is None:
        pytest.skip("no lab speech motion on this machine")
    lab, F, s, fits = _skin_fits(capture_id, relabel)
    parts = mouth.mouth_parts(lab, F)
    oo, _pf = mouth.load_lab_mesh(mouth.LIP_MUSCLE)
    old = mouth.assess(fits[fs.MOUTH_CONTOURS][0], F, parts, s * lab, motion=motion, motion_scale=s)
    X, X0 = fits[fs.MOUTH_GAP]
    P = s * oo + volumetric.eval_field(volumetric.build_field(X0, X - X0), s * oo)
    new = mouth.assess(X, F, parts, s * lab, muscle=P, ref_muscle=s * oo, motion=motion, motion_scale=s)
    assert old["edge_gap_mm"] < 0.35 * old["ref_edge_gap_mm"] and old["speech_ratio"] < 0.6
    assert new["edge_gap_mm"] == pytest.approx(new["ref_edge_gap_mm"], rel=0.1)
    assert new["speech_ratio"] >= mouth.SPEECH_OK and new["verdict"] == "OK"


@needs_lab
def test_base_head_build_keeps_the_mouth_gap():
    """build_base.py's fit (MediaPipe's average face, parted lips, base mode)
    keeps the lab head's resting gap instead of drifting open."""
    from pipeline import capture as cap, fit_subject as fs, fixture as fx, merge_views, synthetic
    lab, F = mouth.load_lab_mesh(mouth.SKIN)
    fixture = fx.load_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        mv = merge_views.merge(cap.load_capture(synthetic.make_guided(Path(tmp) / "avg")), fixture, lab)
    s = mv["scale"]
    parts = mouth.mouth_parts(lab, F)
    X, _X0, _ctx = fs.fit_skin(lab, F, mv, fixture, s, gap_ref=s * lab, base_mode=True, mouth=fs.MOUTH_GAP,
                               verbose=False)
    assert mouth.edge_gap_mm(X, parts) == pytest.approx(mouth.edge_gap_mm(s * lab, parts), rel=0.1)

"""Contour smoothing and ring depth (pipeline/holes.py): optional, off by default."""

import numpy as np

from pipeline import holes


def _lip_line(rng, noise_mm, n=11):
    """An inner-lip-like polyline in x/y/z (mm) with a 2 mm cupid's-bow dip, plus landmark noise."""
    x = np.linspace(-25, 25, n)
    y = 0.012 * x ** 2 - 2.0 * np.exp(-(x / 5.0) ** 2)           # the bow: a 2 mm dip at the middle
    z = 330 + 0.004 * x ** 2
    clean = np.c_[x, y, z]
    return clean, clean + rng.normal(scale=noise_mm, size=clean.shape)


def test_smoothing_off_is_the_identity():
    _, noisy = _lip_line(np.random.default_rng(0), 0.8)
    np.testing.assert_array_equal(holes.smooth_contour(noisy, 0.0), noisy)
    np.testing.assert_array_equal(holes.smooth_contour(noisy[:4], 0.8), noisy[:4])    # too few points


def test_smoothing_removes_jitter_and_keeps_the_cupids_bow():
    rng = np.random.default_rng(1)
    gain, kept = [], []
    for _ in range(40):
        clean, noisy = _lip_line(rng, 0.8)
        sm = holes.smooth_contour(noisy, 0.8)
        gain.append(np.sqrt(np.mean((noisy - clean) ** 2)) / np.sqrt(np.mean((sm - clean) ** 2)))
        mid = len(clean) // 2
        kept.append((np.mean(sm[[mid - 3, mid + 3], 1]) - sm[mid, 1]) /
                    (np.mean(clean[[mid - 3, mid + 3], 1]) - clean[mid, 1]))
    assert np.mean(gain) > 1.12            # closer to the truth than the raw landmarks (about 1.2x)
    assert np.mean(kept) > 0.7             # most of the 2 mm bow survives (about 0.8)


def test_smoothing_never_moves_a_contour_far_from_its_landmarks():
    _, noisy = _lip_line(np.random.default_rng(2), 0.8)
    sm = holes.smooth_contour(noisy, 0.8)
    assert np.linalg.norm(sm - noisy, axis=1).max() < 2.5


def _face_patch(bias_inside_mm=0.0, spacing=0.5, seed=0):
    """A curved 'skin' z(x, y) with an eye-shaped opening whose own readings are biased."""
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(np.arange(-30, 30, spacing), np.arange(-20, 20, spacing)), -1).reshape(-1, 2)
    z = 330 + 0.01 * (g[:, 0] ** 2 + g[:, 1] ** 2) + rng.normal(scale=0.2, size=len(g))
    inside = (g[:, 0] / 12.0) ** 2 + (g[:, 1] / 5.0) ** 2 < 1.0
    z = z + inside * bias_inside_mm
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    outline = np.c_[12.0 * np.cos(th), 5.0 * np.sin(th)]
    return np.c_[g, z], outline


def test_ring_depth_ignores_a_biased_opening():
    pts, outline = _face_patch(bias_inside_mm=3.0)
    q = np.array([[0.0, 0.0], [-8.0, 1.0], [8.0, -1.0]])
    truth = 330 + 0.01 * (q[:, 0] ** 2 + q[:, 1] ** 2)
    z = holes.ring_depth(pts, q, truth + 3.0, outline, True, 8.0)
    assert np.all(np.abs(z - truth) < 0.5)                       # the ring's answer, not the +3 mm readings
    plain = holes.ring_depth(pts, q, truth + 3.0, outline, True, 8.0, r_in=0.0)
    assert np.all(np.abs(plain - truth) < 0.5)                    # and a ring starting at the outline works too


def test_ring_depth_needs_enough_points_and_a_depth_window():
    pts, outline = _face_patch()
    q = np.array([[0.0, 0.0]])
    assert np.isnan(holes.ring_depth(pts[:10], q, np.array([330.0]), outline, True, 8.0)).all()
    far = holes.ring_depth(pts, q, np.array([360.0]), outline, True, 8.0)      # hint 30 mm off: nothing in the window
    assert np.isnan(far).all()


def test_ring_adjust_pulls_back_a_biased_target_and_leaves_agreeing_ones():
    pts, outline = _face_patch(bias_inside_mm=2.5)
    truth = lambda xy: 330 + 0.01 * (xy[:, 0] ** 2 + xy[:, 1] ** 2)
    xy = np.array([[-6.0, 0.0], [0.0, 0.0], [6.0, 0.0]])
    target = np.c_[xy, truth(xy) + np.array([2.5, 2.5, 0.2])]                  # two biased, one within the dead band
    ref_pts, ref_outline = pts, outline                                          # reference == same shape: zero offset
    ref_margin = np.c_[xy, truth(xy)]
    out, n = holes.ring_adjust(target, pts, outline, True, ref_pts, ref_outline, ref_margin)
    expect = truth(xy)
    assert np.all(np.abs(out[:2, 2] - (expect[:2] + holes.RING_DEADBAND_MM)) < 0.35)   # pulled to within the soft threshold
    assert out[2, 2] == target[2, 2] or abs(out[2, 2] - target[2, 2]) < 0.35            # the agreeing target barely moves
    assert n >= 2


def test_ring_adjust_keeps_the_reference_margin_curve():
    pts, outline = _face_patch()
    xy = np.array([[-6.0, 0.0], [0.0, 0.0], [6.0, 0.0]])
    surf = 330 + 0.01 * (xy[:, 0] ** 2 + xy[:, 1] ** 2)
    ref_margin = np.c_[xy, surf - 1.5]                    # on the reference head the margin sits 1.5 mm in front of its ring
    target = np.c_[xy, surf]                              # the person's scan reads the plain surface there
    out, _ = holes.ring_adjust(target, pts, outline, True, pts, outline, ref_margin)
    assert np.all(out[:, 2] < target[:, 2] - 0.5)         # it moves toward the reference's lip/lid projection
    assert np.all(out[:, 2] >= target[:, 2] - holes.RING_CLAMP_MM - 1e-9)


def test_ring_adjust_leaves_a_target_alone_when_the_ring_cannot_be_fitted():
    pts, outline = _face_patch()
    target = np.array([[0.0, 0.0, 330.0], [1.0, 0.0, 330.0]])
    out, n = holes.ring_adjust(target, pts[:5], outline, True, pts, outline, target.copy())
    np.testing.assert_array_equal(out, target)
    assert n == 0


def test_ring_adjust_clamps_the_change():
    pts, outline = _face_patch()
    xy = np.array([[0.0, 0.0]])
    surf = 330 + 0.01 * (xy[:, 0] ** 2 + xy[:, 1] ** 2)
    target = np.c_[xy, surf + 6.0]
    ref_margin = np.c_[xy, surf]
    out, _ = holes.ring_adjust(target, pts, outline, True, pts, outline, ref_margin)
    assert abs(out[0, 2] - target[0, 2]) <= holes.RING_CLAMP_MM + 1e-9


# --- the options wired into constraints() / mouth_gap_constraints(), on a stand-in skin ----------

def _mock_world(bias_mm=0.0):
    """A curved sheet 'skin' with two eye openings and a lip line, a fixture and landmarks on them."""
    from pipeline import canonical as mp
    g = np.stack(np.meshgrid(np.arange(-45, 45.01, 1.5), np.arange(-40, 25.01, 1.5)), -1).reshape(-1, 2)
    surf = lambda xy: 330 + 0.01 * (xy[:, 0] ** 2 + xy[:, 1] ** 2)
    V = np.c_[g, surf(g)]
    lm = np.full((mp.N_LANDMARKS, 3), np.nan)
    vert = [None] * mp.N_LANDMARKS

    def nearest(p):
        return int(np.argmin(np.linalg.norm(V[:, :2] - p[:2], axis=1)))

    def put(ids, xy):
        for i, p in zip(ids, xy):
            lm[i] = [p[0], p[1], surf(np.array([p]))[0]]
            vert[i] = nearest(p)

    eyes = {}
    for name, (up, lo, inner), cx in (("eye_r", (mp.RIGHT_EYE_UPPER, mp.RIGHT_EYE_LOWER, mp.EYE_R_INNER), 18.0),
                                      ("eye_l", (mp.LEFT_EYE_UPPER, mp.LEFT_EYE_LOWER, mp.EYE_L_INNER), -18.0)):
        up, lo = list(up), list(lo)
        sgn = 1.0 if cx > 0 else -1.0
        t_up, t_lo = np.linspace(0, np.pi, len(up)), np.linspace(0, np.pi, len(lo))
        # inner corner first: for the +X eye the inner corner is at the smaller x
        put(up, np.c_[cx - sgn * 7 * np.cos(t_up), 10 + 3.0 * np.sin(t_up)])
        put(lo, np.c_[cx - sgn * 7 * np.cos(t_lo), 10 - 2.5 * np.sin(t_lo)])
        eyes[name] = (up, lo)
    mu, ml = list(mp.LIPS_INNER_UPPER), list(mp.LIPS_INNER_LOWER)
    xs_u, xs_l = np.linspace(9, -9, len(mu)), np.linspace(9, -9, len(ml))
    put(mu, np.c_[xs_u, -22.0 + 0.3 * np.cos(xs_u / 3.0)])
    put(ml, np.c_[xs_l, -22.5 - 0.3 * np.cos(xs_l / 3.0)])
    return V, lm, vert, eyes, (mu, ml), surf


def _cloud_with_bias(surf, bias_mm, lm_eyes):
    g = np.stack(np.meshgrid(np.arange(-45, 45.01, 0.6), np.arange(-40, 25.01, 0.6)), -1).reshape(-1, 2)
    z = surf(g)
    for cx in (18.0, -18.0):                                  # the sensor reads the wet eye deeper
        z = z + bias_mm * (((g[:, 0] - cx) / 7.0) ** 2 + ((g[:, 1] - 10.0) / 3.0) ** 2 < 1.0)
    return np.c_[g, z]


def _run_constraints(V, lm, vert, eyes, cloud, **kw):
    from pipeline import holes as h
    chains = {n: ([int(vv) for vv in [vert[i] for i in up]], [int(vv) for vv in [vert[i] for i in lo]])
              for n, (up, lo) in eyes.items()}
    chains["mouth"] = ([vert[i] for i in holes_mouth_ids(lm, vert)[0]], [vert[i] for i in holes_mouth_ids(lm, vert)[1]])
    h.template_edges = lambda V_, F_: (chains, 0.0)
    fixture = {"vertex": vert}
    valid = np.isfinite(lm).all(axis=1)
    return h.constraints(V, None, lm, valid, cloud, ref_skin=V, fixture=fixture, only=("eye_r", "eye_l"), **kw)


def holes_mouth_ids(lm, vert):
    from pipeline import canonical as mp
    return list(mp.LIPS_INNER_UPPER), list(mp.LIPS_INNER_LOWER)


def test_options_off_change_nothing_and_on_runs_end_to_end(monkeypatch):
    from pipeline import holes as h
    V, lm, vert, eyes, mouth, surf = _mock_world()
    cloud = _cloud_with_bias(surf, 3.0, eyes)
    monkeypatch.setattr(h, "template_edges", h.template_edges)          # restored after the test
    i0, t0 = _run_constraints(V, lm, vert, eyes, cloud)
    i1, t1 = _run_constraints(V, lm, vert, eyes, cloud, smooth_mm=0.0, ring=False)
    np.testing.assert_array_equal(i0, i1)
    np.testing.assert_array_equal(t0, t1)                              # defaults == explicit off == the old behaviour
    h.RING_STATS.clear()
    i2, t2 = _run_constraints(V, lm, vert, eyes, cloud, smooth_mm=0.8, ring=True)
    assert t2.shape == t0.shape and np.isfinite(t2).all()
    assert h.RING_STATS["fitted"] > 0
    # eyes: the scan read +3 mm inside the opening; the ring pulls those targets back toward the surface
    truth = surf(t0[:, :2])
    assert np.abs(t2[:, 2] - truth).mean() < np.abs(t0[:, 2] - truth).mean()
    assert np.abs(t2[:, 2] - t0[:, 2]).max() <= h.RING_CLAMP_MM + 3.0 * 0.8   # smoothing moves xy/z a little more


def test_mouth_gap_options_run(monkeypatch):
    from pipeline import holes as h
    V, lm, vert, eyes, (mu, ml), surf = _mock_world()
    cloud = _cloud_with_bias(surf, 0.0, eyes)
    edges = {"mouth": ([vert[i] for i in mu], [vert[i] for i in ml])}
    valid = np.isfinite(lm).all(axis=1)
    fixture = {"vertex": vert}
    # a reference head whose lip line rests 1 mm proud of its surroundings
    ref = V.copy()
    base = h.mouth_gap_constraints(V, None, lm, valid, ref, cloud, fixture, edges=edges)
    h.RING_STATS.clear()
    both = h.mouth_gap_constraints(V, None, lm, valid, ref, cloud, fixture, edges=edges, smooth_mm=0.8, ring=True)
    assert base[0].shape == both[0].shape and base[1].shape == both[1].shape
    assert np.isfinite(both[1]).all() and both[2] is not None
    assert "targets" in h.RING_STATS

"""Fit the lab template to one phone capture -> rest geometry for the rig.

    capture dir
      -> mirror gate           pipeline/mirror_check.py   (stops unless OK, or --relabel / --force)
      -> views merged          pipeline/merge_views.py    (rigid per view + ICP, fit frame)
      -> uniform scale s       every body and joint scaled by s about the template origin
      -> skin non-rigid fit    pipeline/nonrigid.py       (fixture landmarks + dense cloud, holes masked)
      -> volumetric warp       pipeline/volumetric.py     (tissue full field, bones similarity, joints)
      -> checks                flips, skin-to-skull thickness, bone conformity
      -> template_export/subjects/<id>/<body>.npz (rest_vertices) + joints.npz + report

The skin is fitted in output space (what the camera can see) and the same
per-vertex displacement is applied to its Ziva rest, which differs from the
output only along the top forehead edge ("pull_back", up to 16.6 mm).

Then, in Maya:
    mayapy rig/write_rest.py template_export/subjects/<id> --save <id>
    mayapy rig/sim_check.py template_export/scenes/<id>.mb --subject <id>

    mayapy pipeline/fit_subject.py CAPTURE_DIR [--id NAME] [--relabel] [--force]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import io as cio                             # noqa: E402
from pipeline import canonical as mp                   # noqa: E402
from pipeline import capture as cap                    # noqa: E402
from pipeline import fixture as fx                     # noqa: E402
from pipeline import holes, importance, merge_views, mirror_check, neutral_check, nonrigid, volumetric  # noqa: E402
from core.similarity import apply_similarity                # noqa: E402
from core.geometry import ray_hit_distance as geo_ray       # noqa: E402

SKULL = "middleres_skull"
JAW = "middleres_Jaw"
# Bones placed from skin depth, and the bodies that move rigidly with them
# (TeethUp follows the skull through wrap26; TeethDown is skinned to jaw01).
BONE_GROUPS = {SKULL: [SKULL, "middleres_TeethUp"], JAW: [JAW, "middleres_TeethDown"]}
BONE_RAYS = 3000
BONE_CENTRES = 800
LM_WEIGHT = 20.0
# Trust the scan only to its noise level. Raw TrueDepth depth carries ~1 mm of
# noise; matching it more closely copies the noise into the skin. On the first
# real capture (from the base head): stiffness 20 / full scan pull matched the
# scan to 0.92 mm and looked crumpled (1,047 folds); stiffness 200 / pull 0.3
# matches to 1.12 mm, folds 507, and looks like a face. (Earlier: stiffness 2
# left 2,449 folds, 10 left 1,322.)
NONRIGID = dict(schedule=(400.0, 200.0), iters_per_stage=3, max_dist=10.0, beta=0.3,
                point_to_plane=True, max_tangent=3.0)
DENOISE_K = 32
# Openings and importance (first real capture): pulling the eye and mouth holes
# onto the eyelid and inner-lip contours took them from 4.6 mm to 0.19 mm off;
# stiffness / 5 at the most important skin then gave lips 0.85 mm (was 0.91)
# and openings 0.11 mm, for 866 folds instead of 840. Extra data weight on
# the Poisson-disk anchors measured no gain (every visible vertex already
# follows the scan), so it is off unless asked for.
HOLE_WEIGHT = 10.0
IMPORTANCE_GAIN = 4.0
ANCHOR_DATA_WEIGHT = 4.0
COVERAGE_MM = 3.0
# Skin with no scan point this close keeps 4x the stiffness (unseen areas keep
# the start shape's smooth form instead of being dragged along unevenly).
UNSEEN_MM = 8.0
UNSEEN_STIFFNESS = 4.0
# Weak pull of unseen skin toward the start shape. Without it the unseen side
# of the head swung by up to 14 mm with the choice of views (319 folds); at
# 0.2 it moves at most 5 mm, only where a view adds data (388 folds).
UNSEEN_TETHER = 0.2
BASE_SUBJECT = "base_human"


def load_template_bodies(src_dir=None):
    src = Path(src_dir) if src_dir else config.template_export_dir()
    with open(src / "manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    bodies = {}
    for key, info in manifest["bodies"].items():
        with np.load(src / (key + ".npz"), allow_pickle=False) as z:
            bodies[key] = {"role": info["role"], "out": np.asarray(z["vertices"][0], dtype=np.float64),
                           "rest": np.asarray(z["rest_vertices"], dtype=np.float64),
                           "faces": np.asarray(z["faces"], dtype=np.int64).reshape(-1, 3)}
        if len(bodies[key]["out"]) != info["n_verts"]:
            raise cio.InvariantError("{0}: {1} vertices, manifest says {2}".format(
                key, len(bodies[key]["out"]), info["n_verts"]))
    with np.load(src / "joints.npz", allow_pickle=False) as z:
        joints = dict(zip([str(n) for n in z["names"]], np.asarray(z["positions"], dtype=np.float64)))
    return bodies, joints, manifest


def load_base_bodies(name):
    """The human-proportioned base head (pipeline/build_base.py) as the start
    shape: same meshes, vertex order and joints as the lab head, with each
    body's template output-minus-rest offset kept (the skin's forehead strip,
    the legacy meshes), so the rig sees the same offsets."""
    bodies, joints, manifest = load_template_bodies()
    src = config.template_export_dir() / "subjects" / name
    for k, b in bodies.items():
        with np.load(src / (k + ".npz"), allow_pickle=False) as z:
            rest = np.asarray(z["rest_vertices"], dtype=np.float64)
        if rest.shape != b["rest"].shape:
            raise cio.InvariantError("{0}: base {1} has {2} vertices, template {3}".format(
                k, name, len(rest), len(b["rest"])))
        offset = b["out"] - b["rest"]
        b["rest"], b["out"] = rest, rest + offset
    with np.load(src / "joints.npz", allow_pickle=False) as z:
        joints = dict(zip([str(n) for n in z["names"]], np.asarray(z["positions"], dtype=np.float64)))
    return bodies, joints, manifest


def base_scale_vs_lab_head(name):
    """The base head's own uniform scale against the lab head (from its report)."""
    with open(config.template_export_dir() / "subjects" / name / "report.json", encoding="utf-8") as fh:
        m = json.load(fh)["metrics"]
    return float(m.get("scale_vs_lab_head", m["scale"]))


def relabel(views):
    """Swap MediaPipe's left/right indices in every view (LABELS_SWAPPED)."""
    perm = list(fx.mirror_permutation(mp.load()["vertices"])) + [473, 474, 475, 476, 477, 468, 469, 470, 471, 472]
    for v in views.values():
        v.landmarks = v.landmarks[perm]
        v.unreliable = v.unreliable[perm]


def fit(capture_dir, subject_id, out_root=None, do_relabel=False, force=False, use_anchors=False,
        base=None, base_mode=False, verbose=True):
    """base: start from template_export/subjects/<base> (the human base head)
    instead of the lab head. base_mode: building that base from MediaPipe's
    average face -- its lips are parted (5.8 mm), so lip landmarks, lip-region
    scan data and the mouth opening are left out and the lab head's closed
    mouth is kept."""
    capture_dir = Path(capture_dir)
    out_dir = Path(out_root or config.template_export_dir() / "subjects") / subject_id
    with cio.RunDir(subject_id, "fit", "pipeline/fit_subject.py") as run:
        M = run.metrics
        views = cap.load_capture(capture_dir)
        gate = mirror_check.check(views)
        M["mirror_verdict"] = gate["verdict"]
        M["mirror_views"] = gate["views"]
        if verbose:
            print("mirror check:", gate["verdict"])
        if gate["verdict"] == "LABELS_SWAPPED" and do_relabel:
            relabel(views)
            M["relabelled"] = True
            again = mirror_check.check(views)
            M["mirror_verdict_after_relabel"] = again["verdict"]
            if verbose:
                print("mirror check after relabel:", again["verdict"])
            if again["verdict"] != "OK" and not force:
                raise RuntimeError("relabelling did not fix the mirror check: " + again["verdict"])
        elif gate["verdict"] != "OK" and not force:
            raise RuntimeError("mirror gate: {0}. {1}".format(gate["verdict"], mirror_check.ADVICE.get(gate["verdict"])))

        bodies, joints, manifest = load_base_bodies(base) if base else load_template_bodies()
        M["start_head"] = base or "lab head"
        if verbose:
            print("start head:", M["start_head"])
        fixture = fx.load_fixture()
        skin = bodies[fx.SKIN]
        V, F = skin["out"], skin["faces"]

        # 1. views -> fit frame; template scale
        mv = merge_views.merge(views, fixture, V)
        s = mv["scale"]
        M["scale"] = s
        # against the lab head itself (motion recorded on it is resized by this)
        M["scale_vs_lab_head"] = s * (base_scale_vs_lab_head(base) if base else 1.0)
        M["rigid"] = {k: {kk: vv for kk, vv in v.items() if kk != "residual_mm"} for k, v in mv["per_view"].items()}
        M["rigid_residual_mm"] = mv["per_view"][mv["primary"]]["residual_mm"]
        if verbose:
            p = mv["per_view"][mv["primary"]]
            print("rigid: scale {0:.4f}, residual median {1:.2f} rms {2:.2f} max {3:.2f} mm over {4} landmarks".format(
                s, p["median_used_mm"], p["rms_used_mm"], p["max_used_mm"], p["n_used"]))

        # Neutral-face check from geometry (never from blendshape scores), with
        # where each camera sat as seen from the face, from the rigid fit.
        face_centre = mv["landmarks"][mv["valid"]].mean(axis=0)
        cam_dirs = {k: v["camera_centre"] - face_centre for k, v in mv["per_view"].items()}
        M["neutral"] = neutral_check.check(views, cam_dirs, primary_name=mv["primary"])
        if verbose:
            for k, r in M["neutral"]["views"].items():
                print("  {0:6s} lips {1:.1f} mm, eyelids {2:.1f}/{3:.1f} mm, phone {4:+.0f} deg up/down, {5:+.0f} deg side".format(
                    k, r["lip_gap_mm"] or float("nan"), r["eyelid_opening_r_mm"] or float("nan"),
                    r["eyelid_opening_l_mm"] or float("nan"), r.get("camera_elevation_deg", float("nan")),
                    r.get("camera_side_deg", float("nan"))))
            print("neutral check:", "ok" if M["neutral"]["neutral"] else "; ".join(M["neutral"]["warnings"]))

        # 2. uniform scale of the whole template (bodies and joints) about the origin
        scaled = {k: {"out": s * b["out"], "rest": s * b["rest"]} for k, b in bodies.items()}
        sjoints = {k: s * p for k, p in joints.items()}

        # 3. non-rigid skin fit in output space
        region_s, rim_dist, _loops, _an = fx.skin_regions(V, F)
        data_mask = rim_dist > fx.HOLE_ADJ_MM
        lip_region = np.isin(region_s, ["lip_upper", "lip_lower"])
        if base_mode:
            data_mask &= ~lip_region
        # Landmark targets: not the face oval (silhouette depth is biased even
        # when it passes the edge test; the dense cloud defines the outline),
        # not what the rigid fit rejected as an outlier.
        skip = set(mp.FACE_OVAL) | set(mv["per_view"][mv["primary"]]["rejected"])
        if base_mode:
            skip |= {i for i, r in enumerate(fixture["region"]) if r in ("lip_upper", "lip_lower")}
            skip |= {mp.MOUTH_R, mp.MOUTH_L, 78, 308}
        li, vi, w = [], [], []
        for i, (v, wt) in enumerate(zip(fixture["vertex"], fixture["weight"])):
            if v is not None and mv["valid"][i] and fixture["weight_class"][i] != "zero" and i not in skip:
                li.append(i); vi.append(v); w.append(max(wt, 0.1))
        li, vi, w = np.array(li), np.array(vi), np.array(w) * LM_WEIGHT
        X0 = scaled[fx.SKIN]["out"]
        cloud_s = nonrigid.denoise_cloud(mv["cloud"], DENOISE_K)

        # Openings: the eye and mouth holes follow the person's eyelid and
        # inner-lip contours, keeping the template's own edge-behind-margin
        # offset (pipeline/holes.py).
        finite = np.isfinite(mv["landmarks"]).all(axis=1)
        hidx, htgt = holes.constraints(V, F, mv["landmarks"], finite, cloud_s, ref_skin=X0, fixture=fixture,
                                       only=("eye_r", "eye_l") if base_mode else None)
        lm_i = np.r_[vi, hidx]
        lm_p = np.r_[mv["landmarks"][li], htgt] if len(hidx) else mv["landmarks"][li]
        lm_w = np.r_[w, np.full(len(hidx), HOLE_WEIGHT)]
        # Where the face moves and creases (pipeline/importance.py), the skin
        # may take more detail: shape-keeping stiffness / (1 + gain * importance).
        reg_w = np.ones(len(V))
        data_w = None
        imp_file = config.template_export_dir() / importance.IMPORTANCE_NPZ
        if imp_file.exists():
            imp = importance.load(imp_file)
            reg_w = 1.0 / (1.0 + IMPORTANCE_GAIN * imp["importance"])
            if use_anchors:
                data_w = np.ones(len(V))
                data_w[imp["anchors"]] = ANCHOR_DATA_WEIGHT
        elif verbose:
            print("note: no", imp_file.name, "- uniform stiffness (run pipeline/importance.py)")
        # Skin the camera never saw keeps the start shape's smooth form.
        from scipy.spatial import cKDTree
        # Ramped, not a step: a 1x-to-4x step at the scan's edge left a crease down
        # each cheek. Full stiffness only well away from any scan point.
        dist = cKDTree(cloud_s).query(X0)[0] if len(cloud_s) else np.full(len(V), np.inf)
        ramp = np.clip((dist - UNSEEN_MM) / UNSEEN_MM, 0.0, 1.0)
        unseen = ramp > 0.5
        reg_w = reg_w * (1.0 + (UNSEEN_STIFFNESS - 1.0) * ramp)
        # Stiffness alone lets the unseen side of the head tilt freely about the
        # face's edge; a small tilt there swung it up to 45 mm and folded it
        # over the cheek. A weak pull toward the start shape holds it.
        tether = np.flatnonzero(ramp > 0)
        lm_i = np.r_[lm_i, tether]
        lm_p = np.r_[lm_p, X0[tether]]
        lm_w = np.r_[lm_w, UNSEEN_TETHER * ramp[tether]]
        X, nr = nonrigid.fit(X0, F, lm_idx=lm_i, lm_pts=lm_p, lm_w=lm_w,
                             cloud=cloud_s, cloud_n=mv["cloud_normals"], oriented=True,
                             data_mask=data_mask, data_w=data_w, reg_w=reg_w, verbose=False, **NONRIGID)
        v13, v14 = fixture["vertex"][13], fixture["vertex"][14]
        M["openings"] = {"n_constraints": int(len(hidx)),
                         "off_before_mm": holes.alignment_error(X0, hidx, htgt),
                         "off_after_mm": holes.alignment_error(X, hidx, htgt),
                         "edges_forward_mm": holes.forward_shift(X0, X, hidx),
                         "lip_gap_mm": float(np.linalg.norm(X[v13] - X[v14])),
                         "lip_gap_start_mm": float(np.linalg.norm(X0[v13] - X0[v14])),
                         "unseen_skin_frac": float(unseen.mean()),
                         "importance_stiffness": imp_file.exists(), "anchors_weighted": data_w is not None}
        if verbose and len(hidx):
            o = M["openings"]
            print("openings: {0:.2f} mm -> {1:.2f} mm off the eyelids/lips; edges moved {2:+.1f} mm toward the camera; "
                  "lip gap {3:.1f} mm (start {4:.1f})".format(o["off_before_mm"], o["off_after_mm"],
                                                              o["edges_forward_mm"], o["lip_gap_mm"], o["lip_gap_start_mm"]))
        d_skin = X - X0
        lm_res = np.linalg.norm(X[vi] - mv["landmarks"][li], axis=1)
        cd = nonrigid.cloud_to_mesh_distance(mv["cloud"], X, F) if len(mv["cloud"]) else np.zeros(0)
        covered = np.isfinite(cd)
        from scipy.spatial import cKDTree
        near = cKDTree(mv["cloud"]).query(X, distance_upper_bound=COVERAGE_MM)[0] if len(mv["cloud"]) else np.full(len(X), np.inf)
        face_region = data_mask & (X0[:, 2] < np.percentile(X0[:, 2], 60))
        M["nonrigid"] = {"history": nr["history"],
                         "landmark_residual_median_mm": float(np.median(lm_res)),
                         "landmark_residual_p95_mm": float(np.percentile(lm_res, 95)),
                         "cloud_to_skin_median_mm": float(np.median(cd[covered])) if covered.any() else None,
                         "cloud_to_skin_p95_mm": float(np.percentile(cd[covered], 95)) if covered.any() else None,
                         "cloud_points_off_skin_frac": float(1 - covered.mean()) if len(cd) else None,
                         "skin_coverage_frac": float(np.isfinite(near[face_region]).mean()),
                         "skin_disp_max_mm": float(np.linalg.norm(d_skin, axis=1).max())}
        if verbose:
            n = M["nonrigid"]
            print("non-rigid: landmarks median {0:.2f} p95 {1:.2f} mm; cloud->skin median {2} p95 {3}; coverage {4:.0%}".format(
                n["landmark_residual_median_mm"], n["landmark_residual_p95_mm"],
                None if n["cloud_to_skin_median_mm"] is None else round(n["cloud_to_skin_median_mm"], 2),
                None if n["cloud_to_skin_p95_mm"] is None else round(n["cloud_to_skin_p95_mm"], 2),
                n["skin_coverage_frac"]))

        # 4a. bones from skin depth: skull and jaw placed so the template's
        # skin-to-bone depth (at subject scale) holds under the fitted skin,
        # judged only where the camera saw the face. Each sampled skin vertex
        # belongs to whichever bone its inward ray hits first.
        rng = np.random.default_rng(0)
        seen = np.flatnonzero(face_region & np.isfinite(near))
        sample = np.sort(rng.choice(seen, min(BONE_RAYS, len(seen)), replace=False)) if len(seen) else seen
        inward = volumetric.inward_normals(X0, F, scaled[SKULL]["rest"].mean(axis=0))
        depth = {}
        for bone in BONE_GROUPS:
            bv, bf = scaled[bone]["rest"], bodies[bone]["faces"]
            depth[bone] = geo_ray(X0[sample], inward[sample], bv, bf, max_dist=40.0)
        stack = np.stack([depth[b] for b in BONE_GROUPS])
        owner = np.where(np.isfinite(stack).any(axis=0), np.argmin(stack, axis=0), -1)
        sims, bone_info, pairs = {}, {}, []
        for gi, bone in enumerate(BONE_GROUPS):
            sim, info = volumetric.bone_from_skin(X0, X, F, sample[owner == gi], scaled[bone]["rest"],
                                                  bodies[bone]["faces"])
            if "_pairs" in info:
                pairs.append(info.pop("_pairs"))
            bone_info[bone] = info
        # A neutral capture has a closed mouth, so skull, jaw and both teeth
        # move as one: one similarity from all their depth targets together.
        # (Separate fits turned the jaw 9-10 deg on the first capture, opening
        # the bite.)
        if pairs:
            sim, joint = volumetric.similarity_from_pairs(np.concatenate([p[0] for p in pairs]),
                                                          np.concatenate([p[1] for p in pairs]))
            for bone in BONE_GROUPS:
                for member in BONE_GROUPS[bone]:
                    sims[member] = sim
            bone_info["skull_jaw_teeth"] = joint
        # Teeth stay behind the lips: if a front tooth comes closer to the skin
        # (toward the camera) than half the start head's own clearance, jaw and
        # teeth move back by the shortfall.
        teeth = ["middleres_TeethUp", "middleres_TeethDown"]
        clear0, clear1 = [], []
        for tkey in teeth:
            tv0 = scaled[tkey]["rest"]
            front = tv0[:, 2] < np.percentile(tv0[:, 2], 5)
            c0 = volumetric.clearance_along_view(tv0[front], X0, F)
            c1 = volumetric.clearance_along_view(apply_similarity(tv0[front], *sims[tkey]), X, F)
            clear0.append(c0[np.isfinite(c0)].min() if np.isfinite(c0).any() else np.inf)
            clear1.append(c1[np.isfinite(c1)].min() if np.isfinite(c1).any() else np.inf)
        need = 0.5 * min(clear0) - min(clear1)
        moved_back = float(need) if np.isfinite(need) and need > 0 else 0.0
        if moved_back:
            s_, R_, t_ = sims["middleres_TeethUp"]
            moved = (s_, R_, t_ + np.array([0.0, 0.0, moved_back]))
            for member in ("middleres_TeethUp", "middleres_TeethDown", JAW):
                sims[member] = moved
        bone_info["teeth_clearance"] = {"start_min_mm": float(min(clear0)), "before_guard_mm": float(min(clear1)),
                                        "fitted_min_mm": float(min(clear1) + moved_back),
                                        "moved_back_mm": moved_back}
        M["bones_from_skin"] = bone_info
        if verbose:
            j = bone_info.get("skull_jaw_teeth", {})
            tc = bone_info["teeth_clearance"]
            print("skull, jaw and teeth: extra scale {0:.3f}, turn {1:.1f} deg, depth residual {2:.2f} mm; "
                  "front teeth {3:.1f} mm behind the lips (start {4:.1f}), moved back {5:.1f} mm".format(
                      j.get("extra_scale", float("nan")), j.get("rot_deg", float("nan")),
                      j.get("target_residual_median_mm", float("nan")), tc["fitted_min_mm"], tc["start_min_mm"],
                      tc["moved_back_mm"]))

        # 4b. one field for everything else, pinned at the skin AND at the
        # moved bone surfaces near the face, so muscle and fat fill the space
        # between the two layers instead of being stretched through the bone.
        sk = np.sort(rng.choice(len(X0), min(volumetric.N_CENTRES, len(X0)), replace=False))
        pts, disp = [X0[sk]], [d_skin[sk]]
        lo, hi = X0[sample].min(axis=0) - 40.0, X0[sample].max(axis=0) + 40.0
        for bone in BONE_GROUPS:
            if bone not in sims:
                continue
            bv = scaled[bone]["rest"]
            cand = np.flatnonzero(((bv > lo) & (bv < hi)).all(axis=1))
            pick = np.sort(rng.choice(cand, min(BONE_CENTRES, len(cand)), replace=False)) if len(cand) else cand
            pts.append(bv[pick])
            disp.append(apply_similarity(bv[pick], *sims[bone]) - bv[pick])
        pts, disp = np.concatenate(pts), np.concatenate(disp)
        field = volumetric.build_field(pts, disp, n_centres=len(pts))

        # Eyeballs: rigid, anatomical size, seated behind the fitted eye openings.
        eye_ring = {}
        for k in bodies:
            if k.startswith("middleres_Eyes_"):
                c = scaled[k]["rest"].mean(axis=0)
                ring = min(("eye_r", "eye_l"), key=lambda r: np.linalg.norm(X0[_loops["loops"][r]].mean(axis=0) - c))
                eye_ring[k] = _loops["loops"][ring]

        new_rest, per_body = {}, {}
        for k, b in scaled.items():
            if k in eye_ring:
                ring = eye_ring[k]
                new_rest[k], per_body[k] = volumetric.place_eyeball(b["rest"], X0[ring], X[ring])
                new_rest[k], per_body[k]["pushed_back_mm"] = volumetric.keep_behind(new_rest[k], X, F)
                per_body[k].update(volumetric.flip_report(b["rest"], new_rest[k], bodies[k]["faces"]))
                continue
            if k == fx.SKIN:
                # The rig adds the template's own output-minus-rest offset (the
                # forehead "pull_back", up to 16.6 mm) as a fixed amount that
                # does not scale with the head (round trip: 0.83 mm at 1.05x and
                # 4.16 mm at 0.75x on the strip = 5% and 25% of 16.6 mm). So
                # rest = wanted output - that offset, and Maya shows X.
                new_rest[k] = X - (bodies[k]["out"] - bodies[k]["rest"])
                per_body[k] = {"disp_max_mm": float(np.linalg.norm(d_skin, axis=1).max())}
                continue
            if k in sims:
                new_rest[k] = apply_similarity(b["rest"], *sims[k])
                per_body[k] = {"from": "skin depth", "bone_scale": bone_info.get(k, {}).get("extra_scale")}
            else:
                new_rest[k], info = volumetric.warp_body(bodies[k]["role"], b["rest"], field)
                if "similarity" in info:
                    sims[k] = info.pop("similarity")
                per_body[k] = info
            per_body[k].update(volumetric.flip_report(b["rest"], new_rest[k], bodies[k]["faces"]))
        per_body[fx.SKIN].update(volumetric.flip_report(X0, X, F))
        jnames = sorted(sjoints)
        jpos = np.array([sjoints[n] for n in jnames])
        new_joints = dict(zip(jnames, jpos + volumetric.eval_field(field, jpos)))
        M["bodies"] = per_body
        M["joints_disp_max_mm"] = float(max(np.linalg.norm(new_joints[n] - sjoints[n]) for n in jnames))
        M["thickness"] = volumetric.thickness_check(X0, X, F, scaled[SKULL]["rest"], bodies[SKULL]["faces"],
                                                    sims[SKULL])
        flips = {k: v["n_flipped_or_collapsed"] for k, v in per_body.items() if v.get("n_flipped_or_collapsed")}
        M["flipped_faces"] = flips
        if verbose:
            t = M["thickness"]
            print("thickness skin->skull: template {0:.1f} mm, fitted {1:.1f} mm (ratio median {2:.2f}, "
                  "{3} negative, {4} below 30%)".format(t["template_mm_median"], t["new_mm_median"],
                                                         t["ratio_median"], t["n_negative"], t["n_below_30pct"]))
            print("flipped/collapsed faces:", flips or "none")

        # 5. write
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, r in new_rest.items():
            np.savez(out_dir / (k + ".npz"), rest_vertices=r, faces=bodies[k]["faces"].astype(np.int32),
                     role=bodies[k]["role"], units="mm", subject=subject_id)
        np.savez(out_dir / "joints.npz", names=np.array(jnames), positions=np.array([new_joints[n] for n in jnames]),
                 units="mm")
        np.savez(out_dir / "skin_fit_output.npz", vertices=X, faces=F, cloud=mv["cloud"])
        report = {"capture": str(capture_dir), "subject": subject_id, "metrics": M,
                  "limitations": [
                      "Tissue depth is relative to the template, not the subject's anatomy (artist-fitted skin).",
                      "Regions the phone did not see are extrapolated by the regulariser and the RBF field.",
                      "Joint animation is offset, not re-targeted: expressions move about the warped pivots."]}
        cio.write_json_atomic(out_dir / "report.json", report)
        M["out_dir"] = str(out_dir)
        if verbose:
            print("wrote", out_dir)
        return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--id", default=None)
    ap.add_argument("--relabel", action="store_true", help="swap left/right indices if the gate says LABELS_SWAPPED")
    ap.add_argument("--force", action="store_true", help="proceed past a failing mirror gate (not for real fits)")
    ap.add_argument("--anchors", action="store_true", help="extra scan weight on the Poisson-disk anchors (measured no gain)")
    ap.add_argument("--base", default=None,
                    help="start head: a subject under template_export/subjects (default: {0} if it exists, "
                         "else the lab head)".format(BASE_SUBJECT))
    ap.add_argument("--lab-head", action="store_true", help="start from the lab head even if a base head exists")
    args = ap.parse_args()
    base = args.base
    if base is None and not args.lab_head and (config.template_export_dir() / "subjects" / BASE_SUBJECT).exists():
        base = BASE_SUBJECT
    fit(args.capture, args.id or Path(args.capture).name, do_relabel=args.relabel, force=args.force,
        use_anchors=args.anchors, base=base)


if __name__ == "__main__":
    main()

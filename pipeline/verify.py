"""Verification layer, tier 1: is a fitted head physically valid before Maya?

The fit and the step-by-step morph (pipeline/morph_steps.py) both end in new
rest shapes for the 52 Ziva bodies. Ziva rebuilds its tets from those shapes,
runs with collisions off (collisionDetection False on the solver, self-
collision off on every tissue; template_export/attachment_maps.json), and
re-finds every attachment's closest points from them. So anything wrong in a
rest shape goes straight into the simulation, unseen:

  mesh          same vertex count and faces as the lab head, no NaN
  elements      inverted or collapsed triangles per body
  volume        each closed tissue's volume against the lab head's x scale^3
                (muscle and fat are nearly incompressible)
  penetration   tissue and skin vertices inside skull, jaw or teeth, against
                the lab head's own count (the artist's meshes overlap a little)
  attachments   each attached vertex's distance to its target body, against
                the lab head's x scale (from the attachment vertex maps)
  mouth         lip muscle halves apart; skin mouth edges apart
  teeth, eyes   front teeth behind the lips; eyeballs behind the skin

Each check returns pass / warn / fail with its numbers; verify_rest() gathers
them into one verdict (fail if any check fails). Tier 2 (rig/sim_check.py,
rig/ui_smoke_test.py) adds the simulated checks in Maya.

Limits come from the lab head itself (passes everything by construction),
the fits of both s01 captures (pass), and capture_01 as fitted on 3 Nov,
whose mouth animated sealed (must fail the mouth check).
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import geometry as geo
from pipeline import fixture as fx
from pipeline import mouth, nonrigid, volumetric

PASS, WARN, FAIL = "pass", "warn", "fail"
LEGACY = ("jack01_", "skin_cloth")               # reported, not gated (CLAUDE.md)
SKIN = fx.SKIN
BONES = ("middleres_skull", "middleres_Jaw", "middleres_TeethUp", "middleres_TeethDown")
# limits
# Limits. Measured on both s01 fits (lab head start): the chin muscle had 23-24%
# of its triangles inverted or collapsed; thin muscles kept 0.47-1.5x their
# volume; up to 5% more of a muscle's vertices sat inside bone than on the lab
# head (whose meshes already overlap); the skin-to-cheek-muscle attachments
# opened by 4-4.5 mm. Those warn. A fail is a gross break.
FLIP_WARN, FLIP_FAIL = 0.02, 0.30                # share of a body's triangles inverted or collapsed
VOLUME_WARN, VOLUME_FAIL = 0.25, 0.60            # |volume / (lab x s^3) - 1|
PENETRATION_WARN, PENETRATION_FAIL = 0.03, 0.10  # share of a body's vertices newly inside a bone
OVERLAPPING_TISSUE = 0.2                         # over 20% inside bone on the lab head: penetration only warns
ATTACH_TOL_MM, ATTACH_TOL_REL = 2.0, 0.5         # a pair is off by more than max(2 mm, 50% of its lab distance)
ATTACH_FAR_MM = 8.0                              # ... and far off beyond 8 mm
ATTACH_WARN, ATTACH_FAIL = 0.10, 0.10            # warn: >10% of an attachment's pairs off; fail: >10% far off
# The eyeballs are resized on purpose (34 -> 24 mm), which opens 17-19 mm
# behind them on their skull attachments: reported, not gated.
RESIZED = ("middleres_Eyes_",)
EDGE_GAP_FAIL = 0.5                              # skin mouth edge gap under half the lab head's


def _check(name, status, **numbers):
    return dict(name=name, status=status, **numbers)


def gated(name):
    return not name.startswith(LEGACY)


def signed_volume(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def inside(points, V, F, sign=None):
    """Approximate inside test for a closed mesh: a point is inside if it lies
    behind the outward normal of its nearest vertex. Exact enough for counting
    against the same test on the lab head."""
    if sign is None:
        sign = nonrigid.outward_sign(V, F)
    vn = geo.vertex_normals(V, F) * sign
    lo, hi = V.min(axis=0), V.max(axis=0)
    box = ((points >= lo) & (points <= hi)).all(axis=1)
    out = np.zeros(len(points), dtype=bool)
    if box.any():
        _d, j = cKDTree(V).query(points[box])
        out[np.flatnonzero(box)] = ((points[box] - V[j]) * vn[j]).sum(axis=1) < -1e-6
    return out


def load_attachment_maps(path=None):
    p = Path(path) if path else config.template_export_dir() / "attachment_maps.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh).get("attachments")


def attachment_pairs(lab_rest, maps):
    """Each attachment's source vertices and their nearest target vertex on
    the lab head's rest (Ziva binds a source vertex to the closest point on
    the target; the target maps are unpainted). -> {name: (src, tgt, d0)}."""
    out, trees = {}, {}
    for name, a in (maps or {}).items():
        s, t, sv = a.get("source"), a.get("target"), a.get("source_verts")
        if not sv or s not in lab_rest or t not in lab_rest:
            continue
        if t not in trees:
            trees[t] = cKDTree(lab_rest[t])
        src = np.asarray(sv, dtype=np.int64)
        d0, tgt = trees[t].query(lab_rest[s][src])
        out[name] = (s, t, src, tgt, d0)
    return out


def check_mesh(rest, lab):
    bad = [k for k, v in rest.items() if k in lab and (v.shape != lab[k]["rest"].shape or not np.isfinite(v).all())]
    missing = [k for k in lab if k not in rest]
    return _check("mesh", FAIL if bad or missing else PASS, bodies=len(rest), wrong_or_nan=bad, missing=missing)


def check_elements(rest, lab, scale):
    per, worst = {}, (None, 0.0)
    for k, v in rest.items():
        if not gated(k) or k not in lab:
            continue
        f = geo.flipped_faces(scale * lab[k]["rest"], v, lab[k]["faces"], min_area_ratio=0.05)
        frac = float(f.mean())
        if f.any():
            per[k] = {"n": int(f.sum()), "frac": round(frac, 4)}
        if frac > worst[1]:
            worst = (k, frac)
    status = FAIL if worst[1] > FLIP_FAIL else WARN if worst[1] > FLIP_WARN else PASS
    return _check("elements", status, worst_body=worst[0], worst_frac=round(worst[1], 4), per_body=per,
                  limits={"warn": FLIP_WARN, "fail": FLIP_FAIL})


def check_volume(rest, lab, manifest, scale):
    per, worst = {}, (None, 0.0)
    for k, v in rest.items():
        info = manifest["bodies"].get(k, {})
        if not gated(k) or info.get("role") != "tissue" or not info.get("closed") or k.startswith("middleres_Eyes_"):
            continue
        v0 = abs(signed_volume(lab[k]["rest"], lab[k]["faces"])) * scale ** 3
        r = abs(signed_volume(v, lab[k]["faces"])) / max(v0, 1e-9)
        per[k] = round(r, 3)
        if abs(r - 1) > worst[1]:
            worst = (k, abs(r - 1))
    status = FAIL if worst[1] > VOLUME_FAIL else WARN if worst[1] > VOLUME_WARN else PASS
    return _check("volume", status, worst_body=worst[0], worst_change=round(worst[1], 3), ratio=per,
                  limits={"warn": VOLUME_WARN, "fail": VOLUME_FAIL})


def check_penetration(rest, lab, scale):
    """Vertices of every gated soft body (tissue, skin) inside a bone, new
    against the lab head (scaled) with the same test."""
    bones = [b for b in BONES if b in rest]
    signs = {b: nonrigid.outward_sign(lab[b]["rest"], lab[b]["faces"]) for b in bones}
    per, worst = {}, (None, 0.0, 0.0)
    for k, v in rest.items():
        if not gated(k) or k in BONES or lab[k]["role"] == "bone" or k.startswith("middleres_Eyes_"):
            continue
        new = base = 0
        for b in bones:
            new += int(inside(v, rest[b], lab[b]["faces"], signs[b]).sum())
            base += int(inside(scale * lab[k]["rest"], scale * lab[b]["rest"], lab[b]["faces"], signs[b]).sum())
        extra = max(new - base, 0) / len(v)
        if new or base:
            per[k] = {"inside": new, "lab_inside": base}
        # tissue the artist built overlapping bone (the chin muscle: 46% inside
        # the jaw on the lab head, and the rig runs) can only warn
        level = extra if base / len(v) <= OVERLAPPING_TISSUE else min(extra, PENETRATION_FAIL)
        if level > worst[1] or (level == worst[1] and extra > worst[2]):
            worst = (k, level, extra)
    status = FAIL if worst[1] > PENETRATION_FAIL else WARN if worst[1] > PENETRATION_WARN else PASS
    return _check("penetration", status, worst_body=worst[0], worst_extra_frac=round(worst[2], 4), per_body=per,
                  limits={"warn": PENETRATION_WARN, "fail": PENETRATION_FAIL,
                          "overlapping_tissue_warn_only_above": OVERLAPPING_TISSUE})


def check_attachments(rest, lab, scale, pairs):
    if not pairs:
        return _check("attachments", WARN, note="no attachment maps (run: mayapy rig/probe_scene.py --attachments)")
    per, resized, worst, worst_far = {}, {}, (None, 0.0), (None, 0.0)
    n = 0
    for name, (s, t, src, tgt, d0) in pairs.items():
        if s not in rest or t not in rest or not (gated(s) and gated(t)):
            continue
        d1 = np.linalg.norm(rest[s][src] - rest[t][tgt], axis=1)
        dev = np.abs(d1 - scale * d0)
        off = dev > np.maximum(ATTACH_TOL_MM, ATTACH_TOL_REL * scale * d0)
        far = dev > ATTACH_FAR_MM
        entry = {"source": s, "target": t, "off": int(off.sum()), "far": int(far.sum()), "of": int(len(off)),
                 "worst_mm": round(float(dev.max()), 2)}
        if s.startswith(RESIZED) or t.startswith(RESIZED):
            resized[name] = entry
            continue
        n += 1
        if off.any():
            per[name] = entry
        if off.mean() > worst[1]:
            worst = (name, float(off.mean()))
        if far.mean() > worst_far[1]:
            worst_far = (name, float(far.mean()))
    status = FAIL if worst_far[1] > ATTACH_FAIL else WARN if worst[1] > ATTACH_WARN else PASS
    return _check("attachments", status, n=n, worst=worst[0], worst_frac=round(worst[1], 3),
                  worst_far=worst_far[0], worst_far_frac=round(worst_far[1], 3), per_attachment=per,
                  resized_eyes=resized, limits={"tol_mm": ATTACH_TOL_MM, "far_mm": ATTACH_FAR_MM,
                                                "warn": ATTACH_WARN, "fail": ATTACH_FAIL})


def check_mouth(skin_out, muscle, F, scale):
    lab_sk, _ = mouth.load_lab_mesh(SKIN)
    parts = mouth.mouth_parts(lab_sk, F)
    ref_sk = scale * lab_sk
    gap, ref_gap = mouth.edge_gap_mm(skin_out, parts), mouth.edge_gap_mm(ref_sk, parts)
    out = {"edge_gap_mm": round(gap, 2), "ref_edge_gap_mm": round(ref_gap, 2)}
    status = FAIL if gap < EDGE_GAP_FAIL * ref_gap else PASS
    if muscle is not None:
        m = mouth.muscle_compare(muscle, skin_out, scale * mouth.load_lab_mesh(mouth.LIP_MUSCLE)[0], ref_sk, parts)
        out.update({k: m[k] for k in ("muscle_gap_median_mm", "ref_muscle_gap_median_mm", "muscle_slices_closing",
                                      "muscle_closing_frac")})
        if m["muscle_closing_frac"] is not None and m["muscle_closing_frac"] > mouth.MUSCLE_CLOSING_MAX:
            status = FAIL
    return _check("mouth", status, **out)


def check_teeth_eyes(rest, skin_out, F):
    out, status = {}, PASS
    teeth = [t for t in ("middleres_TeethUp", "middleres_TeethDown") if t in rest]
    if teeth:
        clr = []
        for t in teeth:
            tv = rest[t]
            front = tv[tv[:, 2] < np.percentile(tv[:, 2], 5)]
            c = volumetric.clearance_along_view(front, skin_out, F)
            clr.append(float(c[np.isfinite(c)].min()) if np.isfinite(c).any() else np.inf)
        out["teeth_clearance_mm"] = round(min(clr), 2)
        if min(clr) <= 0:
            status = FAIL
    for e in (k for k in rest if k.startswith("middleres_Eyes_")):
        _v, push = volumetric.keep_behind(rest[e], skin_out, F, margin_mm=0.0)
        out[e + "_poke_mm"] = round(push, 2)
        if push > 0.5:
            status = FAIL
    return _check("teeth_eyes", status, **out)


def verify_rest(rest, scale, lab=None, manifest=None, pairs=None):
    """All tier-1 checks on a set of rest shapes {body: (V,3) mm} at uniform
    scale `scale` against the lab head. -> {"verdict", "checks": [...]}."""
    from pipeline import fit_subject as fs
    if lab is None or manifest is None:
        lab, _j, manifest = fs.load_template_bodies()
    if pairs is None:
        pairs = attachment_pairs({k: b["rest"] for k, b in lab.items()}, load_attachment_maps())
    F = lab[SKIN]["faces"]
    skin_out = rest[SKIN] + (lab[SKIN]["out"] - lab[SKIN]["rest"])
    checks = [check_mesh(rest, lab)]
    if checks[0]["status"] != FAIL:
        checks += [check_elements(rest, lab, scale), check_volume(rest, lab, manifest, scale),
                   check_penetration(rest, lab, scale), check_attachments(rest, lab, scale, pairs),
                   check_mouth(skin_out, rest.get(mouth.LIP_MUSCLE), F, scale),
                   check_teeth_eyes(rest, skin_out, F)]
    statuses = [c["status"] for c in checks]
    verdict = FAIL if FAIL in statuses else WARN if WARN in statuses else PASS
    return {"verdict": verdict, "scale_vs_lab_head": scale, "checks": checks,
            "failed": [c["name"] for c in checks if c["status"] == FAIL],
            "warned": [c["name"] for c in checks if c["status"] == WARN]}


def load_subject_rest(subject_dir):
    subject_dir = Path(subject_dir)
    rest = {}
    for p in subject_dir.glob("*.npz"):
        if p.stem in ("joints", "skin_fit_output") or p.stem.startswith("anim_"):
            continue
        with np.load(p, allow_pickle=False) as z:
            if "rest_vertices" in z:
                rest[p.stem] = np.asarray(z["rest_vertices"], dtype=np.float64)
    with open(subject_dir / "report.json", encoding="utf-8") as fh:
        m = json.load(fh)["metrics"]
    return rest, float(m.get("scale_vs_lab_head", m.get("scale", 1.0)))


def verify_subject(subject_dir, write=True):
    """verify_rest on template_export/subjects/<id>; writes report.json["verify"]."""
    rest, scale = load_subject_rest(subject_dir)
    rep = verify_rest(rest, scale)
    if write:
        from core import io as cio
        p = Path(subject_dir) / "report.json"
        with open(p, encoding="utf-8") as fh:
            report = json.load(fh)
        report["verify"] = rep
        cio.write_json_atomic(p, report)
    return rep


def summary(rep):
    """One line per check, for printing."""
    lines = ["verify: " + rep["verdict"].upper()]
    for c in rep["checks"]:
        keys = [k for k in c if k not in ("name", "status", "per_body", "per_attachment", "ratio", "limits",
                                          "resized_eyes")]
        lines.append("  {0:12s} {1:4s}  {2}".format(c["name"], c["status"], ", ".join(
            "{0}={1}".format(k, c[k]) for k in keys)))
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Tier-1 verification of a fitted subject (no Maya).")
    ap.add_argument("subject", help="subject id under template_export/subjects, or 'lab' for the lab head")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    if args.subject == "lab":
        from pipeline import fit_subject as fs
        lab, _j, manifest = fs.load_template_bodies()
        rep = verify_rest({k: b["rest"] for k, b in lab.items()}, 1.0, lab, manifest)
    else:
        d = Path(args.subject)
        d = d if d.exists() else config.template_export_dir() / "subjects" / args.subject
        rep = verify_subject(d, write=not args.no_write)
    print(summary(rep))
    return 0 if rep["verdict"] != FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Step-by-step morph from the lab head to a fitted head (the professor's idea).

The lab head's rig works. A fitted head may not, and a full jump hides where
it stops working. So the fit is approached in steps: at fraction t every body
is placed part of the way from the lab head (t = 0) to the fit (t = 1), the
rig is checked at that step (pipeline/verify.py, then Maya: rig/morph_search.py),
and the step grows while the head keeps working -- like a diffusion model
walking from a known sample toward the target.

  skin, muscle, fat, joints   rest + t * (fit - rest), vertex by vertex
  bones, teeth, eyeballs      their similarity, interpolated (scale
                              geometrically, rotation along the shortest arc,
                              centroid linearly), so they stay rigid at every t

Same meshes and vertex order, so rig/write_rest.py writes any step unchanged.
Each step's report carries its scale against the lab head, which the keyboard
UI uses to size the phoneme motion.

    mayapy pipeline/morph_steps.py SUBJECT 0.25 0.5 0.75     # -> template_export/subjects/SUBJECT_t025 ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import io as cio                             # noqa: E402
from core.similarity import similarity_transform       # noqa: E402

RIGID_PREFIXES = ("middleres_Eyes_",)


def step_name(subject, t, part_t=None):
    """<subject>_tNNN; with per-body fractions, <subject>_tNNN_<Body>NNN... so a step that
    holds the lip muscle back never overwrites the plain step of the same t."""
    name = "{0}_t{1:03d}".format(subject, int(round(100 * t)))
    for k in sorted(part_t or {}):
        name += "_{0}{1:03d}".format(k.replace("middleres_", ""), int(round(100 * part_t[k])))
    return name


def parse_part(items):
    """['OrbicularisOris=0.125', ...] -> {'OrbicularisOris': 0.125} (names as given; see _part_for)."""
    out = {}
    for it in items or []:
        name, _, val = it.partition("=")
        if not name or not val:
            raise ValueError("--part wants NAME=FRACTION, got {0!r}".format(it))
        v = float(val)
        if not 0.0 <= v <= 1.0:
            raise ValueError("--part fraction must be within 0..1, got {0}".format(v))
        out[name] = v
    return out


def _part_for(body, part_t):
    """Fraction for `body` if part_t names it (full name, or its name without the middleres_ prefix)."""
    if not part_t:
        return None
    for key in (body, body.replace("middleres_", "")):
        if key in part_t:
            return part_t[key]
    return None


def _axis_angle(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if ang < 1e-12:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return w * ang


def _rotation(v):
    ang = np.linalg.norm(v)
    if ang < 1e-12:
        return np.eye(3)
    k = v / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def rigid_step(A, B, t):
    """A body that moved by one similarity from A to B, placed at fraction t:
    scale s**t, rotation t of the way, centroid t of the way. Exact at t = 0
    and t = 1 for any A, B related by a similarity."""
    s, R, _tr = similarity_transform(A, B)
    c0, c1 = A.mean(axis=0), B.mean(axis=0)
    Rt = _rotation(t * _axis_angle(R))
    return (1 - t) * c0 + t * c1 + (s ** t) * (A - c0) @ Rt.T


def interpolate(start, fit, roles, t, part_t=None):
    """start, fit: {body: (V,3) rest}; roles: {body: role}. -> {body: (V,3)}.
    part_t {body: fraction} puts those bodies at their own fraction instead of t."""
    out = {}
    for k, A in start.items():
        B = fit[k]
        tk = _part_for(k, part_t)
        tk = t if tk is None else tk
        if roles.get(k) == "bone" or k.startswith(RIGID_PREFIXES):
            out[k] = rigid_step(A, B, tk)
        else:
            out[k] = (1 - tk) * A + tk * B
    return out


def load_rest_dir(d):
    rest = {}
    for p in Path(d).glob("*.npz"):
        if p.stem in ("joints", "skin_fit_output") or p.stem.startswith("anim_"):
            continue
        with np.load(p, allow_pickle=False) as z:
            if "rest_vertices" in z.files:
                rest[p.stem] = np.asarray(z["rest_vertices"], dtype=np.float64)
    with np.load(Path(d) / "joints.npz", allow_pickle=False) as z:
        joints = (np.array([str(n) for n in z["names"]]), np.asarray(z["positions"], dtype=np.float64))
    return rest, joints


def write_step(subject, t, out_root=None, verify=True, verbose=True, part_t=None):
    """Write template_export/subjects/<subject>_tNNN for fraction t of the way
    from the lab head to the fitted <subject>. -> (dir, report).
    part_t {body: fraction} holds named bodies at their own fraction (e.g. the lip
    muscle at 0.125 while the rest is at t); off by default."""
    from pipeline import fit_subject as fs
    root = Path(out_root or config.template_export_dir() / "subjects")
    fit_dir = root / subject
    with open(fit_dir / "report.json", encoding="utf-8") as fh:
        fit_rep = json.load(fh)
    fm = fit_rep["metrics"]
    if fm.get("start_head", "lab head") != "lab head":
        raise ValueError("{0} started from {1}; the morph runs from the lab head (refit without --base)".format(
            subject, fm.get("start_head")))
    lab, lab_joints, _manifest = fs.load_template_bodies()
    fit_rest, (jnames, jfit) = load_rest_dir(fit_dir)
    start = {k: lab[k]["rest"] for k in fit_rest}
    roles = {k: lab[k]["role"] for k in fit_rest}
    rest = interpolate(start, fit_rest, roles, t, part_t)
    if part_t:
        unknown = [k for k in part_t if all(_part_for(b, {k: 0}) is None for b in rest)]
        if unknown:
            raise ValueError("--part names no body: {0} (bodies: {1})".format(unknown, ", ".join(sorted(rest))))
    j0 = np.array([lab_joints[n] for n in jnames])
    joints = (1 - t) * j0 + t * jfit
    scale = (1 - t) * 1.0 + t * float(fm.get("scale_vs_lab_head", fm["scale"]))
    out = root / step_name(subject, t, part_t)
    out.mkdir(parents=True, exist_ok=True)
    for k, r in rest.items():
        np.savez(out / (k + ".npz"), rest_vertices=r, faces=lab[k]["faces"].astype(np.int32), role=lab[k]["role"],
                 units="mm", subject=out.name)
    np.savez(out / "joints.npz", names=jnames, positions=joints, units="mm")
    # what the skin should show in Maya at this step (rig/sim_check.py checks
    # it vertex by vertex); rest + the rig's fixed forehead offset
    skin = fs.fx.SKIN
    with np.load(fit_dir / "skin_fit_output.npz", allow_pickle=False) as z:
        cloud = np.asarray(z["cloud"])
    np.savez(out / "skin_fit_output.npz", vertices=rest[skin] + (lab[skin]["out"] - lab[skin]["rest"]),
             faces=lab[skin]["faces"], cloud=cloud)
    report = {"capture": fit_rep.get("capture"), "subject": out.name,
              "metrics": {"start_head": "lab head", "scale": scale, "scale_vs_lab_head": scale,
                          "morph": {"of": subject, "t": float(t),
                                    "part_t": {k: float(v) for k, v in (part_t or {}).items()}}}}
    if verify:
        from pipeline import verify as vf
        report["verify"] = vf.verify_rest(rest, scale)
        if verbose:
            print("t = {0:.2f}: {1}".format(t, vf.summary(report["verify"]).replace("\n", "\n  ")))
    cio.write_json_atomic(out / "report.json", report)
    return out, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject", help="a fitted subject under template_export/subjects (fitted from the lab head)")
    ap.add_argument("t", type=float, nargs="+", help="fractions of the way to the fit, 0..1")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--part", action="append", metavar="BODY=T",
                    help="hold one body at its own fraction, e.g. --part OrbicularisOris=0.125 (repeatable)")
    args = ap.parse_args()
    part_t = parse_part(args.part)
    for t in args.t:
        if not 0.0 <= t <= 1.0:
            raise SystemExit("t must be within 0..1, got {0}".format(t))
        out, _rep = write_step(args.subject, t, verify=not args.no_verify, part_t=part_t)
        print("wrote", out)


if __name__ == "__main__":
    main()

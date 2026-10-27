"""Export every Ziva body at the rest frame, plus the skinning joints.

Per body, template_export/<key>.npz in the bts-proj1 layout (vertices
(1, V, 3) mm, world, Maya's vertex order) plus:
    rest_vertices  (V, 3) mm, world -- the geometry the solver takes as rest
                   (rest_target), which is what write_rest.py writes back
    role, scene, frame0
and template_export/joints.npz (names, positions mm), manifest.json.

Then spot-checks the seven landmarks_lab.json vertices: a pass means this
scene matches the one the old bts-proj1 exports came from.

    mayapy rig/export_rest.py [--scene PATH] [--out DIR] [--update-registry]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import io as cio                             # noqa: E402
from rig import _maya as M                             # noqa: E402

SPOT_TOL_MM = 0.01


def export(out_dir, scene, frame):
    cmds = M.cmds
    out_dir.mkdir(parents=True, exist_ok=True)
    cmds.currentTime(frame)
    cmds.refresh(force=True)
    bodies = {}
    for key, info in sorted(M.ziva_meshes().items()):
        shape = info["shape"]
        rest, _chain, fed_by = M.rest_target(shape, info["node"])
        out_pts = M.get_points_mm(shape)
        rest_pts = M.get_points_mm(rest)
        faces = M.triangles(shape)
        if len(rest_pts) != len(out_pts):
            raise cio.InvariantError("{0}: rest {1} verts, output {2}".format(key, len(rest_pts), len(out_pts)))
        M.save_npz(out_dir / (key + ".npz"), key, info["role"], out_pts, faces, rest_mm=rest_pts,
                   frame=frame, scene=scene)
        nb = M.boundary_edge_count(faces)
        bodies[key] = {"role": info["role"], "ziva_node": info["node"], "shape": shape, "rest_target": rest,
                       "rest_fed_by": fed_by, "n_verts": int(len(out_pts)), "n_faces": int(len(faces)),
                       "closed": nb == 0, "boundary_edges": nb,
                       "rest_vs_output_max_mm": float(np.abs(rest_pts - out_pts).max()),
                       "rest_vs_output_median_mm": float(np.median(np.linalg.norm(rest_pts - out_pts, axis=1)))}
        print("  {0:42s} {1:6s} V={2:6d} rest-output max {3:7.3f} mm".format(
            key, info["role"], len(out_pts), bodies[key]["rest_vs_output_max_mm"]))
    joints = M.skin_joints()
    names = sorted(joints)
    np.savez(out_dir / "joints.npz", names=np.array(names), positions=np.array([joints[n] for n in names]),
             units="mm", frame0=frame)
    return bodies, len(names)


def spot_check(out_dir):
    with open(config.LANDMARKS_LAB_JSON, encoding="utf-8") as fh:
        lm = json.load(fh)
    errs = {}
    for key, d in lm["landmarks"].items():
        path = out_dir / (d["mesh"] + ".npz")
        if not path.exists():
            errs[key] = None
            continue
        with np.load(path, allow_pickle=False) as z:
            v = z["vertices"][0]
        errs[key] = float(np.linalg.norm(v[d["vtx"]] - np.asarray(d["rest_pos_mm"])))
    return errs


def update_registry(bodies, scene_md5):
    """Every exported body into config/template.json. Meshes that were already
    required stay required; the rest are optional, so a machine holding only
    the original bts-proj1 exports still loads."""
    reg = cio.load_registry()
    required = {m["name"] for m in reg["meshes"] if not m.get("optional")}
    reg["npz_source"] = {
        "dir_env": "BTS_LAB_DATA", "default_dir": "template_export, else ../bts-proj1/data/lab",
        "provenance": "rig/export_rest.py from the scene above (md5 {0}); frame {1}; "
                      "vertices = solved output, rest_vertices = Ziva rest input. Spot check against "
                      "landmarks_lab.json <= 0.0004 mm, so the old bts-proj1 exports match.".format(
                          scene_md5, config.REST_FRAME),
        "units": "mm", "unit_scale": config.MAYA_UNIT_MM}
    reg["meshes"] = [{"name": k, "role": v["role"], "n_verts": v["n_verts"], "n_faces": v["n_faces"],
                      "closed": v["closed"], "optional": k not in required} for k, v in sorted(bodies.items())]
    cio.write_json_atomic(config.TEMPLATE_JSON, reg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--frame", type=int, default=config.REST_FRAME)
    ap.add_argument("--update-registry", action="store_true")
    args = ap.parse_args()

    M.boot()
    out_dir = Path(args.out) if args.out else config.template_export_dir()
    with cio.RunDir("template", "export", "rig/export_rest.py") as run:
        scene = M.open_scene(args.scene)
        md5 = cio.md5_file(scene)
        bodies, n_joints = export(out_dir, str(scene), args.frame)
        errs = spot_check(out_dir)
        manifest = {"scene": str(scene), "scene_md5": md5, "frame": args.frame, "units": "mm",
                    "unit_scale": config.MAYA_UNIT_MM, "bodies": bodies, "n_joints": n_joints,
                    "spot_check_mm": errs}
        cio.write_json_atomic(out_dir / "manifest.json", manifest)
        run.metrics.update({"scene_md5": md5, "n_bodies": len(bodies), "n_joints": n_joints,
                            "spot_check_mm": errs,
                            "rest_vs_output_max_mm": {k: v["rest_vs_output_max_mm"] for k, v in bodies.items()}})
        print("spot check (mm):", {k: (None if e is None else round(e, 4)) for k, e in errs.items()})
        bad = {k: e for k, e in errs.items() if e is None or e > SPOT_TOL_MM}
        run.metrics["spot_check_pass"] = not bad
        if args.update_registry:
            update_registry(bodies, md5)
            print("updated", config.TEMPLATE_JSON)
        if bad:
            print("SPOT CHECK FAILED: this scene's rest geometry does not match landmarks_lab.json "
                  "(recorded on bts-proj1's 450 MB copy). Not a vertex-order bug if counts match; "
                  "a different scene version. Investigate before trusting the old .npz.")
            sys.exit(2)


if __name__ == "__main__":
    main()

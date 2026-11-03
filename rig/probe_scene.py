"""Read-only probe of the lab scene: what the export, writer and fit must respect.

Answers, as far as a query can:
  - every Ziva body (tissue / bone / cloth), its counts, closure, deformer
    history and the shape that holds its rest geometry;
  - what each zTet is wired to (does it read a live mesh, so re-tets on change?);
  - what each zAttachment stores (mesh-sized arrays = data baked at creation);
  - what is animated, over which frames, and whether a neutral pose exists;
  - optionally (--sequence) the lip aperture over the whole animation.

Nothing is saved to the scene.

    mayapy rig/probe_scene.py [--scene PATH] [--sequence]
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import io as cio                             # noqa: E402
from rig import _maya as M                             # noqa: E402

LIP_UPPER_INNER, LIP_LOWER_INNER = 4007, 4060          # landmarks_lab.json, OrbicularisOris


def section(report, key, fn):
    try:
        report[key] = fn()
    except Exception as e:                                       # recorded, not hidden
        report[key] = {"error": repr(e), "traceback": traceback.format_exc()}
        print("  !! {0}: {1}".format(key, e))


def node_attrs(node, max_len=8):
    """Scalar and short attribute values; mesh-sized arrays reported by size only."""
    cmds = M.cmds
    out, arrays = {}, {}
    for a in cmds.listAttr(node, read=True) or []:
        if "." in a:
            continue
        plug = "{0}.{1}".format(node, a)
        try:
            if cmds.attributeQuery(a, node=node, multi=True):
                n = cmds.getAttr(plug, size=True)
                if n > max_len:
                    arrays[a] = {"multi_size": n}
                    continue
            v = cmds.getAttr(plug)
        except Exception:
            continue
        if isinstance(v, (list, tuple)):
            flat = np.asarray(v, dtype=object).ravel() if v and not isinstance(v[0], (list, tuple)) else v
            if len(flat) > max_len:
                arrays[a] = {"len": len(flat)}
                continue
        out[a] = v
    return out, arrays


def connections(node):
    pairs = M.cmds.listConnections(node, connections=True, plugs=True) or []
    return [[pairs[i], pairs[i + 1]] for i in range(0, len(pairs), 2)]


def probe_meshes():
    cmds = M.cmds
    out = {}
    for name, info in sorted(M.ziva_meshes().items()):
        shape = info["shape"]
        faces = M.triangles(shape)
        entry = {"role": info["role"], "ziva_node": info["node"], "shape": shape,
                 "n_verts": int(M.mesh_fn(shape).numVertices), "n_faces": int(len(faces)),
                 "boundary_edges": M.boundary_edge_count(faces),
                 "world_matrix_identity": bool(np.allclose(M.world_matrix(shape), np.eye(4)))}
        entry["closed"] = entry["boundary_edges"] == 0
        try:
            rest, chain, fed_by = M.rest_target(shape, info["node"])
            entry["rest_target"] = rest
            entry["rest_fed_by"] = fed_by
            entry["history"] = chain
            out_pts = M.get_points_mm(shape, world=False)
            rest_pts = M.get_points_mm(rest, world=False)
            entry["rest_n_verts"] = int(len(rest_pts))
            if len(rest_pts) == len(out_pts):
                entry["rest_vs_output_max_mm"] = float(np.abs(out_pts - rest_pts).max())
            entry["skin_clusters"] = [h for h, t in chain if t == "skinCluster"]
            entry["blend_shapes"] = [h for h, t in chain if t == "blendShape"]
        except Exception as e:
            entry["rest_target_error"] = repr(e)
        out[name] = entry
        print("  {0:40s} {1:7s} V={2:6d} F={3:6d} {4} rest={5}".format(
            name, info["role"], entry["n_verts"], entry["n_faces"],
            "closed" if entry["closed"] else "OPEN({0})".format(entry["boundary_edges"]),
            M.short(entry.get("rest_target", "?"))))
    render = [M.short(x) for x in cmds.ls("*_RENDER", type="transform") or []]
    return {"bodies": out, "render_meshes_excluded": render}


def probe_tets():
    cmds = M.cmds
    out = {}
    for t in M.ziva_nodes("zTet"):
        attrs, arrays = node_attrs(t)
        conns = connections(t)
        mesh_inputs = [c for c in conns
                       if cmds.nodeType(c[1].split(".")[0]) in ("mesh", "zGeo", "zTissue")]
        out[t] = {"attrs": attrs, "arrays": arrays, "connections": conns,
                  "mesh_or_geo_links": mesh_inputs,
                  "inherited": cmds.nodeType(t, inherited=True)}
    return out


def probe_attachments():
    cmds = M.cmds
    out = {}
    for i, a in enumerate(M.ziva_nodes("zAttachment")):
        attrs, arrays = node_attrs(a)
        entry = {"mode": attrs.get("attachmentMode"), "maintainOffset": attrs.get("maintainOffset"),
                 "stiffness": attrs.get("stiffness"), "arrays": arrays}
        meshes = []
        for c in connections(a):
            other = c[1].split(".")[0]
            if cmds.nodeType(other) == "zGeo":
                for m in cmds.listConnections(other, type="mesh") or []:
                    meshes.append(M.short(m))
        entry["meshes"] = sorted(set(meshes))
        if i < 3:                                    # full detail for a sample
            entry["attrs"] = attrs
            entry["connections"] = connections(a)
        out[a] = entry
    return out


def probe_zquery():
    """Ziva's own view of the scene, if the zQuery command is exposed."""
    cmds = M.cmds
    out = {}
    for t in ("zTissue", "zBone", "zCloth", "zAttachment", "zTet"):
        try:
            out[t] = {"nodes": cmds.zQuery(t=t) or [], "meshes": cmds.zQuery(t=t, m=True) or []}
        except Exception as e:
            out[t] = {"error": repr(e)}
    return out


def probe_animation():
    cmds = M.cmds
    curves = {}
    for c in cmds.ls(type="animCurve") or []:
        times = cmds.keyframe(c, q=True, timeChange=True) or []
        driven = cmds.listConnections(c + ".output", plugs=True, source=False) or []
        curves[c] = {"driven": driven, "n_keys": len(times),
                     "t_range": [min(times), max(times)] if times else None,
                     "first_value": (cmds.keyframe(c, q=True, valueChange=True) or [None])[0]}
    driven_nodes = sorted({p.split(".")[0] for v in curves.values() for p in v["driven"]})
    return {"timeline": [cmds.playbackOptions(q=True, minTime=True), cmds.playbackOptions(q=True, maxTime=True)],
            "n_curves": len(curves), "driven_nodes": driven_nodes,
            "audio": cmds.ls(type="audio") or [],
            "dag_poses": cmds.ls(type="dagPose") or [],
            "joints": len(cmds.ls(type="joint") or []),
            "curves": curves}


def probe_joints():
    """Skin-cluster joints: position, which channels are keyed, and whether the
    current pose is the bind pose (bindPreMatrix = inverse world matrix)."""
    cmds = M.cmds
    out = {}
    for j, pos in M.skin_joints().items():
        keyed = sorted({a for a in ("translateX", "translateY", "translateZ", "rotateX", "rotateY", "rotateZ",
                                    "scaleX", "scaleY", "scaleZ")
                        if cmds.listConnections("{0}.{1}".format(j, a), source=True, destination=False,
                                                type="animCurve")})
        out[j] = {"pos_mm": pos.round(4).tolist(), "keyed": keyed}
    bind_err = {}
    for sc in cmds.ls(type="skinCluster") or []:
        worst = 0.0
        for idx in cmds.getAttr(sc + ".matrix", multiIndices=True) or []:
            src = cmds.listConnections("{0}.matrix[{1}]".format(sc, idx), source=True, destination=False)
            if not src:
                continue
            wm = np.array(cmds.getAttr(src[0] + ".worldMatrix[0]")).reshape(4, 4)
            bpm = np.array(cmds.getAttr("{0}.bindPreMatrix[{1}]".format(sc, idx))).reshape(4, 4)
            worst = max(worst, float(np.abs(bpm @ wm - np.eye(4)).max()))
        bind_err[sc] = worst
    return {"joints": out, "n_joints": len(out),
            "n_translate_keyed": sum(1 for v in out.values() if any(k.startswith("translate") for k in v["keyed"])),
            "bind_pose_error_at_open": bind_err}


def probe_sequence(start, end):
    """Lip aperture (inner mid, mm) on the simulated sequence, frame by frame."""
    lips = "middleres_OrbicularisOris"
    rec = M.simulate(list(range(start, end + 1)), [lips], start=start)[lips]
    gap = np.linalg.norm(rec[:, LIP_UPPER_INNER] - rec[:, LIP_LOWER_INNER], axis=1)
    return {"frames": list(range(start, end + 1)), "lip_aperture_mm": gap.round(4).tolist(),
            "min_frame": int(start + gap.argmin()), "min_mm": float(gap.min())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--sequence", action="store_true", help="simulate the animation for lip aperture")
    args = ap.parse_args()

    M.boot()
    out_dir = config.template_export_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    with cio.RunDir("template", "probe", "rig/probe_scene.py") as run:
        scene = M.open_scene(args.scene)
        report = {"env": M.env_info(), "scene_md5": cio.md5_file(scene),
                  "counts": {t: len(M.ziva_nodes(t)) for t in M.ZIVA_TYPES}}
        print("counts:", {k: v for k, v in report["counts"].items() if v})
        section(report, "meshes", probe_meshes)
        section(report, "tets", probe_tets)
        section(report, "attachments", probe_attachments)
        section(report, "zquery", probe_zquery)
        section(report, "animation", probe_animation)
        section(report, "joints", probe_joints)
        if args.sequence:
            t0, t1 = report["animation"].get("timeline", [0, 106]) if isinstance(report.get("animation"), dict) else (0, 106)
            section(report, "sequence", lambda: probe_sequence(int(t0), int(t1)))

        cio.write_json_atomic(out_dir / "scene_probe.json", report)
        cio.write_json_atomic(run.path / "outputs" / "scene_probe.json", report)
        bodies = report["meshes"].get("bodies", {}) if isinstance(report["meshes"], dict) else {}
        run.metrics.update({
            "scene_md5": report["scene_md5"], "counts": report["counts"],
            "n_bodies": len(bodies),
            "bodies": {k: {kk: v.get(kk) for kk in ("role", "n_verts", "n_faces", "closed",
                                                      "rest_target", "rest_vs_output_max_mm")}
                       for k, v in bodies.items()},
            "attachment_array_attrs": sorted({k for a in (report["attachments"] or {}).values()
                                              if isinstance(a, dict) for k in a.get("arrays", {})})
            if isinstance(report["attachments"], dict) and "error" not in report["attachments"] else None,
        })
        print("wrote", out_dir / "scene_probe.json")


if __name__ == "__main__":
    main()

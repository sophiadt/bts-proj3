"""Write rest geometry (and skinning joints) back into the lab rig.

Input: a directory laid out like export_rest.py's output -- <key>.npz with
`rest_vertices` (V, 3) mm world, optional joints.npz (names, positions mm).
Vertex counts are asserted; vertex order is never touched.

Per body the points go to the solver's rest input (rig/_maya.rest_target).
Rest shapes fed live by a polySeparate are cut loose first (geometry
unchanged, asserted), or the written points would be recomputed away.

Ziva keeps a tissue's or cloth's neutral geometry inside its zGeo, captured
when the body was created; points written to the rest shape alone are
ignored by the solver (measured: Mentalis output unchanged after a 1.7 mm
rest change, through solver/tissue toggles and `ziva -r`). Bones are already
wired `shape.worldMesh -> zGeo.iNeutralMesh`; tissues and cloth get the same
connection from their rest shape, after which the solver takes the new rest
exactly. The identity round trip checks this changes nothing at rest.

`wrap` deformers (TeethUp follows the skull through wrap26) store their
driver's base shape; each base is rewritten to its driver's new rest, so the
wrap sees no false delta.

Joints that moved are re-positioned parent-first: a keyed translate channel is
offset by the same local delta on every key, an unkeyed one is set; then every
skinCluster's bindPreMatrix is reset to the inverse of its joint's new world
matrix at the rest frame, so skinning is the identity at rest again.

The canonical scene is never overwritten; --save writes a new .mb under
template_export/scenes/.

    mayapy rig/write_rest.py DIR --save LABEL [--scene PATH]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import io as cio                             # noqa: E402
from rig import _maya as M                             # noqa: E402

JOINT_MOVE_TOL_MM = 1e-4


def load_rest_dir(src_dir):
    """-> ({key: rest (V,3)}, joints {name: (3,)} or None)."""
    src_dir = Path(src_dir)
    rest = {}
    for f in sorted(src_dir.glob("*.npz")):
        if f.name == "joints.npz":
            continue
        with np.load(f, allow_pickle=False) as z:
            if "rest_vertices" in z.files:
                rest[f.stem] = np.asarray(z["rest_vertices"], dtype=np.float64)
    joints = None
    jf = src_dir / "joints.npz"
    if jf.exists():
        with np.load(jf, allow_pickle=False) as z:
            joints = dict(zip([str(n) for n in z["names"]], np.asarray(z["positions"], dtype=np.float64)))
    return rest, joints


def apply_rest_shapes(rest, frame=None):
    """Write {body key: (V,3) mm world} to each body's rest target.
    -> {key: max readback error mm}."""
    cmds = M.cmds
    frame = config.REST_FRAME if frame is None else frame
    cmds.currentTime(frame)
    bodies = M.ziva_meshes()
    missing = sorted(set(rest) - set(bodies))
    if missing:
        raise cio.InvariantError("bodies not in scene: {0}".format(missing))
    err = {}
    for key, pts in sorted(rest.items()):
        info = bodies[key]
        target, _chain, fed_by = M.rest_target(info["shape"], info["node"])
        if M.mesh_fn(target).numVertices != len(pts):
            raise cio.InvariantError("{0}: rest target has {1} vertices, writing {2}".format(
                key, M.mesh_fn(target).numVertices, len(pts)))
        if fed_by:
            M.bake_rest(target)
        M.set_points_mm(target, pts, world=True)
        back = M.get_points_mm(target)
        err[key] = float(np.abs(back - pts).max())
        if err[key] > 1e-3:
            raise cio.InvariantError("{0}: readback differs by {1:.4g} mm".format(key, err[key]))
        if info["role"] in ("tissue", "cloth"):
            connect_neutral(info["node"], target)
    sync_wrap_bases(rest, bodies)
    return err


def connect_neutral(body_node, rest_shape):
    """rest_shape.worldMesh -> zGeo.iNeutralMesh, as bones are already wired."""
    cmds = M.cmds
    zgeo = (cmds.listConnections(body_node + ".iGeo", source=True, destination=False) or [None])[0]
    if zgeo is None or cmds.nodeType(zgeo) != "zGeo":
        raise cio.InvariantError("{0}: no zGeo upstream".format(body_node))
    plug = zgeo + ".iNeutralMesh"
    src = cmds.listConnections(plug, source=True, destination=False, plugs=True, shapes=True) or []
    want = rest_shape + ".worldMesh[0]"
    if src and cmds.ls(src[0].split(".")[0], long=True)[0] == cmds.ls(rest_shape, long=True)[0]:
        return
    cmds.connectAttr(want, plug, force=True)


def sync_wrap_bases(rest, bodies):
    """For every wrap driven by a body being rewritten, set its base mesh to
    that body's new rest. -> [(wrap, base, driver key)]"""
    cmds = M.cmds
    by_shape = {cmds.ls(v["shape"], long=True)[0]: k for k, v in bodies.items()}
    done = []
    for w in cmds.ls(type="wrap") or []:
        n = cmds.getAttr(w + ".driverPoints", size=True)
        for i in range(n):
            drv = cmds.listConnections("{0}.driverPoints[{1}]".format(w, i), source=True, shapes=True) or []
            base = cmds.listConnections("{0}.basePoints[{1}]".format(w, i), source=True, shapes=True) or []
            if not drv or not base:
                continue
            key = by_shape.get(cmds.ls(drv[0], long=True)[0])
            if key is None or key not in rest:
                continue
            M.set_points_mm(base[0], rest[key], world=True)
            done.append((w, base[0], key))
    return done


def _depth(j):
    return j.count("|")


def move_joints(targets, frame=None):
    """Move joints to world positions (mm), parent-first, keeping animation as
    offsets, then rebind every skinCluster at the rest frame.
    -> {"moved": n, "max_residual_mm": ..., "rebound": [...]}"""
    cmds = M.cmds
    frame = config.REST_FRAME if frame is None else frame
    cmds.currentTime(frame)
    moved, worst = 0, 0.0
    for j in sorted(targets, key=_depth):
        if not cmds.objExists(j):
            raise cio.InvariantError("joint not in scene: {0}".format(j))
        cur = np.array(cmds.xform(j, q=True, worldSpace=True, translation=True)) * config.MAYA_UNIT_MM
        dw = (np.asarray(targets[j]) - cur) / config.MAYA_UNIT_MM
        if np.linalg.norm(dw) * config.MAYA_UNIT_MM < JOINT_MOVE_TOL_MM:
            continue
        parent = cmds.listRelatives(j, parent=True, fullPath=True)
        if parent:
            pm = np.array(cmds.getAttr(parent[0] + ".worldMatrix[0]")).reshape(4, 4)
            dl = np.linalg.solve(pm[:3, :3].T, dw)          # row-vector convention
        else:
            dl = dw
        for axis, d in zip("XYZ", dl):
            attr = "translate" + axis
            if cmds.listConnections("{0}.{1}".format(j, attr), source=True, destination=False, type="animCurve"):
                cmds.keyframe(j, attribute=attr, edit=True, relative=True, valueChange=float(d))
            else:
                cmds.setAttr("{0}.{1}".format(j, attr), cmds.getAttr("{0}.{1}".format(j, attr)) + float(d))
        moved += 1
    cmds.currentTime(frame + 1)
    cmds.currentTime(frame)
    for j, want in targets.items():
        got = np.array(cmds.xform(j, q=True, worldSpace=True, translation=True)) * config.MAYA_UNIT_MM
        worst = max(worst, float(np.linalg.norm(got - want)))
    rebound = rebind_skin_clusters()
    return {"moved": moved, "max_residual_mm": worst, "rebound": rebound}


def rebind_skin_clusters():
    """bindPreMatrix[i] = inverse(world matrix of influence i), at the current time."""
    cmds = M.cmds
    done = []
    for sc in cmds.ls(type="skinCluster") or []:
        for idx in cmds.getAttr(sc + ".matrix", multiIndices=True) or []:
            src = cmds.listConnections("{0}.matrix[{1}]".format(sc, idx), source=True, destination=False)
            if not src:
                continue
            wm = np.array(cmds.getAttr(src[0] + ".worldMatrix[0]")).reshape(4, 4)
            cmds.setAttr("{0}.bindPreMatrix[{1}]".format(sc, idx), *np.linalg.inv(wm).ravel().tolist(),
                         type="matrix")
        done.append(sc)
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="directory of <body>.npz with rest_vertices (+ joints.npz)")
    ap.add_argument("--save", required=True, help="label: writes template_export/scenes/<label>.mb")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--no-joints", action="store_true")
    args = ap.parse_args()

    M.boot()
    with cio.RunDir(Path(args.src).name, "write", "rig/write_rest.py") as run:
        M.open_scene(args.scene)
        rest, joints = load_rest_dir(args.src)
        run.metrics["readback_max_mm"] = apply_rest_shapes(rest)
        if joints and not args.no_joints:
            run.metrics["joints"] = move_joints(joints)
        out = M.save_copy(config.template_export_dir() / "scenes" / (args.save + ".mb"))
        run.metrics["saved"] = str(out)
        print("saved", out)


if __name__ == "__main__":
    main()

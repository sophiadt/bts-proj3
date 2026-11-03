"""Shared Maya/Ziva helpers. Run under mayapy only.

The one place in the project that converts between millimetres and Maya units:
`get_points_mm` multiplies by config.MAYA_UNIT_MM, `set_points_mm` divides.

The scene is opened read-only. Nothing here saves over config.SCENE_PATH;
`save_copy` refuses any path outside the export directory.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import io as cio                             # noqa: E402

ZIVA_MESH_TYPES = ("zTissue", "zBone", "zCloth")
ZIVA_TYPES = ("zSolver", "zSolverTransform", "zMaterial", "zTissue", "zFiber", "zBone",
              "zCloth", "zTet", "zAttachment", "zEmbedder", "zGeo", "zRestShape",
              "zLineOfAction", "zRivetToBone")

cmds = None
om = None
_output = []


class LicenceError(RuntimeError):
    """Ziva could not check out a licence: nothing it computes is real."""


def boot():
    """Start Maya standalone and load Ziva. Returns maya.cmds.

    Also listens to Maya's command output: Ziva reports a missing licence only
    there (zTet compute aborts, outputs silently stay at their inputs), so
    open_scene checks it. The batch licence is single-seat: run one Maya job
    at a time on this machine.
    """
    global cmds, om
    import maya.standalone
    maya.standalone.initialize(name="python")
    from maya import cmds as _cmds
    import maya.api.OpenMaya as _om
    import maya.OpenMaya as om1
    cmds, om = _cmds, _om
    om1.MCommandMessage.addCommandOutputCallback(lambda msg, *_: _output.append(msg))
    if not ziva_loaded():
        raise RuntimeError("Ziva plugin did not load: zMaterial is not a registered node type. "
                           "Is MAYA_MODULE_PATH pointing at the Ziva module?")
    return cmds


def check_licence():
    """Raise LicenceError if Ziva has reported a licence failure so far."""
    bad = [m for m in _output if "license" in m.lower() and ("err" in m.lower() or "not valid" in m.lower())]
    if bad:
        raise LicenceError("Ziva licence unavailable (is another Maya/Ziva job running? the batch "
                           "licence is single-seat): " + bad[0].strip())


def ziva_loaded():
    """True once Ziva's node types are registered (plugin name varies by release)."""
    for name in ("ziva", "ZivaVFX", "zivaVFX"):
        try:
            if cmds.pluginInfo(name, q=True, loaded=True):
                return True
        except Exception:
            pass
    for name in ("ziva", "ZivaVFX", "zivaVFX"):
        try:
            cmds.loadPlugin(name, quiet=True)
        except Exception:
            continue
    return "zMaterial" in (cmds.allNodeTypes() or [])


def open_scene(path=None, md5=None, check_md5=True):
    path = Path(path or config.SCENE_PATH)
    if check_md5 and md5 is not False:
        want = md5 or (config.SCENE_MD5 if path == config.SCENE_PATH else None)
        if want:
            got = cio.md5_file(path)
            if got != want:
                raise cio.InvariantError("{0}: md5 {1}, expected {2}".format(path, got, want))
    cmds.file(str(path), open=True, force=True, ignoreVersion=True)
    # Pull one solved body through the solver so a licence failure surfaces now.
    for t in ("zTissue", "zCloth"):
        nodes = ziva_nodes(t)
        if nodes:
            meshes = cmds.zQuery(nodes[0], t=t, m=True) or []
            if meshes:
                get_points_mm(shape_of(meshes[0]))
                break
    check_licence()
    return path


def save_copy(path):
    """Save the open scene to `path`, which must be under the export directory."""
    path = Path(path).resolve()
    root = config.template_export_dir().resolve()
    if root not in path.parents:
        raise ValueError("refusing to save outside {0}: {1}".format(root, path))
    if path == Path(config.SCENE_PATH).resolve():
        raise ValueError("refusing to overwrite the canonical scene")
    path.parent.mkdir(parents=True, exist_ok=True)
    cmds.file(rename=str(path))
    cmds.file(save=True, type="mayaBinary", force=True)
    return path


# --- meshes -----------------------------------------------------------------

def shape_of(node):
    """The non-intermediate mesh shape for a transform or shape name."""
    if cmds.nodeType(node) == "mesh":
        return node
    shapes = cmds.listRelatives(node, shapes=True, noIntermediate=True, fullPath=True, type="mesh") or []
    if len(shapes) != 1:
        raise ValueError("{0}: expected one mesh shape, found {1}".format(node, shapes))
    return shapes[0]


def transform_of(shape):
    if cmds.nodeType(shape) != "mesh":
        return shape
    return cmds.listRelatives(shape, parent=True, fullPath=True)[0]


def short(name):
    return name.split("|")[-1]


def mesh_fn(node):
    sel = om.MSelectionList()
    sel.add(node)
    return om.MFnMesh(sel.getDagPath(0))


def get_points_mm(node, world=True):
    fn = mesh_fn(node)
    space = om.MSpace.kWorld if world else om.MSpace.kObject
    pts = fn.getPoints(space)
    return np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float64) * config.MAYA_UNIT_MM


def set_points_mm(node, pts_mm, world=False):
    """Write points (mm) to a mesh shape, vertex order unchanged.

    Rest targets are intermediate shapes with no transform of their own, so
    object space is the default; world is only valid for a shape whose
    transform chain is identity, which callers must check.
    """
    pts = np.asarray(pts_mm, dtype=np.float64) / config.MAYA_UNIT_MM
    fn = mesh_fn(node)
    if fn.numVertices != len(pts):
        raise cio.InvariantError("{0}: {1} vertices, writing {2}".format(node, fn.numVertices, len(pts)))
    arr = om.MPointArray([om.MPoint(float(x), float(y), float(z)) for x, y, z in pts])
    fn.setPoints(arr, om.MSpace.kWorld if world else om.MSpace.kObject)
    fn.updateSurface()


def triangles(node):
    _, tri = mesh_fn(node).getTriangles()
    return np.array(tri, dtype=np.int64).reshape(-1, 3)


def boundary_edge_count(faces):
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _u, c = np.unique(e, axis=0, return_counts=True)
    return int((c == 1).sum())


def world_matrix(node):
    return np.array(cmds.xform(transform_of(node), q=True, matrix=True, worldSpace=True)).reshape(4, 4)


# --- Ziva structure ---------------------------------------------------------

def ziva_nodes(t):
    return cmds.ls(type=t) or []


def body_key(xf):
    """File-safe unique name for a mesh transform: shortest unique DAG path,
    with '|' replaced (two tongue meshes share a short name)."""
    return cmds.ls(xf)[0].strip("|").replace("|", "__")


def ziva_meshes():
    """{body key: {"role", "node", "shape", "transform"}} for every Ziva body.

    In Ziva 2.2 zTissue/zBone/zCloth are plain DG nodes, not deformers:
    tissue and cloth meshes come from `zQuery -m`; a bone's mesh is whatever
    feeds its `animatedMesh`. *_RENDER meshes are never included: they are
    blendshape blends that bypass the Ziva physics.
    """
    out = {}

    def add(role, node, mesh):
        shp = shape_of(mesh)
        xf = transform_of(shp)
        if short(xf).endswith("_RENDER"):
            return
        out[body_key(xf)] = {"role": role, "node": node, "shape": cmds.ls(shp, long=True)[0],
                             "transform": cmds.ls(xf, long=True)[0]}

    for role, t in (("tissue", "zTissue"), ("cloth", "zCloth")):
        for node in ziva_nodes(t):
            meshes = cmds.zQuery(node, t=t, m=True) or []
            for m in meshes[:1]:
                add(role, node, m)
    for node in ziva_nodes("zBone"):
        src = cmds.listConnections(node + ".animatedMesh", source=True, destination=False, shapes=True) or []
        for m in src[:1]:
            add("bone", node, m)
    return out


def ziva_rest_input(body_node):
    """The mesh shape feeding a body's own first Ziva deformer, or None.

    zTissue.oGeo -> zTet.iGeo and zCloth.oGeo -> zMaterial.iGeo; that
    deformer's input[0].inputGeometry is the geometry the solver takes as rest.
    Bones have no such deformer (zBone reads the live shape via animatedMesh).
    """
    for dst in cmds.listConnections(body_node + ".oGeo", source=False, destination=True) or []:
        if cmds.nodeType(dst) not in ("zTet", "zMaterial"):
            continue
        src = cmds.listConnections(dst + ".input[0].inputGeometry", source=True, destination=False,
                                   shapes=True) or []
        if src and cmds.nodeType(src[0]) == "mesh":
            return cmds.ls(src[0], long=True)[0]
    return None


def rest_target(mesh, body_node=None):
    """The shape whose points the Ziva solver reads as rest geometry.

    Tissue and cloth: the mesh feeding the body's own zTet / zMaterial
    (`ziva_rest_input`). Otherwise (bones): the most upstream intermediate
    shape under the same transform that feeds this shape's history -- every
    bone here is `...ShapeOrig -> [skinCluster] -> zAttachments -> shape`. A
    mesh with no intermediate shape is its own rest target.

    Returns (shape, chain, fed_by): `fed_by` is the plug driving the rest
    shape's inMesh (e.g. a polySeparate for the fat pieces) -- points written
    there are overwritten on evaluation unless the link is cut first.
    """
    shape = shape_of(mesh)
    xf = transform_of(shape)
    hist = cmds.listHistory(shape, pruneDagObjects=True) or []
    chain = [(h, cmds.nodeType(h)) for h in hist]
    if body_node is not None and cmds.nodeType(body_node) in ("zTissue", "zCloth"):
        z = ziva_rest_input(body_node)
        if z is not None:
            src = cmds.listConnections(z + ".inMesh", source=True, destination=False, plugs=True) or []
            return z, chain, (src[0] if src else None)
    upstream = set(cmds.ls(cmds.listHistory(shape) or [], long=True))
    cands = [s for s in cmds.ls(cmds.listRelatives(xf, shapes=True, fullPath=True, type="mesh") or [], long=True)
             if cmds.getAttr(s + ".intermediateObject") and s in upstream]
    if not cands:
        return shape, chain, None
    roots = []
    for c in cands:
        above = set(cmds.ls(cmds.listHistory(c) or [], long=True)) - {c}
        if not any(o in above for o in cands if o != c):
            roots.append(c)
    if len(roots) != 1:
        raise ValueError("{0}: ambiguous rest target among {1}".format(mesh, roots or cands))
    src = cmds.listConnections(roots[0] + ".inMesh", source=True, destination=False, plugs=True) or []
    return roots[0], chain, (src[0] if src else None)


def bake_rest(rest_shape):
    """Cut an incoming inMesh link so written points stick; geometry unchanged."""
    src = cmds.listConnections(rest_shape + ".inMesh", source=True, destination=False, plugs=True) or []
    if not src:
        return None
    before = get_points_mm(rest_shape, world=False)
    cmds.disconnectAttr(src[0], rest_shape + ".inMesh")
    after = get_points_mm(rest_shape, world=False)
    if len(after) != len(before) or np.abs(after - before).max() > 1e-6:
        raise cio.InvariantError("{0}: geometry changed when cutting {1}".format(rest_shape, src[0]))
    return src[0]


def skin_joints():
    """Every joint that influences a skinCluster on a Ziva body, with its
    world position (mm). -> {joint long name: (3,)}"""
    out = {}
    for sc in cmds.ls(type="skinCluster") or []:
        for j in cmds.skinCluster(sc, q=True, influence=True) or []:
            jl = cmds.ls(j, long=True)[0]
            if jl not in out:
                out[jl] = np.array(cmds.xform(jl, q=True, worldSpace=True, translation=True)) * config.MAYA_UNIT_MM
    return out


def mute_animation(mute=True):
    """Mute every animation curve, so only the solver moves geometry."""
    curves = cmds.ls(type="animCurve") or []
    muted = []
    for c in curves:
        for plug in cmds.listConnections(c + ".output", plugs=True, source=False) or []:
            try:
                cmds.mute(plug, disable=not mute, force=True)
                muted.append(plug)
            except Exception:
                pass
    return muted


def simulate(frames, meshes, start=0):
    """Step from `start` to max(frames), never jumping; capture meshes (mm, world)
    at each frame in `frames`. -> {mesh: (len(frames), V, 3)}."""
    want = sorted(set(frames))
    out = {m: [] for m in meshes}
    cmds.currentTime(start)
    cmds.refresh(force=True)
    for f in range(start, want[-1] + 1):
        cmds.currentTime(f)
        cmds.refresh(force=True)
        if f in want:
            for m in meshes:
                out[m].append(get_points_mm(m))
    return {m: np.stack(v) for m, v in out.items()}


def save_npz(path, name, role, verts_mm, faces, rest_mm=None, frame=0, scene=None):
    """Same layout as bts-proj1/export/export_meshes.py: vertices (frames, V, 3) mm."""
    v = np.asarray(verts_mm, dtype=np.float32)
    if v.ndim == 2:
        v = v[None]
    kw = dict(vertices=v, faces=np.asarray(faces, dtype=np.int32), fps=30, units="mm",
              unit_scale=config.MAYA_UNIT_MM, mesh_name=name, role=role, frame0=frame,
              scene=str(scene or ""))
    if rest_mm is not None:
        kw["rest_vertices"] = np.asarray(rest_mm, dtype=np.float64)
    np.savez(path, **kw)


def env_info():
    return {"maya": cmds.about(version=True), "ziva_loaded": ziva_loaded(),
            "scene": cmds.file(q=True, sceneName=True),
            "MAYA_MODULE_PATH": os.environ.get("MAYA_MODULE_PATH")}

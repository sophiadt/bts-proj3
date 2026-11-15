"""
Make the lab face move from a phone recording.   (Maya Script Editor > Python tab)

No joints, no skinCluster. It moves the vertices of a COPY of the skin directly:
every vertex follows the nearby tracked landmarks (a smooth Gaussian blend), and
in the mouth area a vertex only listens to landmarks on its OWN side of the lip
line, so the upper and lower lip do not drag each other.

Needs three files (it asks for any it can't find):
  registration.json         from scan_to_maya (v4 / v4.1)
  landmarks_on_model.json   from scan_to_maya
  displacements_mm.json     from process_recording.py

Run it, then press Play. Nothing is changed on your original skin (it is only hidden).
To stop / go back, run:   scan_animation_off()
"""
import json
import os

import numpy as np

SKIN_MESH = "skin_cloth_copy_v5_pull_back_RENDER"
FPS = 24.0
GAIN = 1.0            # 1.0 = real size of the movement. Try 1.5 or 2 to exaggerate it
SIGMA = 12.0          # Maya units a landmark's movement spreads (about 7 mm)
FAR = 0.5             # vertices far from every landmark fade to no movement
LIP_SPLIT = True      # upper and lower lip do not drag each other
SPLIT_BAND = 80.0     # Maya units above/below the lip line where the split applies
CORNER_FADE = 20.0    # the split fades out this far past the mouth corners
IRIS_FROM = 468       # landmarks 468+ are the irises: no skin there, ignored
MOUTH_UP, MOUTH_LOW, CORNER_L, CORNER_R = 13, 14, 61, 291


def build_weights(base, lm_xyz, mouth_y, x_lo, x_hi, sigma=SIGMA, far=FAR,
                  split=LIP_SPLIT, band=SPLIT_BAND, fade=CORNER_FADE):
    """(V,3) vertices, (K,3) landmark positions -> (V,K) float32 weights."""
    base = np.asarray(base, float)
    lm = np.asarray(lm_xyz, float)
    d2 = (base ** 2).sum(1)[:, None] + (lm ** 2).sum(1)[None, :] - 2.0 * base @ lm.T
    np.maximum(d2, 0.0, out=d2)
    G = np.exp(-d2 / (2.0 * sigma * sigma))
    if split:
        sv = base[:, 1] - mouth_y                    # + above the lip line, - below
        sl = lm[:, 1] - mouth_y
        same = (sv[:, None] * sl[None, :]) > 0
        same |= (np.abs(sl) < 3.0)[None, :]          # landmarks ON the lip line count for both sides
        out_x = np.maximum(np.maximum(x_lo - base[:, 0], base[:, 0] - x_hi), 0.0)
        blend = np.clip(1.0 - out_x / fade, 0.0, 1.0)            # 1 between the mouth corners
        blend = np.where(np.abs(sv) < band, blend, 0.0)          # only near the mouth
        G = G * (blend[:, None] * same + (1.0 - blend)[:, None])
    W = G / (G.sum(1, keepdims=True) + far)
    return W.astype(np.float32)


def resample(times, disp, ids, fps=FPS):
    """Recording -> one displacement array per Maya frame: (F, K, 3) mm, frames 1..F."""
    T = len(times)
    arr = np.zeros((T, len(ids), 3))
    for ti in range(T):
        row = disp[ti]
        for k, i in enumerate(ids):
            if i < len(row) and row[i] is not None:
                arr[ti, k] = row[i]
    end = 1 + int(round(times[-1] * fps))
    f_t = (np.arange(1, end + 1) - 1) / fps
    out = np.zeros((end, len(ids), 3))
    for k in range(len(ids)):
        for c in range(3):
            out[:, k, c] = np.interp(f_t, times, arr[:, k, c])
    return out


def build_deltas(W, frames_mm, M, gain=GAIN):
    """(V,K) weights, (F,K,3) mm, 3x3 mm->Maya matrix -> (F,V,3) float32 Maya-unit offsets."""
    Dk = np.einsum("rc,fkc->fkr", np.asarray(M, float), frames_mm) * gain      # (F,K,3) Maya units
    F, K, _ = Dk.shape
    flat = Dk.transpose(1, 0, 2).reshape(K, F * 3)
    out = (W @ flat.astype(np.float32)).reshape(W.shape[0], F, 3).transpose(1, 0, 2)
    return np.ascontiguousarray(out, dtype=np.float32)


# ---------------------------------------------------------------- Maya side
_STATE = {}


def scan_tick(*_args):
    import maya.cmds as cmds
    import maya.api.OpenMaya as om
    s = _STATE
    if "fn" not in s:
        return
    f = int(round(cmds.currentTime(q=True)))
    i = min(max(f, 1), s["deltas"].shape[0]) - 1
    pts = s["base"] + s["deltas"][i]
    s["fn"].setPoints(om.MPointArray([om.MPoint(float(a), float(b), float(c)) for a, b, c in pts]),
                      om.MSpace.kWorld)
    s["fn"].updateSurface()


def scan_animation_off():
    import maya.cmds as cmds
    for j in cmds.scriptJob(listJobs=True) or []:
        if "scan_tick" in j:
            try:
                cmds.scriptJob(kill=int(j.split(":")[0]), force=True)
            except Exception:
                pass
    for n in ("scan_driven_face",):
        if cmds.objExists(n):
            cmds.delete(n)
    if cmds.objExists(SKIN_MESH):
        cmds.setAttr(SKIN_MESH + ".visibility", 1)
    _STATE.clear()
    print("Animation off. Original skin is visible again.")


def _find(cmds, name, hints, caption):
    for folder in hints:
        p = os.path.join(folder, name)
        if folder and os.path.exists(p):
            return p
    pick = cmds.fileDialog2(fileMode=1, caption=caption, fileFilter="JSON (*.json)")
    return pick[0] if pick else None


def run():
    import maya.cmds as cmds
    import maya.api.OpenMaya as om

    home = os.path.expanduser("~")
    hints = [os.path.join(home, "Downloads"), os.path.join(home, "Downloads", "SFU Fall 2026 Research"),
             os.path.join(home, "Documents")]
    disp_path = _find(cmds, "displacements_mm.json", hints, "Choose displacements_mm.json")
    if not disp_path:
        print("No displacements file chosen.")
        return
    here = [os.path.dirname(disp_path)] + hints
    reg_path = _find(cmds, "registration.json", here, "Choose registration.json")
    pos_path = _find(cmds, "landmarks_on_model.json", here, "Choose landmarks_on_model.json")
    if not reg_path or not pos_path:
        print("Missing registration files.")
        return
    reg = json.load(open(reg_path))
    pos = {int(k): v for k, v in json.load(open(pos_path)).items()}
    rec = json.load(open(disp_path))
    times, disp = np.asarray(rec["t"], float), rec["displacement"]

    M = (reg.get("stats") or {}).get("motion_matrix")
    if not M:
        s, R = reg["scale"], reg["R"]
        M = [[s * R[r][c] for c in range(3)] for r in range(3)]
        print("note: no stretch matrix in registration.json; using rotation + scale only")

    ids = [i for i in sorted(pos) if i < IRIS_FROM and i < len(disp[0])]
    for need in (MOUTH_UP, MOUTH_LOW, CORNER_L, CORNER_R):
        if need not in pos:
            print("landmark %d is missing from landmarks_on_model.json" % need)
            return
    lm_xyz = np.array([pos[i] for i in ids], float)
    mouth_y = 0.5 * (pos[MOUTH_UP][1] + pos[MOUTH_LOW][1])
    xs = [pos[CORNER_L][0], pos[CORNER_R][0]]
    print("recording: %d frames, %.1f s; %d landmarks; lip line y = %.1f" % (len(times), times[-1], len(ids), mouth_y))

    scan_animation_off()
    dup = cmds.duplicate(SKIN_MESH, name="scan_driven_face", returnRootsOnly=True)[0]
    if cmds.listRelatives(dup, parent=True):
        dup = cmds.parent(dup, world=True)[0]
    cmds.setAttr(SKIN_MESH + ".visibility", 0)

    sel = om.MSelectionList()
    sel.add(dup)
    dag = sel.getDagPath(0)
    dag.extendToShape()
    fn = om.MFnMesh(dag)
    base = np.array([[p.x, p.y, p.z] for p in fn.getPoints(om.MSpace.kWorld)], float)

    print("computing the motion (a few seconds)...")
    W = build_weights(base, lm_xyz, mouth_y, min(xs), max(xs))
    frames_mm = resample(times, disp, ids)
    deltas = build_deltas(W, frames_mm, M)
    print("largest vertex move: %.1f Maya units" % float(np.abs(deltas).max()))

    _STATE.update(fn=fn, base=base.astype(np.float32), deltas=deltas)
    cmds.currentUnit(time="film")
    end = deltas.shape[0]
    cmds.playbackOptions(minTime=1, maxTime=end, animationStartTime=1, animationEndTime=end)
    cmds.scriptJob(event=["timeChanged", scan_tick], protected=False)
    cmds.currentTime(1)
    scan_tick()
    print("Done. Press Play (or drag the time slider). To undo: scan_animation_off()")


if __name__ == "__main__":
    try:
        run()
    except ImportError:
        print("Run this inside Maya.")

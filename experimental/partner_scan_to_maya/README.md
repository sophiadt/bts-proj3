# Partner's scan-to-Maya prototype (kept for reference, superseded)

These three scripts are a first, independent attempt to put a phone capture onto the lab head inside
Maya. They were written by the iOS app's author on the school PC, alongside burst depth capture. They are
kept here unchanged as a record. The main line is `pipeline/` + `rig/`; see "Comparison" below.

The two files the prototype wrote for the first burst capture, plus its Maya screenshots, are in
`captures/s01/s01_20261110T212937Z_burst/extras/`.

## What it does, in order

All three scripts are pasted into Maya's Script Editor (Python tab). They need only the open lab scene, and
none of them saves the `.mb`.

1. **`scan_to_maya_v4_3.py`** registers the front view onto the skin. It runs in pure Python with no
   numpy and takes about a minute.
   - Reads `front_landmarks_3d_mm.json` (it asks for the file). Drops landmarks whose depth is more than
     100 mm from the median.
   - Finds the eye and mouth holes in `skin_cloth_copy_v5_pull_back_RENDER` from its border edges.
   - **Scale** = the model's outer-eye-corner distance ÷ the scan's distance between landmarks 33 and 263.
   - **Rigid fit:** a Horn quaternion similarity on 3 anchors (both outer eye corners and the mouth
     centre), trying both eye pairings and keeping the one whose nose points out of the face. Then ICP
     with the scale held fixed.
   - **Non-rigid step:** 14 feature pairs (eye corners and lids, mouth corners and lips, nose tip, chin).
     A least-squares 3×4 affine map, then a Gaussian RBF warp (σ = 0.45 × eye separation).
   - **Snap:** every landmark goes to the nearest skin *vertex* and becomes a locator `lm_###` in
     `scan_landmarks_GRP`.
   - **Writes** `registration.json` (s, R, t, rms, `stats.warped` / `stats.rigid` per landmark) and
     `landmarks_on_model.json` (snapped vertex positions; 437 unique for 468 landmarks) next to the scan.
2. **`fit_skin_to_scan.py`** morphs a *copy* of the skin (`scan_fitted_face`) toward the scan. It hides
   the original skin.
   - Gaussian weights from each vertex to the landmarks; upper and lower lip landmarks don't drag each
     other.
   - `"proportions"` mode moves landmark vertices to the rigid scan positions. `"gentle"` mode only pushes
     along the normal.
   - `scan_fit_dense(a, b)` also pulls vertices along their normals onto `front_face_crop.ply`, inside the
     MediaPipe face outline, then applies 20 Laplacian smoothing passes.
3. **`animate_direct.py`** plays a phone recording on another copy (`scan_driven_face`). It needs
   `displacements_mm.json` from a `process_recording.py`, which is **not included**.
   - Per-landmark displacements, resampled to 24 fps, are blended onto vertices with Gaussian weights.
   - A `timeChanged` scriptJob calls `setPoints` each frame.
   - No joints, skinCluster or Ziva are involved.

The fourth script, the depth noise tool, now lives at `tools/compare_depth_noise.py`.

## Known problems (checked 2026-11-14)

- **Wrong mesh.** All three target `skin_cloth_copy_v5_pull_back_RENDER`, a blendshape output that bypasses
  the Ziva physics (CLAUDE.md: never use `*_RENDER`). The results are disconnected duplicates, so nothing
  reaches the rig: no bones, muscles, fat, joints or attachments move.
- **Scale 1.68× too large.** The scale is taken from the eyes, and the template's eyes are stylised (about
  1.6× human). The fit gives 1.75 Maya units/mm against the true 1/0.962 = 1.04. `MM_PER_UNIT = 1/1.685` in
  `fit_skin_to_scan.py` hard-codes the same error, so every printed "mm" is too small by that factor, and
  "proportions" mode morphs toward a face about 1.7× life size. Many thresholds are tuned to this scale
  (`MAX_MOVE 25`, `SIGMA 14/20/12`, `DENSE_RADIUS 20`, ...).
- **The warp increases the gap it should close.** Skin-to-landmark gap: 3.64 mm after the rigid fit, 4.31 mm
  after the warp (`registration.json` → `stats`).
- **Front view only.** The left and right views are never used, so depth and scale come from one frame. The
  mirror check (`pipeline/mirror_check.py`) needs the turned views to tell a mirror twin from a correct
  capture.
- **Fallback orientations assume a template facing +Z.** Two of them are reflections. The lab head faces −Z.
- **No vertex-count check,** and the `_RENDER` mesh's vertex order has not been checked against the Ziva
  skin.
- **Session side effects.** `animate_direct.py` sets the scene to 24 fps; a failure can leave a
  `polySelectConstraint` on. Saving the scene afterwards would keep the hidden skin and the extra nodes.

## Comparison with `pipeline/`

| | This prototype | `pipeline/` + `rig/` |
| --- | --- | --- |
| Views used | front only | front, left, right merged (ICP + smooth bend) |
| Left/right safety | none | `mirror_check.py` gate (OK / LABELS_SWAPPED / mirror twin ...) |
| Scale | eye spacing (1.68× off) | fixture landmarks weighted toward nose and chin, not the stylised eyes |
| Skin fit | Gaussian blend + RBF on 14 features, nearest-vertex snap | Laplacian non-rigid fit to the dense scan, openings aligned to eyelid and lip contours |
| Inside the head | nothing moves | skull, jaw and teeth placed from skin depth; muscle and fat by an RBF field; joints moved |
| Into the rig | duplicate `_RENDER` copy | `write_rest.py` rewrites Ziva's rest geometry, rebinds joints; `sim_check.py` verifies |
| Animation | vertex scriptJob on a copy | the lab's keyboard UI plays phonemes through Ziva on the fitted head |

What carries over from the partner's work: **burst depth** in the app (16 frames, per-pixel median),
measured by `tools/compare_depth_noise.py`. On s01 it lowered surface roughness from 1.43 to 1.12 mm and
halved the flying pixels (0.58% → 0.28%). This is a different session, so it is indicative, not a
controlled test.

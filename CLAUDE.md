# Ziva workstation — matching capture data to the lab face

## Status on this machine (2026-10-27) — read first

Measured here; where this disagrees with the brief below, this wins.

| Brief's question | Answer | Evidence |
| --- | --- | --- |
| Canonical scene | The repo copy (md5 `b174b0d5…`, 432 MB) matches the old bts-proj1 exports: `landmarks_lab.json` spot check ≤ 0.0004 mm | `rig/export_rest.py` |
| Rest frame 0 or 15 | Moot for geometry: identical to 0.0003 mm. True-neutral question (lips 3.12 mm apart) still open | `config.py` |
| `zTet` nodes | 41, `cacheTetMesh: False` (tetSize 26.4, maxResolution 150): tets are rebuilt from the mesh, not stored | `rig/probe_scene.py` |
| `maintainOffset` snap-back | **No.** Scaled rest at 1.05 and 0.75: frame-0 p99 ≤ 0.17 mm, kept fraction ≥ 0.987 | `rig/roundtrip_test.py` |
| Export round trip | Pass: identity ≤ 0.011 mm | `rig/roundtrip_test.py` |

Facts the brief did not have:

- **52 Ziva bodies**, not 3: 30 tissues (muscles + 14 `polySurface*` fat pieces), 11 bones (incl. skull 28,414 v), 2 cloths. Three `jack01_*` meshes and the 47k `skin_cloth` are legacy and reported but not gated.
- **Writing rest geometry**: points written to a tissue's `…ShapeOrig` are *ignored* by the solver — Ziva keeps the neutral inside its `zGeo`. The writer connects `Orig.worldMesh → zGeo.iNeutralMesh` (bones were already wired this way), cuts `polySeparate` feeds into Orig shapes, and resyncs `wrap` bases (TeethUp follows the skull via `wrap26`).
- **Joints**: 1,216 skin-cluster joints, 1,203 with keyed translates. The writer offsets the keys and resets `bindPreMatrix`; verified to 1e-13 mm.
- **Skin rest ≠ output** along the top forehead edge ("pull_back", 1,254 vertices, up to 16.6 mm, and 0.3–1 mm on 403 more). The rig adds this offset as a **fixed amount that does not scale** with the head: the round-trip strip deviation was 0.83 mm at 1.05× and 4.16 mm at 0.75×, which is 5% and 25% of 16.6 mm. So the written skin rest = wanted face − template offset, and Maya then shows the fitted face exactly (p99 0.000 mm). The earlier "1 mm skin gap" was this offset, not a fitting error.
- **Bones are placed from skin depth**, not by stretching the face's movement to the back of the head. At the face points the camera saw, the template's skin-to-bone depth (at subject scale) is kept under the fitted skin, and one similarity per bone is fitted; teeth follow their bone. On the synthetic face this took the skull's extra shrink from 12% to 0.9% and skin-through-skull spots from 36 to 1. Muscle and fat follow a field pinned at both the skin and the moved bone surfaces.
- **Topology and UVs never change.** Only vertex positions move, so the existing rig (attachments, weights, joints, textures) carries over. `rig/sim_check.py` proves it per run: rest-mesh polygons and UVs identical to the lab scene in all 52 bodies. Compare the rest mesh, not `getTriangles()` or the visible OrbicularisOris (it has a `polyTriangulate1`), because Maya re-picks quad diagonals when a shape changes.
- **First real capture (`captures/s01/s01_20261030T170049Z_guided`, fitted as `capture_01`): `LABELS_SWAPPED`.** Proof that doesn't depend on the check itself: `rgb.jpg`, turned 90° clockwise, shows "Superdry®" reading normally, so it isn't mirrored; on that photo landmark 33 lands on his *left* eye. Cause: the app's `.leftMirrored` mirrors MediaPipe's input. Fixed in `TrueDepth App/.../CameraManager.swift` (flip in `uprightBuffer`, depth row `1 − lm.x`, overlay mirrored for the selfie view), and **verified on a phone**: the burst capture `s01_20261110T212937Z_burst` (partner's build, 2026-11-10) is `OK` without relabelling. The old capture keeps `labels: swapped` in its `capture.json`, so `fit_subject.py` relabels it on its own. The live preview is a mirror image, so the red-dot test is: right hand on the right eye.
- **Never judge expression from MediaPipe blendshapes.** They scored a neutral face eyeSquint ~0.5 and mouthPucker 0.71 (hooded/monolid eyes; low phone). `pipeline/neutral_check.py` measures lips and jaw in mm from reliable landmarks, never warns on eye opening, and takes the phone angle from the rigid fit. Many subjects are Asian; keep it that way.
- **Capture-side robustness learned on the real capture:** keep only depth inside the face-oval outline (a hoodie and hair pulled the jaw skin up to 55 mm); smooth the scan (`denoise_cloud`, k=32) and end the skin fit at stiffness 20 (folds 2,449 → 745). Silhouette landmarks and the chin can carry wrong depth (chin read 80 mm behind the nose); they're filtered.
- **Openings are aligned, not just masked** (professor's direction, Nov 2026). The scan's closest points stay masked within 3 mm of a hole, as the brief says. But each eye and mouth hole is split at its corners, and its upper and lower edges are pulled onto MediaPipe's eyelid and inner-lip contours (`pipeline/holes.py`), with depth taken from the smoothed scan. On the first capture this took the openings from 4.6 mm to about 0.1 mm off the real eyelids and lips.
- **Importance, after the lab's joint placement** (`pipeline/importance.py`, deformation-weighted adaptive Poisson-disk sampling as in `b98-sampling-joint/d2-possion-sampling.py`). Importance = the lab head's speech motion from frame 14, plus normal curvature (creases), plus the density of the 1,200 UI joints. It sets the skin fit's stiffness, ÷(1 + 4·importance): the lips then match the scan 0.85 mm instead of 0.91. Extra scan weight on the Poisson-disk anchors measured no gain, so it's behind `--anchors`. Rebuild with `pipeline/importance.py` after `rig/ui_smoke_test.py` has saved `template_export/anim_ui_generic.npz`.
- **Start head: the lab head** (2026-11-22). `base_human` (`pipeline/build_base.py`, the lab head fitted to MediaPipe's canonical face in base mode) is now opt-in (`--base`). With the detail pass, both s01 captures matched the scan's features as well from the lab head (folds 212/337 vs 190/344), and the base head's lip muscle came out squeezed shut (half-gap median 0.50 mm vs the lab head's 1.67 at that size), which every fit from it inherited.
- **Collisions are off in the rig**: `zSolver1.collisionDetection False`, every zTissue `selfCollisions False` (`rig/probe_scene.py --attachments` -> `template_export/attachment_maps.json`). Nothing in the simulation keeps the lips apart; only the rest shapes do. That is the "sandman mouth" the professor saw (lips stuck together when they move): a rest whose lip muscle halves touch gets tets across the gap.
- **Lips are kept apart in the warp** (`volumetric.split_lip_warp`, `open_lip_muscle`). Lip tissue gets two fields, one from the skin without the lower lip and one without the upper, joined past the corners; the lip muscle's halves are labelled once on the lab head (`mouth.muscle_upper_mask`, topology only), because splitting by the fitted skin's slit misplaced them. On the base head build this took the half-gap from 0.50 to 1.10 mm; a guard then opens any slice under 0.7x the lab head's (symmetric, max 2 mm). `mouth.muscle_gap` uses the same labels. `ui_smoke_test` also gates the lip muscle while speaking (widest gap >= 0.5x the generic head's).
- **Detail pass** (`pipeline/detail.py`): after the stiff fit, seen skin moves along its normals by 0.8x the smoothed distance to the denoised scan (clamp 4 mm, faded near holes and unseen skin). Per-region scan-to-skin medians (burst, lab start): nose 0.61 -> 0.56, brow 0.98 -> 0.72, cheeks 1.11 -> 0.83, chin 1.23 -> 0.99 mm, folds 163 -> 212. `fit_skin` reports `features` per region; `--no-detail` compares.
- **Tissue guards in the warp**: tissue newly inside skull/jaw/teeth is pushed out (only tissue with under 20% lab-head overlap: the chin muscle has 46% of its vertices inside the jaw on the lab head, and pushing only the new ones tore it); a tissue with more than 5% of its triangles folded takes its warp's best affine map, or a similarity when that affine map mirrors it (the chin muscle: affine det -0.65, i.e. the field turned it inside out; 539 folds -> 0).
- **Verification layer** (`pipeline/verify.py`, researcher's suggestion): tier 1 without Maya -- mesh, elements, volume (closed tissues vs lab x s^3), penetration (vs the lab head's own overlap), attachments (each of the 126 mapped attachments' source vertices to their nearest target vertex on the lab head; pair off if > max(2 mm, 50%), far if > 8 mm), mouth, teeth/eyes. Lab head: PASS. Burst fit: WARN. Guided fit: FAIL (a cheek muscle at 1.74x volume). `capture_01` (3 Nov, sealed): FAIL on the mouth. `fit_subject` saves it in `report.json["verify"]`. Eye-to-skull attachments open 17-19 mm because the eyeballs are resized on purpose: reported, not gated.
- **Step-by-step morph** (professor's direction; `pipeline/morph_steps.py`, `rig/morph_search.py`): step t places soft bodies and joints linearly between the lab head and the fit, bones/teeth/eyeballs by an interpolated similarity. Each step: verify, write_rest, sim_check, ui_smoke_test (`--reuse-generic`), adaptive (0.25, x1.5 after a pass, halfway after a fail, max 6). Keeps `<id>_morph.mb`; deletes the other steps' scenes. Each step writes `skin_fit_output.npz` (sim_check compares the scene with it).
- **Morph result (2026-11-22, both captures):** furthest working step t = 0.125 (mouth 87% / 85% of the lab head's), fail from t = 0.157 (68% / 66%), every step passing verify, write_rest and sim_check. The failure is the upper lip's middle following the jaw down. Swap tests on the burst t = 0.157 step: only the lip muscle back at t = 0.125 fixes it; skin, joints, DLI or Mentalis do not. A rigid 2 mm shift changes nothing (Ziva is translation-invariant here), and so do tetSize × scale and painting attachment target maps to the lab binding (tried, reverted). The lip muscle moves only ~0.05 mm between the steps, so the suspect is its rebuilt tet mesh (tetSize 26.4 vs a ~2.5 mm lip gap). Next test, the lab's call: a finer OrbicularisOris tet mesh. Morph step folders `_tNNN` are machine-local (gitignored); the kept step's renders are in its `views/`.
- **Fit fixes measured on the first capture:**
  - *Openings keep the lab head's wrap.* The hole edges sit 7–9 mm (lips) and 4.5–5 mm (lids) behind the visible margin; targets are contour + that offset. Pulling the edges straight onto the contours had opened the mouth and shredded the lids.
  - *Overfitting noise* looked crumpled: stiffness 200, scan pull 0.3 (matches ~0.85 mm, the depth's noise level).
  - *Eyeballs* are rigid, 24 mm, moved with the eye-ring fit, then pushed behind the skin (`keep_behind`). The lab head's are 34 mm even at subject scale.
  - *Skull, jaw and both teeth share one similarity* (separate fits turned the jaw 9–10°). A guard keeps the front teeth at least half the start head's clearance behind the lips (signed: a tooth through the lip counts negative).
  - *No landmark fill from turned views.* Most fills came from one view, unchecked; the set changed with the views taken, and swung the unseen side of the head up to 45 mm into a crease down the cheek. Turned views give dense depth only, after a smooth bend onto the front where they overlap (1.75 → 1.3 mm).
  - *Unseen skin is tethered* (weight 0.2) to the start shape: without it the back of the head moved 14 mm with the choice of views.
  - *Tried and rejected:* two-step fit (thin-plate warp from landmarks, then scan: ~5,000 folds; landmark-only surface step: 3,500+), a grazing-angle filter on turned views (folds 498 → 686), point-to-plane matching (kept; no measurable change).
- **The mouth keeps the rig's resting gap** (2026-11-17). The lab rig rests with its lips apart: mouth-hole edges 8.3 mm apart at the middle, ~272 mm² of the hole visible from the front, the lip muscle's (OrbicularisOris) halves ~2.5 mm apart; its phoneme motion starts from there. Captures have closed lips, so pulling the hole onto MediaPipe's inner-lip contours (which then coincide) shut it: edges 1.2–1.4 mm apart, the lip muscle's halves squeezed together (Ziva rebuilds tets from that shape, self-collision off), and `capture_01` animated with a sealed mouth that `sim_check` and `ui_smoke_test` both passed. Now `holes.mouth_gap_constraints` lays the lab head's gap (× `scale_vs_lab_head`) on the person's lip line, and the scan's lip-contact strip (±2.5 mm) is dropped. Skin-only preview on both captures: opening when speaking 39–49% → 82–94% of the lab head's, lip muscle no longer closing; at rest the lips show ~2 mm apart, as on the lab head. `--close-mouth` gives the old pull, for comparison only. `pipeline/mouth.py` measures it; `ui_smoke_test` and `tools/mouth_check.py` gate it. `fit_subject.fit_skin` is the skin step alone, so `tools/mouth_check.py --preview` runs it off the workstation. `base_human` must be rebuilt (base mode keeps the gap too).
- **Keyboard UI** lives in the lab's repo (`C:\Users\zpa18\a3-stochastic-V1\…\h99-UI-mayascipt-V1_comsciV1.py`); don't edit it. `rig/keyboard_ui.py` runs it on a fitted head from an in-memory copy:
  - phoneme displacements scaled by the fit's scale against the lab head (`scale_vs_lab_head`, not the scale against `base_human`; jaw rotation [900:903] not scaled);
  - feedback control's neutral swapped for the fitted frame-14 joints;
  - reset reopens the fitted `_ui.mb`;
  - its writes go to `template_export/ui_results`.

  `rig/write_rest.py --ui-ready` writes `<id>_ui.mb` with the 1,201 UI joints' keys after frame 14 removed.
- **The iPhone app** is in `TrueDepth App/` (one folder: `.xcodeproj` + source). Commit messages: one short line, no Claude co-author.
- **Captures** live in `captures/<subject>/<capture_id>/`; the id is `<subject>_<UTC YYYYMMDD>T<HHMMSS>Z_<mode>`, the subject an anonymous code (`s01`; never a name), the mode `guided`, `burst`, `photo` or `record`. The app names its session folder and zip that way and writes `capture.json`; `tools/ingest_capture.py` (standard library only) files AirDropped zips from `captures/_inbox/` or `--from ~/Downloads`. Pipeline scripts take a capture id or a folder. Steps for the person AirDropping are in `captures/README.md`.
- **Guided capture is burst capture:** 16 depth frames per pose, per-pixel median (holes = 0). On s01 it lowered depth roughness 1.43 → 1.12 mm and halved flying pixels (`tools/compare_depth_noise.py`).
- **The partner's Maya prototype** (`experimental/partner_scan_to_maya/`) is superseded: front view only, `_RENDER` skin copies, scale from the stylised eyes (1.68× too large). Don't build on it.
- **Timeline** is 0–50, not 0–106.
- `landmarks_lab.json`'s `mouth_corner_R` sits at x = −25.8, which is the subject's **left** (+X is the subject's right). The label is viewer-side.
- The template is 1.42–1.45× a real face overall (eyes 1.6–1.7×, nose 1.2×). The fitted uniform scale lands near 0.69.
- **The Ziva batch licence is single-seat.** Run one Maya job at a time; `rig/_maya.open_scene` raises `LicenceError` rather than returning unsimulated output.
- mayapy 2023 has numpy 2.0.2, scipy 1.13.1 and matplotlib, so the whole fit runs here (user decision, 2026-10-27).

### Run order (each step a gate)

```
MP="C:/Program Files/Autodesk/Maya2023/bin/mayapy.exe"
$MP rig/probe_scene.py                 # read-only scene facts -> template_export/scene_probe.json
$MP rig/export_rest.py                 # all bodies + joints -> template_export/ (committed, except scenes/*.mb), spot check
$MP rig/roundtrip_test.py              # identity + scale 1.05/0.75 gate
$MP pipeline/canonical.py              # MediaPipe canonical mesh out of face_landmarker.task (committed)
$MP pipeline/fixture.py                # config/landmark_to_vertex.json (committed, never per subject)
python tools/ingest_capture.py         # AirDropped zips in captures/_inbox/ -> captures/<subject>/<capture_id>/ (any Python)
$MP pipeline/mirror_check.py ID        # STOP: a person reviews the verdict before any app change; then --record
python tools/mouth_check.py --preview ID --compare   # any machine with the lab skin: skin-only fit + mouth check first
$MP pipeline/fit_subject.py ID         # ID = capture id; output name defaults to it (--id overrides)
$MP rig/write_rest.py template_export/subjects/ID --save ID --ui-ready
$MP rig/sim_check.py template_export/scenes/ID.mb --subject template_export/subjects/ID
$MP tools/render_fit.py ID              # pictures + animation GIFs -> template_export/subjects/ID/views
$MP rig/ui_smoke_test.py ID             # keyboard UI plays "not even you" on the fitted head (runs the lab's code); fails a shut mouth
$MP tools/mouth_check.py ID             # mouth opening vs the lab head from the saved UI frames + pictures (no licence needed)
$MP pipeline/importance.py              # once, after the first UI test: skin importance map + Poisson-disk anchors
$MP pipeline/verify.py ID               # tier-1 physical checks (fit_subject runs them too)
$MP rig/morph_search.py ID              # step-by-step morph from the lab head; keeps the furthest working step
$MP rig/probe_scene.py --attachments    # attachment vertex maps + collision settings -> attachment_maps.json
$MP pipeline/build_base.py              # optional start head (--base)
python tools/check_template_export.py   # after a clone: is template_export/ complete
# Maya GUI: exec(open(r"C:\Users\syd3\bts-proj3\rig\keyboard_ui_fitted.py").read())
PYTHONPATH=".pytest-lib;." $MP -m pytest tests   # pytest via pip --target; mayapy untouched; trimesh tests skip here
```

The mirror check separates OK / LABELS_SWAPPED / GEOMETRY_MIRRORED / MIRROR_TWIN / RGB_DEPTH_MISMATCH, and it needs the guided capture's turned views. A single frontal view is INCONCLUSIVE, because a consistent mirror twin passes every residual.

---

*The original brief follows, unchanged.*

Rename this to `CLAUDE.md` in the repo root on the Ziva machine.

This machine runs **Maya 2023 with Ziva** and holds the lab template scene. Its job in this
project is narrow: get the phone's measured face onto the template, and prove the export
round trip. The research pipeline (registration core, volumetric warp, metrics) lives in a
separate repo on a laptop and is **not** built here.

Work in `mayapy` (Maya's bundled Python). Do not try to install `mediapipe`, `open3d`,
`libigl` or `trimesh` into it — they will fight the bundled interpreter. Anything needing
those libraries is written to disk and handed to the laptop side.

---

## What this machine is being asked to do

In order. Stop at a failing gate.

1. **Correspondence fixture.** Map each of the 478 MediaPipe landmark indices to one
   template skin vertex index. Computed **once**, on the neutral template, committed to the
   repo as a JSON fixture, never regenerated per subject.
2. **Rigid alignment.** Given a captured face, find the rotation, translation and uniform
   scale that best places the template on it. Report per-landmark residuals in mm.
3. **Export round trip.** Get the template meshes out, write them back unchanged, confirm
   the Ziva rig still simulates identically. Then repeat with a 5% uniform scale.

Non-rigid fitting and the volumetric warp come later and are written on the laptop. Do not
start them here.

---

## The data that arrives from the phone

Capture folders are AirDropped off the device. Each contains:

| File | Contents |
| --- | --- |
| `landmarks_3d_mm.json` | 478 entries in MediaPipe index order, each `[x, y, z]` in mm or `null` where depth was missing. **Index position is stable** — nulls hold their slot |
| `depth_<W>x<H>_float32_m.bin` | Raw depth, metres, float32, row-major, sensor-native landscape. `NaN` or `0` means no reading |
| `rgb.jpg` | The colour frame |
| `meta.json` | Intrinsics in px at `intrinsics_ref_width` × `intrinsics_ref_height`. **Scale by `depth_width / intrinsics_ref_width` before unprojecting** |
| `landmarks.ply` | Nulls dropped, so **indices shift**. Viewing only — never use for correspondences |
| `face_crop.ply` | Dense cloud, viewing and future surface fitting |

Guided capture writes the same six files three times, prefixed `front_`, `left_`, `right_`.

**Coordinate frame of the phone data:** camera frame — x along the sensor's long side,
y down, z away from the lens. Right-handed but **not** Y-up, and the head lies sideways
because the sensor is landscape. The template is Y-up with the face toward −Z. Rotating
between the two is the first thing any script must do, once, explicitly.

---

## The template — confirmed facts

| Fact | Value | Source |
| --- | --- | --- |
| Scene | `v68 neutral v13.8 after housekeeping.mb`, ~430–450 MB. Four copies exist; **canonical one unconfirmed** | mentor question |
| Unit scale | **1 Maya unit = 0.962 mm** — confirmed, see below. Scene header says `cm` and is wrong | `bts-proj1/config/units.txt` |
| Orientation | Y-up, face toward −Z (nose tip at z ≈ −46 mm) | checked on exported skin |
| Skin | `skin_cloth_copy_v5_pull_back`: 15,905 verts, 31,225 tris. **Open** — 589 boundary edges at eye, mouth and nose holes | export log |
| Jaw | `middleres_Jaw`: 7,426 verts, closed | export log |
| Lip muscle | `middleres_OrbicularisOris`: 10,240 verts | export log |
| Skull | `middleres_skull` — in the scene, **not yet exported** | `bts-proj1/export/passive_dynamics_test.py` |
| Attachments | 140 `zAttachment` nodes, all fixed mode, `maintainOffset` on | `ziva_attachments.json` |

**Never export the `*_RENDER` meshes.** They are blendshape blends that bypass the Ziva
physics.

### The unit scale is confirmed; the model is stylised

Measured from `bts-proj1/config/landmarks_lab.json`:

| Feature | Template | Adult norm | Ratio |
| --- | --- | --- | --- |
| Mouth width, corner to corner | 52.5 mm | ~50 mm | **1.05** |
| Mouth to chin tip, vertical | 63.7 mm | ~45–50 mm | ~1.4 |
| Eye width, corner to corner | 44.1 mm | ~30 mm | 1.47 |

Mouth width landing on the human norm confirms 0.962 mm/unit — a unit error would have
thrown all three off together. Instead the ratios spread from 1.05 to 1.47, so **no global
scale reconciles them**: the model is genuinely stylised. Human-proportioned mouth,
oversized eyes, elongated lower face.

Two consequences that shape the matching:

- A fit to a real face is a **strongly non-uniform** warp — mouth barely moves, eye region
  compresses ~30%, lower face shortens. Expect tet-quality trouble around the orbits later.
- Anything weighting the eye region heavily will drag the whole alignment toward
  accommodating a stylisation. Weight the mouth and nose, which are closer to human.

The template is also **not bilaterally symmetric**: the mouth corners differ by 1.7 mm in z
and 0.9 mm in x. Any symmetry-based sanity check inherits that as its noise floor.

---

## How the matching works

### Stage 0 — resolve the rest frame first

`bts-proj1/config/landmarks_lab.json` records its seven landmark positions at
**`rest_frame: 15`** of the sequence `"not even you"`, not frame 0.

The pipeline doc elsewhere says export frame 0. **These disagree.** Export frame 0 and every
spot-check against those seven vertices fails, in a way that looks exactly like a
vertex-order bug.

Worse, frame 15 of a speech sequence is a chosen quiet frame, not a true neutral — the lip
aperture there is 3.12 mm, so the lips are nearly but not quite closed. If the rig has a
real neutral pose outside that animation, that is what everything should be built on, and
those seven landmark positions should be re-recorded there.

Settle this before writing any exporter code. It is one line of config and it invalidates
the identity test if wrong.

### Stage 1 — the correspondence fixture

For each MediaPipe landmark index, find the template skin vertex that represents the same
anatomical point. Output: `config/landmark_to_vertex.json`, 478 integer indices.

**Use geodesic distance, not Euclidean.** Straight-line nearest-vertex jumps across the
mouth and eye openings — an upper-lip landmark lands on the lower lip, which is physically
close and anatomically opposite. The 589 boundary edges mark exactly where this happens.

**The holes are masks, not anchors.** This is the most important correction to the current
script. Detecting the eye, mouth and nose boundaries is the right thing to do, but their job
is to *exclude* regions from nearest-point search, not to drive alignment. Three reasons:

1. Depth is least reliable precisely there — eyelashes, wet specular eye surface, dark
   nostril and mouth cavities absorb the IR. The 2 of 478 landmarks missing depth are
   almost certainly in those regions.
2. The mesh boundary is where an artist cut the geometry; MediaPipe's eye contour is the
   visible eyelid margin. Different curves on different objects, so matching them bakes in
   an offset no iteration removes.
3. The eye region is the most distorted part of the template (1.47×). Anchoring on it pulls
   everything else toward the stylisation.

Verify the fixture by colouring snapped vertices by region and inspecting the mouth and
eyelids specifically.

### Stage 2 — rigid alignment

Classic Procrustes with uniform scale, solved in closed form via SVD on the matched pairs.
No iteration, no optimiser, roughly ten lines of numpy.

Weighting matters more than the algorithm:

- **High weight:** nose bridge and tip, chin, jawline, temples, outer face width
- **Low or zero weight:** eye contours, lip contours, anything on a hole boundary
- **Never:** landmarks whose capture entry is `null`

Constraints must be spread widely across the face. Central-only constraints leave scale and
out-of-plane rotation badly conditioned — the fit can pivot about the face centre and still
look locally correct.

Commonly cited MediaPipe indices for the useful anchors: 1 and 4 (nose), 6 (nasion),
152 and 199 (chin), 234 and 454 (face edges), 10 (forehead), 468 and 473 (iris centres).
**Verify these against MediaPipe's canonical face mesh before trusting them**, and treat all
left/right labels as unverified until the mirror check below is done.

Output per run: the 7-parameter transform, and per-landmark residuals in mm. Residuals are
the first real number this project produces — log them from the first run.

### Stage 3 — export round trip

`bts-proj1/export/export_meshes.py` runs in `mayapy`, steps frames 0–106, reads world-space
points with `MFnMesh.getPoints` (which preserves Maya's vertex order), multiplies by 0.962
and writes one `.npz` per mesh.

Three changes needed:

1. Export the rest frame only (see Stage 0 — frame 15 or a true neutral, not 0)
2. Add the skull, the fat, and **every** muscle, not just `OrbicularisOris`
3. Write the matching **writer** that puts vertices back

Then: export untouched → write back → confirm the rig simulates identically. Then apply a
5% uniform scale and confirm it settles at the new shape rather than snapping back.

That second test is the one that matters. `maintainOffset` is on for all 140 attachments,
and if Ziva stores each attachment's rest offset at creation, moved vertices may be pulled
back toward the generic face. An identity warp cannot detect this, because nothing moves.

---

## Before anything else: check for `zTet` nodes

Ziva builds its own tetrahedral meshes for tissue. If `zTet` nodes exist in this scene, the
later volumetric work **deforms those existing tets** rather than generating new ones — a
fresh tetrahedralization has different topology and every attachment index refers to the old
one. This likely removes TetGen and fTetWild from the project's stack entirely.

One query, large consequence. Run it early and record the answer.

---

## Hard invariants

### Vertex order is sacred

Ziva stores attachments **by vertex index** (paper §3.7.1: found by closest point at
creation, then saved as indices). Moving vertices keeps them bound. But anything that
reorders, welds or de-duplicates vertices sends all 140 attachments somewhere random, with
no visible error.

- `trimesh.load(..., process=False)` on the laptop side — the default welds
- Don't trust an OBJ round trip until checked; some exporters split vertices at UV seams
- Assert counts after every load and save: skin 15,905, jaw 7,426, OrbicularisOris 10,240
- Spot-check the seven named vertices in `landmarks_lab.json` — **at the rest frame they
  were recorded at**

### Units

- Everything in this project is **millimetres**
- mm → Maya units is **÷ 0.962**, not ÷ 10. If the current snapping script divides by 10 it
  is wrong by ~4% — small enough to hide behind a scale fit, large enough to corrupt every
  number in the report
- Convert once, at the boundary, never scattered through the code

### Left/right may be swapped

The capture app mirrors the RGB before MediaPipe (`.leftMirrored`) but not the depth. If
landmark numbers sit on the opposite side of the real face, aligning to the template needs a
reflection — which inverts every tetrahedron downstream.

**Check before building the fixture:** open a saved capture, look at where landmark 33 lands
in 3D, confirm it is on the subject's actual left or right as expected. Ten minutes. If it
is flipped, the fix belongs in the app, not in this machine's code — do not compensate here.

---

## Template fidelity — state it, don't hide it

The asset is a store-bought anatomy model with a separate artist-made skin, fitted onto the
anatomy with several millimetres of error. It was not scanned from a person.

So **the template's own skin-to-skull distances are not real tissue depths.** Any skull
inferred by warping skin and propagating inward is plausible relative to this template, not
relative to real anatomy. This belongs in the report as a measured limitation, not as a
caveat in a footnote.

---

## Open questions — ask, don't guess

- Which of the four `.mb` copies is canonical? Project 1 took mesh exports from the 451 MB
  root copy and its materials query from `… housekeeping 3.mb`
- Is the rest pose frame 15 of `"not even you"`, or is there a true neutral elsewhere?
- Does `maintainOffset` pull moved rest vertices back toward the generic shape?
- Is the template meant to be anatomically realistic, or is the stylisation intentional?

---

## Deadline context

December 2026, roughly seven weeks from late October, with **shared booked access to this
workstation**. That booking is the critical path for everything in this file.

Front-load anything that does not need a slot — the mirror check, the template measurements,
reading `landmarks_lab.json`, writing the fixture code against exported `.npz` data. Save
slot time for the exporter work and the round-trip tests, which genuinely need Maya open.

## What this machine does not do

- Not the non-rigid fit — that is laptop-side Python
- Not the volumetric warp
- Not building a rig from scratch; one exists
- Not placing muscles, joints or NURBS CVs by hand — they ride the warp field later

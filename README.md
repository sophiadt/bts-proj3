# 🧬 Automated 3D Asset Generation for Physically Based Facial Animation

**Automated 3D Asset Generation for Physically Based Facial Animation** is an undergraduate research project at Simon Fraser University (GrUVi Lab, School of Computing Science). It investigates how to automate the construction of personalized, physically accurate facial assets from a person's own capture, so that a generic anatomical face model can be reshaped to match them before simulation.

**Goal:** Establish a reliable, automated starting point for facial meshes before the physical solver takes over, replacing the tedious manual setup and topological tweaking that personalized, physically accurate 3D face models currently require.

**Method:** Infer rough skull dimensions, rest muscle anchor points and a personalized skin mesh boundary from a person's own capture, using smartphone-based landmark tracking and, as a more robust alternative, volumetric scans (such as MRI).

This research project runs from September 2026 to Present. This repository will be updated regularly as the work progresses.

---

# 🖼 How it fits together

```
iPhone app (TrueDepth App/)            front, left and right photos: colour, raw depth,
        │                              478 MediaPipe landmarks in 3D (mm)
        ▼  AirDrop: one <capture_id>.zip per session
captures/<subject>/<capture_id>/       filed by tools/ingest_capture.py, committed
        │
        ▼  Ziva workstation, mayapy
Fitting pipeline (pipeline/)           left/right check → align and size the lab head →
        │                              reshape the skin → move muscles, fat, skull, jaw
        ▼
Maya side (rig/)                       write the fitted head into a copy of the lab scene,
        │                              check it simulates
        ▼
Lab keyboard UI (lab's a3-stochastic   plays phonemes such as "not even you"
repo, launched from rig/)              on the fitted head
```

The lab head is never rebuilt: only its existing vertices move. Vertex order, polygons and UVs stay the same, so the existing Ziva rig (attachments, weights, joints) carries over to every person.

---

# 🧱 What's in this repo

```
bts-proj3/
├── TrueDepth App/                       the iPhone app: open the .xcodeproj in Xcode
│   ├── Project 3 - TrueDepth Camera Data.xcodeproj
│   └── Project 3 - TrueDepth Camera Data/
│       ├── CameraManager.swift          capture, MediaPipe, depth lookup, saving
│       ├── ContentView.swift            camera UI, mesh overlay, landmark-33 red dot
│       └── face_landmarker.task         MediaPipe model (also read by the pipeline)
├── captures/                            phone captures: <subject>/<capture_id>/ (see captures/README.md)
├── pipeline/                            fitting (plain Python, runs in mayapy)
├── rig/                                 everything that opens Maya: export, write, checks, UI launcher
├── tools/
│   ├── ingest_capture.py                files AirDropped captures into captures/ (standard library only)
│   ├── compare_depth_noise.py           depth roughness per pose (single frame vs burst)
│   ├── mouth_check.py                   does the fitted mouth open like the lab head's? (no Ziva needed)
│   └── render_fit.py                    pictures and animation GIFs of a fit
├── experimental/partner_scan_to_maya/   the app author's first Maya prototype (superseded, kept for reference)
├── core/, checks/                       shared geometry, IO and template measurements
├── config/                              landmark map, MediaPipe canonical face, template registry
├── tests/                               pytest
├── template_export/                     lab head exports, fitted subjects and their pictures (Maya scenes not committed)
├── v68 neutral v13.8 after housekeeping.mb   the lab head scene (Git LFS)
└── CLAUDE.md                            measured facts about the lab rig, run order
```

Generated files go to `template_export/` and `runs/`. `template_export/` is committed (the lab head exports, each fitted subject and its pictures) apart from its Maya scenes, which are too big for GitHub and are rebuilt by `rig/write_rest.py`. From `runs/` only each run's `metrics.json` is committed.

**Not in this repo:** the keyboard/piano animation UI and its phoneme data are in the lab's `a3-stochastic-V1` repo (`C:\Users\zpa18\a3-stochastic-V1`) and on the lab share. `rig/keyboard_ui_fitted.py` runs that UI on a fitted head without changing it.

---

# 🚀 Quick start

## 1. Take a capture (Mac + iPhone)

**Requirements:** a Mac with Xcode, an iPhone with Face ID (the simulator has no TrueDepth camera). The project targets iOS 27.0. MediaPipe Tasks Vision comes in through Swift Package Manager and resolves when the project opens.

1. Open `TrueDepth App/Project 3 - TrueDepth Camera Data.xcodeproj`.
2. In *Signing & Capabilities*, pick your team, select the iPhone, press **Run** (⌘R), allow camera access.
3. Type the subject's anonymous code (`s01`, `s02`, ...) in the **SUBJECT** field. Never a name.
4. Tap **guided capture** (bottom left). The app takes a front, left and right photo, each after a 4-second countdown, with 16 depth frames merged per photo.
5. When it says `Saved s02_<UTC time>_burst.zip`, tap **Share** and AirDrop it to the laptop: one zip per session.
6. File it and commit it, as in [`captures/README.md`](captures/README.md):
   ```bash
   python3 tools/ingest_capture.py --from ~/Downloads     # or drop the zip in captures/_inbox/ and run without --from
   git add captures/s02 && git commit -m "Add capture <capture_id>" && git push
   ```

## 2. Build the asset (Ziva workstation)

Run in a terminal from the repo root, **one Maya job at a time** (the Ziva batch licence has one seat; a second job gets no licence and the scripts stop with `LicenceError`).

```bash
MP="C:/Program Files/Autodesk/Maya2023/bin/mayapy.exe"

ID=s01_20261110T212937Z_burst                                     # a capture id from captures/

$MP pipeline/mirror_check.py $ID --record    # 1. left/right check: read the verdict; --record stores it in capture.json
$MP pipeline/fit_subject.py $ID              # 2. fit + physical checks (relabels on its own if capture.json says labels: swapped)
$MP rig/write_rest.py template_export/subjects/$ID --save $ID --ui-ready   # 3. write into Maya
$MP rig/sim_check.py template_export/scenes/$ID.mb --subject template_export/subjects/$ID   # 4. check it simulates
$MP tools/render_fit.py $ID                  # 5. pictures and GIFs
$MP rig/ui_smoke_test.py $ID                 # 6. the keyboard UI speaks on it; fails if the mouth or lip muscle stays shut
$MP tools/mouth_check.py $ID                 # 7. mouth opening vs the lab head, with pictures (no licence needed)
```

Or let the morph find how far toward the person the rig still works (steps 3, 4 and 6 for each step, about 15 minutes a step, up to 6 steps):

```bash
$MP rig/morph_search.py $ID                  # -> template_export/scenes/<id>_morph.mb (+ _morph_ui.mb)
```

`pipeline/fit_subject.py` ends with the physical checks (`pipeline/verify.py`, below) and prints a pass / warn / fail table; `$MP pipeline/verify.py $ID` repeats them on any subject.

Before booking the workstation, any machine with the lab skin (`template_export/`, or `../bts-proj1/data/lab`) can preview the skin fit and the mouth: `python tools/mouth_check.py --preview $ID --compare`.

The subject output name defaults to the capture id (`--id` overrides it). The first fit, made before captures had ids, is `capture_01` (from `s01_20261030T170049Z_guided`).

You get:

| Where                                         | What                                                                                                         |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `template_export/scenes/<id>.mb`            | the fitted head in the lab rig, with the lab scene's own keys (frames 0–50: the tissue settling, no speech) |
| `template_export/scenes/<id>_ui.mb`         | the same head with the baked keys after frame 14 removed, for the keyboard UI                                |
| `template_export/subjects/<id>/report.json` | every number from the fit: residuals, folds, skin-to-skull depth, neutral-face check                         |
| `template_export/subjects/<id>/views/`      | before/after pictures, the cut through the head, animation GIFs                                              |

The first time on a new machine, set up the template once (each takes a few minutes):

```bash
$MP rig/export_rest.py --update-registry    # lab head bodies and joints → template_export/
$MP rig/roundtrip_test.py                   # proves the rig accepts and keeps a new shape
$MP rig/ui_smoke_test.py capture_01         # after a first fit: also records the lab head speaking
$MP pipeline/importance.py                  # where the skin needs more detail (from that speech motion)
$MP rig/probe_scene.py --attachments       # which vertices each Ziva attachment binds, collision settings
$MP pipeline/build_base.py                  # optional: a human-proportioned start head (--base)
```

Fits start from the lab head itself. `--base` starts from `base_human` (the lab head fitted once to MediaPipe's average face) instead: it kept the person's features no better, and its lip muscle came out squeezed shut, which every fit from it inherited.

## Working on another machine (no Maya)

`template_export/` is committed: the lab head's 52 bodies, joints, the importance map, the attachment maps and the fitted subjects. Only the Maya scenes (`.mb`, 430 MB each) are not; `rig/write_rest.py` rebuilds one from its `subjects/<id>/` folder on the workstation in a few minutes.

```bash
python tools/check_template_export.py                 # everything there? prints what is missing and how to rebuild it
python tools/mouth_check.py --preview $ID --compare   # skin-only fit and mouth check of a capture
python pipeline/verify.py $ID                         # physical checks of a fitted subject
python -m pytest tests                                # (mayapy on the workstation; trimesh tests run on the laptop)
```

## 3. Play it in the keyboard UI (Maya with the GUI)

1. Open Maya 2023 (Ziva loaded). In the Script Editor's **Python** tab run:
   ```python
   exec(open(r"C:\Users\syd3\bts-proj3\rig\keyboard_ui_fitted.py").read())
   ```
2. It opens `capture_01_ui.mb` and the lab's UI. Type **not even you** and press **Go**, or use keyboard mode.
3. To play another person, change `SUBJECT` at the top of `rig/keyboard_ui_fitted.py`.

The launcher sizes the phoneme motion to the fitted face (it was recorded on the lab head, about 1.45× a real face), uses the fitted head's own neutral for the UI's feedback control, and writes the UI's output to `template_export/ui_results/` instead of the lab's folders.

## Tests

```bash
PYTHONPATH=".pytest-lib;." $MP -m pytest tests
```

pytest is installed for mayapy with `$MP -m pip install --target .pytest-lib pytest`, so mayapy itself is left unchanged.

---

# 📸 Taking a good capture

- Hold the phone **at eye level**, about 30–40 cm away, with the whole face in frame. Holding it low tilts the face and biases the result; the fit warns when the phone was more than 15° above or below the face.
- Even light, hair off the forehead, no glasses.
- Neutral face: lips closed and relaxed, jaw closed, eyes open and relaxed.
- At "turn your head to YOUR LEFT / RIGHT", turn about 30° to **your own** left or right and hold still until "Processing". The turned photos are what the left/right check needs.
- Each pose should report most landmarks with depth, for example `470/478 with depth`. Redo it if it says `NO FACE` or many are missing.
- **Left/right test in the app:** touch your **right** eye with your **right** hand. The red dot (landmark 33, the outer corner of your own right eye) should be under your finger.

The fit never uses MediaPipe's blendshape scores. They are saved in `meta.json` for reference, but they read monolid and hooded eyes as squinting and change with camera angle. The neutral-face check measures the 3D landmarks in millimetres instead, and never warns about eye opening.

---

# 📊 Capture files

Each capture lives in `captures/<subject>/<capture_id>/`, where the capture id is `<subject>_<UTC YYYYMMDD>T<HHMMSS>Z_<mode>` (mode `guided`, `burst`, `photo` or `record`). The app names the session folder and its zip the same way. Next to the app's files are:

- `capture.json`: subject, mode, time, app build, and the reviewed left/right verdict (`labels: ok / swapped / unchecked`);
- `extras/`: anything the app didn't write, such as screenshots or other tools' outputs.

The layout, the naming rules and the steps from phone to repo are in [`captures/README.md`](captures/README.md).

Guided capture writes seven files per pose (`front_`, `left_`, `right_`), 21 in all:

| File                                               | Contents                                                                                                                    |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `<pose>_rgb.jpg`                                 | Colour photo in the sensor's native orientation (sideways); same pixel grid as the depth                                    |
| `<pose>_rgb_upright.jpg`                         | The same photo turned upright, not mirrored, for viewing (new app versions)                                                 |
| `<pose>_depth_<W>x<H>_float32_m.bin`             | Raw depth in metres, one float32 per pixel, row-major                                                                       |
| `<pose>_meta.json`                               | Intrinsics, image sizes, `depth_filtered`, lens distortion tables (saved, not applied), pupil distance, blendshape scores |
| `<pose>_landmarks_3d_mm.json`                    | 478 points `[x, y, z]` in mm, `null` where depth is missing; each index keeps its slot                                  |
| `<pose>_landmarks.ply`, `<pose>_face_crop.ply` | For viewing only (nulls dropped, so indices shift)                                                                          |

A single photo (`..._photo`) writes the same files without a prefix. A recording (`..._record`) writes `landmarks_sequence_mm.json`, `recording_meta.json` and, with RAW on, a `raw/` folder. Every session ends with `capture.json` and is shared as one `<capture_id>.zip`. The fitting pipeline reads guided and burst captures; it cannot read recordings yet.

---

# 🔬 How the fitting works

1. **Left/right check.** Compares the landmarks' handedness with MediaPipe's canonical face, and uses the turned photos to see which side came toward the lens. It separates OK, swapped labels, a mirrored scan and depth read from the wrong pixels.
2. **Start head.** The fit starts from the lab head itself (stylised: about 1.45× a real face, oversized eyes, a long, wide lower face). Same meshes and vertex order throughout, so the rig still applies.
3. **Join the photos and align.** The front photo's landmarks place the start head (one rotation, position and uniform scale on about 270 landmark pairs from `config/landmark_to_vertex.json`, weighted toward the nose, chin, forehead and jaw). Each turned photo is fitted onto the front's depth, bent smoothly where the two overlap so they join without a step, and adds only the skin the front did not see. Landmarks come from the front photo only.
4. **Reshape the skin.** The skin is pulled toward its landmarks and onto the depth surface inside the face outline (hair, ears, neck and clothing are left out), while keeping its own shape. It follows the scan only to the scan's noise level (about 1 mm), so the noise is not copied in. The eye openings follow the person's eyelids, keeping the lab head's own offset behind the visible lid (the skin's edge wraps a few mm inward there). The mouth keeps the lab rig's resting opening, centred on the person's lip line: the rig rests with its lips slightly apart and its speech motion starts from there, so closing it to match a closed-mouth capture would seal the mouth. Where the face moves and creases most (lips, jaw, nose, eye corners: an importance map in the spirit of the lab's deformation-weighted Poisson-disk joint sampling) the skin is allowed more detail. Skin no camera saw (the sides and back of the head) stays stiff and is held gently to the start head's shape. A last **detail pass** moves the skin along its normals toward the smoothed scan (clamped to 4 mm, not near the openings or unseen skin), which brings back the nose, brow, cheekbones and chin the stiff fit softened.
5. **Move the inside.** Skull, jaw and teeth move together as one (the capture is a closed, neutral mouth), placed so the lab head's skin-to-bone depth is kept under the fitted skin where the camera saw the face. The front teeth are kept behind the lips. The eyeballs are real size (24 mm), rigid, seated behind the fitted eye openings. Muscle and fat follow a smooth field pinned at the skin and at the moved bones, with three guards:
   - **Lips stay apart.** The lip muscle's upper half follows the upper lip and its lower half the lower lip (two fields, joined past the mouth corners), and where its halves still come closer than 0.7× the lab head's gap they are moved apart. Ziva runs with collisions off, so nothing else would keep them from sticking together.
   - **No tissue inside bone.** Tissue the warp pushed into skull, jaw or teeth goes back out to the bone's surface.
   - **No inside-out muscle.** A small muscle the field folds (more than 5% of its triangles) takes its warp's best affine map, or a similarity if that would mirror it.
6. **Check it is physically valid** (`pipeline/verify.py`, no licence): same mesh, folded triangles, each muscle's volume, tissue inside bone, every Ziva attachment's distance to its bone, mouth and lip muscle open, teeth and eyes behind the skin, each against the lab head at the person's size. Pass / warn / fail with the numbers, saved in `report.json`.
7. **Write into Maya.** New rest shapes are written into a copy of the lab scene, wired so Ziva uses them; joints are moved and rebound. Checks: the scene shows the fitted face, every body's polygons and UVs match the lab head, and frames 0–50 (the tissue settling under gravity) run without blowing up. Speech is tested through the keyboard UI (`rig/ui_smoke_test.py`).

---

# 📈 First real capture (`captures/s01/s01_20261030T170049Z_guided/`, fitted as `capture_01`)

| Measure                            | Result                                                                                                                                     |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Landmarks with depth               | 478/478 in each pose                                                                                                                       |
| Left/right check                   | `LABELS_SWAPPED` (app mirroring; fitted with `--relabel`, OK after)                                                                    |
| Neutral-face check                 | OK: lips closed (0.3–0.6 mm), eyelids 7–11.5 mm, phone within 15°                                                                       |
| Start head                         | `base_human` (the lab head reshaped to an average human face)                                                                            |
| Size against the lab head          | 71.9%                                                                                                                                      |
| Alignment error, before reshaping  | 5.4 mm typical                                                                                                                             |
| Skin to landmarks, after reshaping | 1.1 mm typical                                                                                                                             |
| Skin to scan                       | 0.85 mm typical, 3.6 mm for 95%                                                                                                            |
| Folded skin triangles              | 340 of 31,225 (745 before the human start head and the fixes below)                                                                        |
| Eye and mouth openings             | 0.5 mm off the person's eyelids and inner lips                                                                                             |
| Mouth                              | pulled shut (lips 3.2 mm apart at the midline); front teeth 8.3 mm behind the lips. This sealed the mouth when animated; see Known issues |
| Eyeballs                           | 24 mm, behind the lids                                                                                                                     |
| Skin-to-skull depth                | 1.01× the lab head's                                                                                                                      |
| In Maya                            | shows the fitted face exactly; polygons and UVs identical to the lab head in all 52 bodies; stable while the tissue settles (frames 0–50) |
| Keyboard UI                        | plays "not even you" on the fitted head (motion scaled ×0.72): lips, chin and jaw move up to 19.9 mm, stable                              |

**Second capture, same subject (`captures/s01/s01_20261110T212937Z_burst/`).** This was the first capture taken with the mirror fix and burst depth:

- The left/right check is **OK** without `--relabel`, so the fix works on the phone.
- 16 depth frames merged per pose lowered the face's depth roughness from 1.43 to 1.12 mm, and halved the flying pixels (0.58% → 0.28%), by `tools/compare_depth_noise.py`.

**Both captures refitted (2026-11-22)** from the lab head, with the detail pass, the split lip field and the tissue guards:

| Measure | Burst (2026-11-10) | Guided (2026-10-30) |
| --- | --- | --- |
| Left/right check | OK | `LABELS_SWAPPED`, relabelled automatically, OK after |
| Size against the lab head | 73.9% | 71.6% |
| Skin to landmarks | 1.15 mm typical, 2.8 mm for 95% | 1.10 mm typical, 3.3 mm for 95% |
| Scan to skin | 0.61 mm typical, 2.2 mm for 95% | 0.76 mm typical, 3.5 mm for 95% |
| Feature match, before → after detail pass (median) | nose 0.61 → 0.56, brow 0.98 → 0.72, cheeks 1.11 → 0.83, chin 1.23 → 0.99 mm | nose 0.69 → 0.60, brow 1.36 → 1.03, cheeks 1.21 → 0.95, chin 0.71 → 0.67 mm |
| Mouth at rest | hole edges 6.1 mm apart (lab head at this size 6.1) | 5.9 mm (5.9) |
| Lip muscle at rest | halves 1.66 mm apart (lab head 1.79), no slice closing | 1.81 mm (1.75), none closing |
| Skin-only speech preview | opens 81% as wide as the lab head | 92% |
| Physical checks (`verify.py`) | WARN (folded triangles in the lower-lip depressors, volume, tissue in bone, attachments) | FAIL: ZygomaticusMinor_r at 1.74× its volume |

**Step-by-step morph** (`rig/morph_search.py`): t is the fraction of the way from the lab head (0) to the fit (1). Every step passed the physical checks, was written into Maya and simulated stably (frames 0–50). The column is the keyboard UI's "not even you": the mouth's widest opening against the lab head's at that size (pass ≥ 70%), and the lip muscle's widest gap.

| t | Burst | Guided |
| --- | --- | --- |
| 0.125 | **pass**: 87%, muscle 96% | **pass**: 85%, muscle 96% |
| 0.157 | fail: 68%, muscle 89% | fail: 66%, muscle 89% |
| 0.188 | fail: 63%, muscle 87% | fail: 61%, muscle 87% |
| 0.25 | fail: 63%, muscle 84% | fail: 64%, muscle 84% |

The furthest working step of each is kept as `template_export/scenes/<id>_morph.mb` (and `_morph_ui.mb` for the keyboard UI). The lips no longer stick (the lip muscle parts at 84–96% of the lab head's at every step); what stops the morph is the middle of the upper lip, which follows the jaw down as the mouth opens. See Known issues.

---

# ⚠️ Known issues

- **The morph stops working at t ≈ 0.15, on both captures.** Past that step the middle of the upper lip follows the jaw down when the mouth opens (at the widest frame it drops 16.6 mm against the lab head's 5.9, and stays about 5 mm low after the sentence), so the opening falls from 87% to 68% of the lab head's. Tests on the burst capture's t = 0.157 step, each putting one part back at t = 0.125:
  - the lip muscle (OrbicularisOris) alone restores it (87%);
  - the skin, the joints, the lower-lip depressors or the chin muscle change nothing (68%);
  - moving the whole head 2 mm changes nothing (the same 779 mm²), and neither does scaling every tissue's `tetSize` with the head (67%) or pinning the attachments' target maps to the lab head's binding (68%).

  The lip muscle moves only ~0.05 mm between the two steps, so the likely cause is Ziva's tet mesh for it. That mesh is rebuilt from the rest shape at `tetSize` 26.4 (about a quarter of the muscle's width), much coarser than the ~2.5 mm gap between the lip halves, so a tiny change can join the halves at the middle. A finer lip-muscle tet mesh is the next test. It changes the lab's tuned rig, so it is a decision for the lab.
- **Fits before 2026-11-17 animate with a sealed mouth** (`capture_01`). The lab rig rests with its lips apart (mouth-hole edges 8.3 mm apart at the middle), and its speech motion starts from there. The fit pulled the mouth hole onto the person's closed lips, which squeezed the hole and the lip muscle shut; when animated, the mouth opened about 40–50% as wide as the lab head's, only in the middle. The fit now keeps the rig's resting opening on the person's lip line, so at rest the lips show about 2 mm apart, as on the lab head (skin-only preview: 82–94% of the lab head's opening). `ui_smoke_test.py` and `tools/mouth_check.py` now fail a mouth that stays shut. Rebuild `base_human` and refit.
- **Left/right labels were swapped by the first app build** (it mirrored the image MediaPipe sees). Fixed in `CameraManager.swift` and verified: the burst capture of 2026-11-10 passes the left/right check without `--relabel`. The first capture keeps `labels: swapped` in its `capture.json`, and `fit_subject.py` relabels it on its own.
- **The lab head is stylised** (about 1.45× a real face, eyes 1.6–1.7×). The human start head absorbs most of that, but the skin is still a little faceted: the phone's depth is about 1 mm noisy and coarse, and the fit follows it no closer than that.
- **Tried and dropped:** a two-step fit (first a smooth warp from the landmarks alone, then the scan) folded the skin far more (3,500–5,000 folded triangles), because a 3D warp crosses the skin's inner wraps at the lips and eyes. Filling missing landmarks from the turned photos put a crease down a cheek: most fills came from one photo with nothing to check them against.
- **Tissue depth comes from the lab head**, scaled to the person, not measured. A phone cannot see bone.
- **Expressions** play from the lab head's recorded motion, sized to the person; they are not re-fitted to the new face.
- **Results so far come from one person**; more subjects are still to be captured.
- No license file has been added yet.

---

# 🛠 Technologies

- Swift and SwiftUI, AVFoundation (TrueDepth: synchronized colour and depth), Core Image
- MediaPipe Tasks Vision, Face Landmarker (Swift Package Manager)
- Autodesk Maya 2023 and Ziva VFX 2.2, with mayapy (numpy, scipy, matplotlib)

---

# 📚 Related Research

> [Add full citation of the primary reference]

- Google. *MediaPipe Face Landmarker.* https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
- Apple. *AVFoundation: capturing depth data (TrueDepth camera).* https://developer.apple.com/documentation/avfoundation

---

# 📄 License

No license file has been added yet. Third-party software (MediaPipe, Autodesk Maya, Ziva), the lab head scene and any captured face data are covered by their own licenses and terms.

---

# 👨‍💻 Authors

**Sophia Don Tranho** | Undergraduate Research Assistant | Simon Fraser University

**William Huang** | Undergraduate Research Assistant | Simon Fraser University

---

# 🙏 Acknowledgements

This project is conducted as part of an undergraduate research position at Simon Fraser University.

Supervised by Zhiyun (Richard) Peng and Eugene Fiume at the GrUVi Lab, School of Computing Science, Simon Fraser University.

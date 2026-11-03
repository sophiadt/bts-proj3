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
        ▼  AirDrop / Share
Capture folder (Capture Data/)
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
├── Capture Data/                        a guided capture: Front/, Left/, Right/
├── pipeline/                            fitting (plain Python, runs in mayapy)
├── rig/                                 everything that opens Maya: export, write, checks, UI launcher
├── tools/render_fit.py                  pictures and animation GIFs of a fit
├── core/, checks/                       shared geometry, IO and template measurements
├── config/                              landmark map, MediaPipe canonical face, template registry
├── tests/                               pytest
├── v68 neutral v13.8 after housekeeping.mb   the lab head scene (Git LFS)
└── CLAUDE.md                            measured facts about the lab rig, run order
```

Generated files go to `template_export/` and `runs/`; both are git-ignored apart from each run's `metrics.json`.

**Not in this repo:** the keyboard/piano animation UI and its phoneme data are in the lab's `a3-stochastic-V1` repo (`C:\Users\zpa18\a3-stochastic-V1`) and on the lab share. `rig/keyboard_ui_fitted.py` runs that UI on a fitted head without changing it.

---

# 🚀 Quick start

## 1. Take a capture (Mac + iPhone)

**Requirements:** a Mac with Xcode, an iPhone with Face ID (the simulator has no TrueDepth camera). The project targets iOS 27.0. MediaPipe Tasks Vision comes in through Swift Package Manager and resolves when the project opens.

1. Open `TrueDepth App/Project 3 - TrueDepth Camera Data.xcodeproj`.
2. In *Signing & Capabilities*, pick your team, select the iPhone, press **Run** (⌘R), allow camera access.
3. Tap **guided capture** (bottom left). The app takes a front, left and right photo, each after a 4-second countdown.
4. When it says `Saved face_<timestamp>`, tap **Share** and AirDrop the files to the Mac.
5. Put the 18 files (or the three `Front/`, `Left/`, `Right/` folders) in `Capture Data/` (or any folder) and copy it to the Ziva workstation.

## 2. Build the asset (Ziva workstation)

Run in a terminal from the repo root, **one Maya job at a time** (the Ziva batch licence has one seat; a second job gets no licence and the scripts stop with `LicenceError`).

```bash
MP="C:/Program Files/Autodesk/Maya2023/bin/mayapy.exe"

$MP pipeline/mirror_check.py "Capture Data"                       # 1. left/right check: read the verdict
$MP pipeline/fit_subject.py "Capture Data" --id capture_01        # 2. fit (add --relabel only if step 1 said LABELS_SWAPPED)
$MP rig/write_rest.py template_export/subjects/capture_01 --save capture_01 --ui-ready   # 3. write into Maya
$MP rig/sim_check.py template_export/scenes/capture_01.mb --subject template_export/subjects/capture_01   # 4. check it simulates
$MP tools/render_fit.py capture_01                                # 5. pictures and GIFs
```

You get:

| Where | What |
|---|---|
| `template_export/scenes/capture_01.mb` | the fitted head in the lab rig, with the lab scene's own keys (frames 0–50: the tissue settling, no speech) |
| `template_export/scenes/capture_01_ui.mb` | the same head with the baked keys after frame 14 removed, for the keyboard UI |
| `template_export/subjects/capture_01/report.json` | every number from the fit: residuals, folds, skin-to-skull depth, neutral-face check |
| `template_export/subjects/capture_01/views/` | before/after pictures, the cut through the head, animation GIFs |

The first time on a new machine, set up the template once (each takes a few minutes):

```bash
$MP rig/export_rest.py --update-registry    # lab head bodies and joints → template_export/
$MP rig/roundtrip_test.py                   # proves the rig accepts and keeps a new shape
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
PYTHONPATH=".pytest-lib;." $MP -m pytest tests/test_pipeline.py
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

Guided capture writes six files per pose (`front_`, `left_`, `right_`), 18 in all:

| File | Contents |
|---|---|
| `<pose>_rgb.jpg` | Colour photo in the sensor's native orientation (sideways); same pixel grid as the depth |
| `<pose>_rgb_upright.jpg` | The same photo turned upright, not mirrored, for viewing (new app versions) |
| `<pose>_depth_<W>x<H>_float32_m.bin` | Raw depth in metres, one float32 per pixel, row-major |
| `<pose>_meta.json` | Intrinsics, image sizes, `depth_filtered`, lens distortion tables (saved, not applied), pupil distance, blendshape scores |
| `<pose>_landmarks_3d_mm.json` | 478 points `[x, y, z]` in mm, `null` where depth is missing; each index keeps its slot |
| `<pose>_landmarks.ply`, `<pose>_face_crop.ply` | For viewing only (nulls dropped, so indices shift) |

A single photo writes `capture_<timestamp>/` with the same files. A recording writes `recording_<timestamp>/` with `landmarks_sequence_mm.json`, `recording_meta.json`, and with RAW on a `raw/` folder and its `.zip`.

---

# 🔬 How the fitting works

1. **Left/right check.** Compares the landmarks' handedness with MediaPipe's canonical face, and uses the turned photos to see which side came toward the lens. It separates OK, swapped labels, a mirrored scan and depth read from the wrong pixels.
2. **Align and size.** One rotation, position and uniform scale place the lab skin on about 270 landmark pairs (from `config/landmark_to_vertex.json`), weighted toward the nose, chin, forehead and jaw. The whole lab head (skin, muscles, bones, joints) is scaled together.
3. **Reshape the skin.** The skin is pulled toward its landmarks and toward the depth points inside the face outline (hair, ears, neck and clothing are left out), while keeping its own shape. The eye, mouth and nose openings are masks: nothing is pulled at their edges.
4. **Move the inside.** Skull and jaw are placed so the lab head's skin-to-bone depth is kept under the fitted skin, where the camera saw the face. Muscle and fat follow a smooth field pinned at the skin and at the moved bones.
5. **Write into Maya.** New rest shapes are written into a copy of the lab scene, wired so Ziva uses them; joints are moved and rebound. Checks: the scene shows the fitted face, every body's polygons and UVs match the lab head, and frames 0–50 (the tissue settling under gravity) run without blowing up. Speech is tested through the keyboard UI (`rig/ui_smoke_test.py`).

---

# 📈 First real capture (`Capture Data/`, November 2026)

| Measure | Result |
|---|---|
| Landmarks with depth | 478/478 in each pose |
| Left/right check | `LABELS_SWAPPED` (app mirroring; fitted with `--relabel`, OK after) |
| Neutral-face check | OK: lips closed (0.3–0.6 mm), eyelids 7–11.5 mm, phone within 15° |
| Size against the lab head | 71.6% |
| Alignment error, before reshaping | 5.2 mm typical |
| Skin to landmarks, after reshaping | 0.9 mm typical |
| Skin to scan | 0.9 mm typical, 3.8 mm for 95% |
| Folded skin triangles | 745 of 31,225 (mostly face edges and jaw sides) |
| Skin-to-skull depth | 0.92× the lab head's |
| In Maya | shows the fitted face exactly; polygons and UVs identical to the lab head in all 52 bodies; stable while the tissue settles (frames 0–50) |
| Keyboard UI | plays "not even you" on the fitted head (motion scaled ×0.72): lips, chin and jaw move up to 21.6 mm, stable |

---

# ⚠️ Known issues

- **Left/right labels were swapped by the app** (it mirrored the image MediaPipe sees). Fixed in `CameraManager.swift`; the fix still has to be built and checked on the iPhone (the red-dot test, then a new guided capture that the left/right check calls OK without `--relabel`).
- **The lab head is stylised** (about 1.45× a real face, eyes 1.6–1.7×), so real faces need a strong reshape around the jaw and eyes, which is where the remaining folds are.
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

**William Huang** | Undergraduate Research Assistant | Simon Fraser University

**Sophia Don Tranho** | Undergraduate Research Assistant | Simon Fraser University

---

# 🙏 Acknowledgements

This project is conducted as part of an undergraduate research position at Simon Fraser University.

Supervised by Zhiyun (Richard) Peng and Eugene Fiume at the GrUVi Lab, School of Computing Science, Simon Fraser University.

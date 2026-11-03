# 🧬 Automated 3D Asset Generation for Physically Based Facial Animation

**Automated 3D Asset Generation for Physically Based Facial Animation** is an undergraduate research project at Simon Fraser University (GrUVi Lab, School of Computing Science). It investigates how to automate the construction of personalized, physically accurate facial assets from a person's own capture, so that a generic anatomical face model can be reshaped to match them before simulation.

**Goal:** Establish a reliable, automated starting point for facial meshes before the physical solver takes over, replacing the tedious manual setup and topological tweaking that personalized, physically accurate 3D face models currently require.

**Method:** Infer rough skull dimensions, rest muscle anchor points and a personalized skin mesh boundary from a person's own capture, using smartphone-based landmark tracking and, as a more robust alternative, volumetric scans (such as MRI).

This research project runs from September 2026 to Present. This repository will be updated regularly as the work progresses.

---

# 🎯 Research Overview

Generating personalized, physically accurate 3D face models for animation currently requires tedious manual setup and topological tweaking. This project explores how to fully automate the construction of personalized facial assets to streamline physically based facial animation. It investigates distinct initialization pathways:

- **Smartphone video and landmark tracking:** a lightweight vision pipeline (for example, MediaPipe) that infers rough skull dimensions, rest muscle anchor points and a personalized skin mesh boundary initialization.
- **Volumetric scans (like MRI):** a highly robust alternative that automates the generation of the same boundaries directly from volumetric scans, used as ground truth.
- **iPhone TrueDepth (this repository):** a depth-based capture pathway that adds measured depth to the smartphone approach. The capture app records color, raw depth and face landmarks, and writes them in a folder layout that the Ziva-side fitting pipeline reads directly.

The outputs of the project are a prototype initialization tool (a script that ingests video or scan data and outputs personalized muscle anchors and skin boundaries), a comparative analysis of the accuracy, computational cost and accessibility of the approaches, and a technical report on the registration pipeline, its success rate and its viability for scaling personalized avatars.

The project introduces core concepts in computer vision, geometric processing and automated pipeline design, including the ambiguities of monocular depth estimation and the trade-offs between accessible RGB video and highly accurate volumetric data. This repository covers the step of turning raw inputs into data a simulation-ready 3D asset can be built from.

---

# 🖼 Pipeline Overview

```
iPhone (TrueDepth camera)
        │  color + raw depth (filtering off)
        ▼
MediaPipe Face Landmarker
        │  478 landmarks + 52 blendshape scores
        ▼
Depth lookup + camera intrinsics
        │  478 3D points in mm
        ▼
Capture folder  (face_<timestamp>/, 18 files)
        │  AirDrop → Mac
        ▼
Fitting pipeline (Ziva side)
        │  shrink head → reshape skin → muscles, fat, skull, jaw follow
        ▼
Fitted lab head, written back into Maya / Ziva
```

---

# 🔬 Method

1. The app reads synchronized color and depth frames from the front TrueDepth camera (structured light, not LiDAR). Depth filtering is off, so the depth is raw.
2. MediaPipe Face Landmarker finds 478 landmarks (468 face points and 10 iris points) and 52 blendshape scores on the color image.
3. Each landmark's depth is read from the depth map, using a 3×3 median and falling back to 7×7 where there is no reading.
4. The 2D landmark and its depth are converted to a 3D point in millimeters with the camera intrinsics (`fx`, `fy`, `cx`, `cy`).
5. In guided capture, the app takes a **front**, **left** and **right** pose, each after a 4-second countdown, and saves the files for each pose.
6. The files are shared to a Mac and passed to the fitting pipeline.

The sensor's native landscape frame is kept, so a face appears on its side in the raw color and depth files. The app rotates the image for MediaPipe and maps the landmarks back to depth pixels.

---

# 📲 Setup and Usage

**Requirements**

- A Mac with Xcode
- An iPhone with Face ID (the simulator will not work)
- iOS 16 or later
- MediaPipe Tasks Vision (iOS) and the `face_landmarker.task` model, added to the app bundle

**Setup**

1. Clone the repo and open the Xcode project (the `.xcworkspace` if there is one, otherwise the `.xcodeproj`).
2. Install dependencies if needed (for CocoaPods: `pod install`; the `Pods/` folder is not committed).
3. Make sure `face_landmarker.task` is in the app target so it ends up in the bundle.
4. In *Signing & Capabilities*, choose your team, select your iPhone, and press Run (Cmd+R). Allow camera access on first launch.

**In the app**

- **Live view:** a blue wireframe shows the detected face; it turns **red** where a landmark has no depth.
- **DEPTH / CAMERA** (top left) switches between the depth view and the normal camera view.
- **Status badge** (top right) shows the landmarks with depth, for example `476/478`, or `NO FACE` / `FACE CUT OFF`.
- **PHOTO / RECORD** with the round shutter saves one frame or a recording. **RAW: ON/OFF** (record mode) also saves the raw color and depth for every 2nd processed frame. It is large and off by default.
- **Guided capture** (bottom left) takes the front, left and right poses. This is the capture to use for fitting.
- **Share** (bottom right) sends the files from the last capture.

**Taking a good capture**

- Phone about 30–40 cm from the face, whole face in frame, even light, hair off the forehead, no glasses.
- Neutral face: lips closed and relaxed, eyes open, no smile.
- At "turn to YOUR LEFT / RIGHT", turn about 30° and hold still until the app says "Processing".
- Each pose should report most landmarks with depth, for example `470/478 with depth`. Redo it if it says `NO FACE` or many are missing.

---

# 📊 Output Files

Guided capture writes `face_<timestamp>/` with six files per pose (`front_`, `left_`, `right_`), 18 files in all:

| File | Contents |
|---|---|
| `<pose>_rgb.jpg` | Color photo (sensor-native orientation) |
| `<pose>_depth_<W>x<H>_float32_m.bin` | Raw depth in meters, one float32 per pixel, row-major; the real resolution is in the filename |
| `<pose>_meta.json` | Intrinsics, image sizes, `depth_filtered`, lens distortion tables (saved, not applied), pupil distance, 52 blendshape scores |
| `<pose>_landmarks_3d_mm.json` | 478 points `[x, y, z]` in mm, `null` where depth is missing |
| `<pose>_landmarks.ply` | Landmarks, for viewing only |
| `<pose>_face_crop.ply` | Depth points around the face, for viewing only |

A single photo writes `capture_<timestamp>/` with the same six files. A recording writes `recording_<timestamp>/` with `landmarks_sequence_mm.json` (per frame: time, 478 points in mm, blendshape scores, raw-frame index), `recording_meta.json`, and, with RAW on, a `raw/` folder and its `.zip`.

Units are millimeters unless stated. Z is the distance from the camera.

---

# 📈 Evaluation Metrics

- **Landmarks with depth:** how many of the 478 landmarks have a depth reading in each pose (target: about 470 or more)
- **Eye width:** the distance between landmarks 33 and 263 should be about 85–100 mm on an adult
- **Pupil distance:** about 54–74 mm on an adult; a value far outside this suggests the depth scale is off
- **Depth filtering:** the status line should read `filtered: no`

---

# 🧱 Repository Structure

```
bts-proj3/
├── Project 3 - TrueDepth Camera Data/
│   └── Project 3 - TrueDepth Camera Data/
│       ├── CameraManager.swift   # capture, MediaPipe, depth fusion, saving
│       ├── ContentView.swift     # camera UI, mesh overlay, shutter
│       └── ...                   # app entry point, assets, face_landmarker.task
├── .gitignore
└── README.md
```

---

# ⚠️ Limitations

> Findings below are from early testing and will be updated as more captures are processed.

- **Left/right labels are being verified.** In a debug test, landmark 33 (the outer corner of the person's own *right* eye) appeared on the person's *left* eye, so the labels may be swapped. The mirroring and left/right code stays unchanged until the check on the guided captures is back.
- Landmarks at the edge of the face (for example 361, 365 and 401) can pick up background depth, and the iris landmarks often have no depth.
- TrueDepth can fail with beards, glasses and dark hair.
- Results so far come from one person; a second subject is still to be captured.

---

# 🛠 Technologies

- Swift and SwiftUI
- AVFoundation (TrueDepth camera, synchronized color and depth)
- MediaPipe Tasks Vision, Face Landmarker
- Core Image
- Xcode, CocoaPods
- Maya and Ziva (fitting pipeline, outside this repo)

---

# 🚀 Future Work

**Planned improvements to the capture app:**

- Settle the left/right and mirroring labels from the guided-capture check
- Reject background depth at the face edge
- Burst depth: save about 20 depth frames per press so noise can be averaged out
- Apply lens distortion correction (the tables are saved but not yet used)
- Capture a second subject

---

# 📚 Related Research

The capture app supports the fitting pipeline for the project below.

> [Add full citation of the primary reference]

- Google. *MediaPipe Face Landmarker.* https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
- Apple. *AVFoundation: capturing depth data (TrueDepth camera).* https://developer.apple.com/documentation/avfoundation

---

# 📄 License

This project is released under the MIT License.

See the [LICENSE](LICENSE) file for details.

Third-party software (such as MediaPipe, Autodesk Maya and Ziva) and any captured face data are covered by their own licenses and terms and are not included.

---

# 👨‍💻 Authors

**William Huang** | Undergraduate Research Assistant | Simon Fraser University

**Sophia Don Tranho** | Undergraduate Research Assistant | Simon Fraser University

---

# 🙏 Acknowledgements

This project is conducted as part of an undergraduate research position at Simon Fraser University.

Supervised by Zhiyun (Richard) Peng and Eugene Fiume at the GrUVi Lab, School of Computing Science, Simon Fraser University.

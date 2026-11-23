# Findings, 22 Nov 2026

Capture: `s01_20261110T212937Z_burst` (10 Nov burst). Rig runs on the Ziva workstation; skin-only previews on a Mac.
The same page, with the pictures inline, is in `report/capture_fit_followup.html`.

## Result

When the full fit says "not even you", its mouth opens **46%** as wide as the lab head's at the same size. The pass mark is 70%.
Nothing tried today moved it past 49%. The lip muscle is not the limit: it opens almost as far as the lab head's and the lips still do not follow.

Mouth opening is compared with the lab head at the same size: 521 mm2 opening, 9.48 mm lip-muscle gap.

| Run (Ziva) | Opening when speaking | Lip muscle gap while speaking | Result |
|---|---|---|---|
| Baseline, full fit, settings as before | 238 mm2 (46%) | 7.84 mm (83%) | LOW |
| `--smooth-contours --ring-depth` | 237 mm2 (46%) | 8.05 mm (85%) | LOW |
| Same plus `--smooth-skin` (10 iterations) | 238 mm2 (46%) | 8.07 mm (85%) | LOW |
| Lip muscle kept as the lab's shape (`--rigid OrbicularisOris`) | 257 mm2 (49%) | 9.35 mm (99%) | LOW |

Pictures: `pictures/baseline/mouth_check.png`, `pictures/smooth/mouth_check.png`, `pictures/rigid_lip_muscle_mouth_check.png`.
At the widest speech frame the opening is nearly as wide as the lab head's and much shorter. The upper lip does not lift.

Sophia's step-by-step morph (First Capture Fit report) stops at t = 0.25 (63%). The full fit is t = 1, and 46% continues that trend.
Her swap test at t about 0.157 pointed at the lip muscle. At the full fit it does not.

### Raw `tools/mouth_check.py` output (Ziva runs)

```
baseline (s01_20261110T212937Z_burst)
  hole edges at the middle  6.2 mm   (lab head 6.0 mm)
  opening at rest           128 mm2  (lab head 150 mm2)
  opening when speaking     238 mm2  (lab head 521 mm2) = 46%
  lip muscle gap            median 1.66 mm, min 0.72 mm  (lab head 1.79 / 0.75); narrowest 66%; 0 of 30 slices closing
  ui_smoke_test: skin moves up to 20.3 mm (generic 27.4); lip muscle widest gap 7.84 mm vs 9.48 mm (83%)

options on (..._burst_opts)
  opening at rest           130 mm2
  opening when speaking     237 mm2  = 46%
  lip muscle gap            median 1.70 mm, min 0.68 mm; narrowest 68%
  ui_smoke_test: lip muscle widest gap 8.05 mm (85%)

lip muscle as the lab's shape (..._t100_OrbicularisOrisRigid)
  opening at rest           128 mm2
  opening when speaking     257 mm2  = 49%
  lip muscle gap            median 1.77 mm, min 0.88 mm; narrowest 95%
  ui_smoke_test: skin moves up to 20.4 mm; lip muscle widest gap 9.35 mm (99%)

smoothed fit (..._smooth)
  opening at rest           130 mm2
  opening when speaking     238 mm2  = 46%
  lip muscle gap            median 1.67 mm, min 0.67 mm; narrowest 69%
  ui_smoke_test: lip muscle widest gap 8.07 mm (85%)
  fit_subject: skin smoothing (10 Taubin iterations): roughness 0.108 -> 0.108 mm, moved median 0.08 max 1.70 mm
  fit_subject: features, scan to skin median mm: lips 0.54, nose 0.56, chin 0.98, brow 0.71, cheeks 0.84
```

## Lip muscle variants, checked before Maya (`pipeline/verify.py`, no licence)

| Variant | Physical check | Ziva |
|---|---|---|
| `--part OrbicularisOris=0.125` (muscle held at 12.5% toward the fit) | FAIL: muscle volume 1.268x, attachments n=72 worst fraction 1.0 (all far) | not run |
| `--rigid OrbicularisOris` | WARN: worst attachment fraction 0.412, none far; mouth edge gap 6.13 mm (lab 6.11); muscle gap median 1.77 mm (lab 1.79) | 49% |
| `--rigid` on OrbicularisOris, LevatorLabiiSuperioris l/r, ZygomaticusMinor l/r (`--tag lift`) | WARN, same as above | written, not run |

Remaining warnings (both WARN runs): elements on DepressorLabiiInferioris_r (fraction 0.0271), volume on DepressorLabiiInferioris_r (0.546), penetration on Mentalis (0.1537).
These match what CLAUDE.md reports for the plain burst fit.

## Skin lumps (skin-only preview on a Mac, about 10 s per run)

`python tools/mouth_check.py --preview s01_20261110T212937Z_burst --start lab --smooth-contours --ring-depth --smooth-skin N`

| Setting | Folded triangles | Scan to skin | Landmarks median | Preview folder |
|---|---|---|---|---|
| No options | 212 | 0.61 mm | | `mac_previews/20261122T195435_6861` |
| Contour smoothing and ring depth | 136 | 0.61 mm | | `mac_previews/20261122T195443_9e4d` |
| plus `--smooth-skin 10` | 141 | 0.60 mm | 1.18 mm | `mac_previews/20261122T221908_b4c2` |
| `--smooth-skin 40` | 123 | 0.60 mm | 1.18 mm | `mac_previews/20261122T221915_6d9f` |
| `--smooth-skin 80` | 110 | 0.59 mm | 1.22 mm | `mac_previews/20261122T221923_a149` |
| `--smooth-skin 80`, `--smooth-contours 1.6` | 119 | 0.59 mm | 1.22 mm | `mac_previews/20261122T223545_31ad` |

(The first two rows' fold counts are from the earlier Mac run; the preview opening of 83 to 84% is a skin-only estimate, not the rig result.)

What the pictures show:
- 80 iterations softens the faceting under the lower lip and on the chin a little, and calms the forehead and cheeks in `pictures/smooth/parts.png` against `pictures/baseline/parts.png`. The change is small.
- The wavy lower edge of the mouth opening and the crack at the right mouth corner stay. The edge comes from the landmark contours, which smoothing leaves alone on purpose. The corner artifacts come from the speech motion and also show on the lab head.
- Folded triangles sit at the eye corners, the nostril base and the mouth corners (see `fit_detail.png`), where smoothing is switched off. The fold count is a poor stand-in for cheek lumpiness; judge by the pictures.
- The first smoothing result (10 iterations) moved the skin by a median of 0.08 mm, which is invisible. More iterations help only a little.

## Ruled out

- Fit-side settings for the mouth (contour smoothing, ring depth, skin smoothing): the opening stays at 46%.
- Lip muscle shape as the limit at the full fit: its gap reaches 99% of the lab head's and the opening is 49%.
- Photogrammetry for the eye and mouth data loss and the M shape: the depth there is present; the M shape is the simulation's upper-lip sag; the fit keeps the rig's own openings.
- Two-view landmark triangulation (`tools/triangulate_check.py`): about 3 mm of noise between views, too noisy to correct depth.
- More burst frames: depth roughness (about 1.1 mm) is almost the same in every frame.

## Code in this branch

| Commit message | Flag |
|---|---|
| Optional contour smoothing and ring depth for eyelid and lip targets | `--smooth-contours [MM]`, `--ring-depth`; also `tools/triangulate_check.py` |
| Morph steps: optional per-body fraction | `pipeline/morph_steps.py ... --part BODY=T` |
| Morph steps: optional rigid (similarity-only) bodies | `--rigid BODY` |
| Morph steps: --tag for short step folder names | `--tag NAME` |
| Optional Taubin skin smoothing after the fit | `fit_subject.py ... --smooth-skin [N]` |

Every option is off unless its flag is passed. Tests: 106 passed, 3 skipped.

## Open

1. Ask the lab to approve a finer tet mesh for the lip muscle (OrbicularisOris), tried on a copy of the scene. It is the one change in the First Capture Fit report nobody has tested.
2. Run the upper-lip lift step on Ziva (about 25 min). `..._t100_lift` is written and passed the physical check with warnings.
3. Test other lip-area muscles one at a time to find what holds the upper lip down.
4. For the lumps: record a slow head sweep in the app and fuse the frames into one surface. The app can record; the fusion is not built.

Not done: FLAME (skipped), and saving the raw burst frames in the app for a trimmed average.

## Not in this folder

The 432 MB scene file, the per-run Ziva scenes and the VM log files stay on the workstation.

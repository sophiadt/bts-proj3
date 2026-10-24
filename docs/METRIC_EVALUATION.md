# Metric evaluation — benchmark comparison brief

Self-contained. Someone with no access to this repo can read this and understand
the project, the design, and what remains open.

**Status: 2026-09-14.** Metric library complete and validated (**89 synthetic
tests**). Design settled on matched landmarks, scale-normalised (§3); the
baseline's own metrics turn out to be unusable here (§2b). Both configs
validate and `results/results_table.{md,csv}` plus `results/figures/` are
generated. Limitations and the retraction record live in the repo `README.md`
and in the per-row caveat block of `results/results_table.md`. FLAME's landmark embedding has landed, which unblocked curvature and
absolute aperture on the baseline — and corrected three earlier results (§6).

---

## 1. What this project is

A research-assistant task: build a **quantitative benchmark comparison** between
a lab biomechanical facial-animation model and a state-of-the-art learned model,
on the same spoken phrase.

**Model A — the lab model.** "Beneath the Skin: Interactive Biomechanical Facial
Simulations via Composable Co-Expressions" (25 pp, in `papers/`). An inside-out
pipeline: an artist sculpts a target pose, a gradient-descent solver minimises
Euclidean distance + elastic strain energy + muscle work (weights 0.340 / 0.330 /
0.330) to produce a deformation trajectory for the internal musculature, which
drives **1200 landmark joints** feeding a Ziva FEM simulation, and a registered
skin layer receives the result. Runs in Maya 2023 with Ziva VFX.

**Model B — the baseline.** **FaceDiffuser** (Stan et al. 2023, *Speech-Driven 3D
Facial Animation Synthesis Using Diffusion*, in `papers/`). A diffusion model that
maps speech audio to per-frame 3D vertex offsets. Chosen deliberately as an
independent SOTA comparison.

Test phrase: **"not even you"** (`notevenyou.m4a`), phoneme sequence
N AA T IY V N Y UW.

**We are not editing the lab paper.** This is a separate benchmark exercise.

---

## 2. Why this is worth doing

The lab paper quantifies its *own solver* — strain-energy profiles, Euclidean
distance to artist-sculpted targets (§4.4.1), skin registration convergence
(Fig. 14, average deviation 11.3 Maya units ≈ 1.09 cm). But its comparison
against a learned model is **entirely qualitative**. §4.4.1 says so outright:

> "we provide a qualitative comparison against a prominent learning-based
> technique via supplementary video evaluations"

§4.5 and Appendix J are three figures plus prose. **The paper contains no
quantitative results table at all** — Tables 1–6 are action units, biomechanical
parameters, a transition matrix, and an ablation reported as checkmarks with
"all five converged".

Meanwhile the prose makes claims that are physically falsifiable and unmeasured:

| Claim | Where |
|---|---|
| "volume preservation" | abstract, intro (3×, never measured) |
| learned methods "hallucinate geometry under stress, frequently resulting in an unnatural loss of volume and structural detail" | §5 |
| the lab model "maintains absolute structural integrity regardless of the extremity of the target pose" | §5 |
| baseline gives "merely a superficial, intersecting touch between the lips"; lab model "resolves the prominent compressive force of the lower lip pressing upward" | §4.5 |
| lower lip keeps a "distinct, tension-driven parabolic curve"; baseline "collapses into a flat, unnatural plateau" | §4.5, Appendix J |

Those are measurable. That is the whole exercise.

One caution: §4.5's figures compare against a *different* learned model
(Media2Face, Zhao et al. 2024). **Never claim to quantify Figs. 15/16/18.** Test
the prose claims, which are stated about learned methods generally, and always
name FaceDiffuser explicitly.

---

## 2b. The baseline's own metrics are unusable here — and that is the point

FaceDiffuser evaluates with **LVE**, **MVE**, **FDD** and **Diversity** (§6.1 of
its paper). The first three are the standard protocol in this literature,
inherited from FaceFormer / CodeTalker / FaceXHuBERT. All three are
**reference-based**:

> "**MVE**. Mean vertex error measures the deviation of all the face vertices of
> a sequence **with respect to the ground truth** by computing the maximal L2
> error for each frame and by taking a mean over all corresponding frame pairs."
>
> "**LVE**. Lip vertex error is identical to MVE. We only consider the lip
> vertices."
>
> "**FDD**. It measures the variation of facial dynamics for a motion sequence
> **against ground truth**."

**There is no ground truth for this comparison.** "Not even you" has no 4D scan,
and even if it did, it would not be registered to a stylized character's topology
or to FLAME. So LVE / MVE / FDD cannot be computed for either model, and the
fourth metric (Diversity) needs multiple subject-conditioned samples we do not
have.

Two consequences:

1. **The comparison must be reference-free.** That is not a compromise forced by
   missing data — it is the only kind of evaluation available for a stylized
   character, which is precisely the case the lab model is built for.
2. **This is the strongest argument for the whole exercise.** The standard
   protocol in speech-driven facial animation cannot evaluate physical
   plausibility at all: it measures distance to a recorded human. A model can
   score well on LVE while losing volume, self-intersecting, and stretching
   tissue past anything anatomical. Reference-free physical metrics measure the
   axis the standard protocol is blind to.

Where their machinery is still worth borrowing: the **vertex sets** (lip region,
upper-face region) and the **statistics** (max L2 per frame, standard deviation
of motion). Reframed as descriptive quantities rather than errors — "lip
displacement" instead of "lip vertex error", "upper-face motion variation"
instead of FDD — they are computable on both models and legible to anyone who
knows the literature.

---

## 3. Comparison design

### The naive design, and why it fails

The obvious approach is to compare the lab model's exported meshes against
FaceDiffuser's exported meshes directly. That runs into three hard walls, all
verified by inspection:

1. **FaceDiffuser's output surface is open.** It is FLAME topology (5023 verts,
   9976 faces) with **62 boundary edges in 2 loops** — the mouth interior (32
   verts) and the neck (30 verts). `enclosed_volume` on it returns 4.1×10⁶ mm³,
   which is garbage. Volume is undefined for an open surface, so the headline
   "volume preservation" claim cannot be tested this way.
   Closing it is not an option: the mouth loop's vertical extent runs 17.8 →
   27.1 mm across the take (**a 53% change**), so any cap grows with the
   deformation being measured.
2. **No teeth.** FLAME has none, so lip↔teeth interpenetration is not computable.
3. **No correspondence, no shared frame.** 5023 FLAME verts vs the lab's 10240
   muscle verts; FaceDiffuser centred near the origin, the lab head at
   y ≈ −82 mm. Nothing can be compared in absolute coordinates.

Under that design the only cross-model metric that runs is per-edge stretch, and
it is badly confounded (§5).

### Primary design: matched anatomical landmarks, scale-normalised

Measure the **same anatomical points on both models, in each model's own
geometry**, and normalise by each model's own facial scale. Landmarks need no
closed surface, no teeth and no vertex correspondence, so all three walls above
simply do not apply. This is the supervised recommendation and it is what ships.

What it supports on both models: displacement, velocity, acceleration, jerk,
straightness, lip aperture, mouth width, lip-contour curvature, lip separation
and contact fraction — each divided by that model's rest mouth width so a
stylized character and a FLAME identity are comparable.

What stays lab-only, stated as such rather than hidden: **volume change**
(the baseline surface is open, so volume is not merely unmeasured but undefined),
**lip↔teeth interpenetration** (no teeth in FLAME), and **passive settling
dynamics** (a per-frame denoiser has no mass or damping, so it has no free
response — legitimately asymmetric, not an unfair omission).

Normalisation is not optional. The two characters differ in proportion, the lab
model is stylized, and absolute millimetres across them would be meaningless.

### Optional extension: same rig, different driver

Built and tested, not on the critical path. It is the **only** route that makes
cross-model volume measurable, so it is worth knowing it exists.

**Drive the lab's own face model with FaceDiffuser's animation.** The lab
pipeline is landmark-driven: `beta_function` in
`h99-UI-mayascipt-V1_comsciV1.py:1077` sets absolute world positions on 1200
joints (`oor_*`, `ooc_l_*`, `ooc_r_*`, `frontalis_*`, 300 each) plus a jaw
rotation, then Ziva simulates. So if FaceDiffuser's motion is converted into
those landmark positions, the lab rig runs it.

Both animations then live on **identical lab geometry**, and every wall above
falls:

| Wall | Under the same-rig design |
|---|---|
| open surface | gone — lab meshes are closed (verified) → **volume works for both** |
| no teeth | gone — lab teeth present → **lip↔teeth works for both** |
| no correspondence | gone — same topology, same world frame |

This is the standard "same rig, different driver" ablation, and it is a
substantially stronger experiment: it isolates the *animation signal* from the
*rendering/simulation substrate*, which is exactly the comparison the paper's
prose claims are about.

`experimental/retarget.py` implements this geometrically, with no Ziva: a one-time
similarity alignment from matched landmarks (Umeyama, reflections suppressed so
left/right cannot silently swap), closest-point-on-surface correspondence with
barycentric weights fixed at rest, then per-frame **displacement** transfer —
`dst[t] = dst[0] + s·R·(src_surface_pt[t] − src_surface_pt[0])`. Only motion
crosses over; the lab model keeps its own rest shape.

Deliberately no Ziva in this path: routing the baseline through the lab solver
would make volume preservation partly the *solver's* achievement and destroy the
attribution. Vertices far from any source surface get a smooth falloff to zero
rather than extrapolated motion — a hard mask would create a seam that a stretch
metric reads as enormous fake deformation.

14 tests, including the two that matter: retargeting a mesh onto itself
reproduces the source motion exactly, and a rigidly moving source induces
p95 stretch of 1.0 on the target (no phantom deformation).

**Not on the sanctioned path.** It lives in `experimental/` per the supervisor's
steer -- get data from both models and measure landmarks, rather than implement
the paper's model onto the lab rig. The lab skin has since been exported, so the
blocker is gone; the reason it stays out is scope, not capability.

Scene: `v68 neutral v13.8 after housekeeping.mb` (repo root, 451 MB).
Animation script: `Eugene-lab-summer26/project code/.../h99-UI-mayascipt-V1_comsciV1.py`.

---

## 4. Verified facts

Lab capture, `data/lab/*.npz` — 92 frames (15–106) @ 30 fps, millimetres:

| Mesh | V | F | Topology |
|---|---|---|---|
| `middleres_OrbicularisOris` | 10240 | 20480 | closed, **genus 1** (ring muscle) |
| `middleres_Jaw` | 7426 | 14848 | closed, single shell |
| `middleres_TeethUp` / `Down` | 24608 | 49152 | closed, **16 shells each** |

Baseline, `data/baseline/facediffuser_notEvenYou_skin.npz` — 102 frames @ 30 fps,
FLAME, mm, open (see §3).

**Units.** 1 Maya unit = 0.962 mm. Source traced to the lab paper Appendix C:
*"the width of the mandible is 155.92 units, and the author's counterpart is
15 cm"* → 0.096203 cm/unit. Cross-checked: `middleres_Jaw` Z extent at frame 0 is
**exactly 155.92 units = 149.99 mm**. That confirms we exported the same mandible
the paper measured and that the export applied the stated scale — it does *not*
independently confirm 0.962, which rests on one author's single measurement.
Volume factor 0.962³ = **0.890277128** (was recorded as 0.890177 — a typo, fixed).

**Measured so far.** The numbers live in **`results/results_table.{md,csv}`**,
regenerated by `analysis/make_results_table.py`, and are not duplicated here —
an earlier copy of this section went stale against a re-export and disagreed with
the generated table by several percent on volume. Read the generated file.

Headline shape, as of the current export (rest = Maya frame 15, speech 16–106):

| Quantity | Lab | FaceDiffuser |
|---|---|---|
| volume change, OrbicularisOris | −9.42% … −1.35% | n/a — open surface |
| volume change, rigid bodies (Jaw, teeth) | < 4e-6 % (sanity check passes) | — |
| stretch p95, lip region | 1.672 | 1.345 |
| lip aperture, min … max | 2.82 … 17.59 mm | 1.44 … 7.11 mm |
| aperture peak-to-peak ratio | 2.16–2.38x lab | — |
| self-contact area fraction, Δ from rest | 0.0078 | 0.0388 |

---

## 5. Two traps in the numbers

**Stretch is confounded three ways** and must not be reported as a
who-wins metric yet:

1. *Surface mismatch* — the lab number is the orbicularis oris muscle, which
   genuinely deforms enormously; the baseline number is a whole FLAME head where
   most edges never move.
2. *Under-articulation* — the baseline articulates less on every measure
   (peak landmark displacement / rest width 0.122 vs 0.149). A model that barely
   moves scores well by doing nothing, which is why the articulation controls
   print adjacent to the stretch rows.
3. *"Ideal = 1" is wrong* — lip tissue really does stretch. 1.73 may be
   physiologically correct and 1.035 may be evidence of failing to deform.

Mitigations built: `stretch_ratios` takes a `vertex_mask` so p95 can be
restricted to a lip region (whole-mesh and lip-region are different
measurements), and `min_rest_len` drops tiny rest edges that dominate the raw
maximum. Report stretch as **descriptive**, alongside an articulation control,
unless a published facial-skin strain limit is supplied to test exceedance
against.

**Interpenetration at rest is 8.6% of lip volume.** Ziva rigs overlap tissue and
bone in the rest pose by design and hold contact with attachment constraints, so
absolute interpenetration is not a defect and not comparable across models. Use
change from rest (−959 … +417 mm³). This belongs in a limitations note, not a
results row: "ours 4544–5920 mm³, baseline n/a — no teeth" reads as the lab model
losing on a metric the baseline dodges by lacking anatomy.

---

## 6. What is built

`analysis/metrics.py` — pure Python, no Maya, importable and testable anywhere.
**89 synthetic tests pass** (`.venv\Scripts\python -m pytest analysis/test_metrics.py experimental/test_retarget.py -v`),
each against a known analytic answer.

| Function | Measures | Works on open surfaces? |
|---|---|---|
| `enclosed_volume` | mm³, divergence theorem; signed (winding diagnostic) | no — closed only |
| `volume_change_pct` | % vs frame 0, whole closed mesh | no |
| `stretch_ratios` | per-edge current/rest; `(max, p95)`; `vertex_mask`, `min_rest_len` | **yes** |
| `interpenetration_volume` | mm³ boolean overlap (trimesh + manifold) | no |
| `min_lip_separation_mm` | nearest-neighbour lip-to-lip distance | **yes** |
| `lip_contact_fraction` | fraction of lip verts in contact within ε | **yes** |
| `self_proximity_mm` | nearest non-adjacent lip vertex, no upper/lower split | **yes** |
| `self_contact_fraction` | **diagnostic only** — fraction of region *vertices* | **yes** |
| `self_contact_area_fraction` | **reportable** — fraction of region *area* | **yes** |
| `circle_curvature_3pt` | Menger curvature, exactly 1/R | **yes** |
| `lip_curvature` | lip-contour curvature per frame; `contour=` is explicit | **yes** |
| `load_flame_embedding` | FLAME iBUG-68 embedding → `(faces, bary)`, no torch | — |
| `boundary_loops` | ordered rim loops; `[]` if closed | — |
| `rigid_angular_velocity` | deg/s via Kabsch + quaternion magnitude | — |
| `velocity` / `acceleration` / `inertial_force` / `straightness` | landmark kinematics | **yes** |
| `vertices_within_mm` | reproducible region mask | **yes** |
| `validate_config` | landmark/region config vs the data | — |

Validated behaviours worth knowing: volume is exact on a cube and a regular
tetrahedron, correct on a **torus** (genus 1, as the lip muscle is) and on
**multi-shell** meshes (as the teeth are), and invariant to a 10⁴ mm
translation. Curvature is exactly 1/R and → 0 for a flat contour, which is
precisely the "parabolic curve vs flat plateau" claim. `rigid_angular_velocity`
uses a quaternion path because `arccos((tr R − 1)/2)` leaks ~5×10⁻⁵ deg/s of
spurious rotation on a purely translating mesh.

**Also done:** `config/units.txt` authoritative source filled in;
`export/export_meshes.py` now exports the skin layer
(`skin_cloth_copy_v5_pull_back`, live mesh not `_RENDER`), captures from
**frame 0** rather than 15, drops missing meshes with a warning instead of dying
mid-walk, and reports boundary-edge counts at export so closure is known before
anything downstream depends on it.

**Now built:** `config/landmarks_baseline.json` (rebuilt by
`analysis/build_baseline_config.py`), `analysis/make_results_table.py`,
`analysis/speech_window.py`, and the repo `README.md`.

Both FLAME assets have since landed. `landmark_embedding.npy` gives the baseline
citable iBUG-68 landmarks, stored **barycentrically** because the embedding places
them inside triangles — nearest-vertex snapping would move the landmarks used
here by up to 2.50 mm, larger than the 1.44 mm minimum aperture they measure.
Reading the
file needs **no torch**: it is an object-dtype `.npy` whose pickle stream is read
with a `find_class` that stubs `torch.*`, and the two entries actually used are
type-checked as real numpy arrays. `FLAME_masks.pkl` supplies the 254-vertex
`lips` cross-check and the `upper_face` region.

That unblocked lip curvature and gave the baseline a genuine inner-lip contact
pair. It also surfaced three errors in what was already built — a lip-region
radius licensed by a coincidence, a contact fraction with a tessellation-dependent
denominator, and a curvature fallback that would have paired different contours
across models. All three are fixed; see the Limitations and Retraction record
sections of the repo `README.md`, and the caveats on the affected rows.

**Two figures reported earlier are retracted:** the contact-fraction gap and the
1.43x aperture ratio's "upper bound" caveat (the bound was **backwards**; the
corrected ratio is 2.16-2.38x). The contact-fraction gap was originally reported
as **17x** (vertex-count fraction, under the superseded absolute-radius lip
region). The same vertex-count metric now reads **15.3x** under the current
relative-radius region -- a different number because the region definition
changed, not a second retraction. The reportable figure is the area-based one,
**5.0x**; vertex count over-states contact on both models, by different factors,
because tessellation density differs between them (see `metrics.py` around line
992). Neither figure self-corrects outside this repo.

Also `experimental/retarget.py` is built and tested but stays out of the
sanctioned path.

---

## 7. Free result nobody has looked at

Frames 0–14 were discarded throughout as "gravity settling, not signal". They are
a **damped free response**: gravity switching on at frame 0 is an impulse, and
the tissue falls, overshoots, rebounds and settles before speech begins around
frame 17. That is the same signal the shake test was built to produce, and the
rig emits it on every run.

The export now captures them. Extractable per landmark: overshoot ratio (peak /
final displacement), settle time (frames to stay within 5% of final), and damping
ratio ζ from log decrement — **only** if two successive same-direction peaks
exist. Caveat: 15 frames at 30 fps is 0.5 s, so resolution is thin, and a single
clean overshoot gives overshoot and settle time but not ζ. This does not replace
a controlled shake test with chosen amplitude and direction; it de-risks it by
showing the response shape in advance.

Passive dynamics is a lab-model-only result, legitimately — a per-frame denoiser
has no mass and no damping, so it has no free response to measure. The
cross-model version of this idea is jerk magnitude / high-frequency spectral
energy over the speech frames: mass and damping low-pass filter motion, so
simulated tissue should be bandlimited where a learned model need not be.

---

## 8. Environment

Windows, Maya 2023 + Ziva VFX (remote lab machine) for anything in `export/`.
Everything in `analysis/` is plain Python:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # numpy trimesh manifold3d scipy pytest
.venv\Scripts\python -m pytest analysis/test_metrics.py -v
```

The boolean engine is registered as **`manifold`** in trimesh (the pip package is
`manifold3d`); `metrics.py` resolves the name at import rather than hardcoding
it. **`librosa` is deliberately not installed** — `analysis/speech_window.py`
uses `wave` + numpy, because the dependency was judged not worth a partial
answer when it would still not give phoneme alignment.

```
analysis/     metrics.py, test_metrics.py, speech_window.py,
              build_baseline_config.py, make_results_table.py    pure Python
export/       export_meshes.py, query_ziva_materials.py,
              passive-dynamics scripts               needs Maya 2023 + Ziva
config/       landmarks_lab.json, landmarks_baseline.json, units.txt
data/lab/     5 npz (incl. the skin layer)
data/baseline/facediffuser_notEvenYou_skin.npz, landmark_embedding.npy,
              FLAME_masks/
results/      results_table.{md,csv}, figures/
papers/       both papers
```

To regenerate everything from scratch:

```
.venv\Scripts\python -m pytest analysis/test_metrics.py experimental/test_retarget.py -q
.venv\Scripts\python analysis/build_baseline_config.py
.venv\Scripts\python analysis/make_results_table.py
```

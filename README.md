# bts-proj3
Automated 3D Asset Generation for Physically Based Facial Animation
# Project 3: Automated 3D Asset Generation for Physically Based Facial Animation

Description: Generating personalized, physically accurate 3D face models for animation currently requires
tedious manual setup and topological tweaking. This project explores how to fully automate the construction
of personalized facial assets to streamline physically based facial animation. The student will investigate two
distinct initialization pathways. First, they will develop a lightweight vision pipeline using smartphone video
and landmark tracking (e.g., MediaPipe) to infer rough skull dimensions, rest muscle anchor points, and
personalized skin mesh boundary initialization. Second, they will evaluate a highly robust alternative:
automating the generation of these boundaries directly from cross-individual volumetric scans (like MRI).
The objective is to establish a reliable, automated starting point for facial meshes before the physical solver
takes over.

To achieve this, the student will build a prototype initialization tool—a script that ingests video or scan data
and outputs personalized muscle anchors and skin boundaries. Alongside this tool, they will produce a
comparative analysis evaluating the accuracy, computational cost, and accessibility of both approaches. The
project will culminate in a technical report detailing the registration pipeline, its success rate, and its
viability for scaling personalized avatars.

Why it teaches research: This project introduces core concepts in computer vision, geometric processing,
and automated pipeline design. Students will tackle the ambiguities of monocular depth estimation and
compare the trade-offs of accessible RGB video versus highly accurate volumetric data, learning how to
translate raw inputs into simulation-ready 3D assets.

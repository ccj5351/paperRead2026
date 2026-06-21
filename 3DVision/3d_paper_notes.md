# 3D Vision Paper Notes

Papers covered, in chronological order:
1. DUSt3R (CVPR 2024)
2. MASt3R (2024, NAVER LABS)
3. VGGT (CVPR 2025)
4. VGGT-Ω (CVPR 2026)

---

## 1. DUSt3R: Geometric 3D Vision Made Easy — CVPR24

**Why / What.** Classical 3D reconstruction (SfM + MVS) requires known/estimated camera intrinsics and extrinsics before dense geometry can be triangulated, and the multi-stage pipeline (matching → pose → triangulation → BA) is brittle and error-propagating. DUSt3R reframes pairwise 3D reconstruction as direct regression from two uncalibrated, unposed images to a dense pointmap, from which cameras, depth, correspondences, and full 3D reconstruction can all be recovered without ever explicitly solving for camera parameters.

**How**
- Two input images are encoded by a shared-weight ViT (Siamese encoder), then processed by two transformer decoders that continuously exchange information via cross-attention so each branch can reason about the other view's geometry.
- Two regression heads output a dense pointmap and confidence map per view, both expressed in the coordinate frame of the *first* image, trained end-to-end with a simple confidence-weighted 3D regression loss (no explicit geometric constraints).
- For scenes with more than two images, a fast global alignment step optimizes per-pair rigid transforms and scale factors to stitch all pairwise pointmaps into one consistent coordinate frame, replacing classical bundle adjustment with direct 3D-space optimization (much faster, converges in a few hundred gradient steps).

<div align="center">
    <img src="figures/dust3r_arch_fig2.png" alt="DUSt3R Architecture" width="800">
    <p><em>Figure: DUSt3R architecture.</em></p>
</div>

**Pros and Cons**:
- **Pros:** removes the need for known/calibrated cameras entirely, gracefully degrades to monocular depth when only one view is informative, and the learned confidence map gives a built-in signal for which regions to trust, all from a single feed-forward pass. 
- **Cons:** the model is fundamentally pairwise, so scenes with more than two images need a separate, slower global alignment optimization step to merge pointmaps into one frame, and reconstruction quality/scale consistency degrades for image pairs with little visual overlap or large viewpoint gaps.

**Q&As**
Q1: Training Loss Understanding

<div align="center">
    <img src="figures/dust3r_eq4_loss.png" alt="DUSt3R Eq. 4 confidence-aware loss" width="400">
    <p><em>Figure: DUSt3R confidence-aware regression loss (Eq. 4).</em></p>
</div>

$\mathcal{L}\_{\text{conf}}$ sums, over both views and all pixels with valid ground truth, the per-pixel regression loss $\ell\_{\text{regr}}$ scaled by a predicted confidence $C_i^{v,1}$, minus $\alpha \log C_i^{v,1}$.

The second term is the regularizer that keeps the confidence weighting honest. Without it, the network could trivially minimize the first term by driving every $C_i^{v,1} \to 0$, since a zero weight zeroes out any regression error regardless of how wrong the prediction is. The $-\alpha \log C$ term grows without bound as $C \to 0$ (recall $C > 1$ by construction, so $\log C > 0$ and this term is always negative, rewarding *larger* C), so pushing confidence down has a cost. The network therefore only lowers confidence on pixels where the regression error genuinely outweighs that cost, i.e., on truly ambiguous or ill-constrained regions (e.g. areas visible in only one view), while keeping confidence high (and thus the regression loss fully weighted) everywhere else. This is the same idea as heteroscedastic/aleatoric uncertainty weighting in regression: jointly learning a per-sample precision and a log-precision penalty prevents the degenerate "infinite confidence, zero loss" solution.


---

## 2. MASt3R: Grounding Image Matching in 3D with MASt3R — 2024

**Why / What.** Image matching is conventionally treated as a 2D pixel-correspondence problem, but matching is fundamentally a 3D problem linked to camera pose and scene geometry. DUSt3R's pointmap regression is surprisingly robust to extreme viewpoint changes but imprecise for matching since it was never trained explicitly for that task; MASt3R augments DUSt3R to deliver dense, pixel-accurate, robust correspondences.

**How**
- MASt3R keeps the DUSt3R backbone (Siamese ViT encoder + cross-attention transformer decoders) and adds a second lightweight head that regresses dense local feature maps per image, trained with an InfoNCE matching loss (a classification-style objective rewarding exact pixel correspondence, unlike the regression loss used for pointmaps).
- Exhaustive reciprocal nearest-neighbor matching over dense feature maps is O(W²H²) and too slow, so the paper introduces a fast reciprocal matching algorithm that iteratively propagates and verifies cycle-consistent matches from a sparse pixel subset, cutting cost to O(kWH) with no loss (in fact a gain) in accuracy.
- A coarse-to-fine scheme (match downscaled images first, then refine within high-resolution window crops) extends this to high-resolution images despite ViT's quadratic attention cost.

<div align="center">
    <img src="figures/mast3r_arch_fig2.png" alt="MASt3R Architecture" width="800">
    <p><em>Figure: MASt3R architecture.</em></p>
</div>

**Pros and Cons**:
- **Pros:** delivers dense, pixel-accurate correspondences directly from the DUSt3R backbone with minimal added overhead, and the fast reciprocal matching algorithm makes dense matching tractable at high resolution without sacrificing accuracy.
- **Cons:** still inherits DUSt3R's pairwise-only formulation, so it needs the same external global alignment to scale beyond two views, and adding the matching head and InfoNCE loss increases training complexity and cost relative to the base model.

---

## 3. VGGT: Visual Geometry Grounded Transformer — CVPR25

**Why / What.** DUSt3R/MASt3R-style models only process image pairs and need post-hoc global alignment/optimization to scale to many views, which is costly. VGGT asks whether a single feed-forward network—essentially a plain large transformer with minimal 3D-specific inductive bias—can directly predict *all* key 3D attributes (camera parameters, depth maps, point maps, point tracks) for one to hundreds of images in a single forward pass, outperforming optimization-heavy alternatives even without any post-processing.

**How**
- Each image is patchified into tokens via a frozen-ish DINO backbone; a learnable camera token and register tokens are appended per frame.
- These tokens pass through L=24 alternating-attention (AA) blocks that interleave frame-wise self-attention (within each image) and global self-attention (across all frames), avoiding any cross-attention.
- A lightweight camera head (self-attention + linear layers) decodes camera tokens into intrinsics/extrinsics, while a shared DPT head decodes dense image tokens into depth maps, point maps, and tracking features (the latter feeding a CoTracker2-based tracking module).
- Trained end-to-end with a multi-task loss (camera + depth + point-map + tracking losses); ablations show this redundant multi-task supervision and the AA design both meaningfully boost accuracy versus single-task or cross-attention/global-only variants.
- Predictions can optionally be refined with classic Bundle Adjustment for extra accuracy at modest extra cost.

<div align="center">
    <img src="figures/vggt_arch_fig2.png" alt="VGGT Architecture" width="800">
    <p><em>Figure: VGGT architecture.</em></p>
</div>

**Pros and Cons**:
- **Pros:** a single feed-forward pass handles anywhere from one to hundreds of images with no post-hoc optimization required, and the multi-task supervision (camera, depth, point map, tracking) acts as a strong regularizer that improves each individual task.
- **Cons:** the alternating-attention design over all frames scales in memory/compute with the number of views, and matching the very best accuracy still benefits from an optional, more expensive Bundle Adjustment refinement step.

---

## 4. VGGT-Ω — CVPR26

**Why / What.** VGGT-Ω studies whether feed-forward reconstruction models scale predictably with model and data size (like LLMs/vision foundation models), and whether this scaling can be pushed further while also extending coverage to dynamic (non-rigid) scenes and reducing training cost — VGGT-Ω shows a consistent power-law-like improvement in reconstruction accuracy from 0.2B→10B parameters and thousands→millions of training sequences.

**How**
- Keeps VGGT's per-frame camera/scene "register" tokens but introduces **register attention**: in 25% of the global-attention layers, cross-frame information exchange is restricted to just the registers (rather than all tokens), which then redistribute information back into per-frame tokens during frame-wise attention — this cuts memory/FLOPs substantially (registers act as an information bottleneck) with no measurable performance loss.
- The trained registers turn out to carry reusable global scene information, useful for VLA/language-alignment tasks beyond reconstruction.
- The architecture is simplified by dropping point-map and tracking heads in favor of a single dense depth head plus a sparse camera head (multi-task *losses* are kept, multi-task *heads* are not).
- Memory-hungry high-resolution DPT convolutional layers are replaced with a lightweight MLP + pixel-shuffle upsampler.
- Training scale is enabled by a new annotation pipeline (VLM pre-filtering + VGGT + COLMAP + image matchers + geometric post-filtering) that mines ~40M internet videos down to 0.8M accurately annotated sequences (including dynamic content).
- A DINO-style self-supervised teacher-student protocol additionally exploits 18M unlabeled videos.

<div align="center">
    <img src="figures/vggtomega_arch_fig2.png" alt="VGGT-Ω Architecture" width="800">
    <p><em>Figure: VGGT-Ω architecture.</em></p>
</div>

**Pros and Cons**:
- **Pros:** register attention cuts memory/FLOPs substantially with no measurable accuracy loss, the model scales predictably (power-law-like) with parameters and data, and it extends coverage to dynamic scenes while also being cheaper to train than VGGT.
- **Cons:** dropping the point-map and tracking heads trades off some task-specific outputs for efficiency, and the gains depend heavily on the new large-scale mined annotation pipeline, which adds significant data-engineering complexity outside the model itself.

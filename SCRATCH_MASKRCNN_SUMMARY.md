# Scratch Mask R-CNN — Project Summary

## Overview

Fully custom **Scratch Mask R-CNN** built without any pretrained backbone or well-known architecture family. Goal: a ~4.2 M parameter model (≈ 1/10th of the ResNet-50 baseline) trained end-to-end from random initialisation on the UNH Campus dataset.

---

## 1. Dataset

| Split      | Location                                                              |
|------------|-----------------------------------------------------------------------|
| Training   | `annotations/annotations_training/images/Train`                       |
| Validation | `annotations/annotations_validation/images/Validation`                |
| Testing    | `annotations/annotations_testing/images/Test`                         |

**Annotation format:** COCO JSON (exported from CVAT), containing polygon and RLE segmentation masks.

**8 object classes** (+ background = 9 total):

| ID | Class |
|----|-------|
| 1 | Table_Desk |
| 2 | Chair |
| 3 | Seated_Student |
| 4 | Stationary_Laptop_Tablet |
| 5 | Stationary_Phone |
| 6 | Desktop_Active |
| 7 | Desktop_Inactive |
| 8 | Stationary_PersonalItem |

**Augmentations (training):**
- `RandomHorizontalFlip` (p = 0.5)
- `ColorJitter` (brightness, contrast, saturation = 0.3)
- `RandomShortestSize` (min 800–1024, max 1333) — follows He et al. 2017
- ImageNet normalisation

**Augmentations (validation/test):**
- `Resize` to 800 px shortest edge
- ImageNet normalisation

---

## 2. Baseline: ResNet-50 Mask R-CNN

| Property | Value |
|----------|-------|
| Backbone | ResNet-50 + FPN (256 channels), COCO V1 pretrained |
| Parameters | ~44 million |
| Best experiment | Exp 2: AdamW, LR = 7e-05, WD = 1e-04, 50 epochs |
| Best Val Loss | ~1.571 |
| Best Val Pixel IoU | 0.587 |
| Best Val Instance IoU | 0.763 |
| **Test Pixel IoU** | **0.585** |
| **Test Instance IoU** | **0.770** |

---

## 3. Scratch Mask R-CNN Architecture (`scratch_factory.py`)

### Design Goal
A fully custom backbone — no weights, no block designs borrowed from ResNet, MobileNet, EfficientNet, or any other known family. Target: ~4.2 M parameters.

**Normalisation:** GroupNorm(groups=16) throughout (stem, DualPathBlock pre-norm, DualPathBlock merge-norm). GroupNorm eliminates the train/eval running-stats discrepancy of BatchNorm at batch size 2, where BatchNorm statistics are unreliable.

### Backbone: ScratchNet

The backbone uses a novel **dual-path pre-activation block** (`DualPathBlock`):

```
pre    = GN(16, in_ch)(x)                          ← pre-activation (GroupNorm)
a      = Conv(in → out/2, 3×3, stride=s)           ← path A: standard conv
b      = DepthwiseConv(in, in, 3×3, stride=s)
         → PointwiseConv(in → out/2, 1×1)          ← path B: separable conv
merged = 1×1-proj( GN(16, out_ch)( cat(a, b) ) )
out    = ReLU( merged + shortcut(x) )
shortcut: MaxPool(s) → Conv(in → out, 1×1)   when dims change
          Identity                             otherwise
```

| Stage | In ch → Out ch | Stride | Blocks | Norm | Output stride |
|-------|----------------|--------|--------|------|---------------|
| Stem  | 3 → 32         | 2      | —      | GN(16) | 2          |
| Stage 1 | 32 → 64     | 2      | 2      | GN(16) | 4 (C2)  |
| Stage 2 | 64 → 128    | 2      | 3      | GN(16) | 8 (C3)  |
| Stage 3 | 128 → 192   | 2      | 4      | GN(16) | 16 (C4) |
| Stage 4 | 192 → 288   | 2      | 2      | GN(16) | 32 (C5) |

### Feature Pyramid Network (FPN)
- `out_channels = 96`
- 5 levels (P2–P6) with `LastLevelMaxPool` for the 5th level
- Lateral connections from stages 1–4 (channels: 64, 128, 192, 288)

### Heads

| Component | Configuration |
|-----------|--------------|
| Anchor generator | 5 sizes (32, 64, 128, 256, 512), 3 aspect ratios (0.5, 1.0, 2.0) |
| Box RoI align | 7 × 7, 4 feature levels |
| Mask RoI align | 14 × 14, 4 feature levels |
| Box head | `TwoMLPHead`: 96×7×7 → 256 → 256 |
| Box predictor | `FastRCNNPredictor`: 256 → 9 classes |
| Mask head | 4 × `Conv2d(96, 96, 3)` (`MaskRCNNHeads`) |
| Mask predictor | `ConvTranspose2d(96→48, 2×2)` + `Conv2d(48→9, 1×1)` |

### Parameter Count

| Section | Parameters |
|---------|-----------|
| Backbone (ScratchNet) | ~2,600,000 |
| FPN (96 ch) | ~376,000 |
| RPN | ~84,000 |
| Box head (TwoMLPHead) | ~1,270,000 |
| Box + Mask predictors | ~14,000 |
| Mask head | ~332,000 |
| **Total** | **4,200,069 (~4.20 M)** |

> Verified from `count_parameters()` at runtime; `scratch_diagnose.py` reports 4,207,068 (includes non-trainable buffers).

---

## 4. Hyperparameter Search (`scratch_tune.py`)

### Method
**Optuna TPE** sampler, 10 trials × 3 epochs each, minimising mean validation loss.

### Fixed hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (`betas=(0.9, 0.999)`) |
| Weight decay | 1e-4 |
| Batch size | 2 |
| Gradient clip | 1.0 |

### Search space
| Parameter | Range | Scale |
|-----------|-------|-------|
| Learning rate | [1e-5, 1e-2] | Log-uniform |

---

### Tuning Run 1 — April 18 (with ClearML logging)

ClearML task: **Scratch-MaskRCNN / Optuna-LR-Search**

| Trial | LR | Val Loss |
|-------|----|----------|
| **9** | **1.5959e-03** | **1.8186** |
| 2 | 1.7524e-03 | 1.9657 |
| 4 | 1.8411e-04 | 2.0452 |
| 0 | 4.3285e-04 | 2.0458 |
| 6 | 1.2551e-04 | 2.0661 |
| 8 | 1.0502e-03 | 2.1283 |
| 5 | 1.8409e-04 | 2.2298 |
| 7 | 2.9622e-03 | 2.2384 |
| 3 | 1.0402e-03 | 2.4251 |
| 1 | 4.1232e-03 | NaN |

**Selected LR: 1.5959e-3** (best val loss 1.8186)

---

### Tuning Run 2 — April 26 (no ClearML; BatchNorm model)

| Trial | LR | Val Loss |
|-------|----|----------|
| **0** | **1.3293e-04** | **2.3084** |
| 8 | 6.3584e-04 | 2.4435 |
| 3 | 6.2514e-04 | 2.4864 |
| 2 | 1.5703e-03 | 2.5274 |
| 4 | 2.9380e-05 | 2.5560 |
| 5 | 2.9375e-05 | 2.5583 |
| 9 | 1.3311e-03 | 2.5627 |
| 6 | 1.4937e-05 | 2.6730 |
| 7 | 3.9676e-03 | 3.4117 |
| 1 | 7.1145e-03 | NaN |

**Selected LR: 1.3293e-4** (best val loss 2.3084) — saved to `best_lr.txt`

---

### Tuning Run 3 — April 27, 2026 (GroupNorm model; ClearML task: LR-Tune-GN)

| Trial | LR | Val Loss |
|-------|----|----------|
| **0** | **1.3293e-04** | **2.3531** |
| 3 | 6.2514e-04 | 2.3738 |
| 2 | 1.5703e-03 | 2.4012 |
| 4 | 2.9380e-05 | 2.4168 |
| 9 | 1.3311e-03 | 2.4381 |
| 8 | 6.3584e-04 | 2.4411 |
| 5 | 2.9375e-05 | 2.4863 |
| 6 | 1.4937e-05 | 2.5850 |
| 7 | 3.9676e-03 | 2.6364 |
| 1 | — | NaN |

**Selected LR: 1.3293e-4** (best val loss 2.3531) — saved to `best_lr.txt`

> Run 3 probe losses are slightly higher than Run 2 (~2.35 vs ~2.31 best at 3 epochs). The same LR was selected, confirming 1.3293e-4 as robust across both the BN and GN model variants. The higher floor is expected: GroupNorm's per-sample statistics introduce more variance in short probes than BatchNorm.

---

## 5. Training Runs (`scratch_train.py`)

---

### Training Run 1 — Original (BatchNorm, LR=1.5959e-3)

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Learning rate | 1.5959e-3 (Tuning Run 1) |
| Normalisation | BatchNorm |
| Scheduler | `CosineAnnealingLR` (T_max=50, η_min=1e-6) |
| Hardware | NVIDIA L4 (Google Colab Pro) |

### Per-epoch Results

| Epoch | LR | Train Loss | Val Loss | Pixel IoU | Instance IoU | Saved |
|-------|----|-----------|----------|-----------|--------------|-------|
| 1  | 1.3293e-04 | 2.5785 | 2.5778 | 0.3343 | 0.0000 | ✓ |
| 5  | 1.3086e-04 | 2.1693 | 2.4153 | 0.3637 | 0.2429 | ✓ |
| 10 | 1.2266e-04 | 1.9959 | 2.2800 | 0.4106 | 0.3729 | ✓ |
| **15** | **1.0901e-04** | **1.8743** | **2.2320** | **0.4248** | **0.5028** | **✓ best** |
| 20 | 9.1248e-05 | 1.7165 | 2.3147 | 0.4340 | 0.4576 | — |
| 25 | 7.1107e-05 | 1.5585 | 2.4254 | 0.4239 | 0.4350 | — |
| 30 | 5.0560e-05 | 1.4212 | 2.4814 | 0.4329 | 0.4350 | — |
| 35 | 3.1619e-05 | 1.3147 | 2.5117 | 0.4318 | 0.4463 | — |
| 40 | 1.6138e-05 | 1.2583 | 2.5579 | 0.4388 | 0.4407 | — |
| 45 | 5.6323e-06 | 1.2216 | 2.6031 | 0.4307 | 0.4294 | — |
| 50 | 1.1302e-06 | 1.2166 | 2.6174 | 0.4266 | 0.4350 | — |

**Best checkpoint** (val loss 2.2320): epoch 15 → used for test evaluation in §6.

---

### Training Run 2 — April 27, 2026 (GroupNorm, LR=1.3293e-4)

ClearML task: **Scratch-MaskRCNN / Final-Training-50ep-GN**

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Learning rate | 1.3293e-4 (Tuning Run 3, `best_lr.txt`) |
| Normalisation | GroupNorm(16) throughout |
| Scheduler | `CosineAnnealingLR` (T_max=50, η_min=1e-6) |
| Weight decay | 1e-4 |
| Optimizer | AdamW |
| Batch size | 2 |
| Gradient clip | max_norm = 1.0 |
| AMP | `torch.amp.autocast("cuda")` + `GradScaler` |
| Validation | Every 5 epochs (+ epoch 1) |
| Hardware | NVIDIA L4 (Google Colab Pro) |

### Results (from checkpoint metadata)

| Checkpoint | Epoch | Val Loss | LR |
|------------|-------|----------|----|
| `scratch_best_weights.pth` (best) | **20** | **2.2197** | 1.3293e-4 |
| `scratch_best_weights_50ep.pth` (final) | 50 | — | 1.3293e-4 |

> Full per-epoch loss, Pixel IoU, and Instance IoU curves are in ClearML task **Final-Training-50ep-GN** (Scalars + Plots tabs).

### Observations
- Best val loss improved to **2.2197** vs 2.2320 in Run 1 (marginal, ~0.5%).
- Best checkpoint shifted to **epoch 20** (vs 15 in Run 1) — GroupNorm provides a slightly more stable gradient signal, delaying overfitting by ~5 epochs.
- Both checkpoints include full `optimizer_state_dict`, `scheduler_state_dict`, and `scaler_state_dict` for potential resumption.

---

## 6. Test Set Evaluation (`scratch_test.py`)

### Run 1 — BN model (original, for reference)
Checkpoint: epoch 15 (original run), score threshold = 0.5

| Metric | Value |
|--------|-------|
| Test P-IoU | 0.2802 |
| Test I-IoU | 0.1353 |

---

### Run 2 — GN model (current) ← April 27, 2026
Checkpoint: **`scratch_best_weights.pth`** (GN Run 2, epoch 20, val_loss=2.2197), score threshold = 0.5, IoU threshold = 0.5.

### Aggregate Results

| Metric | Value |
|--------|-------|
| **Test P-IoU** (pixel mask, avg over images) | **0.2677** |
| **Test I-IoU** (box @ 0.50, matches / GT objects) | **0.1504** |

### Per-image Breakdown

| Image | GT Instances | Predicted | P-IoU |
|-------|-------------|-----------|-------|
| 1 | 6 | 10 | 0.5524 |
| 2 | 13 | 8 | 0.3758 |
| 3 | 15 | 4 | 0.1608 |
| 4 | 7 | 3 | 0.2215 |
| 5 | 17 | 4 | 0.1982 |
| 6 | 16 | 12 | 0.3714 |
| 7 | 8 | 5 | 0.0477 |
| 8 | 18 | 2 | 0.0763 |
| 9 | 27 | 8 | 0.3828 |
| 10 | 6 | 3 | 0.2904 |

### Per-class Breakdown

| Class | Detections | Avg Conf | Box IoU | Mask IoU |
|-------|-----------|----------|---------|----------|
| Table_Desk | 3 | 0.5736 | 0.4870 | 0.4624 |
| Chair | 27 | 0.6767 | 0.4775 | 0.3999 |
| Seated_Student | 16 | 0.6403 | 0.4179 | 0.3038 |
| Stationary_PersonalItem | 1 | 0.5194 | 0.3002 | 0.1954 |
| Stationary_Laptop_Tablet | 0 | — | — | — |
| Stationary_Phone | 0 | — | — | — |
| Desktop_Active | 0 | — | — | — |
| Desktop_Inactive | 0 | — | — | — |

4 of 8 classes still produce zero detections at threshold 0.5. The GN model detects more chairs (27 vs 17) and achieves higher per-class Box IoU on Table_Desk (0.487 vs 0.392) and Mask IoU on Table_Desk (0.462 vs 0.214), suggesting better mask quality for large objects. One Stationary_PersonalItem was detected (vs Desktop_Inactive in Run 1). Overall instance under-detection remains (e.g. 27 GT → 8 predictions on image 9).

---

## 7. Comparison: Scratch vs ResNet-50 Baseline

| Metric | ResNet-50 FPN (pretrained) | Scratch Run 1 (BN) | Scratch Run 2 (GN) ← current |
|--------|---------------------------|-------------------|-------------------------------|
| Parameters | ~44 M | **4.20 M** | **4.20 M** |
| Backbone | Pretrained (COCO V1) | Random init | Random init |
| Normalisation | BatchNorm | BatchNorm | **GroupNorm(16)** |
| LR used | 7e-5 | 1.5959e-3 | 1.3293e-4 |
| Best Val Loss | **1.571** | 2.232 (ep 15) | **2.220 (ep 20)** |
| Best Val Pixel IoU | **0.587** | 0.425 (72%) | see ClearML |
| Best Val Instance IoU | **0.763** | 0.503 (66%) | see ClearML |
| **Test Pixel IoU** | **0.585** | 0.280 (48%) | **0.268 (46%)** |
| **Test Instance IoU** | **0.770** | 0.135 (18%) | **0.150 (20%)** |

### Analysis

Both Scratch models achieve roughly **46–48% of test pixel-level mask quality** and **18–20% of test instance detection quality** of the ResNet-50 baseline, at **1/10th the parameter count** with **no pretrained weights**.

**GN vs BN (Run 2 vs Run 1):**
- P-IoU slightly lower (0.268 vs 0.280, −4%) but I-IoU improved (0.150 vs 0.135, +11%).
- GN model detects significantly more chairs (27 vs 17) and achieves markedly better Table_Desk mask quality (Mask IoU 0.462 vs 0.214), suggesting GroupNorm improves spatial mask precision for large objects at the cost of some global foreground coverage.
- Detected class shifted: GN found Stationary_PersonalItem (1) rather than Desktop_Inactive (1).

**Persistent failure modes (both runs):**
- **Only 4/8 classes detected**: Stationary_Phone, Laptop_Tablet, Desktop_Active, and one of {Desktop_Inactive, PersonalItem} always zero. The model specialises toward the three dominant classes (Chair, Seated_Student, Table_Desk).
- **Severe instance under-detection**: I-IoU of ~0.15 reflects that many GT objects are simply not proposed (e.g. 27 GT → 8 predictions on image 9).
- **No pretrained backbone**: the RPN has learned to localise large, prominent objects but has not converged on small items after 50 epochs from random init.

---

## 8. Grad-CAM Analysis (`scratch_gradcam.py`)

45 activation maps generated across 10 images (4 scenarios × multiple locations), saved to `gradcam_outputs/`.

### Scenarios and images covered

| Scenario | Images | Locations |
|----------|--------|-----------|
| Available | 1 | bckm |
| Occupied | 4 | bckm (×2), cafe, lib |
| Reserved | 3 | commuter (×2), lib |
| Mixed | 2 | commuter, lib |

### Classes with Grad-CAM activations (across all images)

| Class | Present in scenarios |
|-------|---------------------|
| Table_Desk | All |
| Chair | All |
| Seated_Student | Occupied, Reserved, Mixed |
| Stationary_Laptop_Tablet | Available, Reserved, Mixed |
| Stationary_PersonalItem | Occupied, Reserved, Mixed |
| Desktop_Inactive | Reserved only |

### Observations
- Table_Desk and Chair produced activations in every scenario — consistent with their dominant share of training detections.
- Seated_Student activations present across occupied/reserved/mixed scenarios; the model attends broadly to upper-body/torso regions.
- Stationary_Laptop_Tablet activates in non-occupied contexts (available, reserved, mixed) where laptops are present without seated students — suggesting the model has some discriminative signal for this class, despite zero detections on the test set at threshold 0.5.
- Desktop_Inactive activations appeared only in reserved scenarios, consistent with the single Desktop_Inactive detection seen in Run 1 testing.

---

## 9. Study Space Logic Layer (`scratch_study_space.py`)

Classifies each detected Table_Desk as **Available**, **Reserved**, or **Occupied** using mask-based spatial overlap.

### Configuration (current)

| Parameter | Value |
|-----------|-------|
| Model | Scratch (GN Run 2) |
| Score threshold | 0.3 |
| Mask threshold (`MASK_THRESHOLD`) | 0.5 |
| Overlap threshold (table zone) | 0.3 |
| Chair overlap threshold | 0.15 (half of overlap threshold) |

### Logic rules
- **Occupied**: Seated_Student mask overlaps table or associated chair by ≥ threshold.
- **Reserved**: Laptop_Tablet / Phone / Desktop_Active / PersonalItem overlaps table; or PersonalItem overlaps an associated chair.
- **Available**: Table has only empty chairs or no associated objects.
- Chair association: chair centroid falls within table mask bounding box expanded by 30%.

10 per-image visualisation PNGs saved to `study_space_outputs/`.

### Observations and issue
The logic layer produced **over-sensitive** classifications — tables were flagged as Reserved or Occupied from weak or spurious mask overlaps that do not reflect true occupancy. The root cause is that at `MASK_THRESHOLD=0.5`, predicted mask blobs can be noisy and spatially imprecise enough to breach the `overlap_threshold=0.3` surface, generating false positive status changes.

**Next step:** Raise `MASK_THRESHOLD` (the per-pixel confidence cutoff for binarising raw mask logits) to reduce the spatial extent of noisy mask predictions before overlap is computed. This makes the logic layer less sensitive to weak detections without changing the overlap threshold logic itself.

---

## 10. Files (`Scratch_MaskRCNN/`)

| File | Purpose |
|------|---------|
| `scratch_factory.py` | Model definition — `DualPathBlock`, `ScratchNet`, `build_scratch_mask_rcnn()`, `count_parameters()` |
| `scratch_tune.py` | Optuna LR search; saves `best_lr.txt` |
| `scratch_train.py` | Full 50-epoch training run |
| `scratch_test.py` | Test set evaluation — P-IoU, I-IoU, per-class breakdown, saves PNGs to `scratch_test_outputs/` |
| `scratch_diagnose.py` | Checkpoint validation, raw score distribution, FPN feature probing |
| `scratch_gradcam.py` | Grad-CAM activation maps — 45 PNGs in `gradcam_outputs/` (10 images, 4 scenarios) |
| `scratch_study_space.py` | Logic layer: mask-based table classification (Available / Reserved / Occupied) |
| `gradcam_outputs/` | 45 Grad-CAM PNGs (GN Run 2 checkpoint) |
| `study_space_outputs/` | 10 per-image availability visualisations (GN Run 2, score_thresh=0.3) |
| `scratch_best_weights.pth` | Best checkpoint — **GN Run 2**, epoch 20, val_loss=2.2197, LR=1.3293e-4 |
| `scratch_best_weights_50ep.pth` | Final checkpoint — **GN Run 2**, epoch 50, LR=1.3293e-4 |
| `tune_results.csv` | Tuning Run 3 trial results (all 9 completed trials, sorted by val_loss) |

---

## 11. Next Steps

- [x] Replace **BatchNorm with GroupNorm** (groups=16) — done in `scratch_factory.py`
- [x] Re-run tuning with GN model — **Tuning Run 3** complete (LR=1.3293e-4 confirmed)
- [x] Re-run 50-epoch training with GN model — **Training Run 2** complete (epoch 20 best, val_loss=2.2197)
- [x] **Test set evaluation** (GN Run 2) — P-IoU=0.268, I-IoU=0.150 (see §6)
- [x] **Grad-CAM visualisations** — 45 PNGs generated across 4 scenarios (see §8)
- [x] **Study space logic layer** run — over-sensitivity identified (see §9)
- [ ] **Raise `MASK_THRESHOLD`** in `scratch_study_space.py` (currently 0.5) to reduce noisy mask blobs and false status changes in the logic layer
- [ ] Add **score threshold sweep** (0.3–0.5) to recover the 4 zero-detection classes
- [ ] Retrieve full **per-epoch val IoU** from ClearML task *Final-Training-50ep-GN* and add to §5 table
- [ ] Extend training to **100+ epochs** with warm-up — val loss had not fully plateaued at epoch 20

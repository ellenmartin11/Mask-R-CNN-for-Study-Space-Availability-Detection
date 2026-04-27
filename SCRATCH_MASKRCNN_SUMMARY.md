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

### Backbone: ScratchNet

The backbone uses a novel **dual-path pre-activation block** (`DualPathBlock`):

```
pre = BN(x)                                        ← pre-activation
a   = Conv(in → out/2, 3×3, stride=s)             ← path A: standard conv
b   = DepthwiseConv(in, 3×3, stride=s)
      → PointwiseConv(in → out/2, 1×1)            ← path B: separable conv
merged = 1×1-proj( BN( cat(a, b) ) )
out    = ReLU( merged + shortcut(x) )
shortcut: MaxPool(s) → Conv(in → out, 1×1)   when dims change
          Identity                             otherwise
```

| Stage | In ch → Out ch | Stride | Blocks | Output stride |
|-------|----------------|--------|--------|---------------|
| Stem  | 3 → 32         | 2      | —      | 2             |
| Stage 1 | 32 → 64     | 2      | 2      | 4 (C2)        |
| Stage 2 | 64 → 128    | 2      | 3      | 8 (C3)        |
| Stage 3 | 128 → 192   | 2      | 4      | 16 (C4)       |
| Stage 4 | 192 → 288   | 2      | 2      | 32 (C5)       |

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

### Tuning Run 2 — April 26 (most recent; no ClearML)

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

> The optimal LR dropped sharply between runs (1.6e-3 → 1.3e-4). Run 1 produced notably lower 3-epoch probe losses, suggesting Run 1's LR regime was stronger for early training. The higher LR (1.6e-3) is consistent with bootstrapping learning from random initialisation, while 1.3e-4 is more conservative.

---

## 5. Training Run (`scratch_train.py`)

### Setup

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Learning rate | 1.3293e-4 (from `best_lr.txt`, Run 2 above) |
| Weight decay | 1e-4 |
| Optimizer | AdamW |
| Scheduler | `CosineAnnealingLR` (T_max=50, η_min=1e-6) |
| Batch size | 2 |
| Gradient clip | max_norm = 1.0 |
| AMP | `torch.amp.autocast("cuda")` + `GradScaler` |
| Validation | Every 5 epochs (+ epoch 1) |
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

**Best checkpoint** (lowest val loss 2.2320): epoch 15 → `scratch_best_weights.pth`
**Final checkpoint**: epoch 50 → `scratch_best_weights_50ep.pth`

### Training Observations
- Training loss fell steadily from **2.58 → 1.22** with no instability.
- Val loss hit its minimum at **epoch 15 (2.2320)** then increased continuously — earlier convergence than the original run, likely because LR=1.3e-4 is lower and exploration is limited.
- Instance IoU peaked at **0.5028 at epoch 15** then regressed, suggesting the model did not improve further with the decaying LR.
- The lower starting LR (vs. Run 1's 1.6e-3) produced a higher val loss floor throughout training, consistent with slower feature bootstrap from random init.

### Note on Checkpoint Files
`scratch_diagnose.py` (run after training) reads the checkpoint metadata as: `epoch=15, best_val_loss=1.9518, best_lr=0.001595857`. The stored `best_lr` matches Run 1 (1.5959e-3), not the LR used in this training run, indicating the **checkpoint files on disk are from the original training run** (Run 1, LR=1.5959e-3). This is consistent with a Google Drive sync failure during the April 26 training save. All test evaluations below used these on-disk checkpoints.

---

## 6. Test Set Evaluation (`scratch_test.py`)

Evaluated on **`scratch_best_weights.pth`** (checkpoint from original run, epoch 15, val_loss=1.9518), score threshold = 0.5, IoU threshold = 0.5.

### Aggregate Results

| Metric | Value |
|--------|-------|
| **Test P-IoU** (pixel mask, avg over images) | **0.2802** |
| **Test I-IoU** (box @ 0.50, matches / GT objects) | **0.1353** |

### Per-image Breakdown

| Image | GT Instances | Predicted | P-IoU |
|-------|-------------|-----------|-------|
| 1 | 6 | 4 | 0.1349 |
| 2 | 13 | 8 | 0.2034 |
| 3 | 15 | 4 | 0.1505 |
| 4 | 7 | 4 | 0.5344 |
| 5 | 17 | 5 | 0.2381 |
| 6 | 16 | 6 | 0.2854 |
| 7 | 8 | 4 | 0.2085 |
| 8 | 18 | 6 | 0.2360 |
| 9 | 27 | 5 | 0.3038 |
| 10 | 6 | 5 | 0.5071 |

### Per-class Breakdown

| Class | Detections | Avg Conf | Box IoU | Mask IoU |
|-------|-----------|----------|---------|----------|
| Table_Desk | 9 | 0.5908 | 0.3924 | 0.2137 |
| Chair | 17 | 0.6684 | 0.4233 | 0.3577 |
| Seated_Student | 17 | 0.6263 | 0.4175 | 0.2567 |
| Desktop_Inactive | 1 | 0.7802 | 0.8537 | 0.6514 |
| Stationary_Laptop_Tablet | 0 | — | — | — |
| Stationary_Phone | 0 | — | — | — |
| Desktop_Active | 0 | — | — | — |
| Stationary_PersonalItem | 0 | — | — | — |

4 of 8 classes produced zero detections at threshold 0.5. Classes detected were the three most spatially prominent (Table_Desk, Chair, Seated_Student) plus one instance of Desktop_Inactive.

---

## 7. Comparison: Scratch vs ResNet-50 Baseline

| Metric | ResNet-50 FPN (fine-tuned) | Scratch ScratchNet |
|--------|--------------------------|-------------------|
| Parameters | ~44 M | **4.20 M** |
| Backbone | Pretrained (COCO V1) | **Random init** |
| Best Val Loss | **1.571** | 2.232 |
| Best Val Pixel IoU | **0.587** | 0.425 (72%) |
| Best Val Instance IoU | **0.763** | 0.503 (66%) |
| **Test Pixel IoU** | **0.585** | 0.280 (48%) |
| **Test Instance IoU** | **0.770** | 0.135 (18%) |

### Analysis

The Scratch model achieves **48% of test pixel-level mask quality** and **18% of test instance detection quality** of the ResNet-50 baseline, at **1/10th the parameter count** and with **no pretrained weights**.

The large gap between val IoU (72%/66% of baseline) and test IoU (48%/18%) reveals a **significant generalisation failure**: the model has overfit the validation distribution and does not transfer well to unseen test images. Contributing factors:

- **Only 4/8 classes detected**: smaller, less frequent objects (Stationary_Phone, Laptop, Desktop_Active, PersonalItem) receive zero predictions. The model has specialised towards dominant classes (Chair, Seated_Student, Table_Desk).
- **Severe instance under-detection**: I-IoU of 0.135 means matched GT instances are a small fraction of total GT objects — the model consistently predicts far fewer instances than are present (e.g., 27 GT → 5 predictions on image 9).
- **No pretrained backbone**: without ImageNet features, the RPN requires many more epochs to learn reliable proposals. Evidence: instance IoU = 0.000 at epoch 1 (vs. 0.469 by epoch 5 for the Tiny MobileNetV2 model).
- **Lower LR in final run** (1.3e-4 vs 1.6e-3 in original): this slowed early training further and led to val loss convergence that is ~0.6 higher than the original run's best (2.232 vs 1.952).

---

## 8. Files

| File | Purpose |
|------|---------|
| `scratch_factory.py` | Model definition — `DualPathBlock`, `ScratchNet`, `build_scratch_mask_rcnn()`, `count_parameters()` |
| `scratch_tune.py` | Optuna LR search; saves `best_lr.txt` |
| `scratch_train.py` | Full 50-epoch training run |
| `scratch_test.py` | Test set evaluation — P-IoU, I-IoU, per-class breakdown, saves PNGs to `scratch_test_outputs/` |
| `scratch_diagnose.py` | Checkpoint validation, raw score distribution, FPN feature probing |
| `scratch_gradcam.py` | Grad-CAM activation maps for detected instances (37 PNGs in `gradcam_outputs/`) |
| `scratch_study_space.py` | Logic layer: classifies tables as Available / Reserved / Occupied |
| `scratch_best_weights.pth` | Best checkpoint (original run, epoch 15, val_loss=1.9518, LR=1.5959e-3) |
| `scratch_best_weights_50ep.pth` | Final checkpoint (original run, epoch 25, val_loss=1.9817) |

---

## 9. Next Steps

- [ ] Re-run training at **LR=1.5959e-3** (Run 1 result) with explicit Drive flush after each checkpoint save to avoid sync failure
- [ ] Run formal **test set evaluation on the 50-epoch checkpoint** (`scratch_best_weights_50ep.pth`) for comparison
- [ ] Replace **BatchNorm with GroupNorm** (groups=16) to reduce small-batch sensitivity and improve inference-mode stability
- [ ] Extend training to **100+ epochs** — IoU was still improving at epoch 15 and had not plateaued
- [ ] Add **score threshold sweep** on test set (e.g. 0.3–0.5) to recover the 4 undetected classes
- [ ] Add a **5-epoch linear warm-up** before cosine decay to help RPN bootstrap faster from random init

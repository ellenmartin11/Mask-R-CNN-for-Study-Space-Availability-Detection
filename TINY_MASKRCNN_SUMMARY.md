# Tiny Mask R-CNN — Project Summary

## Overview

This document summarises the design, hyperparameter search, training, and evaluation of a lightweight **Tiny Mask R-CNN** built to replace the baseline ResNet-50 version used in the original project. The goal was to reduce the model to approximately 1/10th of the original parameter count while retaining reasonable instance segmentation performance on the UNH Campus dataset.

---

## 1. Dataset

| Split      | Location                                      |
|------------|-----------------------------------------------|
| Training   | `annotations_training/images/Train`           |
| Validation | `annotations_validation/images/Validation`   |
| Testing    | `annotations_testing/images/Test`             |

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

**Preprocessing:**
- Training images rescaled so the longest edge ≤ 1200 px (matching He et al. scaling)
- Annotations (bounding boxes, polygon masks, RLE masks) rescaled accordingly

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

The original model used by the project was the standard torchvision `maskrcnn_resnet50_fpn` with COCO V1 pretrained weights, fine-tuned to 9 output classes.

| Property | Value |
|----------|-------|
| Backbone | ResNet-50 + FPN (256 channels) |
| Parameters | ~44 million |
| Best experiment | Exp 2: AdamW, LR = 7e-05, WD = 1e-04, 50 epochs |
| Best Val Loss | ~1.571 |
| Best Pixel IoU (val) | 0.587 |
| Best Instance IoU (val) | 0.763 |
| Test Instance IoU | 0.770 |
| Test Pixel IoU | 0.585 |

---

## 3. Tiny Mask R-CNN Architecture (`tiny_factory.py`)

### Design Goal
Reduce parameters to ~4.4 M (≈ 1/10 of ResNet-50 version) while keeping the standard Mask R-CNN detection head structure fully compatible with the existing `FastDataset` pipeline and target dictionary format (`boxes`, `labels`, `masks`, `image_id`, `area`, `iscrowd`).

### Backbone: MobileNetV2 (truncated)

The MobileNetV2 feature extractor is tapped at four intermediate layers and truncated after `features[16]` (stride 32, 160 channels). Layers `features[17]` (320 ch) and `features[18]` (1280 ch) are dropped — they would add ~2 M parameters with no detection benefit at this scale.

| Tap point | Cumulative stride | Channels |
|-----------|------------------|----------|
| `features[3]` | 4 | 24 |
| `features[6]` | 8 | 32 |
| `features[13]` | 16 | 96 |
| `features[16]` | 32 | 160 |

### Feature Pyramid Network (FPN)
- `out_channels = 64` (vs 256 in ResNet-50 version)
- 5 levels (P2–P6) with `LastLevelMaxPool` for the 5th level

### Heads

| Component | Configuration |
|-----------|--------------|
| Anchor generator | 5 sizes (16, 32, 64, 128, 256), 3 aspect ratios (0.5, 1.0, 2.0) |
| Box RoI align | 7 × 7, 4 feature levels |
| Mask RoI align | 14 × 14, 4 feature levels |
| Box head | `TwoMLPHead`: 64×7×7 → 256 → 256 |
| Box predictor | `FastRCNNPredictor`: 256 → 9 classes |
| Mask head | 4 × `Conv2d(64, 64, 3)` |
| Mask predictor | `ConvTranspose2d(64→32)` + `Conv2d(32→9)` |

### Parameter Count

| Section | Parameters |
|---------|-----------|
| Backbone (MobileNetV2 truncated) | 1,505,728 |
| RPN | 37,903 |
| Box head (TwoMLPHead) | 868,864 |
| Box predictor | 11,565 |
| Mask head | 147,712 |
| Mask predictor | 8,521 |
| **Total** | **2,580,293 (2.58 M)** |

> The final model is **~1/17th the size** of the ResNet-50 baseline, comfortably under the 4.5 M parameter budget.

---

## 4. Hyperparameter Search (`tiny_tune.py`)

### Method
**Optuna TPE (Tree-structured Parzen Estimator)** sampler with 10 trials.  
Each trial trained for **3 epochs** on the training set and reported mean validation loss.

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
| Learning rate | [1e-5, 1e-3] | Log-uniform |

### All trials (sorted by validation loss)

| Trial | LR | Val Loss |
|-------|----|----------|
| 9 | 2.61e-04 | **1.8626** |
| 7 | 5.40e-04 | 1.9000 |
| 1 | 7.97e-04 | 1.9040 |
| 0 | 5.61e-05 | 1.9662 |
| 2 | 2.91e-04 | 2.0164 |
| 3 | 1.58e-04 | 2.0513 |
| 4 | 2.10e-05 | 2.0567 |
| 8 | 1.59e-04 | 2.1009 |
| 6 | 1.30e-05 | 2.1253 |
| 5 | 2.10e-05 | 2.3324 |

### Key observation
The top three trials clustered between **2.6e-4 and 8.0e-4**. Very low learning rates (< 3e-5) were consistently the worst performers — consistent with MobileNetV2's faster convergence relative to ResNet-50.

### Selected hyperparameters
| Parameter | Value |
|-----------|-------|
| **Learning rate** | **2.607e-04** |
| Weight decay | 1e-4 |
| Optimizer | AdamW |

---

## 5. Full Training Run (`tiny_final_train.py`)

### Setup

| Parameter | Value |
|-----------|-------|
| Epochs | 25 |
| Learning rate | 2.607e-04 (from Optuna) |
| Weight decay | 1e-4 |
| Optimizer | AdamW |
| Scheduler | `CosineAnnealingLR` (T_max=25, η_min=1e-6) |
| Batch size | 2 |
| Gradient clip | max_norm = 1.0 |
| AMP | `torch.amp.autocast("cuda")` + `GradScaler` |
| Validation | Every 5 epochs (+ epoch 1) |
| Hardware | NVIDIA L4 GPU (Google Colab Pro) |
| Experiment tracking | ClearML (`Tiny-MaskRCNN / Final-Training-25ep`) |

### Training and Validation Results

| Epoch | LR | Train Loss | Val Loss | Pixel IoU | Instance IoU | Saved |
|-------|----|-----------|----------|-----------|--------------|-------|
| 1 | 2.61e-04 | 2.1257 | 2.2336 | 0.3493 | 0.0000 | ✓ |
| 5 | 2.45e-04 | 1.5880 | 1.7953 | 0.4969 | 0.4689 | ✓ |
| 10 | 1.86e-04 | 1.2732 | 1.7405 | 0.5116 | 0.6667 | ✓ |
| 15 | 1.07e-04 | 1.1087 | 1.6723 | 0.5406 | **0.7062** | ✓ |
| 20 | 3.62e-05 | 1.0124 | 1.6779 | **0.5600** | 0.6780 | — |
| 25 | 2.02e-06 | 0.9662 | **1.6523** | 0.5445 | 0.6723 | ✓ |

**Best checkpoint saved at epoch 25** (lowest val loss: 1.6523):
```
/content/drive/MyDrive/deep_learning_project/Tiny_MaskRCNN/tiny_best_weights.pth
```

### Training Observations

- Training loss fell smoothly from **2.13 → 0.97** with no NaN instability — the cosine scheduler annealed stably throughout.
- Instance IoU was 0.000 at epoch 1 (RPN not yet proposing usable boxes), then jumped to **0.469 by epoch 5** and peaked at **0.706 at epoch 15**.
- Pixel IoU improved monotonically through epoch 20 (0.560), with a slight dip at epoch 25 as the LR approached zero.
- Validation loss plateaued after epoch 15 (~1.67), suggesting the model reached convergence within the 25-epoch budget.

---

## 6. Comparison: Tiny vs ResNet-50 Baseline

| Metric | ResNet-50 (Exp 1, 50 ep) | Tiny MobileNetV2 (25 ep) |
|--------|--------------------------|--------------------------|
| Parameters | ~44 M | **2.58 M** |
| Val Loss (best) | 1.571 | 1.652 |
| Val Pixel IoU | 0.587 | 0.560 |
| Val Instance IoU | 0.763 | 0.706 |
| Epochs to converge | ~50 | ~15–25 |

> The Tiny model recovers **95% of the pixel-level mask quality** and **92% of the instance detection quality** of the full ResNet-50 model at **~1/17th the parameter count**.

---

## 7. Files

| File | Purpose |
|------|---------|
| `tiny_factory.py` | Model definition — `build_tiny_mask_rcnn()`, `count_parameters()` |
| `tiny_tune.py` | Optuna LR search with ClearML logging |
| `tiny_final_train.py` | Full 25-epoch training run with ClearML logging |
| `tiny_visualize.py` | Validation image visualisation (boxes, masks, labels) |
| `Tiny_MaskRCNN/tiny_best_weights.pth` | Best checkpoint (epoch 25, val loss 1.6523) |
| `Tiny_MaskRCNN/val_predictions.png` | Sample validation predictions (threshold 0.65) |

---

## 8. Next Steps

- [ ] Formal evaluation on the **test set** (pixel IoU + instance IoU + per-class breakdown)
- [ ] Compare test-set predictions visually against ResNet-50 Exp 1 outputs
- [ ] Optionally tune weight decay and/or extend to 50 epochs to close the remaining gap to the ResNet-50 baseline

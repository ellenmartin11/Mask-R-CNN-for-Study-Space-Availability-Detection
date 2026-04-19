# -*- coding: utf-8 -*-
"""
tiny_tune.py — Optuna LR search for Tiny Mask R-CNN, tracked with ClearML.

Usage (Colab):
    !python tiny_tune.py

Pipeline:
  1. Mount Google Drive and copy data to local runtime (mirrors dl_project_finetuning.py).
  2. Build FastDataset with IDENTICAL target keys (boxes, labels, masks, …).
  3. Run an Optuna TPE study over the learning rate.
  4. Log every trial and the best result to a ClearML task.
"""

# ============================================================
# 0. Imports
# ============================================================
import gc
import os
import random
import shutil
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.amp as amp
from PIL import Image
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from clearml import Task

from tiny_factory import build_tiny_mask_rcnn, count_parameters

# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 42

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

set_seed(SEED)

# ============================================================
# 2. Data paths
#
# drive.mount() is an IPython-only API and cannot be called from a
# subprocess (!python …).  Mount the drive in a notebook cell first:
#
#   from google.colab import drive
#   drive.mount("/content/drive")
#
# This script then copies data to the local runtime if it hasn't been
# copied yet, or reuses what's already there.
# ============================================================

PROJECT_PATH = Path("/content/drive/MyDrive/deep_learning_project")

TRAIN_IMG_DIR  = PROJECT_PATH / "annotations/annotations_training/images/Train"
TRAIN_ANN_FILE = PROJECT_PATH / "annotations/annotations_training/annotations/instances_Train.json"

VAL_IMG_DIR   = PROJECT_PATH / "annotations/annotations_validation/images/Validation"
VAL_ANN_FILE  = PROJECT_PATH / "annotations/annotations_validation/annotations/instances_Validation.json"

LOCAL_DATA = Path("/content/data")
LOCAL_DATA.mkdir(exist_ok=True)

if not (LOCAL_DATA / "train_ann.json").exists():
    if not PROJECT_PATH.exists():
        raise RuntimeError(
            "Google Drive is not mounted. Run this in a notebook cell first:\n"
            "  from google.colab import drive\n"
            "  drive.mount('/content/drive')"
        )
    print("Copying training data to runtime …")
    shutil.copytree(TRAIN_IMG_DIR, LOCAL_DATA / "train_images", dirs_exist_ok=True)
    shutil.copy(TRAIN_ANN_FILE,    LOCAL_DATA / "train_ann.json")
    print("Copying validation data to runtime …")
    shutil.copytree(VAL_IMG_DIR, LOCAL_DATA / "val_images", dirs_exist_ok=True)
    shutil.copy(VAL_ANN_FILE,    LOCAL_DATA / "val_ann.json")
else:
    print("Local data already present — skipping copy.")

TRAIN_IMG_LOCAL = str(LOCAL_DATA / "train_images")
TRAIN_ANN_LOCAL = str(LOCAL_DATA / "train_ann.json")
VAL_IMG_LOCAL   = str(LOCAL_DATA / "val_images")
VAL_ANN_LOCAL   = str(LOCAL_DATA / "val_ann.json")

# ============================================================
# 3. Transforms  (identical to dl_project_finetuning.py)
# ============================================================

train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    v2.RandomShortestSize(min_size=list(range(800, 1025)), max_size=1333),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=800),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================================
# 4. FastDataset  — IDENTICAL to dl_project_finetuning.py
# ============================================================

def collate_fn(batch):
    return tuple(zip(*batch))


class FastDataset(Dataset):
    def __init__(self, root_orig, annotation_orig, target_size=1200, transforms=None):
        self.root = root_orig
        self.coco = COCO(annotation_orig)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.target_size = target_size
        self.cache = []

        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]

            ann_ids  = self.coco.getAnnIds(imgIds=img_id)
            coco_anns = self.coco.loadAnns(ann_ids)

            boxes, labels, masks, areas = [], [], [], []

            for ann in coco_anns:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])
                areas.append(ann["area"])

                seg = ann["segmentation"]
                if isinstance(seg, dict):
                    m = maskUtils.decode(maskUtils.frPyObjects(seg, seg["size"][0], seg["size"][1]))
                else:
                    m = self.coco.annToMask(ann)

                if len(m.shape) > 2:
                    m = m[:, :, 0]
                masks.append(m)

            self.cache.append({
                "path":     os.path.join(self.root, img_info["file_name"]),
                "boxes":    torch.as_tensor(boxes,           dtype=torch.float32),
                "labels":   torch.as_tensor(labels,          dtype=torch.int64),
                "masks":    torch.as_tensor(np.array(masks), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area":     torch.as_tensor(areas,           dtype=torch.float32),
            })

    def __getitem__(self, index):
        c = self.cache[index]
        image = Image.open(c["path"]).convert("RGB")

        target = {
            "boxes":    tv_tensors.BoundingBoxes(c["boxes"], format="XYXY", canvas_size=image.size[::-1]),
            "masks":    tv_tensors.Mask(c["masks"]),
            "labels":   c["labels"],
            "image_id": c["image_id"],
            "area":     c["area"],
            "iscrowd":  torch.zeros((len(c["labels"]),), dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.ids)


# ============================================================
# 5. Datasets & loaders
# ============================================================

print("Building datasets …")
train_dataset = FastDataset(
    root_orig=TRAIN_IMG_LOCAL,
    annotation_orig=TRAIN_ANN_LOCAL,
    target_size=1200,
    transforms=train_transform,
)
val_dataset = FastDataset(
    root_orig=VAL_IMG_LOCAL,
    annotation_orig=VAL_ANN_LOCAL,
    target_size=1200,
    transforms=val_transform,
)

BATCH_SIZE  = 2
NUM_WORKERS = 0

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate_fn, num_workers=NUM_WORKERS,
    worker_init_fn=seed_worker, pin_memory=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_fn, num_workers=NUM_WORKERS,
    worker_init_fn=seed_worker, pin_memory=True,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

# ============================================================
# 6. ClearML task
# ============================================================

task   = Task.init(project_name="Tiny-MaskRCNN", task_name="Optuna-LR-Search")
logger = task.get_logger()

# Log architecture metadata once
tiny_probe = build_tiny_mask_rcnn(num_classes=9, pretrained_backbone=False)
n_params   = count_parameters(tiny_probe)
del tiny_probe
task.set_parameter("architecture/total_params_M", round(n_params / 1e6, 3))
task.set_parameter("architecture/backbone",       "MobileNetV2-FPN")
print(f"Tiny Mask R-CNN — {n_params:,} trainable parameters ({n_params/1e6:.2f} M)")

# ============================================================
# 7. Optuna objective
# ============================================================

# Tune only LR here; weight_decay fixed at a good default from Exp 2
FIXED_WD    = 1e-4
N_TUNE_EPOCHS = 3   # short trials — Optuna explores efficiently with TPE


def run_one_trial(lr: float) -> float:
    """Train for N_TUNE_EPOCHS, return mean validation loss."""
    model = build_tiny_mask_rcnn(num_classes=9, pretrained_backbone=True).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=FIXED_WD, betas=(0.9, 0.999),
    )
    scaler = amp.GradScaler("cuda")

    for epoch in range(N_TUNE_EPOCHS):
        model.train()
        for images, targets in train_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with amp.autocast("cuda"):
                loss_dict = model(images, targets)
                losses    = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses):
                continue

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

    # Validation loss (train mode so Mask R-CNN returns losses)
    model.train()
    val_total = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with amp.autocast("cuda"):
                loss_dict = model(images, targets)
            val_total += sum(v.item() for v in loss_dict.values())

    val_loss = val_total / len(val_loader)

    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return val_loss


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    print(f"\n[Trial {trial.number}] lr={lr:.2e}")
    val_loss = run_one_trial(lr)
    print(f"[Trial {trial.number}] val_loss={val_loss:.4f}")

    # Log scalars to ClearML
    logger.report_scalar(title="val_loss", series="per_trial", value=val_loss,    iteration=trial.number)
    logger.report_scalar(title="lr",       series="per_trial", value=lr,          iteration=trial.number)

    # Expose current best to ClearML for the live chart
    if trial.study.best_value is not None:
        logger.report_scalar(
            title="best_val_loss", series="running",
            value=trial.study.best_value, iteration=trial.number,
        )

    return val_loss


# ============================================================
# 8. Run study
# ============================================================

N_TRIALS = 10

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    study_name="tiny_maskrcnn_lr_search",
)

print(f"\nStarting Optuna study — {N_TRIALS} trials × {N_TUNE_EPOCHS} epochs each …\n")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ============================================================
# 9. Report best result
# ============================================================

best_lr       = study.best_params["lr"]
best_val_loss = study.best_value

print("\n" + "=" * 50)
print(f"Best LR        : {best_lr:.6e}")
print(f"Best Val Loss  : {best_val_loss:.4f}")
print("=" * 50)

# Persist best params in ClearML
task.set_parameter("best/lr",       best_lr)
task.set_parameter("best/val_loss", best_val_loss)
task.set_parameter("search/n_trials",     N_TRIALS)
task.set_parameter("search/epochs_each",  N_TUNE_EPOCHS)
task.set_parameter("search/fixed_wd",     FIXED_WD)

# Log the full trials dataframe
import pandas as pd
df = study.trials_dataframe()[["number", "value", "params_lr"]]
df.columns = ["trial", "val_loss", "lr"]
print("\nAll trials:")
print(df.sort_values("val_loss").to_string(index=False))

task.close()
print("\nClearML task closed. Results saved.")

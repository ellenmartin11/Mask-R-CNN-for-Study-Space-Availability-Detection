"""
scratch_tune.py
---------------
Optuna hyperparameter search for Scratch Mask R-CNN.
Searches learning rate over 10 trials × 3 epochs each.
All trials are logged to ClearML (project: Scratch-MaskRCNN / task: LR-Tune).

Usage (Colab):
    !python scratch_tune.py
"""

# ── 0. Install dependencies ────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "optuna", "clearml", "-q"])

# ── 1. Imports ─────────────────────────────────────────────────────────────
import os, random, shutil, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

import optuna
from clearml import Task

# ── 2. Reproducibility ────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

set_seed(SEED)

# ── 3. Paths ───────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")

PROJECT_PATH = Path("/content/drive/MyDrive/deep_learning_project")
SCRATCH_PATH = PROJECT_PATH / "Scratch_MaskRCNN"

TRAIN_IMG_DIR  = PROJECT_PATH / "annotations/annotations_training/images/Train"
TRAIN_ANN_FILE = PROJECT_PATH / "annotations/annotations_training/annotations/instances_Train.json"
VAL_IMG_DIR    = PROJECT_PATH / "annotations/annotations_validation/images/Validation"
VAL_ANN_FILE   = PROJECT_PATH / "annotations/annotations_validation/annotations/instances_Validation.json"

# Copy data to local runtime for speed
LOCAL = Path("/content/data")
LOCAL.mkdir(exist_ok=True)

def copy_data():
    print("Copying data to local runtime …")
    shutil.copytree(TRAIN_IMG_DIR, LOCAL / "train_images", dirs_exist_ok=True)
    shutil.copy(TRAIN_ANN_FILE,    LOCAL / "train_ann.json")
    shutil.copytree(VAL_IMG_DIR,   LOCAL / "val_images",   dirs_exist_ok=True)
    shutil.copy(VAL_ANN_FILE,      LOCAL / "val_ann.json")
    print("Done.")

copy_data()

# ── 4. Model factory (import from Drive) ──────────────────────────────────
sys.path.insert(0, str(SCRATCH_PATH))
from scratch_factory import build_scratch_mask_rcnn, count_parameters

# ── 5. Transforms ─────────────────────────────────────────────────────────
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

# ── 6. Dataset ─────────────────────────────────────────────────────────────
def collate_fn(batch):
    return tuple(zip(*batch))


class FastDataset(Dataset):
    def __init__(self, root_orig, annotation_orig, target_size=1200, transforms=None):
        self.root       = root_orig
        self.coco       = COCO(annotation_orig)
        self.ids        = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.target_size = target_size
        self.cache      = []

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
                "boxes":    torch.as_tensor(boxes,            dtype=torch.float32),
                "labels":   torch.as_tensor(labels,           dtype=torch.int64),
                "masks":    torch.as_tensor(np.array(masks),  dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area":     torch.as_tensor(areas,            dtype=torch.float32),
            })

    def __getitem__(self, index):
        c     = self.cache[index]
        image = Image.open(c["path"]).convert("RGB")
        target = {
            "boxes":    tv_tensors.BoundingBoxes(c["boxes"], format="XYXY",
                                                  canvas_size=image.size[::-1]),
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


# Build datasets once (they cache annotations in __init__)
print("Loading datasets …")
train_dataset = FastDataset("/content/data/train_images", "/content/data/train_ann.json",
                             transforms=train_transform)
val_dataset   = FastDataset("/content/data/val_images",   "/content/data/val_ann.json",
                             transforms=val_transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                          collate_fn=collate_fn, num_workers=0,
                          worker_init_fn=seed_worker, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False,
                          collate_fn=collate_fn, num_workers=0,
                          worker_init_fn=seed_worker, pin_memory=True)

# ── 7. Helpers ─────────────────────────────────────────────────────────────
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    running = 0.0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast("cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
    return running / len(loader)


@torch.no_grad()
def evaluate_val_loss(model, loader, device):
    model.train()  # MaskRCNN only returns losses in train mode
    total = 0.0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.amp.autocast("cuda"):
            loss_dict = model(images, targets)
        total += sum(v.item() for v in loss_dict.values())
    return total / len(loader)


# ── 8. Optuna objective ────────────────────────────────────────────────────
TUNE_EPOCHS  = 3
WEIGHT_DECAY = 1e-4

# ClearML parent task for the study
parent_task = Task.init(
    project_name="Scratch-MaskRCNN",
    task_name="LR-Tune",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False,
)
logger = parent_task.get_logger()
trial_results = []


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    set_seed(SEED + trial.number)

    model     = build_scratch_mask_rcnn(num_classes=9).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler    = torch.amp.GradScaler("cuda")

    for epoch in range(1, TUNE_EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, scaler, device)

    val_loss = evaluate_val_loss(model, val_loader, device)

    trial_results.append({"trial": trial.number, "lr": lr, "val_loss": val_loss})
    logger.report_scalar("val_loss", "trial", value=val_loss, iteration=trial.number)
    logger.report_scalar("lr",       "trial", value=lr,       iteration=trial.number)
    print(f"  Trial {trial.number:2d} | lr={lr:.4e} | val_loss={val_loss:.4f}")

    del model
    torch.cuda.empty_cache()
    return val_loss


# ── 9. Run study ───────────────────────────────────────────────────────────
print("\nStarting Optuna study (10 trials × 3 epochs) …\n")
sampler = optuna.samplers.TPESampler(seed=SEED)
study   = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=10)

# ── 10. Report results ─────────────────────────────────────────────────────
best = study.best_trial
print(f"\n{'='*50}")
print(f"Best trial : {best.number}")
print(f"Best LR    : {best.params['lr']:.6e}")
print(f"Best val loss: {best.value:.4f}")
print(f"{'='*50}\n")

print("All trials (sorted by val_loss):")
sorted_trials = sorted(trial_results, key=lambda x: x["val_loss"])
for t in sorted_trials:
    print(f"  Trial {t['trial']:2d} | lr={t['lr']:.4e} | val_loss={t['val_loss']:.4f}")

# Save best LR to Drive so scratch_train.py can pick it up
best_lr_path = SCRATCH_PATH / "best_lr.txt"
best_lr_path.write_text(f"{best.params['lr']:.8f}")
print(f"\nBest LR saved to {best_lr_path}")

parent_task.upload_artifact("best_lr", best.params["lr"])
parent_task.close()

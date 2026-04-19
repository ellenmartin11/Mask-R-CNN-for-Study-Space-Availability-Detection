"""
scratch_train.py
----------------
Full 50-epoch training of Scratch Mask R-CNN on the UNH Campus dataset.

  • Reads best LR from scratch_tune.py output (best_lr.txt) or uses the
    value found in the original run (1.595857e-3) as a fallback.
  • Optimizer  : AdamW  (weight_decay=1e-4)
  • Scheduler  : CosineAnnealingLR  (T_max=50, η_min=1e-6)
  • AMP        : torch.amp.autocast + GradScaler
  • Grad clip  : max_norm=1.0
  • Validation : every 5 epochs (+ epoch 1)
  • Tracking   : ClearML  (project: Scratch-MaskRCNN / task: Final-Training-50ep)
  • Saves best checkpoint (lowest val loss) to:
        Scratch_MaskRCNN/scratch_best_weights.pth
    and final epoch checkpoint to:
        Scratch_MaskRCNN/scratch_best_weights_50ep.pth

Usage (Colab):
    !python scratch_train.py
"""

# ── 0. Install dependencies ────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "clearml", "-q"])

# ── 1. Imports ─────────────────────────────────────────────────────────────
import os, random, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

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
BEST_WEIGHTS = SCRATCH_PATH / "scratch_best_weights.pth"
FINAL_WEIGHTS = SCRATCH_PATH / "scratch_best_weights_50ep.pth"

TRAIN_IMG_DIR  = PROJECT_PATH / "annotations/annotations_training/images/Train"
TRAIN_ANN_FILE = PROJECT_PATH / "annotations/annotations_training/annotations/instances_Train.json"
VAL_IMG_DIR    = PROJECT_PATH / "annotations/annotations_validation/images/Validation"
VAL_ANN_FILE   = PROJECT_PATH / "annotations/annotations_validation/annotations/instances_Validation.json"

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

# ── 4. Model factory ───────────────────────────────────────────────────────
sys.path.insert(0, str(SCRATCH_PATH))
from scratch_factory import build_scratch_mask_rcnn, count_parameters

# ── 5. Best LR ─────────────────────────────────────────────────────────────
FALLBACK_LR  = 1.595857e-3   # value found in the original Optuna run
best_lr_file = SCRATCH_PATH / "best_lr.txt"

if best_lr_file.exists():
    LEARNING_RATE = float(best_lr_file.read_text().strip())
    print(f"Using LR from tune file: {LEARNING_RATE:.6e}")
else:
    LEARNING_RATE = FALLBACK_LR
    print(f"best_lr.txt not found — using fallback LR: {LEARNING_RATE:.6e}")

WEIGHT_DECAY = 1e-4
NUM_EPOCHS   = 50
VAL_INTERVAL = 5    # validate every N epochs (+ epoch 1)

# ── 6. Transforms ─────────────────────────────────────────────────────────
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

# ── 7. Dataset ─────────────────────────────────────────────────────────────
def collate_fn(batch):
    return tuple(zip(*batch))


class FastDataset(Dataset):
    def __init__(self, root_orig, annotation_orig, target_size=1200, transforms=None):
        self.root        = root_orig
        self.coco        = COCO(annotation_orig)
        self.ids         = list(sorted(self.coco.imgs.keys()))
        self.transforms  = transforms
        self.target_size = target_size
        self.cache       = []

        for img_id in self.ids:
            img_info  = self.coco.loadImgs(img_id)[0]
            ann_ids   = self.coco.getAnnIds(imgIds=img_id)
            coco_anns = self.coco.loadAnns(ann_ids)

            boxes, labels, masks, areas = [], [], [], []
            for ann in coco_anns:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])
                areas.append(ann["area"])

                seg = ann["segmentation"]
                if isinstance(seg, dict):
                    m = maskUtils.decode(
                        maskUtils.frPyObjects(seg, seg["size"][0], seg["size"][1])
                    )
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

# ── 8. Evaluation helpers ──────────────────────────────────────────────────
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")


@torch.no_grad()
def evaluate_val_loss(model, loader, device):
    """Validation loss (model stays in train mode — MaskRCNN requires it for losses)."""
    model.train()
    total = 0.0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.amp.autocast("cuda"):
            loss_dict = model(images, targets)
        total += sum(v.item() for v in loss_dict.values())
    return total / len(loader)


@torch.no_grad()
def evaluate_pixel_iou(model, loader, device, threshold: float = 0.5) -> float:
    model.eval()
    total_iou, n = 0.0, 0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        outputs = model(images)
        for i in range(len(images)):
            gt_masks = targets[i]["masks"].to(device)
            if gt_masks.shape[0] == 0:
                continue
            H, W = gt_masks.shape[-2:]
            full_gt = (gt_masks.sum(dim=0) > 0).float()

            pred_masks = outputs[i]["masks"]
            if pred_masks.shape[0] > 0:
                pred_masks = pred_masks.squeeze(1)
                if pred_masks.shape[-2:] != (H, W):
                    pred_masks = F.interpolate(
                        pred_masks.unsqueeze(1), size=(H, W), mode="nearest"
                    ).squeeze(1)
                full_pred = (pred_masks.sum(dim=0) > threshold).float()
            else:
                full_pred = torch.zeros((H, W), device=device)

            intersection = (full_pred * full_gt).sum()
            union        = (full_pred + full_gt).clamp(0, 1).sum()
            total_iou   += (intersection / (union + 1e-6)).item()
            n            += 1
    return total_iou / n if n > 0 else 0.0


@torch.no_grad()
def evaluate_instance_iou(model, loader, device,
                           score_threshold: float = 0.2,
                           iou_threshold:   float = 0.5) -> float:
    model.eval()
    matches, total_gt = 0, 0
    for images, targets in loader:
        images  = list(img.to(device) for img in images)
        outputs = model(images)
        for i, output in enumerate(outputs):
            keep       = output["scores"] > score_threshold
            pred_boxes = output["boxes"][keep]
            gt_boxes   = targets[i]["boxes"].to(device)
            total_gt  += len(gt_boxes)
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                iou_mat = box_iou(pred_boxes, gt_boxes)
                max_ious, _ = iou_mat.max(dim=0)
                matches += (max_ious >= iou_threshold).sum().item()
    return matches / total_gt if total_gt > 0 else 0.0


# ── 9. ClearML task ────────────────────────────────────────────────────────
task = Task.init(
    project_name="Scratch-MaskRCNN",
    task_name="Final-Training-50ep",
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False,
)
task.connect({
    "learning_rate":  LEARNING_RATE,
    "weight_decay":   WEIGHT_DECAY,
    "epochs":         NUM_EPOCHS,
    "batch_size":     2,
    "grad_clip":      1.0,
    "scheduler":      "CosineAnnealingLR",
    "optimizer":      "AdamW",
    "fpn_channels":   96,
    "num_classes":    9,
})
logger = task.get_logger()

# ── 10. Model, optimiser, scheduler ───────────────────────────────────────
model = build_scratch_mask_rcnn(num_classes=9).to(device)
count_parameters(model)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
scaler    = torch.amp.GradScaler("cuda")

# ── 11. Training loop ──────────────────────────────────────────────────────
best_val_loss = float("inf")
best_lr_seen  = LEARNING_RATE

print(f"\nTraining for {NUM_EPOCHS} epochs "
      f"(LR={LEARNING_RATE:.4e}, WD={WEIGHT_DECAY}).\n")

for epoch in range(1, NUM_EPOCHS + 1):
    # ── train ──────────────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0
    current_lr   = optimizer.param_groups[0]["lr"]

    for images, targets in train_loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast("cuda"):
            loss_dict = model(images, targets)
            loss      = sum(loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    scheduler.step()

    logger.report_scalar("loss",         "train", value=train_loss,  iteration=epoch)
    logger.report_scalar("learning_rate","train", value=current_lr,  iteration=epoch)
    print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | LR={current_lr:.4e} | "
          f"Train loss={train_loss:.4f}", end="")

    # ── validate ───────────────────────────────────────────────────────────
    if epoch == 1 or epoch % VAL_INTERVAL == 0:
        val_loss     = evaluate_val_loss(model,    val_loader, device)
        pixel_iou    = evaluate_pixel_iou(model,   val_loader, device)
        instance_iou = evaluate_instance_iou(model, val_loader, device)

        logger.report_scalar("loss",         "val",          value=val_loss,     iteration=epoch)
        logger.report_scalar("pixel_iou",    "val",          value=pixel_iou,    iteration=epoch)
        logger.report_scalar("instance_iou", "val",          value=instance_iou, iteration=epoch)

        print(f" | Val loss={val_loss:.4f} | "
              f"Pixel IoU={pixel_iou:.4f} | Instance IoU={instance_iou:.4f}", end="")

        # ── checkpoint ─────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lr_seen  = current_lr
            checkpoint = {
                "epoch":             epoch,
                "model_state_dict":  model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "best_val_loss":     best_val_loss,
                "best_lr":           best_lr_seen,
            }
            torch.save(checkpoint, BEST_WEIGHTS)
            print(" ✓ saved", end="")

    print()

# ── 12. Save final epoch checkpoint ───────────────────────────────────────
final_checkpoint = {
    "epoch":              NUM_EPOCHS,
    "model_state_dict":   model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict":    scaler.state_dict(),
    "best_val_loss":      best_val_loss,
    "best_lr":            best_lr_seen,
}
torch.save(final_checkpoint, FINAL_WEIGHTS)
print(f"\nFinal checkpoint saved → {FINAL_WEIGHTS}")
print(f"Best checkpoint (val loss {best_val_loss:.4f}) → {BEST_WEIGHTS}")

task.upload_artifact("best_weights",  str(BEST_WEIGHTS))
task.upload_artifact("final_weights", str(FINAL_WEIGHTS))
task.close()
print("Training complete.")

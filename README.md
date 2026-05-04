# Mask R-CNN for Study Space Availability Detection
Project submitted for Deep Learning at the University of New Haven. Uses Mask R-CNN with a ResNet-50-FPN backbone to detect 9 custom classes using a custom dataset (N=100). Also includes a Tiny R-CNN model (fine tuned MobileNetV2), generated using Claude Code. 

- [Finetuning Jupyter Notebook - Augmentation, Dataloading, Random Search, Model Training, Validation, Testing and CAM](https://github.com/ellenmartin11/Mask-R-CNN-for-Study-Space-Availability-Detection/blob/main/dl_project_finetuning_CAM_final.md)
- [Dataset & Fine-Tuned Models](https://drive.google.com/drive/folders/16Mj66gZysbH1EZXZO-neVbpdzy3ptoQP?usp=sharing)
- [Mask R-CNN Repo](https://github.com/facebookresearch/Detectron)
- [Gradient CAM for Fine-Tuned Mask R-CNN 10 Epoch Model - Exp1](https://github.com/ellenmartin11/Mask-R-CNN-for-Study-Space-Availability-Detection/blob/main/CAM.pdf)
- [Gradio Deployment on HuggingFace](https://huggingface.co/spaces/ellenmartin11/Instance_Segmentation_of_UNH_Study_Spaces)


## Scratch Model - AI Agent Architecture
- [DualPath Backbone + FPN + RoiAlign + GroupNorm](https://github.com/ellenmartin11/Mask-R-CNN-for-Study-Space-Availability-Detection/blob/main/DL_Project_Claude%20(1).md)
- [Scratch R-CNN Summary - AI Agent Fine-Tuned](https://github.com/ellenmartin11/Mask-R-CNN-for-Study-Space-Availability-Detection/blob/main/SCRATCH_MASKRCNN_SUMMARY.md)

## Model Comparison

## 7. Comparison: Scratch (GN) vs ResNet-50 Baseline

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


### Block Diagram of Architecture
![Block Diagram of Scratch R-CNN Architecture](https://github.com/ellenmartin11/Mask-R-CNN-for-Study-Space-Availability-Detection/blob/main/scratch_arch-1.png)

### Validation Loss
![Validation Loss Across 50 Epochs](https://github.com/ellenmartin11/Mask-R-CNN-for-Study-Space-Availability-Detection/blob/main/scratch_val_loss.png)

### Pixel IoU
![Pixel IOU Across 50 Epochs](https://github.com/ellenmartin11/Mask-R-CNN-for-Study-Space-Availability-Detection/blob/main/pixel_iout.png)

### Instance IoU
![Instance IOU Across 50 Epochs](https://github.com/ellenmartin11/Mask-R-CNN-for-Study-Space-Availability-Detection/blob/main/instance_iou.png)

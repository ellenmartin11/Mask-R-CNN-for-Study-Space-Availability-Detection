```python
!pip install gradio -q
import gradio as gr
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import tv_tensors
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import v2
from torchvision.datasets import CocoDetection
from torchvision import tv_tensors
from google.colab import drive
import os
from pathlib import Path
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import random
import shutil
import json
import itertools
import pandas as pd
import matplotlib.patches as patches
import gc
import torch.amp as amp
from torch.optim.lr_scheduler import MultiStepLR
```


```python
drive.mount('/content/drive')

PROJECT_PATH = Path("/content/drive/MyDrive/deep_learning_project")
```

    Mounted at /content/drive



```python
!pip install optuna clearml torchsummary -q
```

    [?25l   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/419.5 kB[0m [31m?[0m eta [36m-:--:--[0m[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m419.5/419.5 kB[0m [31m41.4 MB/s[0m eta [36m0:00:00[0m
    [?25h[?25l   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/1.4 MB[0m [31m?[0m eta [36m-:--:--[0m[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.4/1.4 MB[0m [31m85.4 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
import clearml

```


```python
!clearml-init

```

    ClearML SDK setup process
    
    Please create new clearml credentials through the settings page in your `clearml-server` web app (e.g. http://localhost:8080//settings/workspace-configuration) 
    Or create a free account at https://app.clear.ml/settings/workspace-configuration
    
    In settings page, press "Create new credentials", then press "Copy to clipboard".
    
    Paste copied configuration here:
    api {   # Ellen Martin's workspace   web_server: https://app.clear.ml/   api_server: https://api.clear.ml   files_server: https://files.clear.ml   credentials {     "access_key" = "F6T2A7ITP7GMOV7DOIB409LOZRTP7O"     "secret_key" = "9o9_2cthggjJdTTrOZqi79bjDW5Bes6pMuneYOTtqOTI0fbpO9Ha21QwDZJLARVWuB0"   } }
    Detected credentials key="F6T2A7ITP7GMOV7DOIB409LOZRTP7O" secret="9o9_***"
    
    ClearML Hosts configuration:
    Web App: https://app.clear.ml/
    API: https://api.clear.ml
    File Store: https://files.clear.ml
    
    Verifying credentials ...
    Credentials verified!
    
    New configuration stored in /root/clearml.conf
    ClearML setup completed successfully.



```python
api {
  # Ellen Martin's workspace
  web_server: https://app.clear.ml/
  api_server: https://api.clear.ml
  files_server: https://files.clear.ml
  credentials {
    "access_key" = "F6T2A7ITP7GMOV7DOIB409LOZRTP7O"
    "secret_key" = "9o9_2cthggjJdTTrOZqi79bjDW5Bes6pMuneYOTtqOTI0fbpO9Ha21QwDZJLARVWuB0"
  }
}
```

### Hyperparameter Tuning


```python
!python /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_tune.py
```

    Copying data to local runtime …
    Done.
    Loading datasets …
    loading annotations into memory...
    Done (t=0.27s)
    creating index...
    index created!
    loading annotations into memory...
    Done (t=0.04s)
    creating index...
    index created!
    Device: cuda  |  Train batches: 40  |  Val batches: 5
    ClearML Task: created new task id=f53c29f6731c40e893b3202571e77f02
    2026-04-27 17:29:27.525166: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2026-04-27 17:29:27.548301: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1777310967.579754   13731 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1777310967.590219   13731 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    W0000 00:00:1777310967.615638   13731 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1777310967.615676   13731 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1777310967.615679   13731 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1777310967.615683   13731 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    2026-04-27 17:29:27.622757: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    ClearML results page: https://app.clear.ml/projects/4d32501f5d00486696384e1635cea7af/experiments/f53c29f6731c40e893b3202571e77f02/output/log
    ClearML initialised — results page: https://app.clear.ml/projects/4d32501f5d00486696384e1635cea7af/experiments/f53c29f6731c40e893b3202571e77f02/output/log
    
    Starting Optuna study (3 epochs × 10 trials) …
    
    [32m[I 2026-04-27 17:29:33,361][0m A new study created in memory with name: scratch_lr_search_gn[0m
      Trial  0 | lr=1.3293e-04 | val_loss=2.3531
    [32m[I 2026-04-27 17:32:21,758][0m Trial 0 finished with value: 2.353129682317376 and parameters: {'lr': 0.0001329291894316216}. Best is trial 0 with value: 2.353129682317376.[0m
      Trial  1 | lr=7.1145e-03 | val_loss=nan
    [33m[W 2026-04-27 17:34:53,219][0m Trial 1 failed with parameters: {'lr': 0.0071144760093434225} because of the following error: The value nan is not acceptable.[0m
    [33m[W 2026-04-27 17:34:53,220][0m Trial 1 failed with value nan.[0m
      Trial  2 | lr=1.5703e-03 | val_loss=2.4012
    [32m[I 2026-04-27 17:37:20,441][0m Trial 2 finished with value: 2.4012351483106613 and parameters: {'lr': 0.001570297088405539}. Best is trial 0 with value: 2.353129682317376.[0m
      Trial  3 | lr=6.2514e-04 | val_loss=2.3738
    [32m[I 2026-04-27 17:39:48,634][0m Trial 3 finished with value: 2.373786073923111 and parameters: {'lr': 0.0006251373574521745}. Best is trial 0 with value: 2.353129682317376.[0m
      Trial  4 | lr=2.9380e-05 | val_loss=2.4168
    [32m[I 2026-04-27 17:42:18,485][0m Trial 4 finished with value: 2.416781906783581 and parameters: {'lr': 2.9380279387035334e-05}. Best is trial 0 with value: 2.353129682317376.[0m
      Trial  5 | lr=2.9375e-05 | val_loss=2.4863
    [32m[I 2026-04-27 17:44:48,201][0m Trial 5 finished with value: 2.486261298507452 and parameters: {'lr': 2.9375384576328295e-05}. Best is trial 0 with value: 2.353129682317376.[0m
      Trial  6 | lr=1.4937e-05 | val_loss=2.5850
    [32m[I 2026-04-27 17:47:15,997][0m Trial 6 finished with value: 2.5850156459957363 and parameters: {'lr': 1.493656855461762e-05}. Best is trial 0 with value: 2.353129682317376.[0m
      Trial  7 | lr=3.9676e-03 | val_loss=2.6364
    [32m[I 2026-04-27 17:49:43,903][0m Trial 7 finished with value: 2.6364390533417463 and parameters: {'lr': 0.003967605077052989}. Best is trial 0 with value: 2.353129682317376.[0m
      Trial  8 | lr=6.3584e-04 | val_loss=2.4411
    [32m[I 2026-04-27 17:52:11,101][0m Trial 8 finished with value: 2.4411316085606813 and parameters: {'lr': 0.0006358358856676254}. Best is trial 0 with value: 2.353129682317376.[0m
      Trial  9 | lr=1.3311e-03 | val_loss=2.4381
    [32m[I 2026-04-27 17:54:38,079][0m Trial 9 finished with value: 2.438124815374613 and parameters: {'lr': 0.001331121608073689}. Best is trial 0 with value: 2.353129682317376.[0m
    
    ==================================================
    Best trial    : 0
    Best LR       : 1.329292e-04
    Best val loss : 2.3531
    ==================================================
    
    All trials (sorted by val_loss):
      Trial  0 | lr=1.3293e-04 | val_loss=2.3531  ← best
      Trial  3 | lr=6.2514e-04 | val_loss=2.3738
      Trial  2 | lr=1.5703e-03 | val_loss=2.4012
      Trial  4 | lr=2.9380e-05 | val_loss=2.4168
      Trial  9 | lr=1.3311e-03 | val_loss=2.4381
      Trial  8 | lr=6.3584e-04 | val_loss=2.4411
      Trial  5 | lr=2.9375e-05 | val_loss=2.4863
      Trial  6 | lr=1.4937e-05 | val_loss=2.5850
      Trial  7 | lr=3.9676e-03 | val_loss=2.6364
    /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_tune.py:436: ExperimentalWarning:
    
    optuna.visualization.matplotlib._optimization_history.plot_optimization_history is experimental (supported from v2.2.0). The interface can change in the future.
    
    Skipping optimization history plot: plot_optimization_history() got an unexpected keyword argument 'ax'
    /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_tune.py:448: ExperimentalWarning:
    
    optuna.visualization.matplotlib._param_importances.plot_param_importances is experimental (supported from v2.2.0). The interface can change in the future.
    
    Skipping param importances plot: plot_param_importances() got an unexpected keyword argument 'ax'
    Best LR saved to /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/best_lr.txt
    ClearML task closed.



```python
from IPython.display import Image
Image("/content/val_predictions.png")
```




    
![png](output_8_0.png)
    



# Scratch Model


### Prompt 1

"I need you to act as a Senior Deep Learning Engineer. Our goal is to create a 'Tiny Mask R-CNN' to replace my current ResNet-50 version.

Analyze: Read content/dl_project_finetuning.py. Pay special attention to the FastDataset class and the target dictionary keys (boxes, labels, masks). We must keep these identical.

Map the Data: I have my images in /content/drive/MyDrive/deep_learning_project/annotations/. This contains three subfolders, annotations_testing, annotations_training, annotations_validation. Each subfolder contains and images and annotations folder.

The Goal: Build a new model from scratch. We need to hit a target of ~4.4 million parameters (roughly $1/10$ of the ResNet-50 version).

Task: Create a file called scractch_factory.py that defines this architecture and a script called scratch_tune.py that uses Optuna to find the best Learning Rate and ClearML to track the results."


```python
!python /content/scratch_tune.py
```

    Local data already present — skipping copy.
    Building datasets …
    loading annotations into memory...
    Done (t=0.27s)
    creating index...
    index created!
    loading annotations into memory...
    Done (t=0.04s)
    creating index...
    index created!
    Device: cuda
    ClearML Task: overwriting (reusing) task id=f515e56a551249d4ac5b81075f0e1bac
    2026-04-18 22:09:19,183 - clearml.Task - INFO - No repository found, storing script code instead
    2026-04-18 22:09:21.082381: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2026-04-18 22:09:21.102301: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1776550161.126746   42426 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1776550161.134942   42426 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    W0000 00:00:1776550161.155774   42426 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1776550161.155801   42426 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1776550161.155804   42426 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1776550161.155808   42426 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    2026-04-18 22:09:21.161259: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    ClearML results page: https://app.clear.ml/projects/4d32501f5d00486696384e1635cea7af/experiments/f515e56a551249d4ac5b81075f0e1bac/output/log
    CampusNet Mask R-CNN — 4,200,069 parameters (4.20 M)
    [32m[I 2026-04-18 22:09:29,529][0m A new study created in memory with name: campusnet_lr_search[0m
    
    Starting Optuna study — 10 trials × 3 epochs each …
    
      0% 0/10 [00:00<?, ?it/s]
    [Trial 0] lr=4.33e-04
    [Trial 0] val_loss=2.0458
    [32m[I 2026-04-18 22:12:15,883][0m Trial 0 finished with value: 2.045839512348175 and parameters: {'lr': 0.00043284502212938834}. Best is trial 0 with value: 2.045839512348175.[0m
    Best trial: 0. Best value: 2.04584:  10% 1/10 [02:46<24:57, 166.35s/it]
    [Trial 1] lr=4.12e-03
    ClearML Monitor: Could not detect iteration reporting, falling back to iterations as seconds-from-start
    [Trial 1] val_loss=nan
    2026-04-18 22:14:34,133 - clearml - INFO - NaN value encountered. Reporting it as '0.0'. Use clearml.Logger.set_reporting_nan_value to assign another value
    [33m[W 2026-04-18 22:14:34,135][0m Trial 1 failed with parameters: {'lr': 0.004123206532618728} because of the following error: The value nan is not acceptable.[0m
    [33m[W 2026-04-18 22:14:34,136][0m Trial 1 failed with value nan.[0m
    Best trial: 0. Best value: 2.04584:  20% 2/10 [05:04<19:58, 149.82s/it]
    [Trial 2] lr=1.75e-03
    [Trial 2] val_loss=1.9657
    [32m[I 2026-04-18 22:16:53,130][0m Trial 2 finished with value: 1.9656950786709786 and parameters: {'lr': 0.0017524101118128151}. Best is trial 2 with value: 1.9656950786709786.[0m
    Best trial: 2. Best value: 1.9657:  30% 3/10 [07:23<16:54, 144.88s/it]
    [Trial 3] lr=1.04e-03
    [Trial 3] val_loss=2.4251
    [32m[I 2026-04-18 22:19:13,331][0m Trial 3 finished with value: 2.4250568807125092 and parameters: {'lr': 0.0010401663679887319}. Best is trial 2 with value: 1.9656950786709786.[0m
    Best trial: 2. Best value: 1.9657:  40% 4/10 [09:43<14:18, 143.03s/it]
    [Trial 4] lr=1.84e-04
    [Trial 4] val_loss=2.0452
    [32m[I 2026-04-18 22:21:34,387][0m Trial 4 finished with value: 2.045159028470516 and parameters: {'lr': 0.00018410729205738696}. Best is trial 2 with value: 1.9656950786709786.[0m
    Best trial: 2. Best value: 1.9657:  50% 5/10 [12:04<11:51, 142.32s/it]
    [Trial 5] lr=1.84e-04
    [Trial 5] val_loss=2.2298
    [32m[I 2026-04-18 22:23:54,664][0m Trial 5 finished with value: 2.229790261387825 and parameters: {'lr': 0.00018408992080552527}. Best is trial 2 with value: 1.9656950786709786.[0m
    Best trial: 2. Best value: 1.9657:  60% 6/10 [14:25<09:26, 141.62s/it]
    [Trial 6] lr=1.26e-04
    [Trial 6] val_loss=2.0661
    [32m[I 2026-04-18 22:26:17,289][0m Trial 6 finished with value: 2.066123093664646 and parameters: {'lr': 0.00012551115172973836}. Best is trial 2 with value: 1.9656950786709786.[0m
    Best trial: 2. Best value: 1.9657:  70% 7/10 [16:47<07:05, 141.95s/it]
    [Trial 7] lr=2.96e-03
    [Trial 7] val_loss=2.2384
    [32m[I 2026-04-18 22:28:41,840][0m Trial 7 finished with value: 2.2383530117571353 and parameters: {'lr': 0.0029621516588303515}. Best is trial 2 with value: 1.9656950786709786.[0m
    Best trial: 2. Best value: 1.9657:  80% 8/10 [19:12<04:45, 142.78s/it]
    [Trial 8] lr=1.05e-03
    [Trial 8] val_loss=2.1283
    [32m[I 2026-04-18 22:31:05,294][0m Trial 8 finished with value: 2.1282687932252884 and parameters: {'lr': 0.0010502105436744284}. Best is trial 2 with value: 1.9656950786709786.[0m
    Best trial: 2. Best value: 1.9657:  90% 9/10 [21:35<02:22, 142.99s/it]
    [Trial 9] lr=1.60e-03
    [Trial 9] val_loss=1.8186
    [32m[I 2026-04-18 22:33:28,134][0m Trial 9 finished with value: 1.8186010658740996 and parameters: {'lr': 0.0015958573588141277}. Best is trial 9 with value: 1.8186010658740996.[0m
    Best trial: 9. Best value: 1.8186: 100% 10/10 [23:58<00:00, 143.86s/it]
    
    ==================================================
    Best LR        : 1.595857e-03
    Best Val Loss  : 1.8186
    ==================================================
    
    All trials:
     trial  val_loss       lr
         9  1.818601 0.001596
         2  1.965695 0.001752
         4  2.045159 0.000184
         0  2.045840 0.000433
         6  2.066123 0.000126
         8  2.128269 0.001050
         5  2.229790 0.000184
         7  2.238353 0.002962
         3  2.425057 0.001040
         1       NaN 0.004123
    
    ClearML task closed.


- best LR: 1.595e-03
- best val loss: 1.819

### Prompt 2

Run a script to count the total trainable parameters of the new model. If it is over 4.5 million, suggest ways to prune the FPN (Feature Pyramid Network) channels or the Box/Mask heads until we are under the limit.


```python
import shutil
import os

source_folder_plotneuralnet = '/content/PlotNeuralNet'
source_folder_scratch_diagram = '/content/scratch_diagram'

destination_base_path = PROJECT_PATH / "Scratch_MaskRCNN"

# Create the destination directory if it doesn't exist
os.makedirs(destination_base_path, exist_ok=True)

try:
    shutil.move(source_folder_plotneuralnet, destination_base_path)
    print(f"Moved '{source_folder_plotneuralnet}' to '{destination_base_path}'")
except Exception as e:
    print(f"Error moving '{source_folder_plotneuralnet}': {e}")

try:
    shutil.move(source_folder_scratch_diagram, destination_base_path)
    print(f"Moved '{source_folder_scratch_diagram}' to '{destination_base_path}'")
except Exception as e:
    print(f"Error moving '{source_folder_scratch_diagram}': {e}")
```

    Moved '/content/PlotNeuralNet' to '/content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN'
    Moved '/content/scratch_diagram' to '/content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN'


### Training


```python
!python /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_train.py
```

    Using LR from best_lr.txt: 1.329300e-04
    ClearML Task: created new task id=0735745c085c4b13b78d94c2ebf98465
    2026-04-27 18:09:45.615318: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2026-04-27 18:09:45.635355: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1777313385.659611   24451 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1777313385.667667   24451 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    W0000 00:00:1777313385.689162   24451 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1777313385.689201   24451 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1777313385.689205   24451 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1777313385.689211   24451 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    2026-04-27 18:09:45.694658: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    ClearML results page: https://app.clear.ml/projects/4d32501f5d00486696384e1635cea7af/experiments/0735745c085c4b13b78d94c2ebf98465/output/log
    ClearML initialised — results page: https://app.clear.ml/projects/4d32501f5d00486696384e1635cea7af/experiments/0735745c085c4b13b78d94c2ebf98465/output/log
    Local data already present — skipping copy.
    Loading datasets …
    loading annotations into memory...
    Done (t=0.27s)
    creating index...
    index created!
    loading annotations into memory...
    Done (t=0.04s)
    creating index...
    index created!
    Device: cuda
    Trainable parameters: 4,200,069  (~4.20 M)
    
    Training for 50 epochs (LR=1.3293e-04, WD=0.0001).
    
    Epoch   1/50 | LR=1.3293e-04 | Train loss=2.6552 | Val loss=2.5481 | Pixel IoU=0.2695 | Instance IoU=0.1130 ✓ saved
    Epoch   2/50 | LR=1.3280e-04 | Train loss=2.2589
    Epoch   3/50 | LR=1.3241e-04 | Train loss=2.3001
    Epoch   4/50 | LR=1.3176e-04 | Train loss=2.2546
    Epoch   5/50 | LR=1.3086e-04 | Train loss=2.1960 | Val loss=2.2573 | Pixel IoU=0.3694 | Instance IoU=0.1864 ✓ saved
    Epoch   6/50 | LR=1.2970e-04 | Train loss=2.1873
    Epoch   7/50 | LR=1.2830e-04 | Train loss=2.1336
    Epoch   8/50 | LR=1.2665e-04 | Train loss=2.1116
    Epoch   9/50 | LR=1.2477e-04 | Train loss=2.0868
    Epoch  10/50 | LR=1.2266e-04 | Train loss=2.0684 | Val loss=2.3363 | Pixel IoU=0.4299 | Instance IoU=0.2994
    Epoch  11/50 | LR=1.2033e-04 | Train loss=2.0412
    Epoch  12/50 | LR=1.1779e-04 | Train loss=2.0050
    Epoch  13/50 | LR=1.1505e-04 | Train loss=2.0075
    Epoch  14/50 | LR=1.1212e-04 | Train loss=1.9413
    Epoch  15/50 | LR=1.0901e-04 | Train loss=1.9081 | Val loss=2.2518 | Pixel IoU=0.3974 | Instance IoU=0.4576 ✓ saved
    Epoch  16/50 | LR=1.0574e-04 | Train loss=1.8960
    Epoch  17/50 | LR=1.0231e-04 | Train loss=1.8670
    Epoch  18/50 | LR=9.8744e-05 | Train loss=1.8182
    Epoch  19/50 | LR=9.5052e-05 | Train loss=1.8129
    Epoch  20/50 | LR=9.1248e-05 | Train loss=1.7651 | Val loss=2.2197 | Pixel IoU=0.4382 | Instance IoU=0.4350 ✓ saved
    Epoch  21/50 | LR=8.7349e-05 | Train loss=1.7502
    Epoch  22/50 | LR=8.3370e-05 | Train loss=1.7197
    Epoch  23/50 | LR=7.9326e-05 | Train loss=1.7237
    Epoch  24/50 | LR=7.5233e-05 | Train loss=1.6521
    Epoch  25/50 | LR=7.1107e-05 | Train loss=1.5988 | Val loss=2.2455 | Pixel IoU=0.4543 | Instance IoU=0.4633
    Epoch  26/50 | LR=6.6965e-05 | Train loss=1.6066
    Epoch  27/50 | LR=6.2823e-05 | Train loss=1.5407
    Epoch  28/50 | LR=5.8697e-05 | Train loss=1.5267
    Epoch  29/50 | LR=5.4604e-05 | Train loss=1.4935
    Epoch  30/50 | LR=5.0560e-05 | Train loss=1.4646 | Val loss=2.3160 | Pixel IoU=0.4534 | Instance IoU=0.4972
    Epoch  31/50 | LR=4.6581e-05 | Train loss=1.4382
    Epoch  32/50 | LR=4.2682e-05 | Train loss=1.4163
    Epoch  33/50 | LR=3.8878e-05 | Train loss=1.3971
    Epoch  34/50 | LR=3.5186e-05 | Train loss=1.3637
    Epoch  35/50 | LR=3.1619e-05 | Train loss=1.3724 | Val loss=2.4123 | Pixel IoU=0.4598 | Instance IoU=0.4802
    Epoch  36/50 | LR=2.8192e-05 | Train loss=1.3328
    Epoch  37/50 | LR=2.4917e-05 | Train loss=1.3175
    Epoch  38/50 | LR=2.1809e-05 | Train loss=1.3212
    Epoch  39/50 | LR=1.8879e-05 | Train loss=1.2983
    Epoch  40/50 | LR=1.6138e-05 | Train loss=1.2823 | Val loss=2.4217 | Pixel IoU=0.4787 | Instance IoU=0.4746
    Epoch  41/50 | LR=1.3598e-05 | Train loss=1.2759
    Epoch  42/50 | LR=1.1269e-05 | Train loss=1.2553
    Epoch  43/50 | LR=9.1594e-06 | Train loss=1.2580
    Epoch  44/50 | LR=7.2781e-06 | Train loss=1.2437
    Epoch  45/50 | LR=5.6323e-06 | Train loss=1.2416 | Val loss=2.4870 | Pixel IoU=0.4803 | Instance IoU=0.4237
    Epoch  46/50 | LR=4.2286e-06 | Train loss=1.2335
    Epoch  47/50 | LR=3.0724e-06 | Train loss=1.2271
    Epoch  48/50 | LR=2.1684e-06 | Train loss=1.2310
    Epoch  49/50 | LR=1.5202e-06 | Train loss=1.2168
    Epoch  50/50 | LR=1.1302e-06 | Train loss=1.2211 | Val loss=2.4936 | Pixel IoU=0.4834 | Instance IoU=0.4463
    
    Final checkpoint saved  → /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_best_weights_50ep.pth
    Best checkpoint (val loss 2.2197) → /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_best_weights.pth
                                               0% | 0.00/48.24 MB [00:00<?, ?MB/s]: 
    ██████▍                          21% | 10.00/48.24 MB [00:00<00:01, 22.87MB/s]: 
    █████████▋                       31% | 15.00/48.24 MB [00:00<00:02, 16.24MB/s]: 
    ████████████▊                    41% | 20.00/48.24 MB [00:01<00:02, 14.11MB/s]: 
    ████████████████                 52% | 25.00/48.24 MB [00:01<00:01, 13.11MB/s]: 
    ███████████████████▎             62% | 30.00/48.24 MB [00:02<00:01, 12.55MB/s]: 
    ██████████████████████▍          73% | 35.00/48.24 MB [00:02<00:01, 12.21MB/s]: 
    █████████████████████████▋       83% | 40.00/48.24 MB [00:03<00:00, 12.00MB/s]: 
    ████████████████████████████▉    93% | 45.00/48.24 MB [00:03<00:00, 11.86MB/s]: 
    ██████████████████████████████▉ 100% | 48.23/48.24 MB [00:03<00:00, 12.36MB/s]: 
    ███████████████████████████████ 100% | 48.24/48.24 MB [00:04<00:00, 11.37MB/s]: 
    ███████████████████████████████ 100% | 48.24/48.24 MB [00:04<00:00, 11.23MB/s]: 
    Training complete.


### Gradient CAM

Read through "/content/drive/MyDrive/deep_learning_project/
  Scratch_MaskRCNN/SCRATCH_MASKRCNN_SUMMARY.md   
  to understand what we've been working on.Then read through                              
  /content/drive/MyDrive/deep_learning_project/  
  dl_project_finetuning.py to see how I          
  implemented Gradient Class Activation Mapping.
   I want to do the same thing but with my       
  scratch_best_weights_50ep.pth in the           
  Scratch_MaskRCNN folder.       


```python
!pip install grad-cam
```

    Collecting grad-cam
      Downloading grad-cam-1.5.5.tar.gz (7.8 MB)
    [?25l     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/7.8 MB[0m [31m?[0m eta [36m-:--:--[0m[2K     [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m [32m7.8/7.8 MB[0m [31m261.0 MB/s[0m eta [36m0:00:01[0m[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.8/7.8 MB[0m [31m135.4 MB/s[0m eta [36m0:00:00[0m
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from grad-cam) (2.0.2)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.12/dist-packages (from grad-cam) (11.3.0)
    Requirement already satisfied: torch>=1.7.1 in /usr/local/lib/python3.12/dist-packages (from grad-cam) (2.10.0+cu128)
    Requirement already satisfied: torchvision>=0.8.2 in /usr/local/lib/python3.12/dist-packages (from grad-cam) (0.25.0+cu128)
    Collecting ttach (from grad-cam)
      Downloading ttach-0.0.3-py3-none-any.whl.metadata (5.2 kB)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (from grad-cam) (4.67.3)
    Requirement already satisfied: opencv-python in /usr/local/lib/python3.12/dist-packages (from grad-cam) (4.13.0.92)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.12/dist-packages (from grad-cam) (3.10.0)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (from grad-cam) (1.6.1)
    Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (3.28.0)
    Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (4.15.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (75.2.0)
    Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (1.14.0)
    Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (3.6.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (3.1.6)
    Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (2025.3.0)
    Requirement already satisfied: cuda-bindings==12.9.4 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (12.9.4)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.8.93 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (12.8.93)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.8.90 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (12.8.90)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.8.90 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (12.8.90)
    Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (9.10.2.21)
    Requirement already satisfied: nvidia-cublas-cu12==12.8.4.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (12.8.4.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.3.3.83 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (11.3.3.83)
    Requirement already satisfied: nvidia-curand-cu12==10.3.9.90 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (10.3.9.90)
    Requirement already satisfied: nvidia-cusolver-cu12==11.7.3.90 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (11.7.3.90)
    Requirement already satisfied: nvidia-cusparse-cu12==12.5.8.93 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (12.5.8.93)
    Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (0.7.1)
    Requirement already satisfied: nvidia-nccl-cu12==2.27.5 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (2.27.5)
    Requirement already satisfied: nvidia-nvshmem-cu12==3.4.5 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (3.4.5)
    Requirement already satisfied: nvidia-nvtx-cu12==12.8.90 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (12.8.90)
    Requirement already satisfied: nvidia-nvjitlink-cu12==12.8.93 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (12.8.93)
    Requirement already satisfied: nvidia-cufile-cu12==1.13.1.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (1.13.1.3)
    Requirement already satisfied: triton==3.6.0 in /usr/local/lib/python3.12/dist-packages (from torch>=1.7.1->grad-cam) (3.6.0)
    Requirement already satisfied: cuda-pathfinder~=1.1 in /usr/local/lib/python3.12/dist-packages (from cuda-bindings==12.9.4->torch>=1.7.1->grad-cam) (1.5.3)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib->grad-cam) (1.3.3)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.12/dist-packages (from matplotlib->grad-cam) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib->grad-cam) (4.62.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib->grad-cam) (1.5.0)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib->grad-cam) (26.1)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib->grad-cam) (3.3.2)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.12/dist-packages (from matplotlib->grad-cam) (2.9.0.post0)
    Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn->grad-cam) (1.16.3)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn->grad-cam) (1.5.3)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn->grad-cam) (3.6.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.7->matplotlib->grad-cam) (1.17.0)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch>=1.7.1->grad-cam) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch>=1.7.1->grad-cam) (3.0.3)
    Downloading ttach-0.0.3-py3-none-any.whl (9.8 kB)
    Building wheels for collected packages: grad-cam
      Building wheel for grad-cam (pyproject.toml) ... [?25l[?25hdone
      Created wheel for grad-cam: filename=grad_cam-1.5.5-py3-none-any.whl size=44286 sha256=c1d60c133a7316d3bd5be2c10020ebf999e8c35b383e82cc1b33edc8806262f8
      Stored in directory: /root/.cache/pip/wheels/fb/3b/09/2afc520f3d69bc26ae6bd87416759c820a3f7d05c1a077bbf6
    Successfully built grad-cam
    Installing collected packages: ttach, grad-cam
    Successfully installed grad-cam-1.5.5 ttach-0.0.3



```python
!python /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_gradcam.py
```

    Using device: cuda
    Loaded checkpoint — epoch 20, val loss 2.2197
    
    Found 10 test images. Saving PNGs to: /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/gradcam_outputs
    
      avail_bckm_1: detected classes → ['Table_Desk', 'Chair', 'Stationary_Laptop_Tablet', 'Stationary_PersonalItem']
    Figure(1200x800)
        [avail_bckm_1] Table_Desk (conf=0.65) → gradcam_avail_bckm_1_Table_Desk.png
    Figure(1200x800)
        [avail_bckm_1] Chair (conf=0.87) → gradcam_avail_bckm_1_Chair.png
    Figure(1200x800)
        [avail_bckm_1] Stationary_Laptop_Tablet (conf=0.39) → gradcam_avail_bckm_1_Stationary_Laptop_Tablet.png
    Figure(1200x800)
        [avail_bckm_1] Stationary_PersonalItem (conf=0.42) → gradcam_avail_bckm_1_Stationary_PersonalItem.png
      mixed_commuter_2: detected classes → ['Table_Desk', 'Chair', 'Seated_Student', 'Stationary_Laptop_Tablet', 'Stationary_PersonalItem']
    Figure(1200x800)
        [mixed_commuter_2] Table_Desk (conf=0.59) → gradcam_mixed_commuter_2_Table_Desk.png
    Figure(1200x800)
        [mixed_commuter_2] Chair (conf=0.89) → gradcam_mixed_commuter_2_Chair.png
    Figure(1200x800)
        [mixed_commuter_2] Seated_Student (conf=0.76) → gradcam_mixed_commuter_2_Seated_Student.png
    Figure(1200x800)
        [mixed_commuter_2] Stationary_Laptop_Tablet (conf=0.34) → gradcam_mixed_commuter_2_Stationary_Laptop_Tablet.png
    Figure(1200x800)
        [mixed_commuter_2] Stationary_PersonalItem (conf=0.52) → gradcam_mixed_commuter_2_Stationary_PersonalItem.png
      mixed_lib_4: detected classes → ['Table_Desk', 'Chair', 'Seated_Student', 'Stationary_Laptop_Tablet', 'Stationary_PersonalItem']
    Figure(1200x800)
        [mixed_lib_4] Table_Desk (conf=0.37) → gradcam_mixed_lib_4_Table_Desk.png
    Figure(1200x800)
        [mixed_lib_4] Chair (conf=0.84) → gradcam_mixed_lib_4_Chair.png
    Figure(1200x800)
        [mixed_lib_4] Seated_Student (conf=0.72) → gradcam_mixed_lib_4_Seated_Student.png
    Figure(1200x800)
        [mixed_lib_4] Stationary_Laptop_Tablet (conf=0.43) → gradcam_mixed_lib_4_Stationary_Laptop_Tablet.png
    Figure(1200x800)
        [mixed_lib_4] Stationary_PersonalItem (conf=0.44) → gradcam_mixed_lib_4_Stationary_PersonalItem.png
      occupied_bckm_1: detected classes → ['Chair', 'Seated_Student', 'Stationary_PersonalItem']
    Figure(1200x800)
        [occupied_bckm_1] Chair (conf=0.78) → gradcam_occupied_bckm_1_Chair.png
    Figure(1200x800)
        [occupied_bckm_1] Seated_Student (conf=0.49) → gradcam_occupied_bckm_1_Seated_Student.png
    Figure(1200x800)
        [occupied_bckm_1] Stationary_PersonalItem (conf=0.40) → gradcam_occupied_bckm_1_Stationary_PersonalItem.png
      occupied_bckm_11: detected classes → ['Table_Desk', 'Chair', 'Seated_Student', 'Stationary_PersonalItem']
    Figure(1200x800)
        [occupied_bckm_11] Table_Desk (conf=0.40) → gradcam_occupied_bckm_11_Table_Desk.png
    Figure(1200x800)
        [occupied_bckm_11] Chair (conf=0.56) → gradcam_occupied_bckm_11_Chair.png
    Figure(1200x800)
        [occupied_bckm_11] Seated_Student (conf=0.57) → gradcam_occupied_bckm_11_Seated_Student.png
    Figure(1200x800)
        [occupied_bckm_11] Stationary_PersonalItem (conf=0.36) → gradcam_occupied_bckm_11_Stationary_PersonalItem.png
      occupied_cafe_4: detected classes → ['Table_Desk', 'Chair', 'Seated_Student']
    Figure(1200x800)
        [occupied_cafe_4] Table_Desk (conf=0.41) → gradcam_occupied_cafe_4_Table_Desk.png
    Figure(1200x800)
        [occupied_cafe_4] Chair (conf=0.84) → gradcam_occupied_cafe_4_Chair.png
    Figure(1200x800)
        [occupied_cafe_4] Seated_Student (conf=0.79) → gradcam_occupied_cafe_4_Seated_Student.png
      occupied_lib_4: detected classes → ['Table_Desk', 'Chair', 'Seated_Student']
    Figure(1200x800)
        [occupied_lib_4] Table_Desk (conf=0.62) → gradcam_occupied_lib_4_Table_Desk.png
    Figure(1200x800)
        [occupied_lib_4] Chair (conf=0.69) → gradcam_occupied_lib_4_Chair.png
    Figure(1200x800)
        [occupied_lib_4] Seated_Student (conf=0.80) → gradcam_occupied_lib_4_Seated_Student.png
      reserved_commuter_1: detected classes → ['Table_Desk', 'Chair', 'Seated_Student', 'Stationary_Laptop_Tablet', 'Desktop_Inactive', 'Stationary_PersonalItem']
    Figure(1200x800)
        [reserved_commuter_1] Table_Desk (conf=0.31) → gradcam_reserved_commuter_1_Table_Desk.png
    Figure(1200x800)
        [reserved_commuter_1] Chair (conf=0.58) → gradcam_reserved_commuter_1_Chair.png
    Figure(1200x800)
        [reserved_commuter_1] Seated_Student (conf=0.64) → gradcam_reserved_commuter_1_Seated_Student.png
    Figure(1200x800)
        [reserved_commuter_1] Stationary_Laptop_Tablet (conf=0.45) → gradcam_reserved_commuter_1_Stationary_Laptop_Tablet.png
    Figure(1200x800)
        [reserved_commuter_1] Desktop_Inactive (conf=0.30) → gradcam_reserved_commuter_1_Desktop_Inactive.png
    Figure(1200x800)
        [reserved_commuter_1] Stationary_PersonalItem (conf=0.41) → gradcam_reserved_commuter_1_Stationary_PersonalItem.png
      reserved_commuter_2: detected classes → ['Table_Desk', 'Chair', 'Seated_Student', 'Stationary_PersonalItem']
    Figure(1200x800)
        [reserved_commuter_2] Table_Desk (conf=0.53) → gradcam_reserved_commuter_2_Table_Desk.png
    Figure(1200x800)
        [reserved_commuter_2] Chair (conf=0.83) → gradcam_reserved_commuter_2_Chair.png
    Figure(1200x800)
        [reserved_commuter_2] Seated_Student (conf=0.53) → gradcam_reserved_commuter_2_Seated_Student.png
    Figure(1200x800)
        [reserved_commuter_2] Stationary_PersonalItem (conf=0.35) → gradcam_reserved_commuter_2_Stationary_PersonalItem.png
      reserved_lib_3: detected classes → ['Table_Desk', 'Chair', 'Seated_Student', 'Desktop_Inactive', 'Stationary_PersonalItem']
    Figure(1200x800)
        [reserved_lib_3] Table_Desk (conf=0.40) → gradcam_reserved_lib_3_Table_Desk.png
    Figure(1200x800)
        [reserved_lib_3] Chair (conf=0.85) → gradcam_reserved_lib_3_Chair.png
    Figure(1200x800)
        [reserved_lib_3] Seated_Student (conf=0.39) → gradcam_reserved_lib_3_Seated_Student.png
    Figure(1200x800)
        [reserved_lib_3] Desktop_Inactive (conf=0.43) → gradcam_reserved_lib_3_Desktop_Inactive.png
    Figure(1200x800)
        [reserved_lib_3] Stationary_PersonalItem (conf=0.44) → gradcam_reserved_lib_3_Stationary_PersonalItem.png
    
    Done. 42 PNGs saved to /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/gradcam_outputs


# Testing performance on test data


```python
!python /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_diagnose.py
```

    Device: cuda
    loading annotations into memory...
    Done (t=0.03s)
    creating index...
    index created!
    Test images: 10
    
    ============================================================
      DIAGNOSING: best_weights  (scratch_best_weights.pth)
    ============================================================
    
    ============================================================
      CHECK 1 — Checkpoint file
    ============================================================
      Path  : /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_best_weights.pth
      Size  : 48.3 MB
      Type  : dict  (keys: ['epoch', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict', 'best_val_loss', 'best_lr'])
      epoch           : 15
      best_val_loss   : 1.9517597377300262
      best_lr         : 0.001595857
      Tensors in state dict : 206
      Total parameters      : 4,207,068  (~4.21 M)
    
    ============================================================
      CHECK 2 — State-dict key alignment
    ============================================================
      PASS  : all keys match perfectly.
      PASS  : all tensor shapes match.
    
      load_state_dict (strict=True) : OK
    
    ============================================================
      CHECK 3 — Raw score distribution (score_thresh=0.001)
    ============================================================
      Running on 10 test image(s) …
    
      Image    1  GT= 6  Total proposals=  0  Top-10 scores: []
      Image    2  GT=13  Total proposals=  0  Top-10 scores: []
      Image    3  GT=15  Total proposals=  0  Top-10 scores: []
      Image    4  GT= 7  Total proposals=  0  Top-10 scores: []
      Image    5  GT=17  Total proposals=  0  Top-10 scores: []
      Image    6  GT=16  Total proposals=  0  Top-10 scores: []
      Image    7  GT= 8  Total proposals=  0  Top-10 scores: []
      Image    8  GT=18  Total proposals=  0  Top-10 scores: []
      Image    9  GT=27  Total proposals=  0  Top-10 scores: []
      Image   10  GT= 6  Total proposals=  0  Top-10 scores: []
    
    ============================================================
      CHECK 4 — Score summary across all images
    ============================================================
      WARNING : model returned zero proposals on all images.
      This suggests a model architecture or weight loading problem.
    
    ============================================================
      CHECK 5 — Visualization at score_threshold=0.05
    ============================================================
      Saved: diag_0001.png  (0 detections)
      Saved: diag_0002.png  (0 detections)
      Saved: diag_0003.png  (0 detections)
      Saved: diag_0004.png  (0 detections)
      Saved: diag_0005.png  (0 detections)
      Saved: diag_0006.png  (0 detections)
      Saved: diag_0007.png  (0 detections)
      Saved: diag_0008.png  (0 detections)
      Saved: diag_0009.png  (0 detections)
      Saved: diag_0010.png  (0 detections)
    
    ============================================================
      DIAGNOSING: best_weights_50ep  (scratch_best_weights_50ep.pth)
    ============================================================
    
    ============================================================
      CHECK 1 — Checkpoint file
    ============================================================
      Path  : /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_best_weights_50ep.pth
      Size  : 48.3 MB
      Type  : dict  (keys: ['epoch', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict', 'best_val_loss', 'best_lr'])
      epoch           : 25
      best_val_loss   : 1.9816815614700318
      best_lr         : 0.001595857
      Tensors in state dict : 206
      Total parameters      : 4,207,068  (~4.21 M)
    
    ============================================================
      CHECK 2 — State-dict key alignment
    ============================================================
      PASS  : all keys match perfectly.
      PASS  : all tensor shapes match.
    
      load_state_dict (strict=True) : OK
    
    ============================================================
      CHECK 3 — Raw score distribution (score_thresh=0.001)
    ============================================================
      Running on 10 test image(s) …
    
      Image    1  GT= 6  Total proposals=  0  Top-10 scores: []
      Image    2  GT=13  Total proposals=  0  Top-10 scores: []
      Image    3  GT=15  Total proposals=  0  Top-10 scores: []
      Image    4  GT= 7  Total proposals=  0  Top-10 scores: []
      Image    5  GT=17  Total proposals=  0  Top-10 scores: []
      Image    6  GT=16  Total proposals=  0  Top-10 scores: []
      Image    7  GT= 8  Total proposals=  0  Top-10 scores: []
      Image    8  GT=18  Total proposals=  0  Top-10 scores: []
      Image    9  GT=27  Total proposals=  0  Top-10 scores: []
      Image   10  GT= 6  Total proposals=  0  Top-10 scores: []
    
    ============================================================
      CHECK 4 — Score summary across all images
    ============================================================
      WARNING : model returned zero proposals on all images.
      This suggests a model architecture or weight loading problem.
    
    ============================================================
      CHECK 5 — Visualization at score_threshold=0.05
    ============================================================
      Saved: diag_0001.png  (0 detections)
      Saved: diag_0002.png  (0 detections)
      Saved: diag_0003.png  (0 detections)
      Saved: diag_0004.png  (0 detections)
      Saved: diag_0005.png  (0 detections)
      Saved: diag_0006.png  (0 detections)
      Saved: diag_0007.png  (0 detections)
      Saved: diag_0008.png  (0 detections)
      Saved: diag_0009.png  (0 detections)
      Saved: diag_0010.png  (0 detections)
    
    ============================================================
      DIAGNOSIS SUMMARY
    ============================================================
    
      Interpret results:
    
      If CHECK 3 shows:
        • Total proposals = 0 on all images
          → Model is producing NO output at all. This is a fundamental
            architecture/weight problem. Check key mismatches in CHECK 2.
    
        • Proposals exist but max score < 0.3
          → The model learned something but is under-confident. The 50-epoch
            checkpoint (scratch_best_weights_50ep.pth) may be better trained.
            Try lowering --score_threshold to 0.1 or 0.05 in scratch_test.py.
    
        • Proposals exist with scores >= 0.3 but original test showed nothing
          → The issue was a threshold that was too high. Use --score_threshold 0.1.
    
      If CHECK 2 shows key mismatches:
        → The checkpoint was saved from a different version of the architecture.
          The factory file may have been updated after training.
    
      Visualizations are saved to: scratch_diag_outputs/
        



```python
!python /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_probe.py
```

    Device: cuda
    loading annotations into memory...
    Done (t=0.03s)
    creating index...
    index created!
    Test dataset: 10 images  |  probing image 0
    Checkpoint: /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_best_weights.pth
    
    ==============================================================
      CHECK 1 — NaN / Inf in weights
    ==============================================================
      PASS — no NaN or Inf found in any weight tensor.
    
    ==============================================================
      CHECK 2 — Weight magnitude (trained vs. random-init)
    ==============================================================
    
      Layer                                          Trained std    Random std  Status
      --------------------------------------------- ------------  ------------  ----------
      backbone.body.stem.0.weight                       0.086576      0.110105  OK
      backbone.body.stage1.0.path_a.weight              0.085501      0.033931  OK
      backbone.body.stage2.0.path_a.weight              0.062923      0.024040  OK
      backbone.body.stage3.0.path_a.weight              0.052053      0.016991  OK
      backbone.body.stage4.0.path_a.weight              0.043563      0.013886  OK
      backbone.fpn.inner_blocks.0.0.weight              0.126854      0.125561  OK
      rpn.head.conv.0.0.weight                          0.018515      0.009966  OK
      rpn.head.cls_logits.weight                        0.022298      0.009914  OK
      roi_heads.box_predictor.cls_score.weight          0.042337      0.036123  OK
      roi_heads.mask_predictor.mask_fcn_logits.weight     0.480716      0.497785  OK
    
    ==============================================================
      CHECK 3 — Backbone + FPN feature stats
    ==============================================================
    
      Input tensor after transform:  transformed_input                    shape=(1, 3, 800, 1088)  mean=-1.6275  std=4.6119  min=-11.366392  max=+9.928890
    
      FPN level [0]                        shape=(1, 96, 200, 272)  mean=-18.9644  std=444.3141  min=-2752.153076  max=+2933.392578
      FPN level [1]                        shape=(1, 96, 100, 136)  mean=-61.6836  std=461.0372  min=-3215.431885  max=+2316.169434
      FPN level [2]                        shape=(1, 96, 50, 68)  mean=-7.9013  std=409.6845  min=-2473.065918  max=+2286.465088
      FPN level [3]                        shape=(1, 96, 25, 34)  mean=+12.5846  std=324.4200  min=-1879.753540  max=+2171.573975
      FPN level [pool]                     shape=(1, 96, 13, 17)  mean=+11.8332  std=318.8182  min=-1879.753540  max=+2171.573975
    
      Backbone features look non-zero.
    
    ==============================================================
      CHECK 4 — RPN objectness logits (before sigmoid / NMS)
    ==============================================================
      FPN level 0: 163200 anchors  logit mean=+37.3863  sig mean=0.9977  sig>0.5: 162889  sig>0.1: 163042
      FPN level 1:  40800 anchors  logit mean=+6.0170  sig mean=0.5726  sig>0.5: 23396  sig>0.1: 25295
      FPN level 2:  10200 anchors  logit mean=-112.2471  sig mean=0.3048  sig>0.5: 3103  sig>0.1: 3139
      FPN level 3:   2550 anchors  logit mean=-65.8488  sig mean=0.4290  sig>0.5: 1094  sig>0.1: 1114
      FPN level 4:    663 anchors  logit mean=-60.1735  sig mean=0.4381  sig>0.5: 291  sig>0.1: 294
    
      Total anchors across all levels: 217,413
      Objectness logit range: [-871.6190, +420.7303]
      Objectness sigmoid range: [0.000000, 1.000000]
    
      RPN objectness looks normal.
    
    ==============================================================
      CHECK 5 — RPN proposals (before ROI box-head score filter)
    ==============================================================
    
      RPN proposals generated (before ROI head): 6
      First 5 proposal boxes: [[0.0, 84.16473388671875, 1066.0, 800.0], [444.623779296875, 0.0, 444.62481689453125, 800.0], [0.0, 0.0, 1066.0, 800.0], [0.0, 0.0, 141.9404296875, 800.0], [0.0, 0.0, 319.2255859375, 800.0]]
    
    ==============================================================
      CHECK 6 — Trained vs. random-init objectness comparison
    ==============================================================
      trained  →  proposals=   0  max_score=0.0000
      random   →  proposals= 100  max_score=0.1505
    
      If random model also gives 0 proposals:
        → The architecture itself is the issue (RPN misconfigured, or
          backbone output shape incompatible with RPN anchor sizes).
      If random model gives proposals but trained model does not:
        → Training diverged or weights were corrupted on save.
    
    ==============================================================
      INTERPRETATION GUIDE
    ==============================================================
    
      CHECK 3 — FPN features all zero:
        → Backbone weights are dead/wrong. Key names may match but values
          loaded into incorrect layers (architecture mismatch).
    
      CHECK 4 — Objectness sigmoid uniformly < 0.05:
        → RPN never learned to fire. Either training diverged early, or
          the backbone is dead (see Check 3).
    
      CHECK 4 — Objectness sigmoid max = 0.5 (i.e. all logits near 0):
        → Weights are near-random initialization. The checkpoint may not
          contain a trained model (e.g. saved at epoch 0 before training).
    
      CHECK 5 — Zero RPN proposals:
        → Backbone features are dead OR anchor sizes don't match image scale.
          Check anchor_generator sizes vs. the actual image size after transform.
    
      CHECK 6 — Random model also gives 0 proposals:
        → Architecture issue in build_scratch_mask_rcnn itself. The FPN
          output channels or anchor config may be incompatible.
    
      CHECK 6 — Random model gives proposals, trained gives 0:
        → Training failure: weights loaded but the model never converged,
          or converged to a degenerate solution. Try scratch_best_weights_50ep.pth.
        



```python
!python /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_test.py
```

    Device     : cuda
    Checkpoint : /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_best_weights.pth
    Model loaded.
    
    loading annotations into memory...
    Done (t=1.79s)
    creating index...
    index created!
    Test images  : 10
    Score thresh : 0.5
    IoU thresh   : 0.5
    
    Running evaluation …
      [ 1/10] Image    1  GT= 6  Pred=10  P-IoU=0.5524
      [ 2/10] Image    2  GT=13  Pred= 8  P-IoU=0.3758
      [ 3/10] Image    3  GT=15  Pred= 4  P-IoU=0.1608
      [ 4/10] Image    4  GT= 7  Pred= 3  P-IoU=0.2215
      [ 5/10] Image    5  GT=17  Pred= 4  P-IoU=0.1982
      [ 6/10] Image    6  GT=16  Pred=12  P-IoU=0.3714
      [ 7/10] Image    7  GT= 8  Pred= 5  P-IoU=0.0477
      [ 8/10] Image    8  GT=18  Pred= 2  P-IoU=0.0763
      [ 9/10] Image    9  GT=27  Pred= 8  P-IoU=0.3828
      [10/10] Image   10  GT= 6  Pred= 3  P-IoU=0.2904
    
    =========================================================================
      PER-CLASS BREAKDOWN
    =========================================================================
      Class                          |   Det |  Avg Conf |  Box IoU |  Mask IoU
    -------------------------------------------------------------------------
      Table_Desk                     |     3 |   0.5736  |  0.4870 |   0.4624
      Chair                          |    27 |   0.6767  |  0.4775 |   0.3999
      Seated_Student                 |    16 |   0.6403  |  0.4179 |   0.3038
      Stationary_PersonalItem        |     1 |   0.5194  |  0.3002 |   0.1954
    =========================================================================
    
    ====================================================
      FINAL TEST RESULTS — ScratchNet Mask R-CNN
    ====================================================
      P-IoU (pixel mask, avg over images) : 0.2677
      I-IoU (box @ 0.50, matches/GT objects) : 0.1504
    ====================================================
    
    Saving visualizations to: /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_test_outputs
    Saved 10 image(s). Done.


# Logic Layer - Study Space Detection

### Scratch model


```python
!python /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_study_space.py --score_threshold 0.3 --model scratch
```

    Device: cuda
    Model      : scratch
    Checkpoint : /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_best_weights.pth
    Model loaded.
    loading annotations into memory...
    Done (t=0.04s)
    creating index...
    index created!
    Validation images: 10
    
    Score threshold : 0.3
    Overlap threshold: 0.3
    Output directory: /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/study_space_outputs
    
    Running inference …
      Image    1: 4 table(s)  Available=4  Reserved=0  Occupied=0
      Image    2: 1 table(s)  Available=1  Reserved=0  Occupied=0
      Image    3: 5 table(s)  Available=3  Reserved=0  Occupied=2
      Image    4: 4 table(s)  Available=4  Reserved=0  Occupied=0
      Image    5: 2 table(s)  Available=2  Reserved=0  Occupied=0
      Image    6: 9 table(s)  Available=7  Reserved=1  Occupied=1
      Image    7: 10 table(s)  Available=10  Reserved=0  Occupied=0
      Image    8: 3 table(s)  Available=3  Reserved=0  Occupied=0
      Image    9: 8 table(s)  Available=8  Reserved=0  Occupied=0
      Image   10: 3 table(s)  Available=3  Reserved=0  Occupied=0
    
    ==============================================================
      STUDY SPACE AVAILABILITY REPORT — Validation Set
    ==============================================================
      Image    1: 4 table(s)  Available=4  Reserved=0  Occupied=0
        Table 0: AVAILABLE
        Table 1: AVAILABLE
        Table 2: AVAILABLE
        Table 3: AVAILABLE
      Image    2: 1 table(s)  Available=1  Reserved=0  Occupied=0
        Table 0: AVAILABLE
      Image    3: 5 table(s)  Available=3  Reserved=0  Occupied=2
        Table 0: AVAILABLE
        Table 1: AVAILABLE
        Table 2: OCCUPIED  [Seated_Student]
        Table 3: AVAILABLE
        Table 4: OCCUPIED  [Seated_Student]
      Image    4: 4 table(s)  Available=4  Reserved=0  Occupied=0
        Table 0: AVAILABLE
        Table 1: AVAILABLE
        Table 2: AVAILABLE
        Table 3: AVAILABLE
      Image    5: 2 table(s)  Available=2  Reserved=0  Occupied=0
        Table 0: AVAILABLE
        Table 1: AVAILABLE
      Image    6: 9 table(s)  Available=7  Reserved=1  Occupied=1
        Table 0: AVAILABLE
        Table 1: AVAILABLE
        Table 2: AVAILABLE
        Table 3: AVAILABLE
        Table 4: AVAILABLE
        Table 5: AVAILABLE
        Table 6: RESERVED  [Stationary_Laptop_Tablet]
        Table 7: OCCUPIED  [Seated_Student]
        Table 8: AVAILABLE
      Image    7: 10 table(s)  Available=10  Reserved=0  Occupied=0
        Table 0: AVAILABLE
        Table 1: AVAILABLE
        Table 2: AVAILABLE
        Table 3: AVAILABLE
        Table 4: AVAILABLE
        Table 5: AVAILABLE
        Table 6: AVAILABLE
        Table 7: AVAILABLE
        Table 8: AVAILABLE
        Table 9: AVAILABLE
      Image    8: 3 table(s)  Available=3  Reserved=0  Occupied=0
        Table 0: AVAILABLE
        Table 1: AVAILABLE
        Table 2: AVAILABLE
      Image    9: 8 table(s)  Available=8  Reserved=0  Occupied=0
        Table 0: AVAILABLE
        Table 1: AVAILABLE
        Table 2: AVAILABLE
        Table 3: AVAILABLE
        Table 4: AVAILABLE
        Table 5: AVAILABLE
        Table 6: AVAILABLE
        Table 7: AVAILABLE
      Image   10: 3 table(s)  Available=3  Reserved=0  Occupied=0
        Table 0: AVAILABLE
        Table 1: AVAILABLE
        Table 2: AVAILABLE
    --------------------------------------------------------------
      TOTAL  : 49 table(s)  Available=45  Reserved=1  Occupied=3
               Available     91.8%
               Reserved       2.0%
               Occupied       6.1%
    ==============================================================
    
    Per-image visualisations saved to: /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/study_space_outputs


### MaskRCNN ResNet50 Exp 1 (Epoch 10)


```python
!python /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_study_space.py --score_threshold 0.5

```

    Device: cuda
    Model      : resnet50
    Checkpoint : /content/drive/MyDrive/deep_learning_project/exp1_checkpoint_10.pth
    Downloading: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth" to /root/.cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
    100% 170M/170M [00:01<00:00, 171MB/s]
    Model loaded.
    loading annotations into memory...
    Done (t=0.04s)
    creating index...
    index created!
    Validation images: 10
    
    Score threshold : 0.5
    Overlap threshold: 0.3
    Output directory: /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/study_space_outputs
    
    Running inference …
      Image    1: 2 table(s)  Available=2  Reserved=0  Occupied=0
      Image    2: 8 table(s)  Available=8  Reserved=0  Occupied=0
      Image    3: 5 table(s)  Available=3  Reserved=2  Occupied=0
      Image    4: 5 table(s)  Available=5  Reserved=0  Occupied=0
      Image    5: 5 table(s)  Available=2  Reserved=3  Occupied=0
      Image    6: 5 table(s)  Available=5  Reserved=0  Occupied=0
      Image    7: 3 table(s)  Available=3  Reserved=0  Occupied=0
      Image    8: 1 table(s)  Available=1  Reserved=0  Occupied=0
      Image    9: 4 table(s)  Available=4  Reserved=0  Occupied=0
      Image   10: 2 table(s)  Available=2  Reserved=0  Occupied=0
    
    ==============================================================
      STUDY SPACE AVAILABILITY REPORT — Validation Set
    ==============================================================
      Image    1: 2 table(s)  Available=2  Reserved=0  Occupied=0
        Table 0: AVAILABLE
        Table 1: AVAILABLE
      Image    2: 8 table(s)  Available=8  Reserved=0  Occupied=0
        Table 0: AVAILABLE
        Table 1: AVAILABLE
        Table 2: AVAILABLE
        Table 3: AVAILABLE
        Table 4: AVAILABLE
        Table 5: AVAILABLE
        Table 6: AVAILABLE
        Table 7: AVAILABLE
      Image    3: 5 table(s)  Available=3  Reserved=2  Occupied=0
        Table 0: RESERVED  [Stationary_PersonalItem]
        Table 1: AVAILABLE
        Table 2: AVAILABLE
        Table 3: AVAILABLE
        Table 4: RESERVED  [Stationary_PersonalItem]
      Image    4: 5 table(s)  Available=5  Reserved=0  Occupied=0
        Table 0: AVAILABLE
        Table 1: AVAILABLE
        Table 2: AVAILABLE
        Table 3: AVAILABLE
        Table 4: AVAILABLE
      Image    5: 5 table(s)  Available=2  Reserved=3  Occupied=0
        Table 0: RESERVED  [Stationary_PersonalItem]
        Table 1: RESERVED  [Stationary_PersonalItem, Stationary_PersonalItem, Stationary_PersonalItem]
        Table 2: RESERVED  [Stationary_PersonalItem]
        Table 3: AVAILABLE
        Table 4: AVAILABLE
      Image    6: 5 table(s)  Available=5  Reserved=0  Occupied=0
        Table 0: AVAILABLE
        Table 1: AVAILABLE
        Table 2: AVAILABLE
        Table 3: AVAILABLE
        Table 4: AVAILABLE
      Image    7: 3 table(s)  Available=3  Reserved=0  Occupied=0
        Table 0: AVAILABLE
        Table 1: AVAILABLE
        Table 2: AVAILABLE
      Image    8: 1 table(s)  Available=1  Reserved=0  Occupied=0
        Table 0: AVAILABLE
      Image    9: 4 table(s)  Available=4  Reserved=0  Occupied=0
        Table 0: AVAILABLE
        Table 1: AVAILABLE
        Table 2: AVAILABLE
        Table 3: AVAILABLE
      Image   10: 2 table(s)  Available=2  Reserved=0  Occupied=0
        Table 0: AVAILABLE
        Table 1: AVAILABLE
    --------------------------------------------------------------
      TOTAL  : 40 table(s)  Available=35  Reserved=5  Occupied=0
               Available     87.5%
               Reserved      12.5%
               Occupied       0.0%
    ==============================================================
    
    Per-image visualisations saved to: /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/study_space_outputs



```python
%run /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/fetch_clearml_iou.py
```

    Fetched task: Final-Training-50ep-GN  (id=0735745c085c4b13b78d94c2ebf98465)
    Val epochs logged: [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    | Epoch | LR | Train Loss | Val Loss | Pixel IoU | Instance IoU | Saved |
    |-------|----|-----------|----------|-----------|--------------|-------|
    | 1 | 1.3293e-04 | 2.6552 | 2.5481 | 0.2695 | 0.1130 | ✓ |
    | 5 | 1.3086e-04 | 2.1960 | 2.2573 | 0.3694 | 0.1864 | — |
    | 10 | 1.2266e-04 | 2.0684 | 2.3363 | 0.4299 | 0.2994 | — |
    | 15 | 1.0901e-04 | 1.9081 | 2.2518 | 0.3974 | 0.4576 | — |
    | **20** | **9.1248e-05** | **1.7651** | **2.2197** | **0.4382** | **0.4350** | **✓ best** |
    | 25 | 7.1107e-05 | 1.5988 | 2.2455 | 0.4543 | 0.4633 | — |
    | 30 | 5.0560e-05 | 1.4646 | 2.3160 | 0.4534 | 0.4972 | — |
    | 35 | 3.1619e-05 | 1.3724 | 2.4123 | 0.4598 | 0.4802 | — |
    | 40 | 1.6138e-05 | 1.2823 | 2.4217 | 0.4787 | 0.4746 | — |
    | 45 | 5.6323e-06 | 1.2416 | 2.4870 | 0.4803 | 0.4237 | — |
    | 50 | 1.1302e-06 | 1.2211 | 2.4936 | 0.4834 | 0.4463 | — |


### Logic Layer for Study Space Detection
- confidence must be 0.7 for students and 0.4 for everything else.
- overlap threshold currently 0.4


```python
!python /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_study_space.py --model scratch --score_threshold 0.25 --overlap_threshold 0.7 --student_threshold 0.7
```

    Device: cuda
    Model      : scratch
    Checkpoint : /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/scratch_best_weights.pth
    Model loaded.
    loading annotations into memory...
    Done (t=0.04s)
    creating index...
    index created!
    Validation images: 10
    
    Score threshold  : 0.25  (Seated_Student: 0.7)
    Overlap threshold: 0.7
    Output directory : /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/study_space_outputs
    
    Running inference …
      Image    1: 6 table(s)  Available=0  Reserved=3  Occupied=3
      Image    2: 4 table(s)  Available=4  Reserved=0  Occupied=0
      Image    3: 7 table(s)  Available=0  Reserved=0  Occupied=7
      Image    4: 8 table(s)  Available=0  Reserved=3  Occupied=5
      Image    5: 2 table(s)  Available=0  Reserved=0  Occupied=2
      Image    6: 13 table(s)  Available=1  Reserved=12  Occupied=0
      Image    7: 10 table(s)  Available=5  Reserved=5  Occupied=0
      Image    8: 3 table(s)  Available=0  Reserved=3  Occupied=0
      Image    9: 13 table(s)  Available=11  Reserved=2  Occupied=0
      Image   10: 3 table(s)  Available=0  Reserved=1  Occupied=2
    
    ==============================================================
      STUDY SPACE AVAILABILITY REPORT — Validation Set
    ==============================================================
      Image    1: 6 table(s)  Available=0  Reserved=3  Occupied=3
        Table 0: OCCUPIED  [Seated_Student, Seated_Student, Seated_Student, Seated_Student, Seated_Student]
        Table 1: RESERVED  [Stationary_PersonalItem]
        Table 2: RESERVED  [Stationary_PersonalItem]
        Table 3: OCCUPIED  [Seated_Student]
        Table 4: RESERVED  [Stationary_PersonalItem, Stationary_PersonalItem]
        Table 5: OCCUPIED  [Seated_Student, Seated_Student, Seated_Student, Seated_Student]
      Image    2: 4 table(s)  Available=4  Reserved=0  Occupied=0
        Table 0: AVAILABLE
        Table 1: AVAILABLE
        Table 2: AVAILABLE
        Table 3: AVAILABLE
      Image    3: 7 table(s)  Available=0  Reserved=0  Occupied=7
        Table 0: OCCUPIED  [Seated_Student]
        Table 1: OCCUPIED  [Seated_Student]
        Table 2: OCCUPIED  [Seated_Student, Seated_Student, Seated_Student]
        Table 3: OCCUPIED  [Seated_Student, Seated_Student]
        Table 4: OCCUPIED  [Seated_Student, Seated_Student]
        Table 5: OCCUPIED  [Seated_Student]
        Table 6: OCCUPIED  [Seated_Student]
      Image    4: 8 table(s)  Available=0  Reserved=3  Occupied=5
        Table 0: OCCUPIED  [Seated_Student]
        Table 1: RESERVED  [Stationary_PersonalItem, Stationary_PersonalItem]
        Table 2: OCCUPIED  [Seated_Student]
        Table 3: OCCUPIED  [Seated_Student]
        Table 4: OCCUPIED  [Seated_Student]
        Table 5: OCCUPIED  [Seated_Student]
        Table 6: RESERVED  [Stationary_PersonalItem, Stationary_PersonalItem]
        Table 7: RESERVED  [Stationary_PersonalItem, Stationary_PersonalItem]
      Image    5: 2 table(s)  Available=0  Reserved=0  Occupied=2
        Table 0: OCCUPIED  [Seated_Student]
        Table 1: OCCUPIED  [Seated_Student]
      Image    6: 13 table(s)  Available=1  Reserved=12  Occupied=0
        Table 0: RESERVED  [Stationary_PersonalItem]
        Table 1: RESERVED  [Stationary_PersonalItem, Stationary_PersonalItem]
        Table 2: AVAILABLE
        Table 3: RESERVED  [Stationary_PersonalItem, Stationary_PersonalItem]
        Table 4: RESERVED  [Stationary_PersonalItem, Stationary_PersonalItem]
        Table 5: RESERVED  [Stationary_PersonalItem, Stationary_PersonalItem]
        Table 6: RESERVED  [Stationary_PersonalItem, Stationary_Laptop_Tablet, Stationary_Laptop_Tablet, Stationary_Laptop_Tablet, Stationary_PersonalItem]
        Table 7: RESERVED  [Stationary_PersonalItem, Stationary_Laptop_Tablet, Stationary_PersonalItem]
        Table 8: RESERVED  [Stationary_PersonalItem, Stationary_Laptop_Tablet]
        Table 9: RESERVED  [Stationary_PersonalItem, Stationary_PersonalItem]
        Table 10: RESERVED  [Stationary_PersonalItem, Stationary_PersonalItem]
        Table 11: RESERVED  [Stationary_PersonalItem]
        Table 12: RESERVED  [Stationary_PersonalItem]
      Image    7: 10 table(s)  Available=5  Reserved=5  Occupied=0
        Table 0: RESERVED  [Stationary_PersonalItem, Stationary_PersonalItem]
        Table 1: AVAILABLE
        Table 2: RESERVED  [Stationary_PersonalItem]
        Table 3: RESERVED  [Stationary_PersonalItem, Stationary_PersonalItem]
        Table 4: AVAILABLE
        Table 5: AVAILABLE
        Table 6: AVAILABLE
        Table 7: RESERVED  [Stationary_PersonalItem]
        Table 8: AVAILABLE
        Table 9: RESERVED  [Stationary_PersonalItem]
      Image    8: 3 table(s)  Available=0  Reserved=3  Occupied=0
        Table 0: RESERVED  [Stationary_PersonalItem, Stationary_PersonalItem, Stationary_PersonalItem, Stationary_PersonalItem, Stationary_PersonalItem]
        Table 1: RESERVED  [Stationary_PersonalItem, Stationary_PersonalItem, Stationary_PersonalItem, Stationary_PersonalItem, Stationary_PersonalItem]
        Table 2: RESERVED  [Stationary_PersonalItem, Stationary_PersonalItem, Stationary_PersonalItem, Stationary_PersonalItem, Stationary_PersonalItem]
      Image    9: 13 table(s)  Available=11  Reserved=2  Occupied=0
        Table 0: RESERVED  [Stationary_Laptop_Tablet]
        Table 1: RESERVED  [Stationary_Laptop_Tablet, Stationary_PersonalItem]
        Table 2: AVAILABLE
        Table 3: AVAILABLE
        Table 4: AVAILABLE
        Table 5: AVAILABLE
        Table 6: AVAILABLE
        Table 7: AVAILABLE
        Table 8: AVAILABLE
        Table 9: AVAILABLE
        Table 10: AVAILABLE
        Table 11: AVAILABLE
        Table 12: AVAILABLE
      Image   10: 3 table(s)  Available=0  Reserved=1  Occupied=2
        Table 0: OCCUPIED  [Seated_Student]
        Table 1: OCCUPIED  [Seated_Student]
        Table 2: RESERVED  [Stationary_PersonalItem]
    --------------------------------------------------------------
      TOTAL  : 69 table(s)  Available=21  Reserved=29  Occupied=19
               Available     30.4%
               Reserved      42.0%
               Occupied      27.5%
    ==============================================================
    
    Per-image visualisations saved to: /content/drive/MyDrive/deep_learning_project/Scratch_MaskRCNN/study_space_outputs



```python

```

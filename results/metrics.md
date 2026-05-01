# Experimental Results

All runs: YOLOv8n, COCO128, 50 epochs, 640x640, batch 16, Tesla T4 (14 913 MiB), seed 0.

## Validation metrics (mAP, val split = train split for COCO128)

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| From scratch | ~0.00 | ~0.00 | ~0.00 | ~0.00 |
| Transfer learning | **0.729** | **0.553** | **0.729** | **0.668** |

> Observed during training: the from-scratch model produced zero detections
> on the validation set through epoch 50, confirming the regime of severe
> data starvation (128 training images, 80 classes).

## Transfer learning — learning curve snapshots

| Epoch | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| 1 | 0.615 | 0.453 |
| 3 | 0.634 | 0.466 |
| 5 | 0.671 | 0.496 |
| 7 | 0.691 | 0.516 |
| 9 | 0.714 | 0.537 |
| 12 | 0.729 | 0.553 |

## Environment

```
ultralytics  8.3.253
torch        2.11.0+cu130
torchvision  0.26.0
Python       3.11.12
GPU          Tesla T4 (14 913 MiB)
CUDA         13.0
```

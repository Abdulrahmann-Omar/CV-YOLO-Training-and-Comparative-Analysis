# Experimental Results

All runs: YOLOv8n, coco128.yaml, 50 epochs, 640x640, batch 16, Tesla T4 (14 913 MiB), seed 0.

## Validation metrics

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Detections |
|---|---|---|---|---|---|
| From scratch | 0.004 | 0.001 | 0.007 | 0.002 | 0 |
| Transfer learning | 0.843 | 0.671 | 0.852 | 0.767 | 4 |

## Notes

- The figures in results/figures/ are generated from the same run.
- The metrics above are written directly from the Modal output.

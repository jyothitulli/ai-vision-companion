# Dataset and fine-tuning

Do not train from scratch. Start from `yolo11n.pt`.

Layout:

```
datasets/accessibility/
  data.yaml
  train/images|labels
  validation/images|labels
  testing/images|labels
  annotations/
```

Process:

1. Baseline evaluation of pretrained YOLO on a held-out set
2. Collect failure metadata with `training/collect_failures.py` (images off by default)
3. Annotate YOLO boxes for stairs, ramps, curbs, potholes, blocked walkways, etc.
4. `python training/train_accessibility.py --data datasets/accessibility/data.yaml`
5. Compare baseline vs fine-tuned with `evaluation/evaluate.py --predictions ...`

This repository does not contain fabricated mAP numbers or training logs.

# Detection and Counting

This package trains a YOLO detector on the synthetic dataset and then counts moths by counting the predicted bounding boxes.

The synthetic `labels/` files are the training supervision for YOLO. 
The pretrained starting checkpoint lives in
[assets/pretrained/README.md](../../assets/pretrained/README.md) and is
expected at `assets/pretrained/yolo11n.pt`. If that local file exists, the trainer uses it directly. Otherwise it falls back to the Ultralytics checkpoint name `yolo11n.pt`.

## Train on synthetic data

```bash
uv run seed-moth-train \
  --synthetic-root data/synthetic \
  --dataset-root results/detection/yolo_dataset \
  --output-dir results/models/detection
```

Outputs:

- `results/detection/yolo_dataset/dataset.yaml`
- `results/detection/yolo_dataset/split.json`
- `results/models/detection/train/weights/best.pt`
- `results/models/detection/best.pt`

The training script creates a train/validation split from `data/synthetic`, writes a YOLO `dataset.yaml`, and fine-tunes a pretrained detector.

## Predict and count

```bash
uv run seed-moth-predict \
  --model results/models/detection/best.pt \
  --inputs data/test_images \
  --output-dir results/predictions/detection
```

Outputs:

- `results/predictions/detection/predictions.json`
- `results/predictions/detection/counts.json`
- `results/predictions/detection/labels/`
- `results/predictions/detection/preview/`

The predicted label files are in YOLO format and the `preview/` folder contains the same images with boxes drawn on top. The count is just the number of boxes after YOLO inference and NMS (box with the bigest confidence score).

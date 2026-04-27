# Evaluation

This package evaluates a trained YOLO detector against labeled images.

The default target is the synthetic validation split that is created by
`seed-moth-train`, so you can measure the model against the held-out synthetic
ground truth immediately after training.

## Evaluate on synthetic validation data

```bash
uv run seed-moth-evaluate \
  --model results/models/detection/best.pt
```

Defaults used by the command above:

- inputs: `results/detection/yolo_dataset/images/val`
- labels: `results/detection/yolo_dataset/labels/val`
- output: `results/eval/val`

Outputs:

- `results/eval/val/metrics.json`
- `results/eval/val/per_image.json`

The report includes:

- detection `precision`, `recall`, `f1`, and mean IoU
- count `bias`, `mae`, `mse`, and `rmse`
- per-image TP / FP / FN and count errors

## Evaluate on labeled real images

For real images that have labels, point the evaluator at the labeled folder.

For example:
```bash
uv run seed-moth-evaluate \
  --model results/models/detection/best.pt \
  --inputs data/test_images \
  --labels-root results/test_images/labels \
  --output-dir results/eval/test
```

This uses the same metrics, but on real held-out data instead of synthetic validation data.

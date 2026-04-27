# Pretrained Weights

Pretrained YOLO checkpoints for a fully local, reproducible
training start.

Recommended file:

- `assets/pretrained/yolo11n.pt`

If that file exists, `seed-moth-train` uses it directly and does not ask
Ultralytics to resolve a checkpoint name.

If the file is absent, the trainer falls back to the Ultralytics checkpoint `yolo11n.pt`, which can come from the local Ultralytics cache or be downloaded on demand.

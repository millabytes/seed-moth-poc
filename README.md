# seed-moth-poc

Initial proof of concept for detecting and counting avocado seed moths
(`Stenoma catenifer`) in pheromone-trap images.

This repository contains the implementation scaffold, reproducible Python environment, and technical documentation for the interview assignment.


## Setup

Install `uv` globally (recommended):

```bash
python3 -m pip install --user pipx
python3 -m pipx install uv
```

From the project root, sync the environment:

```bash
uv sync
```

This repository keeps the default runtime environment free of third-party packages, so `uv sync` only creates the local `.venv`.

To install the full repo, including the YOLO detection stack, use:

```bash
uv sync --all-extras
```

After that, the main console scripts are available through `uv run`:

- `uv run seed-moth-annotate`
- `uv run seed-moth-extract`
- `uv run seed-moth-backgrounds`
- `uv run seed-moth-synthetic`
- `uv run seed-moth-train`
- `uv run seed-moth-predict`
- `uv run seed-moth-evaluate`

## Full Pipeline

To run the whole pipeline step by step, use the root shell script:

```bash
./run.sh
```

The script:

- syncs the environment with `uv sync --all-extras --locked`
- skips manual annotation, mask/cutout extraction, background generation, and synthetic generation when their outputs are already complete
- always reruns YOLO training, synthetic validation evaluation, and inference when the required inputs exist
- if `data/test_images` and `results/test_images/labels/` exist, also runs the labeled real-image evaluation

Useful overrides:

- `SKIP_SYNC=1 ./run.sh` skips the initial `uv sync`
- `FORCE=1 ./run.sh` reruns every step even if outputs already exist
- `DEVICE=mps ./run.sh` forces Apple Silicon GPU training/inference
- `DEVICE=0 ./run.sh` forces the first CUDA GPU if you are on NVIDIA hardware
- `EPOCHS=60 BATCH_SIZE=16 PATIENCE=20 ./run.sh` changes the YOLO training plan
- `BACKGROUND_COUNT=40 SYNTHETIC_COUNT=200 ./run.sh` changes data-generation volume

## Structure 

Generated data-preparation and synthetic artifacts stay under `data/`, while YOLO training artifacts, predictions, and evaluation metrics go under `results/`.

Practical split:

- `data/` = raw inputs, annotations, cutouts, backgrounds, synthetic images
- `results/` = YOLO datasets, trained weights, inference previews, metrics
- `assets/pretrained/` = optional local pretrained checkpoints such as
  `yolo11n.pt`

### Data Preparation

The data-preparation documentation lives in
[src/seed_moth_poc/data_prep/README.md](src/seed_moth_poc/data_prep/README.md).

### Data Generation

Synthetic data generation lives in
[src/seed_moth_poc/synthetic/README.md](src/seed_moth_poc/synthetic/README.md).

### Detection

Detection and counting live in
[src/seed_moth_poc/detection/README.md](src/seed_moth_poc/detection/README.md).

### Evaluation

Evaluation lives in
[src/seed_moth_poc/evaluation/README.md](src/seed_moth_poc/evaluation/README.md).


## Notes

- `.venv/` is ignored by Git
- `uv.lock` is committed for reproducible installs
- `pyproject.toml` is the project metadata and build config
- The implementation is intended as a proof of concept, not a production-ready system.
- Third-party public data should be tracked through manifests before being downloaded or committed.

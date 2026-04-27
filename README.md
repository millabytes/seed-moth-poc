# seed-moth-poc

Initial proof of concept for detecting and counting avocado seed moths
(`Stenoma catenifer`) in pheromone-trap images.

This repository contains the implementation scaffold, reproducible Python environment,
and technical documentation for the interview assignment.


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

This repository keeps the default runtime environment free of third-party
packages, so `uv sync` only creates the local `.venv`.

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

Generated data-preparation and synthetic artifacts stay under `data/`, while
YOLO training artifacts, predictions, and evaluation metrics go under
`results/`.

Practical split:

- `data/` = raw inputs, annotations, cutouts, backgrounds, synthetic images
- `results/` = YOLO datasets, trained weights, inference previews, metrics

## Development Tools

Formatting, linting, and test tooling can be added later if needed. They are not
required to run the data-preparation scripts in this proof of concept.

## Notes

- `.venv/` is ignored by Git
- `uv.lock` is committed for reproducible installs
- `pyproject.toml` is the project metadata and build config
- The implementation is intended as a proof of concept, not a production-ready system.
- Third-party public data should be tracked through manifests before being downloaded or
  committed.

## Data Preparation

The data-preparation documentation lives in
[src/seed_moth_poc/data_prep/README.md](src/seed_moth_poc/data_prep/README.md).

Synthetic data generation lives in
[src/seed_moth_poc/synthetic/README.md](src/seed_moth_poc/synthetic/README.md).

Detection and counting live in
[src/seed_moth_poc/detection/README.md](src/seed_moth_poc/detection/README.md).

Evaluation lives in
[src/seed_moth_poc/evaluation/README.md](src/seed_moth_poc/evaluation/README.md).

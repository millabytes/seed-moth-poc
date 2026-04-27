# Synthetic Generation

This package generates synthetic trap images by compositing bbox-derived moth
cutouts onto trap-like backgrounds.

If `uv` reports a cache permission error on your machine, prefix the command
with `UV_CACHE_DIR=/tmp/uv-cache`.

The generator uses the `Stenoma catenifer` screening aid PDF as a prior for:

- forewing length of `8.0-15.0 mm`
- yellowish-tan wings with numerous black spots
- a rough C-shaped dark-spot pattern near the distal wing end

The visual source comes from:

- `data/reference/derived/cutouts/images/`
- `data/backgrounds/generated/`

## Prerequisites

Run the data-preparation pipeline first, described
[here](src/seed_moth_poc/data_prep/README.md).

## Generate synthetic data

```bash
uv run src/seed_moth_poc/synthetic/generator.py \
  --backgrounds data/backgrounds/generated \
  --sources-root data/reference/derived/cutouts/images \
  --output-root data/synthetic \
  --count 120 \
  --seed 42
```

Outputs:

- `data/synthetic/images/`
- `data/synthetic/labels/`
- `data/synthetic/manifest.json`

The generator writes YOLO labels and a JSON manifest with the source cutout,
background file, and bounding boxes used for each synthetic image.

By default the generator avoids empty scenes. If you want negative examples, set
`--empty-prob` explicitly.

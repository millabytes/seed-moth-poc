# Data Preparation

This package contains the data-preparation tools for the seed moth POC.

The scripts are designed to be run from the repository root.
If `uv` reports a cache permission error, prefix the command with
`UV_CACHE_DIR=/tmp/uv-cache`.

## Annotate reference images

Open the reference images one by one in a browser and write YOLO bbox labels to
`data/reference/target/labels/`.

```bash
uv run seed-moth-annotate \
  --images data/reference/target/images \
  --labels data/reference/target/labels
```

The annotator starts a local HTTP server, and opens your browser. It runs with the default project setup and does not need any third-party packages.

Add `--no-browser` if you want to start the server manually and open the URL
yourself.

Draw a box by dragging on the image, then click `Save` or press `n` / `Enter`
to save the current image and move to the next one. Labels are written as YOLO
`.txt` files alongside the reference image names.

Shortcuts:
- `n` or `Enter`: next image
- `p`: previous image
- `s`: save
- `u` or `Backspace`: undo last box
- `r`: clear boxes for the current image
- `q`: save and quit

## Extract masks and cutouts

Build binary masks and transparent cutouts from the original reference images.
The extractor uses the YOLO labels as crop guidance by default so the cutout is
centered on the annotated moth.

```bash
uv run seed-moth-extract \
  --inputs data/reference/target/images \
  --output-root data/reference/derived \
  --labels-root data/reference/target/labels
```

Outputs:
- `data/reference/derived/masks/`
- `data/reference/derived/cutouts/`
- `data/reference/derived/manifest.json`

## Review cutouts manually

If a few automatic cutouts are still poor, review them before synthetic
generation. The reviewer opens the extracted cutouts one by one, lets you draw
a polygon around the moth, and writes corrected cutouts and masks to a separate
reviewed output tree.

```bash
uv run seed-moth-cutout-review \
  --inputs data/reference/derived/cutouts/images \
  --output-root data/reference/reviewed
```

Useful controls:
- `Replace` mode uses only the manual polygon and is the default
- `Add` mode merges your polygon with the existing alpha mask
- `Save` writes the reviewed cutout and mask
- `Next` / `Enter` saves and advances to the next cutout

You can pass a whole directory or a few individual cutout PNGs to `--inputs`.
By default the reviewer skips cutouts that already have a reviewed override.


The synthetic generator prefers reviewed cutouts first and falls back to the
automatic cutouts for files that were not reviewed. The reviewed tree is meant
to be a sparse override set, not a full duplicate of the automatic cutouts.

## Generate procedural backgrounds

Create trap-like backgrounds with gradients, noise, stains, and lighting
variation.

```bash
uv run seed-moth-backgrounds \
  --output-dir data/backgrounds/generated \
  --count 24
```

Outputs:
- `data/backgrounds/generated/`
- `data/backgrounds/generated/manifest.json`

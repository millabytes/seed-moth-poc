"""Generate synthetic trap images with article-informed moth priors."""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seed_moth_poc.data_prep.commons import (
    Box,
    ImageBuffer,
    collect_image_files,
    crop_image,
    ensure_directory,
    fit_to_box,
    load_image,
    save_png,
    write_yolo_boxes,
)
from seed_moth_poc.data_prep.mask_extractor import cutout_from_mask, extract_foreground
from seed_moth_poc.synthetic.image_ops import (
    add_c_shape_spots,
    adjust_brightness,
    adjust_contrast,
    blend_toward_color,
    crop_to_alpha,
    draw_shadow,
    paste_rgba,
    resize_rgba,
    rotate_rgba,
)
from seed_moth_poc.synthetic.priors import (
    DEFAULT_EMPTY_PROBABILITY,
    DEFAULT_MAX_OBJECTS,
    DEFAULT_MIN_OBJECTS,
    SHADOW_COLOR,
    article_cue_probability,
    sample_object_count,
    sample_pixel_length,
    sample_source_weight,
    sample_tint_color,
)


@dataclass(slots=True)
class SourceRecord:
    """Metadata for one available moth cutout source."""

    path: Path
    kind: str
    weight: float


@dataclass(slots=True)
class SynthObject:
    """Description of one object pasted into a synthetic image."""

    source_path: Path
    kind: str
    bbox: Box
    angle_deg: float
    long_edge_px: float


@dataclass(slots=True)
class TransformConfig:
    """Parameters that control how many moths are placed in one scene."""

    min_objects: int
    max_objects: int
    empty_probability: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic moth images and YOLO labels.",
    )
    parser.add_argument(
        "--backgrounds",
        type=Path,
        default=Path("data/backgrounds/generated"),
        help="Directory containing trap-like background PNGs.",
    )
    parser.add_argument(
        "--sources-root",
        "--cutout-root",
        dest="sources_root",
        type=Path,
        default=Path("data/reference/derived/cutouts/images"),
        help="Directory containing bbox-derived moth cutouts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/synthetic"),
        help="Directory where synthetic images, labels, and manifest will be saved.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=120,
        help="Number of synthetic images to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--empty-prob",
        type=float,
        default=DEFAULT_EMPTY_PROBABILITY,
        help="Probability that an image contains zero moths.",
    )
    parser.add_argument(
        "--min-objects",
        type=int,
        default=DEFAULT_MIN_OBJECTS,
        help="Minimum number of moths per image.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=DEFAULT_MAX_OBJECTS,
        help="Maximum number of moths per image.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="manifest.json",
        help="Name of the metadata manifest file.",
    )
    return parser.parse_args()


def detect_kind(path: Path) -> str:
    """Classify one cutout as a reference image source."""
    return "reference"


def collect_sources(sources_root: Path) -> list[SourceRecord]:
    """Collect available moth cutouts from the extracted reference directory."""
    paths = collect_image_files([sources_root], recursive=True)
    records: list[SourceRecord] = []
    for path in paths:
        kind = detect_kind(path)
        records.append(
            SourceRecord(
                path=path,
                kind=kind,
                weight=sample_source_weight(kind),
            )
        )
    return records


def collect_backgrounds(backgrounds_root: Path) -> list[Path]:
    """Collect procedural backgrounds from the generated backgrounds directory."""
    return collect_image_files([backgrounds_root], recursive=True)


def weighted_choice(rng: random.Random, items: list[Any], weights: list[float]) -> Any:
    """Choose one item from a weighted list."""
    if not items:
        raise ValueError("Cannot choose from an empty list.")
    return rng.choices(items, weights=weights, k=1)[0]


def object_overlap_ratio(candidate: Box, existing: list[Box]) -> float:
    """Return the maximum IoU between a candidate box and existing boxes."""
    best = 0.0
    for box in existing:
        candidate_box = candidate.ordered()
        existing_box = box.ordered()
        left = max(candidate_box.x1, existing_box.x1)
        top = max(candidate_box.y1, existing_box.y1)
        right = min(candidate_box.x2, existing_box.x2)
        bottom = min(candidate_box.y2, existing_box.y2)
        if right <= left or bottom <= top:
            continue
        intersection = (right - left) * (bottom - top)
        area_a = candidate.width() * candidate.height()
        area_b = box.width() * box.height()
        union = area_a + area_b - intersection
        if union <= 0:
            continue
        best = max(best, intersection / union)
    return best


def position_object(
    rng: random.Random,
    background: ImageBuffer,
    sprite: ImageBuffer,
    placed_boxes: list[Box],
) -> tuple[int, int, Box]:
    """Pick a position for one sprite, avoiding heavy overlap when possible."""
    max_x = max(0, background.width - sprite.width)
    max_y = max(0, background.height - sprite.height)
    margin_x = max(0, sprite.width // 10)
    margin_y = max(0, sprite.height // 10)

    last_x = 0
    last_y = 0
    last_box = Box(0.0, 0.0, float(sprite.width - 1), float(sprite.height - 1))
    for _ in range(12):
        x = rng.randint(-margin_x, max_x + margin_x)
        y = rng.randint(-margin_y, max_y + margin_y)
        candidate = Box(
            float(x),
            float(y),
            float(x + sprite.width - 1),
            float(y + sprite.height - 1),
        ).clamp(background.width, background.height)
        overlap = object_overlap_ratio(candidate, placed_boxes)
        last_x, last_y, last_box = x, y, candidate
        if overlap <= 0.35 or rng.random() < 0.25:
            return x, y, candidate
    return last_x, last_y, last_box


def apply_article_treatment(
    rng: random.Random,
    sprite: ImageBuffer,
    kind: str,
) -> ImageBuffer:
    """Warm the reference cutout and optionally reinforce the article cue."""
    tinted = blend_toward_color(sprite, sample_tint_color(rng), rng.uniform(0.06, 0.16))
    if rng.random() < article_cue_probability(kind):
        spot_density = 0.55
        tinted = add_c_shape_spots(tinted, density=spot_density)
    return tinted


def transform_sprite(
    rng: random.Random,
    source: SourceRecord,
    background: ImageBuffer,
) -> tuple[ImageBuffer, float, float]:
    """Prepare one cutout sprite for placement on a background."""
    source_image = load_image(source.path)
    mask, bbox, _ = extract_foreground(source_image, threshold=48)
    sprite = cutout_from_mask(source_image, mask)
    if bbox != (0, 0, 0, 0):
        x1, y1, x2, y2 = bbox
        sprite = crop_image(sprite, x1, y1, x2 + 1, y2 + 1, fill=(0, 0, 0, 0))
    sprite = crop_to_alpha(sprite)
    if sprite.width <= 0 or sprite.height <= 0:
        raise ValueError(f"Empty cutout source: {source.path}")

    long_edge_px = sample_pixel_length(rng, background.width, background.height)
    long_edge_px *= rng.uniform(0.90, 1.10)
    sprite = apply_article_treatment(rng, sprite, source.kind)

    source_aspect = sprite.width / max(1, sprite.height)
    if source_aspect >= 1.0:
        target_width = int(round(long_edge_px))
        target_height = max(1, int(round(long_edge_px / source_aspect)))
    else:
        target_height = int(round(long_edge_px))
        target_width = max(1, int(round(long_edge_px * source_aspect)))

    target_width, target_height, _ = fit_to_box(
        target_width,
        target_height,
        max(1, int(background.width * 0.92)),
        max(1, int(background.height * 0.92)),
        allow_upscale=False,
    )
    sprite = resize_rgba(sprite, target_width, target_height)
    long_edge_px = float(max(sprite.width, sprite.height))

    if rng.random() < 0.15:
        sprite = adjust_contrast(sprite, rng.uniform(0.90, 1.08))
    if rng.random() < 0.35:
        sprite = adjust_brightness(sprite, rng.uniform(0.90, 1.10))
    if rng.random() < 0.12:
        # A small horizontal flip helps break symmetry across synthetic samples.
        sprite = flip_horizontal(sprite)

    angle_deg = rng.uniform(0.0, 360.0)
    sprite = rotate_rgba(sprite, angle_deg, expand=True)
    sprite = crop_to_alpha(sprite)

    sprite = fit_rotated_sprite(sprite, background)
    return sprite, angle_deg, long_edge_px


def fit_rotated_sprite(sprite: ImageBuffer, background: ImageBuffer) -> ImageBuffer:
    """Ensure a rotated sprite still fits comfortably on the background."""
    target_width, target_height, _ = fit_to_box(
        sprite.width,
        sprite.height,
        max(1, int(background.width * 0.96)),
        max(1, int(background.height * 0.96)),
        allow_upscale=False,
    )
    if target_width != sprite.width or target_height != sprite.height:
        sprite = resize_rgba(sprite, target_width, target_height)
        sprite = crop_to_alpha(sprite)
    return sprite


def flip_horizontal(image: Any) -> Any:
    """Flip an RGBA image horizontally."""
    result = image.copy()
    width = image.width
    height = image.height
    flipped = bytearray(width * height * 4)
    for y in range(height):
        for x in range(width):
            source = (y * width + x) * 4
            dest = (y * width + (width - 1 - x)) * 4
            flipped[dest : dest + 4] = result.pixels[source : source + 4]
    return type(image)(width, height, flipped)


def build_scene(
    rng: random.Random,
    backgrounds: list[Path],
    sources: list[SourceRecord],
    *,
    transform_config: TransformConfig,
) -> tuple[ImageBuffer, list[SynthObject], Path]:
    """Compose one synthetic scene from one background and one or more moths."""
    background_path = weighted_choice(rng, backgrounds, [1.0] * len(backgrounds))
    canvas = load_image(background_path)

    object_count = sample_object_count(
        rng,
        min_objects=transform_config.min_objects,
        max_objects=transform_config.max_objects,
        empty_probability=transform_config.empty_probability,
    )

    objects: list[SynthObject] = []
    placed_boxes: list[Box] = []
    if object_count <= 0:
        return canvas, objects, background_path

    source_weights = [record.weight for record in sources]
    for index in range(object_count):
        source = weighted_choice(rng, sources, source_weights)
        sprite, angle_deg, long_edge_px = transform_sprite(rng, source, canvas)
        if index > 0:
            reduction = rng.uniform(0.55, 0.82)
            sprite = resize_rgba(
                sprite,
                max(1, int(round(sprite.width * reduction))),
                max(1, int(round(sprite.height * reduction))),
            )
            sprite = crop_to_alpha(sprite)

        x, y, bbox = position_object(rng, canvas, sprite, placed_boxes)
        shadow_dx = rng.randint(1, 5)
        shadow_dy = rng.randint(2, 6)
        draw_shadow(
            canvas,
            sprite,
            x,
            y,
            offset_x=shadow_dx,
            offset_y=shadow_dy,
            opacity=rng.uniform(0.16, 0.28),
            color=SHADOW_COLOR,
        )
        paste_rgba(canvas, sprite, x, y)
        placed_boxes.append(bbox)
        objects.append(
            SynthObject(
                source_path=source.path,
                kind=source.kind,
                bbox=bbox,
                angle_deg=angle_deg,
                long_edge_px=long_edge_px,
            )
        )

    return canvas, objects, background_path


def synthesize(
    count: int,
    output_root: Path,
    backgrounds: list[Path],
    sources: list[SourceRecord],
    rng: random.Random,
    *,
    transform_config: TransformConfig,
) -> list[dict[str, Any]]:
    """Generate a batch of synthetic images and return their manifest records."""
    images_root = ensure_directory(output_root / "images")
    labels_root = ensure_directory(output_root / "labels")

    manifest: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        canvas, objects, background_path = build_scene(
            rng,
            backgrounds,
            sources,
            transform_config=transform_config,
        )
        file_name = f"synth_{index:04d}.png"
        label_name = f"synth_{index:04d}.txt"
        image_path = images_root / file_name
        label_path = labels_root / label_name

        boxes = [obj.bbox for obj in objects]
        save_png(canvas, image_path)
        write_yolo_boxes(label_path, boxes, width=canvas.width, height=canvas.height)

        record = {
            "index": index,
            "file": file_name,
            "label": label_name,
            "background": str(background_path),
            "width": canvas.width,
            "height": canvas.height,
            "object_count": len(objects),
            "objects": [
                {
                    "source": str(obj.source_path),
                    "kind": obj.kind,
                    "bbox": [
                        obj.bbox.x1,
                        obj.bbox.y1,
                        obj.bbox.x2,
                        obj.bbox.y2,
                    ],
                    "angle_deg": obj.angle_deg,
                    "long_edge_px": obj.long_edge_px,
                }
                for obj in objects
            ],
        }
        manifest.append(record)
        print(f"Saved {image_path}")
    return manifest


def main() -> None:
    args = parse_args()
    transform_config = TransformConfig(
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        empty_probability=args.empty_prob,
    )

    rng = random.Random(args.seed)
    backgrounds = collect_backgrounds(args.backgrounds)
    if not backgrounds:
        raise SystemExit(
            f"No background images found in {args.backgrounds}. "
            "Run the background generator first."
        )

    sources = collect_sources(args.sources_root)
    if not sources:
        raise SystemExit(
            f"No cutout images found in {args.sources_root}. "
            "Run the mask extractor first."
        )

    output_root = ensure_directory(args.output_root)
    manifest = synthesize(
        args.count,
        output_root,
        backgrounds,
        sources,
        rng,
        transform_config=transform_config,
    )
    manifest_path = output_root / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()

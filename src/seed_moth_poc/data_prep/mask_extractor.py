"""Extract approximate masks and cutouts from reference and morphology images."""

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from collections import deque
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from seed_moth_poc.data_prep.commons import (
    Box,
    ImageBuffer,
    collect_image_files,
    crop_image,
    ensure_directory,
    load_image,
    read_yolo_boxes,
    save_png,
)


@dataclass(slots=True)
class Component:
    """Connected foreground component found during thresholding."""

    area: int
    x1: int
    y1: int
    x2: int
    y2: int
    touches_border: bool
    pixels: list[int]

    @property
    def width(self) -> int:
        """Return the component width in pixels."""
        return self.x2 - self.x1 + 1

    @property
    def height(self) -> int:
        """Return the component height in pixels."""
        return self.y2 - self.y1 + 1


def luma(r: int, g: int, b: int) -> int:
    """Compute integer luma for an RGB color."""
    return int(round(0.299 * r + 0.587 * g + 0.114 * b))


def color_distance(pixel: tuple[int, int, int, int], background: tuple[int, int, int]) -> int:
    """Compute a simple Manhattan distance to the estimated background color."""
    return abs(pixel[0] - background[0]) + abs(pixel[1] - background[1]) + abs(
        pixel[2] - background[2]
    )


def median_abs_deviation(values: list[int]) -> float:
    """Return the median absolute deviation for a sample."""
    if not values:
        return 0.0
    med = statistics.median(values)
    deviations = [abs(value - med) for value in values]
    return float(statistics.median(deviations))


def border_indices(width: int, height: int) -> list[int]:
    """Return flattened pixel indices for the image border."""
    indices: list[int] = []
    for x in range(width):
        indices.append(x)
        indices.append((height - 1) * width + x)
    for y in range(1, height - 1):
        indices.append(y * width)
        indices.append(y * width + width - 1)
    return indices


def estimate_background_color(image: ImageBuffer) -> tuple[int, int, int]:
    """Estimate a dominant border color and treat it as background."""
    border_pixels: list[tuple[int, int, int, int]] = []
    for index in border_indices(image.width, image.height):
        base = index * 4
        border_pixels.append(tuple(image.pixels[base : base + 4]))  # type: ignore[arg-type]
    if not border_pixels:
        return (255, 255, 255)

    border_pixels.sort(key=lambda pixel: luma(pixel[0], pixel[1], pixel[2]))
    keep = border_pixels[len(border_pixels) // 4 :]
    if not keep:
        keep = border_pixels
    rs = [pixel[0] for pixel in keep]
    gs = [pixel[1] for pixel in keep]
    bs = [pixel[2] for pixel in keep]
    return (
        int(statistics.median(rs)),
        int(statistics.median(gs)),
        int(statistics.median(bs)),
    )


def build_scores(
    image: ImageBuffer,
    background: tuple[int, int, int],
) -> tuple[bytearray, list[int]]:
    """Score every pixel by how different it is from the background."""
    scores = bytearray(image.width * image.height)
    border_scores: list[int] = []
    bg_luma = luma(*background)
    border_set = set(border_indices(image.width, image.height))

    for index in range(image.width * image.height):
        base = index * 4
        pixel = tuple(image.pixels[base : base + 4])  # type: ignore[arg-type]
        pixel_luma = luma(pixel[0], pixel[1], pixel[2])
        score = max(
            color_distance(pixel, background),
            abs(pixel_luma - bg_luma) * 2,
            255 - pixel[3],
        )
        score = min(255, score)
        scores[index] = score
        if index in border_set:
            border_scores.append(score)

    return scores, border_scores


def threshold_scores(scores: bytearray, threshold: int) -> bytearray:
    """Convert foreground scores into a binary mask."""
    return bytearray(1 if score > threshold else 0 for score in scores)


def flood_background(scores: bytearray, width: int, height: int, threshold: int) -> bytearray:
    """Mark low-score pixels connected to the crop border as background."""
    background = bytearray(width * height)
    queue: deque[int] = deque()
    for index in border_indices(width, height):
        if scores[index] <= threshold:
            queue.append(index)

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        current = queue.popleft()
        if background[current]:
            continue
        background[current] = 1
        y = current // width
        x = current % width
        for dx, dy in neighbors:
            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            neighbor = ny * width + nx
            if background[neighbor]:
                continue
            if scores[neighbor] <= threshold:
                queue.append(neighbor)

    return background


def connected_components(mask: bytearray, width: int, height: int) -> list[Component]:
    """Group neighboring foreground pixels into connected components."""
    visited = bytearray(width * height)
    components: list[Component] = []
    neighbors = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ]

    for index in range(width * height):
        if mask[index] == 0 or visited[index]:
            continue

        stack = [index]
        visited[index] = 1
        pixels: list[int] = []
        area = 0
        x1 = width
        y1 = height
        x2 = 0
        y2 = 0
        touches_border = False

        while stack:
            current = stack.pop()
            pixels.append(current)
            area += 1
            y = current // width
            x = current % width
            if x < x1:
                x1 = x
            if y < y1:
                y1 = y
            if x > x2:
                x2 = x
            if y > y2:
                y2 = y
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                touches_border = True

            for dx, dy in neighbors:
                nx = x + dx
                ny = y + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                neighbor_index = ny * width + nx
                if mask[neighbor_index] and not visited[neighbor_index]:
                    visited[neighbor_index] = 1
                    stack.append(neighbor_index)

        components.append(
            Component(area, x1, y1, x2, y2, touches_border, pixels)
        )

    return components


def select_components(
    components: list[Component],
    *,
    image_width: int,
    image_height: int,
) -> list[Component]:
    """Keep components that are plausible foreground objects."""
    image_area = image_width * image_height
    min_area = max(24, int(image_area * 0.0002))
    selected: list[Component] = []

    for component in components:
        if component.area < min_area:
            continue
        if component.width <= 1 or component.height <= 1:
            continue
        if component.touches_border and (
            component.width <= max(6, image_width // 25)
            or component.height <= max(6, image_height // 25)
        ):
            continue
        fill_ratio = component.area / (component.width * component.height)
        if fill_ratio < 0.005 and component.area < image_area * 0.1:
            continue
        selected.append(component)

    if selected:
        return selected

    if not components:
        return []

    return [max(components, key=lambda component: component.area)]


def finalize_mask(
    components: list[Component],
    width: int,
    height: int,
) -> tuple[bytearray, int, int, int, int] | None:
    """Merge selected components into one binary mask and bounding box."""
    if not components:
        return None

    mask = bytearray(width * height)
    x1 = width
    y1 = height
    x2 = 0
    y2 = 0
    foreground_pixels = 0

    for component in components:
        for index in component.pixels:
            mask[index] = 1
        foreground_pixels += component.area
        if component.x1 < x1:
            x1 = component.x1
        if component.y1 < y1:
            y1 = component.y1
        if component.x2 > x2:
            x2 = component.x2
        if component.y2 > y2:
            y2 = component.y2

    if foreground_pixels == 0:
        return None

    return mask, x1, y1, x2, y2


def mask_from_alpha(image: ImageBuffer, alpha_threshold: int = 8) -> bytearray:
    """Build a mask from pixels that are visibly opaque."""
    mask = bytearray(image.width * image.height)
    for index in range(image.width * image.height):
        if image.pixels[index * 4 + 3] > alpha_threshold:
            mask[index] = 1
    return mask


def extract_foreground(image: ImageBuffer, threshold: int) -> tuple[bytearray, tuple[int, int, int, int], int]:
    """Estimate the foreground mask, bounding box, and threshold used."""
    alpha_values = [image.pixels[index + 3] for index in range(0, len(image.pixels), 4)]
    if alpha_values and min(alpha_values) < 255:
        alpha_mask = mask_from_alpha(image)
        alpha_components = select_components(
            connected_components(alpha_mask, image.width, image.height),
            image_width=image.width,
            image_height=image.height,
        )
        alpha_result = finalize_mask(alpha_components, image.width, image.height)
        if alpha_result is not None:
            mask, x1, y1, x2, y2 = alpha_result
            return mask, (x1, y1, x2, y2), len(alpha_components)

    background = estimate_background_color(image)
    scores, border_scores = build_scores(image, background)
    border_median = statistics.median(border_scores) if border_scores else 0
    border_mad = median_abs_deviation(border_scores)
    estimated = max(threshold, int(border_median + 3 * border_mad + 6))
    candidates = sorted(
        {
            max(12, threshold),
            max(12, estimated - 12),
            max(12, estimated - 6),
            estimated,
            min(255, estimated + 6),
        }
    )

    best_result: tuple[bytearray, tuple[int, int, int, int], int] | None = None
    best_area = -1

    for candidate in candidates:
        background_mask = flood_background(scores, image.width, image.height, candidate)
        mask = bytearray(1 if value == 0 else 0 for value in background_mask)
        components = select_components(
            connected_components(mask, image.width, image.height),
            image_width=image.width,
            image_height=image.height,
        )
        result = finalize_mask(components, image.width, image.height)
        if result is None:
            continue
        final_mask, x1, y1, x2, y2 = result
        area = sum(final_mask)
        if area > best_area:
            best_result = (final_mask, (x1, y1, x2, y2), candidate)
            best_area = area
        ratio = area / (image.width * image.height)
        if 0.0005 <= ratio <= 0.45:
            return final_mask, (x1, y1, x2, y2), candidate

    if best_result is not None:
        return best_result

    return bytearray(image.width * image.height), (0, 0, 0, 0), threshold


def mask_to_image(mask: bytearray, width: int, height: int) -> ImageBuffer:
    """Render a binary mask as a white-on-black image."""
    output = ImageBuffer.blank(width, height, (0, 0, 0, 255))
    for index, value in enumerate(mask):
        if value:
            base = index * 4
            output.pixels[base] = 255
            output.pixels[base + 1] = 255
            output.pixels[base + 2] = 255
            output.pixels[base + 3] = 255
    return output


def cutout_from_mask(image: ImageBuffer, mask: bytearray) -> ImageBuffer:
    """Copy masked pixels into a transparent cutout image."""
    output = ImageBuffer.blank(image.width, image.height, (0, 0, 0, 0))
    for index, value in enumerate(mask):
        if not value:
            continue
        base = index * 4
        output.pixels[base : base + 4] = image.pixels[base : base + 4]
    return output


def relative_output_path(source_root: Path, file_path: Path, suffix: str) -> Path:
    """Map an input file to a relative derived path with a new suffix."""
    try:
        relative = file_path.relative_to(source_root)
    except ValueError:
        relative = Path(file_path.name)
    if relative == Path("."):
        relative = Path(file_path.name)
    return relative.with_suffix(suffix)


def union_boxes(boxes: list[Box], image_width: int, image_height: int) -> Box:
    """Merge multiple boxes into one enclosing box."""
    left = min(box.ordered().x1 for box in boxes)
    top = min(box.ordered().y1 for box in boxes)
    right = max(box.ordered().x2 for box in boxes)
    bottom = max(box.ordered().y2 for box in boxes)
    return Box(left, top, right, bottom).clamp(image_width, image_height)


def bbox_from_mask(mask: bytearray, width: int, height: int) -> tuple[int, int, int, int]:
    """Compute the tight bounding box around a binary mask."""
    x1 = width
    y1 = height
    x2 = 0
    y2 = 0
    found = False
    for index, value in enumerate(mask):
        if not value:
            continue
        found = True
        y = index // width
        x = index % width
        if x < x1:
            x1 = x
        if y < y1:
            y1 = y
        if x > x2:
            x2 = x
        if y > y2:
            y2 = y
    if not found:
        return (0, 0, 0, 0)
    return (x1, y1, x2, y2)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the mask extractor."""
    parser = argparse.ArgumentParser(
        description="Extract masks and cutouts from reference images."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            Path("data/reference/target/images"),
            Path("data/reference/target/morphology"),
        ],
        help="Input image directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/reference/derived"),
        help="Directory where masks and cutouts will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=28,
        help="Base threshold used for foreground scoring.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="manifest.json",
        help="File name for the JSON summary saved in the output root.",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("data/reference/target/labels"),
        help="Directory containing YOLO label files used as optional crop hints.",
    )
    return parser


def process_image(
    image_path: Path,
    input_root: Path,
    root_name: str,
    masks_root: Path,
    cutouts_root: Path,
    labels_root: Path,
    threshold: int,
) -> dict[str, object]:
    """Process one image and write its derived mask and cutout files."""
    image = load_image(image_path)
    crop_left = 0
    crop_top = 0
    analysis_image = image
    crop_box: list[int] | None = None

    label_path = labels_root / f"{image_path.stem}.txt"
    label_boxes = read_yolo_boxes(
        label_path,
        width=image.width,
        height=image.height,
    )
    if label_boxes:
        union = union_boxes(label_boxes, image.width, image.height)
        padding = max(8, int(round(max(union.width(), union.height()) * 0.15)))
        crop_left = max(0, int(math.floor(union.x1 - padding)))
        crop_top = max(0, int(math.floor(union.y1 - padding)))
        crop_right = min(image.width, int(math.ceil(union.x2 + padding)))
        crop_bottom = min(image.height, int(math.ceil(union.y2 + padding)))
        if crop_right > crop_left and crop_bottom > crop_top:
            crop_box = [crop_left, crop_top, crop_right, crop_bottom]
            analysis_image = crop_image(
                image,
                crop_left,
                crop_top,
                crop_right,
                crop_bottom,
            )

    mask, bbox, used_threshold = extract_foreground(analysis_image, threshold)
    full_mask = bytearray(image.width * image.height)
    for index, value in enumerate(mask):
        if not value:
            continue
        local_y = index // analysis_image.width
        local_x = index % analysis_image.width
        global_x = crop_left + local_x
        global_y = crop_top + local_y
        if 0 <= global_x < image.width and 0 <= global_y < image.height:
            full_mask[global_y * image.width + global_x] = 1

    mask_image = mask_to_image(full_mask, image.width, image.height)
    cutout_image = cutout_from_mask(image, full_mask)
    full_bbox = bbox_from_mask(full_mask, image.width, image.height)
    if full_bbox != (0, 0, 0, 0):
        x1, y1, x2, y2 = full_bbox
        mask_image = crop_image(
            mask_image,
            x1,
            y1,
            x2 + 1,
            y2 + 1,
            fill=(0, 0, 0, 255),
        )
        cutout_image = crop_image(
            cutout_image,
            x1,
            y1,
            x2 + 1,
            y2 + 1,
            fill=(0, 0, 0, 0),
        )

    mask_path = masks_root / root_name / relative_output_path(
        input_root, image_path, ".png"
    )
    cutout_path = cutouts_root / root_name / relative_output_path(
        input_root, image_path, ".png"
    )
    ensure_directory(mask_path.parent)
    ensure_directory(cutout_path.parent)
    save_png(mask_image, mask_path)
    save_png(cutout_image, cutout_path)

    return {
        "source": str(image_path),
        "label_path": str(label_path),
        "mask": str(mask_path),
        "cutout": str(cutout_path),
        "bbox": list(full_bbox),
        "analysis_bbox": list(bbox),
        "crop_box": crop_box,
        "threshold": used_threshold,
        "foreground_pixels": int(sum(full_mask)),
        "image_width": image.width,
        "image_height": image.height,
    }


def main() -> None:
    """Parse CLI arguments, process all inputs, and write the manifest."""
    parser = build_arg_parser()
    args = parser.parse_args()

    input_roots = [Path(path) for path in args.inputs]
    output_root = args.output_root
    masks_root = output_root / "masks"
    cutouts_root = output_root / "cutouts"
    ensure_directory(masks_root)
    ensure_directory(cutouts_root)

    manifest: list[dict[str, object]] = []
    images: list[tuple[Path, str, Path]] = []
    for input_root in input_roots:
        root_name = input_root.name if input_root.is_dir() else (input_root.parent.name or input_root.stem)
        images.extend(
            (input_root, root_name, image_path)
            for image_path in collect_image_files([input_root])
        )

    if not images:
        raise SystemExit("No input images found.")

    for input_root, root_name, image_path in images:
        record = process_image(
            image_path,
            input_root,
            root_name,
            masks_root,
            cutouts_root,
            args.labels_root,
            args.threshold,
        )
        manifest.append(record)
        print(
            f"Processed {image_path.name} -> "
            f"{Path(record['mask']).relative_to(output_root)}"
        )

    manifest_path = output_root / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()

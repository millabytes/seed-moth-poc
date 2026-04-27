"""Generate procedural trap-like backgrounds for synthetic data."""

import argparse
import json
import random
from pathlib import Path

from seed_moth_poc.data_prep.commons import (
    ImageBuffer,
    draw_filled_circle,
    draw_filled_ellipse,
    draw_line,
    resize_bilinear,
    save_png,
)

PALETTES: list[dict[str, object]] = [
    {
        "name": "paper_white",
        "weight": 3.5,
        "top_left": (251, 249, 244),
        "top_right": (248, 245, 238),
        "bottom_left": (244, 238, 228),
        "bottom_right": (238, 231, 220),
        "noise": 6,
        "stain_strength": 8,
        "speck_count": (4, 10),
    },
    {
        "name": "warm_ivory",
        "weight": 3.0,
        "top_left": (248, 244, 232),
        "top_right": (245, 241, 227),
        "bottom_left": (238, 232, 216),
        "bottom_right": (232, 225, 208),
        "noise": 7,
        "stain_strength": 10,
        "speck_count": (6, 14),
    },
    {
        "name": "soft_yellow",
        "weight": 2.0,
        "top_left": (247, 241, 220),
        "top_right": (244, 237, 211),
        "bottom_left": (237, 229, 198),
        "bottom_right": (230, 221, 188),
        "noise": 8,
        "stain_strength": 12,
        "speck_count": (7, 16),
    },
    {
        "name": "dirty_cream",
        "weight": 1.1,
        "top_left": (244, 236, 220),
        "top_right": (240, 231, 212),
        "bottom_left": (232, 223, 202),
        "bottom_right": (225, 215, 193),
        "noise": 9,
        "stain_strength": 14,
        "speck_count": (8, 18),
    },
]


def clamp(value: int) -> int:
    """Clamp a channel value into the 8-bit range."""
    return max(0, min(255, value))


def lerp_channel(start: int, end: int, t: float) -> int:
    """Interpolate one color channel."""
    return int(round(start + (end - start) * t))


def lerp_color(start: tuple[int, int, int], end: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Interpolate between two RGB colors."""
    return (
        lerp_channel(start[0], end[0], t),
        lerp_channel(start[1], end[1], t),
        lerp_channel(start[2], end[2], t),
    )


def multiply_color(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Scale an RGB color by one factor."""
    return (
        clamp(int(round(color[0] * factor))),
        clamp(int(round(color[1] * factor))),
        clamp(int(round(color[2] * factor))),
    )


def blend_rgb(base: tuple[int, int, int], overlay: tuple[int, int, int], opacity: float) -> tuple[int, int, int]:
    """Blend two RGB colors using a fixed opacity."""
    return (
        clamp(int(round(base[0] * (1.0 - opacity) + overlay[0] * opacity))),
        clamp(int(round(base[1] * (1.0 - opacity) + overlay[1] * opacity))),
        clamp(int(round(base[2] * (1.0 - opacity) + overlay[2] * opacity))),
    )


def fill_gradient(
    image: ImageBuffer,
    top_left: tuple[int, int, int],
    top_right: tuple[int, int, int],
    bottom_left: tuple[int, int, int],
    bottom_right: tuple[int, int, int],
) -> None:
    """Fill an image with a four-corner gradient."""
    for y in range(image.height):
        y_ratio = 0.0 if image.height == 1 else y / (image.height - 1)
        left = lerp_color(top_left, bottom_left, y_ratio)
        right = lerp_color(top_right, bottom_right, y_ratio)
        for x in range(image.width):
            x_ratio = 0.0 if image.width == 1 else x / (image.width - 1)
            color = lerp_color(left, right, x_ratio)
            image.set_pixel(x, y, (color[0], color[1], color[2], 255))


def add_low_frequency_noise(
    image: ImageBuffer,
    rng: random.Random,
    *,
    strength: int,
) -> None:
    """Overlay coarse color noise to break up flat regions."""
    small_width = max(8, image.width // 24)
    small_height = max(8, image.height // 24)
    noise = ImageBuffer.blank(small_width, small_height, (0, 0, 0, 255))
    for y in range(small_height):
        for x in range(small_width):
            r = clamp(128 + rng.randint(-strength, strength))
            g = clamp(128 + rng.randint(-strength, strength))
            b = clamp(128 + rng.randint(-strength, strength))
            noise.set_pixel(x, y, (r, g, b, 255))

    noise = resize_bilinear(noise, image.width, image.height)
    opacity = 0.04 + rng.random() * 0.05
    for index in range(image.width * image.height):
        base = index * 4
        image.pixels[base] = clamp(
            int(round(image.pixels[base] * (1.0 - opacity) + noise.pixels[base] * opacity))
        )
        image.pixels[base + 1] = clamp(
            int(round(
                image.pixels[base + 1] * (1.0 - opacity)
                + noise.pixels[base + 1] * opacity
            ))
        )
        image.pixels[base + 2] = clamp(
            int(round(
                image.pixels[base + 2] * (1.0 - opacity)
                + noise.pixels[base + 2] * opacity
            ))
        )


def add_soft_stains(image: ImageBuffer, rng: random.Random, palette_name: str, strength: int) -> None:
    """Add translucent stain-like blobs and streaks."""
    stain_width = max(8, image.width // 16)
    stain_height = max(8, image.height // 16)
    stains = ImageBuffer.blank(stain_width, stain_height, (0, 0, 0, 0))
    stain_count = rng.randint(2, 5)
    for _ in range(stain_count):
        cx = rng.randint(0, stain_width - 1)
        cy = rng.randint(0, stain_height - 1)
        rx = rng.randint(max(2, stain_width // 20), max(4, stain_width // 6))
        ry = rng.randint(max(2, stain_height // 20), max(4, stain_height // 6))
        if palette_name == "soft_yellow":
            tone = (rng.randint(236, 248), rng.randint(228, 242), rng.randint(192, 214))
        else:
            tone = (rng.randint(238, 248), rng.randint(232, 244), rng.randint(220, 236))
        alpha = rng.randint(max(4, strength // 4), max(10, strength))
        draw_filled_ellipse(stains, cx, cy, rx, ry, (tone[0], tone[1], tone[2], alpha))

    for _ in range(rng.randint(1, 3)):
        x1 = rng.randint(0, stain_width - 1)
        y1 = rng.randint(0, stain_height - 1)
        x2 = rng.randint(0, stain_width - 1)
        y2 = rng.randint(0, stain_height - 1)
        tone = (
            rng.randint(180, 220),
            rng.randint(170, 210),
            rng.randint(150, 195),
            rng.randint(6, max(14, strength)),
        )
        draw_line(stains, x1, y1, x2, y2, tone, thickness=rng.randint(1, 2))

    stains = resize_bilinear(stains, image.width, image.height)
    for index in range(image.width * image.height):
        base = index * 4
        alpha = stains.pixels[base + 3] / 255.0
        if alpha <= 0:
            continue
        image.pixels[base] = clamp(
            int(round(image.pixels[base] * (1.0 - alpha) + stains.pixels[base] * alpha))
        )
        image.pixels[base + 1] = clamp(
            int(round(
                image.pixels[base + 1] * (1.0 - alpha)
                + stains.pixels[base + 1] * alpha
            ))
        )
        image.pixels[base + 2] = clamp(
            int(round(
                image.pixels[base + 2] * (1.0 - alpha)
                + stains.pixels[base + 2] * alpha
            ))
        )


def add_sheen(image: ImageBuffer, rng: random.Random, palette_name: str) -> None:
    """Add a broad glossy highlight that mimics sticky trap shine."""
    sheen_width = max(16, image.width // 10)
    sheen_height = max(16, image.height // 10)
    sheen = ImageBuffer.blank(sheen_width, sheen_height, (0, 0, 0, 0))
    sheen_count = rng.randint(2, 4)
    for _ in range(sheen_count):
        cx = rng.randint(0, sheen_width - 1)
        cy = rng.randint(0, sheen_height - 1)
        rx = rng.randint(max(3, sheen_width // 8), max(5, sheen_width // 3))
        ry = rng.randint(max(2, sheen_height // 10), max(4, sheen_height // 5))
        if palette_name == "soft_yellow":
            tone = (255, 249, 233)
        else:
            tone = (255, 252, 244)
        alpha = rng.randint(4, 14)
        draw_filled_ellipse(sheen, cx, cy, rx, ry, (tone[0], tone[1], tone[2], alpha))

    for _ in range(rng.randint(1, 2)):
        x1 = rng.randint(0, sheen_width - 1)
        y1 = rng.randint(0, sheen_height - 1)
        x2 = rng.randint(0, sheen_width - 1)
        y2 = rng.randint(0, sheen_height - 1)
        tone = (255, 250, 238, rng.randint(3, 8))
        draw_line(sheen, x1, y1, x2, y2, tone, thickness=1)

    sheen = resize_bilinear(sheen, image.width, image.height)
    for index in range(image.width * image.height):
        base = index * 4
        alpha = sheen.pixels[base + 3] / 255.0
        if alpha <= 0:
            continue
        image.pixels[base] = clamp(
            int(round(image.pixels[base] * (1.0 - alpha) + sheen.pixels[base] * alpha))
        )
        image.pixels[base + 1] = clamp(
            int(round(
                image.pixels[base + 1] * (1.0 - alpha)
                + sheen.pixels[base + 1] * alpha
            ))
        )
        image.pixels[base + 2] = clamp(
            int(round(
                image.pixels[base + 2] * (1.0 - alpha)
                + sheen.pixels[base + 2] * alpha
            ))
        )


def add_debris(image: ImageBuffer, rng: random.Random, count_range: tuple[int, int]) -> None:
    """Draw small dust specks and faint scratches that mimic trap debris."""
    count = rng.randint(count_range[0], count_range[1])
    for _ in range(count):
        x = rng.randint(0, image.width - 1)
        y = rng.randint(0, image.height - 1)
        radius = rng.randint(1, 2)
        if rng.random() < 0.8:
            tone = (
                rng.randint(90, 145),
                rng.randint(82, 132),
                rng.randint(60, 110),
                rng.randint(40, 120),
            )
        else:
            tone = (
                rng.randint(190, 235),
                rng.randint(182, 228),
                rng.randint(160, 214),
                rng.randint(30, 80),
            )
        draw_filled_circle(image, x, y, radius, tone)

    for _ in range(rng.randint(0, 2)):
        x1 = rng.randint(0, image.width - 1)
        y1 = rng.randint(0, image.height - 1)
        x2 = min(image.width - 1, max(0, x1 + rng.randint(-120, 120)))
        y2 = min(image.height - 1, max(0, y1 + rng.randint(-60, 60)))
        tone = (
            rng.randint(140, 190),
            rng.randint(128, 176),
            rng.randint(110, 150),
            rng.randint(18, 60),
        )
        draw_line(image, x1, y1, x2, y2, tone, thickness=rng.randint(1, 2))


def apply_lighting(image: ImageBuffer, rng: random.Random) -> None:
    """Apply lighting falloff, vignette, and mild color shifts."""
    center_x = rng.uniform(0.25, 0.75) * image.width
    center_y = rng.uniform(0.2, 0.7) * image.height
    radius_x = max(1.0, image.width * rng.uniform(0.7, 1.2))
    radius_y = max(1.0, image.height * rng.uniform(0.7, 1.2))
    strength = rng.uniform(0.02, 0.08)
    vignette = rng.uniform(0.04, 0.14)
    hue_shift = rng.uniform(0.98, 1.03)

    for y in range(image.height):
        for x in range(image.width):
            index = image.idx(x, y)
            dx = (x - center_x) / radius_x
            dy = (y - center_y) / radius_y
            light = max(0.0, 1.0 - (dx * dx + dy * dy))
            light_factor = 1.0 + strength * light

            nx = (x / max(1, image.width - 1)) * 2.0 - 1.0
            ny = (y / max(1, image.height - 1)) * 2.0 - 1.0
            edge_distance = (nx * nx + ny * ny) ** 0.5
            vignette_factor = 1.0 - vignette * min(1.0, edge_distance * edge_distance)
            factor = max(0.65, min(1.25, light_factor * vignette_factor))
            image.pixels[index] = clamp(int(round(image.pixels[index] * factor * hue_shift)))
            image.pixels[index + 1] = clamp(
                int(round(image.pixels[index + 1] * factor * (2.0 - hue_shift)))
            )
            image.pixels[index + 2] = clamp(int(round(image.pixels[index + 2] * factor)))


def build_background(rng: random.Random, width: int, height: int) -> tuple[ImageBuffer, dict[str, object]]:
    """Build one procedural background and return its metadata."""
    preset = rng.choices(PALETTES, weights=[float(p["weight"]) for p in PALETTES], k=1)[0]
    base = ImageBuffer.blank(width, height, (0, 0, 0, 255))
    fill_gradient(
        base,
        preset["top_left"],
        preset["top_right"],
        preset["bottom_left"],
        preset["bottom_right"],
    )
    add_low_frequency_noise(base, rng, strength=int(preset["noise"]))
    add_soft_stains(base, rng, str(preset["name"]), int(preset["stain_strength"]))
    add_sheen(base, rng, str(preset["name"]))
    add_debris(base, rng, tuple(preset["speck_count"]))  # type: ignore[arg-type]
    apply_lighting(base, rng)

    meta = {
        "preset": preset["name"],
        "width": width,
        "height": height,
    }
    return base, meta


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the background generator."""
    parser = argparse.ArgumentParser(
        description="Generate trap-like background images procedurally."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/backgrounds/generated"),
        help="Where generated backgrounds will be saved.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=24,
        help="Number of backgrounds to generate.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Output image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Output image height.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="manifest.json",
        help="File name for the metadata manifest.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments, generate backgrounds, and write the manifest."""
    parser = build_arg_parser()
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for index in range(args.count):
        background, meta = build_background(rng, args.width, args.height)
        file_name = f"bg_{index + 1:04d}.png"
        output_path = output_dir / file_name
        save_png(background, output_path)
        meta.update({"file": file_name, "index": index + 1})
        manifest.append(meta)
        print(f"Saved {output_path}")

    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()

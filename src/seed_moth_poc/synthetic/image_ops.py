"""Image transformation helpers used by the synthetic generator."""

import math
from dataclasses import dataclass
from pathlib import Path

from seed_moth_poc.data_prep.commons import (
    Box,
    ImageBuffer,
    blend_pixel,
    crop_image,
    draw_filled_circle,
    fit_to_box,
    load_image,
    resize_bilinear,
    save_png,
)

ALPHA_THRESHOLD = 8


@dataclass(slots=True)
class SourceImage:
    """One synthetic source cutout loaded from disk."""

    path: Path
    kind: str
    image: ImageBuffer


def alpha_bbox(image: ImageBuffer, alpha_threshold: int = ALPHA_THRESHOLD) -> Box | None:
    """Return the tight bounding box of non-transparent pixels."""
    left = image.width
    top = image.height
    right = -1
    bottom = -1
    for index in range(image.width * image.height):
        if image.pixels[index * 4 + 3] <= alpha_threshold:
            continue
        y = index // image.width
        x = index % image.width
        if x < left:
            left = x
        if y < top:
            top = y
        if x > right:
            right = x
        if y > bottom:
            bottom = y

    if right < left or bottom < top:
        return None
    return Box(float(left), float(top), float(right), float(bottom))


def crop_to_alpha(
    image: ImageBuffer,
    alpha_threshold: int = ALPHA_THRESHOLD,
) -> ImageBuffer:
    """Crop an image to its visible alpha bounds."""
    bounds = alpha_bbox(image, alpha_threshold=alpha_threshold)
    if bounds is None:
        return image.copy()
    left = int(bounds.x1)
    top = int(bounds.y1)
    right = int(bounds.x2) + 1
    bottom = int(bounds.y2) + 1
    return crop_image(image, left, top, right, bottom, fill=(0, 0, 0, 0))


def load_source_image(path: Path, kind: str) -> SourceImage:
    """Load a source cutout from disk."""
    return SourceImage(path=path, kind=kind, image=load_image(path))


def _sample_premultiplied(image: ImageBuffer, x: float, y: float) -> tuple[int, int, int, int]:
    """Sample an RGBA pixel using bilinear interpolation in premultiplied space."""
    if x < 0.0 or y < 0.0 or x > image.width - 1 or y > image.height - 1:
        return (0, 0, 0, 0)

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(image.width - 1, x0 + 1)
    y1 = min(image.height - 1, y0 + 1)
    tx = x - x0
    ty = y - y0

    accum_r = 0.0
    accum_g = 0.0
    accum_b = 0.0
    accum_a = 0.0

    for src_y, wy in ((y0, 1.0 - ty), (y1, ty)):
        for src_x, wx in ((x0, 1.0 - tx), (x1, tx)):
            weight = wx * wy
            if weight <= 0.0:
                continue
            index = image.idx(src_x, src_y)
            r = image.pixels[index] / 255.0
            g = image.pixels[index + 1] / 255.0
            b = image.pixels[index + 2] / 255.0
            a = image.pixels[index + 3] / 255.0
            accum_r += r * a * weight
            accum_g += g * a * weight
            accum_b += b * a * weight
            accum_a += a * weight

    if accum_a <= 0.0:
        return (0, 0, 0, 0)

    r = int(round((accum_r / accum_a) * 255.0))
    g = int(round((accum_g / accum_a) * 255.0))
    b = int(round((accum_b / accum_a) * 255.0))
    a = int(round(accum_a * 255.0))
    return (
        max(0, min(255, r)),
        max(0, min(255, g)),
        max(0, min(255, b)),
        max(0, min(255, a)),
    )


def resize_rgba(image: ImageBuffer, new_width: int, new_height: int) -> ImageBuffer:
    """Resize an RGBA image while preserving transparency edges."""
    if new_width <= 0 or new_height <= 0:
        raise ValueError("New dimensions must be positive.")
    if new_width == image.width and new_height == image.height:
        return image.copy()

    result = bytearray(new_width * new_height * 4)
    for dest_y in range(new_height):
        source_y = (dest_y + 0.5) * image.height / new_height - 0.5
        for dest_x in range(new_width):
            source_x = (dest_x + 0.5) * image.width / new_width - 0.5
            r, g, b, a = _sample_premultiplied(image, source_x, source_y)
            target = (dest_y * new_width + dest_x) * 4
            result[target] = r
            result[target + 1] = g
            result[target + 2] = b
            result[target + 3] = a
    return ImageBuffer(new_width, new_height, result)


def rotate_rgba(image: ImageBuffer, degrees: float, *, expand: bool = True) -> ImageBuffer:
    """Rotate an RGBA image around its center."""
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Image dimensions must be positive.")

    radians = math.radians(degrees % 360.0)
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    center_x = (image.width - 1) / 2.0
    center_y = (image.height - 1) / 2.0

    corners = [
        (0.0 - center_x, 0.0 - center_y),
        ((image.width - 1.0) - center_x, 0.0 - center_y),
        (0.0 - center_x, (image.height - 1.0) - center_y),
        ((image.width - 1.0) - center_x, (image.height - 1.0) - center_y),
    ]
    rotated = [
        (cos_a * x - sin_a * y, sin_a * x + cos_a * y)
        for x, y in corners
    ]
    min_x = min(point[0] for point in rotated)
    max_x = max(point[0] for point in rotated)
    min_y = min(point[1] for point in rotated)
    max_y = max(point[1] for point in rotated)

    if expand:
        out_width = max(1, int(math.ceil(max_x - min_x + 1.0)))
        out_height = max(1, int(math.ceil(max_y - min_y + 1.0)))
    else:
        out_width = image.width
        out_height = image.height
        min_x = -center_x
        min_y = -center_y

    result = bytearray(out_width * out_height * 4)
    for dest_y in range(out_height):
        for dest_x in range(out_width):
            rotated_x = dest_x + min_x
            rotated_y = dest_y + min_y
            source_x = cos_a * rotated_x + sin_a * rotated_y + center_x
            source_y = -sin_a * rotated_x + cos_a * rotated_y + center_y
            r, g, b, a = _sample_premultiplied(image, source_x, source_y)
            target = (dest_y * out_width + dest_x) * 4
            result[target] = r
            result[target + 1] = g
            result[target + 2] = b
            result[target + 3] = a
    return ImageBuffer(out_width, out_height, result)


def adjust_brightness(image: ImageBuffer, factor: float) -> ImageBuffer:
    """Scale visible pixels by one brightness factor."""
    result = image.copy()
    for index in range(image.width * image.height):
        base = index * 4
        alpha = result.pixels[base + 3]
        if alpha == 0:
            continue
        result.pixels[base] = max(0, min(255, int(round(result.pixels[base] * factor))))
        result.pixels[base + 1] = max(
            0, min(255, int(round(result.pixels[base + 1] * factor)))
        )
        result.pixels[base + 2] = max(
            0, min(255, int(round(result.pixels[base + 2] * factor)))
        )
    return result


def adjust_contrast(image: ImageBuffer, factor: float) -> ImageBuffer:
    """Apply a simple contrast adjustment around the midpoint."""
    result = image.copy()
    for index in range(image.width * image.height):
        base = index * 4
        alpha = result.pixels[base + 3]
        if alpha == 0:
            continue
        for channel in range(3):
            value = result.pixels[base + channel]
            adjusted = int(round((value - 128.0) * factor + 128.0))
            result.pixels[base + channel] = max(0, min(255, adjusted))
    return result


def blend_toward_color(
    image: ImageBuffer,
    tint: tuple[int, int, int],
    opacity: float,
) -> ImageBuffer:
    """Blend visible pixels toward a warm tint."""
    result = image.copy()
    opacity = max(0.0, min(1.0, opacity))
    inverse = 1.0 - opacity
    for index in range(image.width * image.height):
        base = index * 4
        if result.pixels[base + 3] == 0:
            continue
        result.pixels[base] = max(
            0,
            min(
                255,
                int(round(result.pixels[base] * inverse + tint[0] * opacity)),
            ),
        )
        result.pixels[base + 1] = max(
            0,
            min(
                255,
                int(round(result.pixels[base + 1] * inverse + tint[1] * opacity)),
            ),
        )
        result.pixels[base + 2] = max(
            0,
            min(
                255,
                int(round(result.pixels[base + 2] * inverse + tint[2] * opacity)),
            ),
        )
    return result


def add_c_shape_spots(
    image: ImageBuffer,
    *,
    spot_color: tuple[int, int, int] = (37, 30, 22),
    density: float = 1.0,
) -> ImageBuffer:
    """Overlay a C-shaped trail of dark spots inspired by the article description."""
    bounds = alpha_bbox(image)
    if bounds is None:
        return image.copy()

    result = image.copy()
    left = int(bounds.x1)
    top = int(bounds.y1)
    right = int(bounds.x2)
    bottom = int(bounds.y2)
    width = max(1, right - left + 1)
    height = max(1, bottom - top + 1)
    count = max(4, int(round((width / 50.0 + height / 40.0) * density)))

    arc_cx = left + int(round(width * 0.73))
    arc_cy = top + int(round(height * 0.48))
    arc_rx = max(2, int(round(width * 0.22)))
    arc_ry = max(2, int(round(height * 0.34)))
    for index in range(count):
        t = 0.15 + (index / max(1, count - 1)) * 0.75
        angle = math.radians(215.0 + 120.0 * t)
        x = int(round(arc_cx + arc_rx * math.cos(angle)))
        y = int(round(arc_cy + arc_ry * math.sin(angle)))
        radius = max(1, int(round(min(width, height) * (0.018 + 0.01 * density))))
        alpha = min(160, 90 + int(round(40 * density)))
        draw_filled_circle(result, x, y, radius, (spot_color[0], spot_color[1], spot_color[2], alpha))

    scatter_count = max(2, int(round(4 * density)))
    for _ in range(scatter_count):
        x = int(round(left + 0.12 * width + (0.76 * width) * (_ / max(1, scatter_count - 1))))
        x += int(round((math.sin(_ + width) * 0.07) * width))
        y = int(round(top + rng_like_noise(width, height, _) * height))
        radius = max(1, int(round(min(width, height) * 0.012)))
        draw_filled_circle(result, x, y, radius, (spot_color[0], spot_color[1], spot_color[2], 120))

    return result


def rng_like_noise(width: int, height: int, index: int) -> float:
    """Return a deterministic small noise term for spot placement."""
    seed = (width * 1315423911) ^ (height * 2654435761) ^ (index * 97531)
    return ((seed & 0xFFFF) / 0xFFFF) * 0.2 - 0.1


def paste_rgba(
    base: ImageBuffer,
    overlay: ImageBuffer,
    x: int,
    y: int,
    *,
    opacity: float = 1.0,
) -> None:
    """Alpha-blend one image onto another."""
    opacity = max(0.0, min(1.0, opacity))
    if opacity <= 0.0:
        return
    for overlay_y in range(overlay.height):
        dest_y = y + overlay_y
        if dest_y < 0 or dest_y >= base.height:
            continue
        for overlay_x in range(overlay.width):
            dest_x = x + overlay_x
            if dest_x < 0 or dest_x >= base.width:
                continue
            src_index = overlay.idx(overlay_x, overlay_y)
            alpha = int(round(overlay.pixels[src_index + 3] * opacity))
            if alpha <= 0:
                continue
            blend_pixel(
                base,
                dest_x,
                dest_y,
                (
                    overlay.pixels[src_index],
                    overlay.pixels[src_index + 1],
                    overlay.pixels[src_index + 2],
                    alpha,
                ),
            )


def draw_shadow(
    base: ImageBuffer,
    overlay: ImageBuffer,
    x: int,
    y: int,
    *,
    offset_x: int = 3,
    offset_y: int = 4,
    opacity: float = 0.22,
    color: tuple[int, int, int] = (78, 60, 40),
) -> None:
    """Draw a soft shadow underneath a pasted moth cutout."""
    opacity = max(0.0, min(1.0, opacity))
    if opacity <= 0.0:
        return
    for overlay_y in range(overlay.height):
        dest_y = y + overlay_y + offset_y
        if dest_y < 0 or dest_y >= base.height:
            continue
        for overlay_x in range(overlay.width):
            dest_x = x + overlay_x + offset_x
            if dest_x < 0 or dest_x >= base.width:
                continue
            src_index = overlay.idx(overlay_x, overlay_y)
            alpha = overlay.pixels[src_index + 3]
            if alpha <= 0:
                continue
            shadow_alpha = int(round(alpha * opacity))
            if shadow_alpha <= 0:
                continue
            blend_pixel(
                base,
                dest_x,
                dest_y,
                (color[0], color[1], color[2], shadow_alpha),
            )


def load_and_crop_source(path: Path, kind: str) -> SourceImage:
    return SourceImage(path=path, kind=kind, image=crop_to_alpha(load_image(path)))


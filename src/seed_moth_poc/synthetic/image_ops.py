"""Image transformation helpers used by the synthetic generator."""

import math
import random
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
MASK_THRESHOLD = 24


@dataclass(slots=True)
class SourceImage:
    """One synthetic source cutout loaded from disk."""

    path: Path
    image: ImageBuffer


def _clamp_channel(value: float) -> int:
    """Clamp a floating-point channel value to the 8-bit range."""
    return max(0, min(255, int(round(value))))


def foreground_mask_from_image(
    image: ImageBuffer,
    *,
    threshold: int = MASK_THRESHOLD,
    keep_largest_component: bool = True,
) -> bytearray:
    """Build a binary foreground mask from a mostly black/white template image."""
    mask = bytearray(image.width * image.height)
    for index in range(image.width * image.height):
        base = index * 4
        pixel = image.pixels[base : base + 4]
        if pixel[3] < 250:
            value = pixel[3]
        else:
            value = (pixel[0] * 299 + pixel[1] * 587 + pixel[2] * 114) // 1000
        if value >= threshold:
            mask[index] = 1

    if keep_largest_component:
        mask = keep_largest_foreground_component(mask, image.width, image.height)
    return mask


def keep_largest_foreground_component(
    mask: bytearray,
    width: int,
    height: int,
) -> bytearray:
    """Keep only the largest 8-connected foreground component in a binary mask."""
    visited = bytearray(width * height)
    best_component: list[int] = []
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
        component: list[int] = []

        while stack:
            current = stack.pop()
            component.append(current)
            cy = current // width
            cx = current % width
            for dx, dy in neighbors:
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                neighbor = ny * width + nx
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)

        if len(component) > len(best_component):
            best_component = component

    result = bytearray(width * height)
    for index in best_component:
        result[index] = 1
    return result


def mask_bbox(mask: bytearray, width: int, height: int) -> Box | None:
    """Return the tight bounding box for a binary foreground mask."""
    left = width
    top = height
    right = -1
    bottom = -1
    for index in range(width * height):
        if mask[index] == 0:
            continue
        y = index // width
        x = index % width
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


def mask_coverage(mask: bytearray, width: int, height: int) -> float:
    """Return the fraction of pixels marked as foreground in a binary mask."""
    total = width * height
    if total <= 0:
        return 0.0
    return sum(mask) / total


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


def load_source_image(path: Path) -> SourceImage:
    """Load a source cutout from disk."""
    return SourceImage(path=path, image=load_image(path))


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


def add_scattered_spots(
    image: ImageBuffer,
    mask: bytearray,
    *,
    spot_color: tuple[int, int, int] = (37, 30, 22),
    density: float = 1.0,
    rng: random.Random | None = None,
) -> ImageBuffer:
    """Overlay many small dark spots inside the visible foreground area."""
    result = image.copy()
    bounds = mask_bbox(mask, image.width, image.height)
    if bounds is None:
        return result

    rng = rng or random.Random(0)
    left = int(bounds.x1)
    top = int(bounds.y1)
    right = int(bounds.x2)
    bottom = int(bounds.y2)
    width = max(1, right - left + 1)
    height = max(1, bottom - top + 1)
    count = max(6, int(round((width + height) / 18.0 * density)))

    for _ in range(count):
        chosen_x = None
        chosen_y = None
        for _attempt in range(32):
            x = rng.randint(left, right)
            y = rng.randint(top, bottom)
            if mask[y * image.width + x]:
                chosen_x = x
                chosen_y = y
                break
        if chosen_x is None or chosen_y is None:
            continue

        radius = max(1, int(round(min(width, height) * rng.uniform(0.012, 0.026))))
        alpha = rng.randint(90, 170)
        draw_filled_circle(
            result,
            chosen_x,
            chosen_y,
            radius,
            (spot_color[0], spot_color[1], spot_color[2], alpha),
        )

    return result


def render_procedural_moth(
    template: ImageBuffer,
    *,
    base_tint: tuple[int, int, int],
    rng: random.Random,
) -> ImageBuffer:
    """Render a clean moth sprite from a binary template."""
    mask = foreground_mask_from_image(template)
    bounds = mask_bbox(mask, template.width, template.height)
    if bounds is None:
        return ImageBuffer.blank(1, 1, (0, 0, 0, 0))

    left = int(bounds.x1)
    top = int(bounds.y1)
    right = int(bounds.x2)
    bottom = int(bounds.y2)
    width = max(1, right - left + 1)
    height = max(1, bottom - top + 1)

    light_tint = (
        _clamp_channel(base_tint[0] * rng.uniform(1.04, 1.12)),
        _clamp_channel(base_tint[1] * rng.uniform(1.02, 1.08)),
        _clamp_channel(base_tint[2] * rng.uniform(0.96, 1.03)),
    )
    mid_tint = (
        _clamp_channel(base_tint[0] * rng.uniform(0.94, 1.00)),
        _clamp_channel(base_tint[1] * rng.uniform(0.90, 0.98)),
        _clamp_channel(base_tint[2] * rng.uniform(0.82, 0.92)),
    )
    dark_tint = (
        _clamp_channel(base_tint[0] * rng.uniform(0.72, 0.84)),
        _clamp_channel(base_tint[1] * rng.uniform(0.64, 0.78)),
        _clamp_channel(base_tint[2] * rng.uniform(0.58, 0.72)),
    )

    result = ImageBuffer.blank(template.width, template.height, (0, 0, 0, 0))
    for index in range(template.width * template.height):
        if mask[index] == 0:
            continue
        y = index // template.width
        x = index % template.width
        x_norm = 0.0 if width == 1 else (x - left) / (width - 1)
        y_norm = 0.0 if height == 1 else (y - top) / (height - 1)
        center_bias = 1.0 - min(1.0, abs(x_norm - 0.5) * 2.0)
        wing_bias = 1.0 - y_norm
        shade = 0.62 + 0.26 * wing_bias + 0.12 * center_bias
        if y_norm < 0.45:
            blend = y_norm / 0.45 if 0.45 > 0 else 0.0
            tone = (
                _clamp_channel(dark_tint[0] * (1.0 - blend) + mid_tint[0] * blend),
                _clamp_channel(dark_tint[1] * (1.0 - blend) + mid_tint[1] * blend),
                _clamp_channel(dark_tint[2] * (1.0 - blend) + mid_tint[2] * blend),
            )
        else:
            blend = (y_norm - 0.45) / 0.55 if 0.55 > 0 else 0.0
            tone = (
                _clamp_channel(mid_tint[0] * (1.0 - blend) + light_tint[0] * blend),
                _clamp_channel(mid_tint[1] * (1.0 - blend) + light_tint[1] * blend),
                _clamp_channel(mid_tint[2] * (1.0 - blend) + light_tint[2] * blend),
            )
        result.pixels[index * 4] = _clamp_channel(tone[0] * shade)
        result.pixels[index * 4 + 1] = _clamp_channel(tone[1] * shade)
        result.pixels[index * 4 + 2] = _clamp_channel(tone[2] * shade)
        result.pixels[index * 4 + 3] = 255

    result = add_scattered_spots(
        result,
        mask,
        spot_color=(35, 28, 20),
        density=rng.uniform(0.75, 1.10),
        rng=rng,
    )

    return crop_image(result, left, top, right + 1, bottom + 1, fill=(0, 0, 0, 0))


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


def load_and_crop_source(path: Path) -> SourceImage:
    return SourceImage(path=path, image=crop_to_alpha(load_image(path)))

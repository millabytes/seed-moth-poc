"""Low-level image buffer helpers used by the data-prep tools."""

import binascii
import math
import struct
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SUFFIXES = {".jpg", ".jpeg"}


@dataclass(slots=True)
class ImageBuffer:
    """In-memory RGBA image stored as a flat bytearray."""

    width: int
    height: int
    pixels: bytearray

    def __post_init__(self) -> None:
        """Validate the backing buffer size."""
        expected = self.width * self.height * 4
        if len(self.pixels) != expected:
            raise ValueError(
                f"Expected {expected} RGBA bytes, got {len(self.pixels)}."
            )

    @classmethod
    def blank(
        cls,
        width: int,
        height: int,
        color: tuple[int, int, int, int] = (255, 255, 255, 255),
    ) -> "ImageBuffer":
        """Create a solid-color image."""
        pixels = bytearray(width * height * 4)
        fill = bytes(color)
        pixels[:] = fill * (width * height)
        return cls(width, height, pixels)

    def copy(self) -> "ImageBuffer":
        """Return a deep copy of the image."""
        return ImageBuffer(self.width, self.height, bytearray(self.pixels))

    def idx(self, x: int, y: int) -> int:
        """Return the byte index for a pixel coordinate."""
        return (y * self.width + x) * 4

    def get_pixel(self, x: int, y: int) -> tuple[int, int, int, int]:
        """Read one RGBA pixel."""
        index = self.idx(x, y)
        return tuple(self.pixels[index : index + 4])  # type: ignore[return-value]

    def set_pixel(self, x: int, y: int, color: Sequence[int]) -> None:
        """Write one RGBA pixel."""
        index = self.idx(x, y)
        self.pixels[index] = int(color[0]) & 255
        self.pixels[index + 1] = int(color[1]) & 255
        self.pixels[index + 2] = int(color[2]) & 255
        self.pixels[index + 3] = int(color[3]) & 255

    def fill(self, color: Sequence[int]) -> None:
        """Fill the entire image with one RGBA color."""
        self.pixels[:] = bytes(int(c) & 255 for c in color[:4]) * (
            self.width * self.height
        )


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Build one PNG chunk."""
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", binascii.crc32(chunk_type + data) & 0xFFFFFFFF)
    return length + chunk_type + data + crc


def _unfilter_scanline(
    filter_type: int,
    scanline: bytearray,
    previous: bytes,
    bytes_per_pixel: int,
) -> None:
    """Invert one PNG scanline filter in-place."""
    if filter_type == 0:
        return

    if filter_type == 1:
        for index in range(bytes_per_pixel, len(scanline)):
            scanline[index] = (scanline[index] + scanline[index - bytes_per_pixel]) & 255
        return

    if filter_type == 2:
        for index in range(len(scanline)):
            scanline[index] = (scanline[index] + previous[index]) & 255
        return

    if filter_type == 3:
        for index in range(len(scanline)):
            left = scanline[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            up = previous[index]
            scanline[index] = (scanline[index] + ((left + up) // 2)) & 255
        return

    if filter_type == 4:
        for index in range(len(scanline)):
            left = scanline[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            up = previous[index]
            up_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            p = left + up - up_left
            pa = abs(p - left)
            pb = abs(p - up)
            pc = abs(p - up_left)
            if pa <= pb and pa <= pc:
                predictor = left
            elif pb <= pc:
                predictor = up
            else:
                predictor = up_left
            scanline[index] = (scanline[index] + predictor) & 255
        return

    raise ValueError(f"Unsupported PNG filter type: {filter_type}.")


def load_png_bytes(data: bytes) -> ImageBuffer:
    """Decode a PNG byte string into an ImageBuffer."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("Data does not look like a PNG image.")

    position = len(PNG_SIGNATURE)
    width = height = None
    bit_depth = None
    color_type = None
    interlace = None
    idat_parts: list[bytes] = []

    while position < len(data):
        length = struct.unpack(">I", data[position : position + 4])[0]
        position += 4
        chunk_type = data[position : position + 4]
        position += 4
        chunk_data = data[position : position + length]
        position += length
        position += 4  # CRC

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError("PNG header is incomplete.")
    if bit_depth != 8:
        raise ValueError(f"Unsupported PNG bit depth: {bit_depth}.")
    if interlace not in (0, None):
        raise ValueError("Interlaced PNGs are not supported.")

    if color_type == 6:
        channels = 4
    elif color_type == 2:
        channels = 3
    elif color_type == 0:
        channels = 1
    elif color_type == 4:
        channels = 2
    else:
        raise ValueError(f"Unsupported PNG color type: {color_type}.")

    decompressed = zlib.decompress(b"".join(idat_parts))
    stride = width * channels
    expected = height * (stride + 1)
    if len(decompressed) < expected:
        raise ValueError("PNG payload is truncated.")

    pixels = bytearray(width * height * 4)
    previous = bytearray(stride)
    source = 0
    target = 0

    for _ in range(height):
        filter_type = decompressed[source]
        source += 1
        scanline = bytearray(decompressed[source : source + stride])
        source += stride
        _unfilter_scanline(filter_type, scanline, previous, channels)
        previous = scanline

        if channels == 4:
            pixels[target : target + width * 4] = scanline
        elif channels == 3:
            for offset in range(width):
                base = offset * 3
                dest = target + offset * 4
                pixels[dest] = scanline[base]
                pixels[dest + 1] = scanline[base + 1]
                pixels[dest + 2] = scanline[base + 2]
                pixels[dest + 3] = 255
        elif channels == 2:
            for offset in range(width):
                base = offset * 2
                value = scanline[base]
                pixels[target + offset * 4 : target + offset * 4 + 4] = bytes(
                    (value, value, value, scanline[base + 1])
                )
        else:
            for offset in range(width):
                value = scanline[offset]
                pixels[target + offset * 4 : target + offset * 4 + 4] = bytes(
                    (value, value, value, 255)
                )
        target += width * 4

    return ImageBuffer(width, height, pixels)


def _convert_jpeg_to_png_bytes(path: Path) -> bytes:
    """Convert a JPEG file to PNG bytes using `sips` on macOS."""
    cache_dir = Path(tempfile.gettempdir()) / "seed_moth_poc_jpeg_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{path.stem}-{path.stat().st_mtime_ns}.png"
    if not target.exists():
        result = subprocess.run(
            ["sips", "-s", "format", "png", str(path), "--out", str(target)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to convert JPEG via sips: {result.stderr.strip()}"
            )
    return target.read_bytes()


def load_image(path: Path | str) -> ImageBuffer:
    """Load a PNG or JPEG image into an ImageBuffer."""
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix in JPEG_SUFFIXES:
        png_bytes = _convert_jpeg_to_png_bytes(input_path)
        return load_png_bytes(png_bytes)
    return load_png_bytes(input_path.read_bytes())


def save_png(image: ImageBuffer, path: Path | str) -> None:
    """Encode an ImageBuffer as a PNG file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    row_stride = image.width * 4
    for row in range(image.height):
        start = row * row_stride
        rows.append(b"\x00" + bytes(image.pixels[start : start + row_stride]))
    payload = zlib.compress(b"".join(rows), level=6)
    ihdr = struct.pack(">IIBBBBB", image.width, image.height, 8, 6, 0, 0, 0)
    png = b"".join(
        [
            PNG_SIGNATURE,
            _chunk(b"IHDR", ihdr),
            _chunk(b"IDAT", payload),
            _chunk(b"IEND", b""),
        ]
    )
    output_path.write_bytes(png)


def fit_to_box(
    width: int,
    height: int,
    max_width: int,
    max_height: int,
    *,
    allow_upscale: bool = True,
) -> tuple[int, int, float]:
    """Scale an image to fit inside a target box."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    scale = min(max_width / width, max_height / height)
    if not allow_upscale:
        scale = min(scale, 1.0)
    scale = max(scale, 0.01)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return new_width, new_height, scale


def resize_nearest(image: ImageBuffer, new_width: int, new_height: int) -> ImageBuffer:
    """Resize an image with nearest-neighbor sampling."""
    if new_width == image.width and new_height == image.height:
        return image.copy()

    result = bytearray(new_width * new_height * 4)
    x_scale = image.width / new_width
    y_scale = image.height / new_height

    for dest_y in range(new_height):
        source_y = min(image.height - 1, int(dest_y * y_scale))
        source_row = source_y * image.width * 4
        target_row = dest_y * new_width * 4
        for dest_x in range(new_width):
            source_x = min(image.width - 1, int(dest_x * x_scale))
            source_index = source_row + source_x * 4
            target_index = target_row + dest_x * 4
            result[target_index : target_index + 4] = image.pixels[
                source_index : source_index + 4
            ]

    return ImageBuffer(new_width, new_height, result)


def resize_bilinear(image: ImageBuffer, new_width: int, new_height: int) -> ImageBuffer:
    """Resize an image with bilinear interpolation."""
    if new_width == image.width and new_height == image.height:
        return image.copy()

    result = bytearray(new_width * new_height * 4)
    x_positions: list[tuple[int, int, float]] = []
    y_positions: list[tuple[int, int, float]] = []

    for dest_x in range(new_width):
        source_x = (dest_x + 0.5) * image.width / new_width - 0.5
        left = int(math.floor(source_x))
        if left < 0:
            left = 0
            right = 0
            weight = 0.0
        elif left >= image.width - 1:
            left = image.width - 1
            right = left
            weight = 0.0
        else:
            right = left + 1
            weight = source_x - left
        x_positions.append((left, right, weight))

    for dest_y in range(new_height):
        source_y = (dest_y + 0.5) * image.height / new_height - 0.5
        top = int(math.floor(source_y))
        if top < 0:
            top = 0
            bottom = 0
            weight = 0.0
        elif top >= image.height - 1:
            top = image.height - 1
            bottom = top
            weight = 0.0
        else:
            bottom = top + 1
            weight = source_y - top
        y_positions.append((top, bottom, weight))

    source = image.pixels
    source_stride = image.width * 4
    for dest_y, (top, bottom, y_weight) in enumerate(y_positions):
        top_row = top * source_stride
        bottom_row = bottom * source_stride
        target_row = dest_y * new_width * 4
        for dest_x, (left, right, x_weight) in enumerate(x_positions):
            target_index = target_row + dest_x * 4
            top_left = top_row + left * 4
            top_right = top_row + right * 4
            bottom_left = bottom_row + left * 4
            bottom_right = bottom_row + right * 4
            for channel in range(4):
                top_value = source[top_left + channel] * (1.0 - x_weight) + source[
                    top_right + channel
                ] * x_weight
                bottom_value = source[bottom_left + channel] * (1.0 - x_weight) + source[
                    bottom_right + channel
                ] * x_weight
                value = top_value * (1.0 - y_weight) + bottom_value * y_weight
                result[target_index + channel] = int(round(value))

    return ImageBuffer(new_width, new_height, result)


def crop_image(
    image: ImageBuffer,
    left: int,
    top: int,
    right: int,
    bottom: int,
    *,
    fill: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> ImageBuffer:
    """Crop a rectangular region from an image."""
    width = max(1, right - left)
    height = max(1, bottom - top)
    result = ImageBuffer.blank(width, height, fill)
    for dest_y in range(height):
        source_y = top + dest_y
        if source_y < 0 or source_y >= image.height:
            continue
        for dest_x in range(width):
            source_x = left + dest_x
            if source_x < 0 or source_x >= image.width:
                continue
            source_index = image.idx(source_x, source_y)
            target_index = result.idx(dest_x, dest_y)
            result.pixels[target_index : target_index + 4] = image.pixels[
                source_index : source_index + 4
            ]
    return result


def to_photoimage_data(image: ImageBuffer) -> bytes:
    """Return PNG bytes that Tkinter can load as a PhotoImage."""
    rows = []
    row_stride = image.width * 4
    for row in range(image.height):
        start = row * row_stride
        rows.append(b"\x00" + bytes(image.pixels[start : start + row_stride]))
    payload = zlib.compress(b"".join(rows), level=6)
    ihdr = struct.pack(">IIBBBBB", image.width, image.height, 8, 6, 0, 0, 0)
    return b"".join(
        [
            PNG_SIGNATURE,
            _chunk(b"IHDR", ihdr),
            _chunk(b"IDAT", payload),
            _chunk(b"IEND", b""),
        ]
    )


def blend_pixel(image: ImageBuffer, x: int, y: int, color: Sequence[int]) -> None:
    """Alpha-blend one pixel onto an image."""
    if x < 0 or y < 0 or x >= image.width or y >= image.height:
        return
    index = image.idx(x, y)
    sr, sg, sb, sa = (int(color[0]), int(color[1]), int(color[2]), int(color[3]))
    if sa <= 0:
        return
    if sa >= 255:
        image.pixels[index] = sr & 255
        image.pixels[index + 1] = sg & 255
        image.pixels[index + 2] = sb & 255
        image.pixels[index + 3] = 255
        return

    inv = 255 - sa
    dr = image.pixels[index]
    dg = image.pixels[index + 1]
    db = image.pixels[index + 2]
    da = image.pixels[index + 3]
    image.pixels[index] = (sr * sa + dr * inv + 127) // 255
    image.pixels[index + 1] = (sg * sa + dg * inv + 127) // 255
    image.pixels[index + 2] = (sb * sa + db * inv + 127) // 255
    image.pixels[index + 3] = min(255, sa + (da * inv + 127) // 255)


def draw_filled_circle(
    image: ImageBuffer,
    center_x: int,
    center_y: int,
    radius: int,
    color: Sequence[int],
) -> None:
    """Draw a filled circle with alpha blending."""
    if radius <= 0:
        return
    x_min = max(0, center_x - radius)
    x_max = min(image.width - 1, center_x + radius)
    y_min = max(0, center_y - radius)
    y_max = min(image.height - 1, center_y + radius)
    radius_sq = radius * radius
    for y in range(y_min, y_max + 1):
        dy_sq = (y - center_y) * (y - center_y)
        if dy_sq > radius_sq:
            continue
        x_delta = int(math.sqrt(radius_sq - dy_sq))
        start = max(x_min, center_x - x_delta)
        end = min(x_max, center_x + x_delta)
        for x in range(start, end + 1):
            blend_pixel(image, x, y, color)


def draw_filled_ellipse(
    image: ImageBuffer,
    center_x: int,
    center_y: int,
    radius_x: int,
    radius_y: int,
    color: Sequence[int],
) -> None:
    """Draw a filled ellipse with alpha blending."""
    if radius_x <= 0 or radius_y <= 0:
        return
    x_min = max(0, center_x - radius_x)
    x_max = min(image.width - 1, center_x + radius_x)
    y_min = max(0, center_y - radius_y)
    y_max = min(image.height - 1, center_y + radius_y)
    for y in range(y_min, y_max + 1):
        normalized_y = (y - center_y) / radius_y
        value = 1.0 - normalized_y * normalized_y
        if value <= 0.0:
            continue
        x_delta = int(radius_x * math.sqrt(value))
        start = max(x_min, center_x - x_delta)
        end = min(x_max, center_x + x_delta)
        for x in range(start, end + 1):
            blend_pixel(image, x, y, color)


def draw_line(
    image: ImageBuffer,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Sequence[int],
    *,
    thickness: int = 1,
) -> None:
    """Draw an anti-aliased line approximation using blended circles."""
    distance = max(abs(x2 - x1), abs(y2 - y1))
    if distance == 0:
        draw_filled_circle(image, x1, y1, max(1, thickness // 2), color)
        return
    for step in range(distance + 1):
        t = step / distance
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        draw_filled_circle(image, x, y, max(1, thickness // 2), color)

"""Shared bbox and file helpers for seed moth data preparation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(slots=True)
class Box:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def ordered(self) -> "Box":
        """Return the box with coordinates normalized to left-top-right-bottom."""
        left = min(self.x1, self.x2)
        right = max(self.x1, self.x2)
        top = min(self.y1, self.y2)
        bottom = max(self.y1, self.y2)
        return Box(left, top, right, bottom)

    def clamp(self, width: int, height: int) -> "Box":
        """Clamp the box coordinates to the image bounds."""
        ordered = self.ordered()
        x1 = max(0.0, min(float(width - 1), ordered.x1))
        y1 = max(0.0, min(float(height - 1), ordered.y1))
        x2 = max(0.0, min(float(width - 1), ordered.x2))
        y2 = max(0.0, min(float(height - 1), ordered.y2))
        return Box(x1, y1, x2, y2)

    def width(self) -> float:
        """Return the box width in pixels."""
        ordered = self.ordered()
        return max(0.0, ordered.x2 - ordered.x1)

    def height(self) -> float:
        """Return the box height in pixels."""
        ordered = self.ordered()
        return max(0.0, ordered.y2 - ordered.y1)

    def is_valid(self, min_size: float = 2.0) -> bool:
        """Return whether the box is large enough to keep."""
        return self.width() >= min_size and self.height() >= min_size

    def to_yolo(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Convert the box to YOLO normalized coordinates."""
        ordered = self.clamp(width, height)
        box_width = max(0.0, ordered.x2 - ordered.x1)
        box_height = max(0.0, ordered.y2 - ordered.y1)
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive.")
        center_x = (ordered.x1 + ordered.x2) / 2.0 / width
        center_y = (ordered.y1 + ordered.y2) / 2.0 / height
        return (
            center_x,
            center_y,
            box_width / width,
            box_height / height,
        )

    @classmethod
    def from_yolo(
        cls,
        center_x: float,
        center_y: float,
        box_width: float,
        box_height: float,
        width: int,
        height: int,
    ) -> "Box":
        """Build a pixel-space box from YOLO normalized coordinates."""
        x1 = (center_x - box_width / 2.0) * width
        y1 = (center_y - box_height / 2.0) * height
        x2 = (center_x + box_width / 2.0) * width
        y2 = (center_y + box_height / 2.0) * height
        return cls(x1, y1, x2, y2).clamp(width, height)


def ensure_directory(path: Path | str) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def collect_image_files(
    paths: Sequence[Path | str],
    *,
    recursive: bool = False,
) -> list[Path]:
    """Collect image files from the provided paths."""
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                files.append(path)
            continue
        if not path.exists():
            continue
        iterator: Iterable[Path]
        if recursive:
            iterator = path.rglob("*")
        else:
            iterator = path.iterdir()
        for child in iterator:
            if child.is_file() and child.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                files.append(child)
    return sorted(files)


def read_yolo_boxes(
    label_path: Path | str,
    *,
    width: int,
    height: int,
) -> list[Box]:
    """Read YOLO labels from a text file and convert them to boxes."""
    path = Path(label_path)
    if not path.exists():
        return []

    boxes: list[Box] = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            _, cx, cy, bw, bh = parts
            box = Box.from_yolo(
                float(cx),
                float(cy),
                float(bw),
                float(bh),
                width,
                height,
            )
        except ValueError:
            continue
        boxes.append(box)
    return boxes


def write_yolo_boxes(
    label_path: Path | str,
    boxes: Sequence[Box],
    *,
    width: int,
    height: int,
    class_id: int = 0,
) -> None:
    """Write a sequence of boxes to a YOLO label file."""
    path = Path(label_path)
    ensure_directory(path.parent)
    lines = []
    for box in boxes:
        ordered = box.clamp(width, height)
        if not ordered.is_valid():
            continue
        cx, cy, bw, bh = ordered.to_yolo(width, height)
        lines.append(
            f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""))

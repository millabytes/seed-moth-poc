"""Lightweight priors for synthetic Stenoma catenifer samples.

The values below are derived from the [IDTools screening aid article](https://idtools.org/pdfs/Stenoma_catenifer.pdf):
- forewing length: 8.0-15.0 mm
- forewings: yellowish-tan
- adults: similar coloration between sexes, with females slightly larger
"""

import random

FOREWING_LENGTH_MM_RANGE: tuple[float, float] = (8.0, 15.0)

DEFAULT_EMPTY_PROBABILITY = 0.0
DEFAULT_MIN_OBJECTS = 2
DEFAULT_MAX_OBJECTS = 10

# Warm, tan-like tones that keep the synthetic moths in the right visual range.
TAN_TINTS: tuple[tuple[int, int, int], ...] = (
    (243, 224, 175),
    (236, 213, 160),
    (228, 201, 148),
    (220, 190, 137),
    (247, 231, 189),
)

SPOT_COLOR: tuple[int, int, int] = (35, 28, 20)
SHADOW_COLOR: tuple[int, int, int] = (72, 58, 40)


def sample_object_count(
    rng: random.Random,
    *,
    min_objects: int = DEFAULT_MIN_OBJECTS,
    max_objects: int = DEFAULT_MAX_OBJECTS,
    empty_probability: float = DEFAULT_EMPTY_PROBABILITY,
) -> int:
    """Sample how many moths to place on one synthetic image."""
    if max_objects < min_objects:
        raise ValueError("max_objects must be greater than or equal to min_objects.")

    if min_objects <= 0 and rng.random() < empty_probability:
        return 0

    count = rng.choices(
        population=list(range(min_objects, max_objects + 1)),
        weights=_object_count_weights(min_objects, max_objects),
        k=1,
    )[0]
    return count


def _object_count_weights(min_objects: int, max_objects: int) -> list[float]:
    """Return a count distribution biased toward low-to-medium crowding."""
    counts = list(range(min_objects, max_objects + 1))
    if not counts:
        return []

    weights: list[float] = []
    for count in counts:
        if count <= 0:
            weights.append(0.8)
        elif count == 1:
            weights.append(0.4)
        elif count == 2:
            weights.append(2.4)
        elif count == 3:
            weights.append(2.8)
        elif count == 4:
            weights.append(2.6)
        elif count == 5:
            weights.append(2.2)
        elif count == 6:
            weights.append(1.9)
        elif count == 7:
            weights.append(1.5)
        elif count == 8:
            weights.append(1.2)
        elif count == 9:
            weights.append(1.0)
        elif count == 10:
            weights.append(0.8)
        else:
            weights.append(max(0.3, 0.8 - 0.05 * (count - 10)))
    return weights


def sample_forewing_length_mm(rng: random.Random) -> float:
    """Sample a forewing length in millimeters from the PDF prior."""
    return rng.uniform(*FOREWING_LENGTH_MM_RANGE)


def sample_pixel_length(
    rng: random.Random,
    background_width: int,
    background_height: int,
) -> float:
    """Map the article's millimeter size prior into a plausible pixel scale."""
    if background_width <= 0 or background_height <= 0:
        raise ValueError("Background dimensions must be positive.")

    short_edge = min(background_width, background_height)
    px_per_mm = short_edge / rng.uniform(26.0, 38.0)
    return sample_forewing_length_mm(rng) * px_per_mm


def sample_tint_color(rng: random.Random) -> tuple[int, int, int]:
    """Pick a warm tan tint for the moth body and wings."""
    return rng.choice(TAN_TINTS)

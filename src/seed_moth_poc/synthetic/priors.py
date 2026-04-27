"""Article-informed priors for synthetic Stenoma catenifer samples.

The values below are derived from the [IDTools screening aid article](https://idtools.org/pdfs/Stenoma_catenifer.pdf):
- forewing length: 8.0-15.0 mm
- forewings: yellowish-tan with numerous black spots
- distal wing spots: rough C-shaped outline
- adults: similar coloration between sexes, with females slightly larger
"""

import random

FOREWING_LENGTH_MM_RANGE: tuple[float, float] = (8.0, 15.0)

# The source cutouts from morphology and reference imagery are mixed together,
# but morphology sources get slightly higher weight because they reinforce the
# posture/shape prior from the PDF.
REFERENCE_SOURCE_WEIGHT = 1.0
MORPHOLOGY_SOURCE_WEIGHT = 1.35

DEFAULT_EMPTY_PROBABILITY = 0.15
DEFAULT_MIN_OBJECTS = 0
DEFAULT_MAX_OBJECTS = 3

# Article cues are strongest on morphology cutouts, lighter on reference cutouts.
REFERENCE_ARTICLE_CUE_PROBABILITY = 0.35
MORPHOLOGY_ARTICLE_CUE_PROBABILITY = 0.9

# Warm, tan-like tones that keep the synthetic moths in the right visual range.
TAN_TINTS: tuple[tuple[int, int, int], ...] = (
    (235, 211, 161),
    (228, 200, 150),
    (220, 191, 141),
    (244, 226, 182),
)

SPOT_COLOR: tuple[int, int, int] = (37, 30, 22)
SHADOW_COLOR: tuple[int, int, int] = (78, 60, 40)


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
    """Return a simple count distribution biased toward one or two moths."""
    counts = list(range(min_objects, max_objects + 1))
    if not counts:
        return []

    weights: list[float] = []
    for count in counts:
        if count <= 0:
            weights.append(0.8)
        elif count == 1:
            weights.append(3.8)
        elif count == 2:
            weights.append(2.6)
        elif count == 3:
            weights.append(1.4)
        else:
            weights.append(max(0.4, 1.1 - 0.1 * (count - 3)))
    return weights


def sample_forewing_length_mm(rng: random.Random) -> float:
    """Sample a forewing length in millimeters from the PDF prior."""
    return rng.uniform(*FOREWING_LENGTH_MM_RANGE)


def sample_source_weight(kind: str) -> float:
    """Return a sampling weight for a cutout source kind."""
    if kind == "morphology":
        return MORPHOLOGY_SOURCE_WEIGHT
    return REFERENCE_SOURCE_WEIGHT


def article_cue_probability(kind: str) -> float:
    """Return how often to add article-specific cues to a cutout."""
    if kind == "morphology":
        return MORPHOLOGY_ARTICLE_CUE_PROBABILITY
    return REFERENCE_ARTICLE_CUE_PROBABILITY


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


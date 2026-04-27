# Trap-like backgrounds

This folder contains trap-like background images used for synthetic dataset generation.

The backgrounds should represent visual conditions expected in pheromone trap images, such as glue texture, debris, lighting variation, and perspective effects.

The procedural generator writes its outputs to `data/backgrounds/generated/`:

```bash
uv run python src/seed_moth_poc/data_prep/background_generator.py \
  --output-dir data/backgrounds/generated \
  --count 24
```

The manifest in the same directory records the parameters used for each generated image.

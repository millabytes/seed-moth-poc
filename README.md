# seed-moth-poc

Initial proof of concept for detecting and counting avocado seed moths
(`Stenoma catenifer`) in pheromone-trap images.

This repository contains the implementation scaffold, reproducible Python environment,
and technical documentation for the interview assignment.


## Setup

Install `uv` globally (recommended):

```bash
python3 -m pip install --user pipx
python3 -m pipx install uv
```

From the project root, sync the environment:

```bash
uv sync
```

## Formatting & pre-commit

Install the git hook:

```bash
uv run pre-commit install
```

Run formatting and linting manually:

```bash
uv run black .
uv run ruff check src tests
uv run pre-commit run --all-files
```

## Notes

- `.venv/` is ignored by Git
- `uv.lock` is committed for reproducible installs
- `pyproject.toml` is the project metadata and build config
- The implementation is intended as a proof of concept, not a production-ready system.
- Third-party public data should be tracked through manifests before being downloaded or
  committed.

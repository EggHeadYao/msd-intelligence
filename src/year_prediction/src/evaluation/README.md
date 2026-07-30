# Year Prediction Evaluation

This package provides shared evaluation utilities and model-specific evaluation entry points.

- `metrics.py`: computes clipped and raw errors, coverage rates, median error, and decade-level quality metrics.
- `compare_models.py`: compares compatible evaluation reports produced by different models.
- `ridge/`: evaluates and validates the custom Ridge model; see `ridge/README.md`.
- `lightgbm/`: contains the LightGBM evaluation boundary; see `lightgbm/README.md`.

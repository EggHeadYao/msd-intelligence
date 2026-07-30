import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[2] / "training"
sys.path.insert(0, str(TRAINING_DIR))

from kernel_sgd.evaluator import evaluate  # noqa: E402

__all__ = ("evaluate",)

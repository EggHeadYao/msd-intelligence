import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel_sgd_evaluation.cli import main  # noqa: E402


if __name__ == "__main__":
    main("t90_rff")

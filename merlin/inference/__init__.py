"""Pure-Python inference interfaces for the MERLIN recommender."""

from .pipeline import MerlinPipeline
from .ranker import LogisticRanker
from .types import Candidate, Recommendation

__all__ = ["Candidate", "LogisticRanker", "MerlinPipeline", "Recommendation"]

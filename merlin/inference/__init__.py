"""Pure-Python inference interfaces for the MERLIN recommender."""

from .retrieval.faiss import FaissTrackIndex
from .ranking.features import FEATURE_ORDER, FEATURE_SCHEMA, RankerFeatureComputer
from .ranking.model import LogisticRanker
from .recall import RecallPipeline, validate_recall_pipeline, write_recall_report
from .recall.factory import load_recall_pipeline
from .runtime.factory import load_inference_pipeline
from .runtime.pipeline import ColdAudioAudit, ColdAudioPipeline, MerlinPipeline
from .runtime.validation import validate_pipeline, write_validation_report
from .types import Candidate, Recommendation

__all__ = [
    "Candidate", "ColdAudioAudit", "ColdAudioPipeline", "FEATURE_ORDER",
    "FEATURE_SCHEMA", "FaissTrackIndex", "LogisticRanker", "MerlinPipeline",
    "RankerFeatureComputer",
    "RecallPipeline", "Recommendation", "load_inference_pipeline", "load_recall_pipeline",
    "validate_pipeline", "validate_recall_pipeline", "write_recall_report", "write_validation_report",
]

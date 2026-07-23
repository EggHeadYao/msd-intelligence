"""Pure-Python inference interfaces for the MERLIN recommender."""

from .faiss_index import FaissTrackIndex
from .feature_schema import RANKER_V2_FEATURES, RANKER_V2_SCHEMA_VERSION
from .features_v2 import RankerV2FeatureComputer
from .ranking.model import LogisticRanker
from .recall import RecallPipeline, validate_recall_pipeline, write_recall_report
from .recall_factory import load_recall_pipeline
from .runtime.cold_audio import ColdAudioAudit, ColdAudioPipeline
from .runtime.factory import load_inference_pipeline
from .runtime.pipeline import MerlinPipeline
from .runtime.validation import validate_pipeline, write_validation_report
from .types import Candidate, Recommendation

InferenceFeatureComputer = RankerV2FeatureComputer

__all__ = [
    "Candidate", "ColdAudioAudit", "ColdAudioPipeline", "FaissTrackIndex",
    "InferenceFeatureComputer", "LogisticRanker",
    "MerlinPipeline", "RANKER_V2_FEATURES", "RANKER_V2_SCHEMA_VERSION", "RankerV2FeatureComputer",
    "RecallPipeline", "Recommendation", "load_inference_pipeline", "load_recall_pipeline",
    "validate_pipeline", "validate_recall_pipeline", "write_recall_report", "write_validation_report",
]

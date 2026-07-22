"""Pure-Python inference interfaces for the MERLIN recommender."""

from .faiss_index import FaissTrackIndex
from .cold_audio import ColdAudioAudit, ColdAudioPipeline
from .feature_schema import RANKER_V2_FEATURES, RANKER_V2_SCHEMA_VERSION
from .features_v2 import RankerV2FeatureComputer
from .factory import load_inference_pipeline
from .pipeline import MerlinPipeline
from .recall import RecallPipeline, validate_recall_pipeline, write_recall_report
from .recall_factory import load_recall_pipeline
from .ranker import LogisticRanker
from .types import Candidate, Recommendation
from .validation import validate_pipeline, write_validation_report

InferenceFeatureComputer = RankerV2FeatureComputer

__all__ = [
    "Candidate", "ColdAudioAudit", "ColdAudioPipeline", "FaissTrackIndex",
    "InferenceFeatureComputer", "LogisticRanker",
    "MerlinPipeline", "RANKER_V2_FEATURES", "RANKER_V2_SCHEMA_VERSION", "RankerV2FeatureComputer",
    "RecallPipeline", "Recommendation", "load_inference_pipeline", "load_recall_pipeline",
    "validate_pipeline", "validate_recall_pipeline", "write_recall_report", "write_validation_report",
]

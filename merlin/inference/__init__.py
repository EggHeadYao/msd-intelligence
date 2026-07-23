"""Pure-Python inference interfaces for the MERLIN recommender."""

import sys as _sys

from .faiss_index import FaissTrackIndex
from .feature_schema import RANKER_V2_FEATURES, RANKER_V2_SCHEMA_VERSION
from .features_v2 import RankerV2FeatureComputer
from .ranking import artifacts as _ranker_artifacts
from .ranking import features as _ranker_features
from .ranking import lineage as _ranker_lineage
from .ranking import model as _ranker
from .ranking import selection as _ranker_selection
from .ranking.model import LogisticRanker
from .recall import RecallPipeline, validate_recall_pipeline, write_recall_report
from .recall_factory import load_recall_pipeline
from .runtime import artifacts as _artifacts
from .runtime import cold_audio as _cold_audio
from .runtime import factory as _factory
from .runtime import pipeline as _pipeline
from .runtime import validation as _validation
from .runtime.cold_audio import ColdAudioAudit, ColdAudioPipeline
from .runtime.factory import load_inference_pipeline
from .runtime.pipeline import MerlinPipeline
from .runtime.validation import validate_pipeline, write_validation_report
from .training import pairs as _training_pairs
from .training import validation_groups as _validation_groups
from .training import weak_labels as _weak_labels
from .types import Candidate, Recommendation


# Preserve imports used by downstream callers while keeping implementations grouped.
for _legacy_name, _module in {
    "artifacts": _artifacts,
    "cold_audio": _cold_audio,
    "factory": _factory,
    "pipeline": _pipeline,
    "ranker": _ranker,
    "ranker_artifacts": _ranker_artifacts,
    "ranker_features": _ranker_features,
    "ranker_lineage": _ranker_lineage,
    "ranker_selection": _ranker_selection,
    "training_pairs": _training_pairs,
    "validation": _validation,
    "validation_groups": _validation_groups,
    "weak_labels": _weak_labels,
}.items():
    _sys.modules.setdefault(f"{__name__}.{_legacy_name}", _module)

del _legacy_name, _module

InferenceFeatureComputer = RankerV2FeatureComputer

__all__ = [
    "Candidate", "ColdAudioAudit", "ColdAudioPipeline", "FaissTrackIndex",
    "InferenceFeatureComputer", "LogisticRanker",
    "MerlinPipeline", "RANKER_V2_FEATURES", "RANKER_V2_SCHEMA_VERSION", "RankerV2FeatureComputer",
    "RecallPipeline", "Recommendation", "load_inference_pipeline", "load_recall_pipeline",
    "validate_pipeline", "validate_recall_pipeline", "write_recall_report", "write_validation_report",
]

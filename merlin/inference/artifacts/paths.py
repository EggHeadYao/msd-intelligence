"""Canonical MERLIN artifact locations used by pure-Python inference."""

from dataclasses import dataclass
from pathlib import Path


PARQUETS_ROOT = Path("parquets_new")
PREPARED_ROOT = PARQUETS_ROOT / "prepared"
PREPARED_MANIFEST_PATH = PREPARED_ROOT / "prepared_manifest.json"
MERLIN_ROOT = PARQUETS_ROOT / "merlin"
AUDIO_ROOT = MERLIN_ROOT / "audio"
GRAPH_ROOT = MERLIN_ROOT / "graph"
RANKER_ROOT = MERLIN_ROOT / "ranker"
TUNING_ROOT = RANKER_ROOT / "tuning"
SET_C_EVALUATION_ROOT = RANKER_ROOT / "set_c_evaluation"
NO_HARD_NEG_ROOT = RANKER_ROOT / "no_hard_neg_model"
NO_HARD_NEG_PAIRS_PATH = RANKER_ROOT / "no_hard_neg_training_pairs.parquet"
NO_HARD_NEG_PAIRS_MANIFEST_PATH = RANKER_ROOT / "no_hard_neg_training_pairs_manifest.json"
NO_HARD_NEG_RAW_FEATURES_PATH = RANKER_ROOT / "no_hard_neg_raw_pair_features.parquet"
NO_HARD_NEG_RAW_FEATURES_MANIFEST_PATH = RANKER_ROOT / "no_hard_neg_raw_pair_features_manifest.json"

AUDIO_INDEX_PATH = AUDIO_ROOT / "index_audio.faiss"
AUDIO_MAPPING_PATH = AUDIO_ROOT / "index_audio_track_ids.parquet"
AUDIO_MANIFEST_PATH = AUDIO_ROOT / "index_audio_manifest.json"
AUDIO_ENCODER_METADATA_PATH = AUDIO_ROOT / "audio_encoder_metadata.json"
AUDIO_C1_MANIFEST_PATH = AUDIO_ROOT / "c1_manifest.json"
AUDIO_SCALER_MODEL_PATH = AUDIO_ROOT / "scaler_model"
RAW_AUDIO_FEATURES_PATH = PREPARED_ROOT / "song_audio_features_raw.parquet"

GRAPH_INDEX_PATH = GRAPH_ROOT / "index_graph.faiss"
GRAPH_MAPPING_PATH = GRAPH_ROOT / "index_graph_track_ids.parquet"
GRAPH_MANIFEST_PATH = GRAPH_ROOT / "index_graph_manifest.json"
GRAPH_ENCODER_METADATA_PATH = GRAPH_ROOT / "graph_encoder_metadata.json"

SONGS_METADATA_PATH = PREPARED_ROOT / "songs_metadata.parquet"
GRAPH_EDGES_PATH = PREPARED_ROOT / "graph_edges.parquet"

RANKER_SCHEMA_PATH = RANKER_ROOT / "ranker_feature_schema.json"
RANKER_SCALER_PATH = RANKER_ROOT / "ranker_scaler.json"
RANKER_COEFFICIENTS_PATH = RANKER_ROOT / "ranker_coefficients.json"
CANDIDATE_POLICY_PATH = RANKER_ROOT / "candidate_policy_manifest.json"
RANKER_TRAINING_MANIFEST_PATH = RANKER_ROOT / "training_manifest.json"
TAG_IDF_PATH = RANKER_ROOT / "tag_idf.json"
CANDIDATE_POOL_PATH = RANKER_ROOT / "candidate_pool.parquet"
CANDIDATE_POOL_MANIFEST_PATH = RANKER_ROOT / "candidate_pool_manifest.json"
SET_B_CANDIDATE_POOL_PATH = RANKER_ROOT / "set_b_candidate_pool.parquet"
SET_B_CANDIDATE_POOL_MANIFEST_PATH = RANKER_ROOT / "set_b_candidate_pool_manifest.json"
SPLIT_ASSIGNMENTS_PATH = RANKER_ROOT / "split_assignments.parquet"
SPLIT_MANIFEST_PATH = RANKER_ROOT / "split_manifest.json"
WEAK_LABEL_THRESHOLDS_PATH = RANKER_ROOT / "weak_label_thresholds.json"
WEAK_POSITIVES_PATH = RANKER_ROOT / "weak_positives.parquet"
WEAK_POSITIVES_MANIFEST_PATH = RANKER_ROOT / "weak_positives_manifest.json"
CANDIDATE_AUDIT_PATH = RANKER_ROOT / "candidate_audit.json"
TUNING_TRAINING_PAIRS_PATH = TUNING_ROOT / "training_pairs.parquet"
TUNING_TRAINING_PAIRS_MANIFEST_PATH = TUNING_ROOT / "training_pairs_manifest.json"
TUNING_RAW_PAIR_FEATURES_PATH = TUNING_ROOT / "raw_pair_features.parquet"
TUNING_RAW_PAIR_FEATURES_MANIFEST_PATH = TUNING_ROOT / "raw_pair_features_manifest.json"
TRAINING_PAIRS_PATH = RANKER_ROOT / "training_pairs.parquet"
TRAINING_PAIRS_MANIFEST_PATH = RANKER_ROOT / "training_pairs_manifest.json"
RAW_PAIR_FEATURES_PATH = RANKER_ROOT / "raw_pair_features.parquet"
RAW_PAIR_FEATURES_MANIFEST_PATH = RANKER_ROOT / "raw_pair_features_manifest.json"
VALIDATION_GROUP_THRESHOLDS_PATH = RANKER_ROOT / "validation_group_thresholds.json"
VALIDATION_GROUP_POSITIVES_PATH = RANKER_ROOT / "validation_group_positives.parquet"
VALIDATION_PAIRS_PATH = RANKER_ROOT / "validation_pairs.parquet"
VALIDATION_GROUPS_MANIFEST_PATH = RANKER_ROOT / "validation_groups_manifest.json"
VALIDATION_RAW_FEATURES_PATH = RANKER_ROOT / "validation_raw_features.parquet"
VALIDATION_RAW_FEATURES_MANIFEST_PATH = RANKER_ROOT / "validation_raw_features_manifest.json"
RECALL_VALIDATION_PATH = MERLIN_ROOT / "recall_validation_report.json"
INFERENCE_VALIDATION_PATH = MERLIN_ROOT / "inference_validation_report.json"
SET_C_PROTOCOL_PATH = SET_C_EVALUATION_ROOT / "protocol.json"
SET_C_CANDIDATE_POOL_PATH = SET_C_EVALUATION_ROOT / "candidate_pool.parquet"
SET_C_CANDIDATE_POOL_MANIFEST_PATH = SET_C_EVALUATION_ROOT / "candidate_pool_manifest.json"
SET_C_POSITIVES_PATH = SET_C_EVALUATION_ROOT / "validation_group_positives.parquet"
SET_C_VALIDATION_PAIRS_PATH = SET_C_EVALUATION_ROOT / "validation_pairs.parquet"
SET_C_GROUPS_MANIFEST_PATH = SET_C_EVALUATION_ROOT / "validation_groups_manifest.json"
SET_C_RAW_FEATURES_PATH = SET_C_EVALUATION_ROOT / "raw_pair_features.parquet"
SET_C_RAW_FEATURES_MANIFEST_PATH = SET_C_EVALUATION_ROOT / "raw_pair_features_manifest.json"
SET_C_EVALUATION_REPORT_PATH = SET_C_EVALUATION_ROOT / "evaluation_report.json"
NO_HARD_NEG_TRAINING_MANIFEST_PATH = NO_HARD_NEG_ROOT / "training_manifest.json"
NO_HARD_NEG_SCHEMA_PATH = NO_HARD_NEG_ROOT / "ranker_feature_schema.json"
NO_HARD_NEG_SCALER_PATH = NO_HARD_NEG_ROOT / "ranker_scaler.json"
NO_HARD_NEG_COEFFICIENTS_PATH = NO_HARD_NEG_ROOT / "ranker_coefficients.json"


@dataclass(frozen=True, slots=True)
class InferenceArtifactPaths:
    audio_index: Path = AUDIO_INDEX_PATH
    audio_mapping: Path = AUDIO_MAPPING_PATH
    audio_manifest: Path = AUDIO_MANIFEST_PATH
    audio_encoder_metadata: Path = AUDIO_ENCODER_METADATA_PATH
    audio_c1_manifest: Path = AUDIO_C1_MANIFEST_PATH
    audio_scaler_model: Path = AUDIO_SCALER_MODEL_PATH
    raw_audio_features: Path = RAW_AUDIO_FEATURES_PATH
    prepared_manifest: Path = PREPARED_MANIFEST_PATH
    graph_index: Path = GRAPH_INDEX_PATH
    graph_mapping: Path = GRAPH_MAPPING_PATH
    graph_manifest: Path = GRAPH_MANIFEST_PATH
    graph_encoder_metadata: Path = GRAPH_ENCODER_METADATA_PATH
    songs_metadata: Path = SONGS_METADATA_PATH
    graph_edges: Path = GRAPH_EDGES_PATH
    ranker_schema: Path = RANKER_SCHEMA_PATH
    ranker_scaler: Path = RANKER_SCALER_PATH
    ranker_coefficients: Path = RANKER_COEFFICIENTS_PATH
    ranker_training_manifest: Path = RANKER_TRAINING_MANIFEST_PATH
    candidate_policy: Path = CANDIDATE_POLICY_PATH
    tag_idf: Path = TAG_IDF_PATH
    candidate_pool: Path = CANDIDATE_POOL_PATH
    candidate_pool_manifest: Path = CANDIDATE_POOL_MANIFEST_PATH
    set_b_candidate_pool: Path = SET_B_CANDIDATE_POOL_PATH
    set_b_candidate_pool_manifest: Path = SET_B_CANDIDATE_POOL_MANIFEST_PATH
    split_assignments: Path = SPLIT_ASSIGNMENTS_PATH
    split_manifest: Path = SPLIT_MANIFEST_PATH
    weak_label_thresholds: Path = WEAK_LABEL_THRESHOLDS_PATH
    weak_positives: Path = WEAK_POSITIVES_PATH
    weak_positives_manifest: Path = WEAK_POSITIVES_MANIFEST_PATH
    candidate_audit: Path = CANDIDATE_AUDIT_PATH
    training_pairs: Path = TRAINING_PAIRS_PATH
    training_pairs_manifest: Path = TRAINING_PAIRS_MANIFEST_PATH
    raw_pair_features: Path = RAW_PAIR_FEATURES_PATH
    raw_pair_features_manifest: Path = RAW_PAIR_FEATURES_MANIFEST_PATH
    final_training_pairs: Path = FINAL_TRAINING_PAIRS_PATH
    final_training_pairs_manifest: Path = FINAL_TRAINING_PAIRS_MANIFEST_PATH
    final_raw_features: Path = FINAL_RAW_FEATURES_PATH
    final_raw_features_manifest: Path = FINAL_RAW_FEATURES_MANIFEST_PATH
    validation_group_thresholds: Path = VALIDATION_GROUP_THRESHOLDS_PATH
    validation_group_positives: Path = VALIDATION_GROUP_POSITIVES_PATH
    validation_pairs: Path = VALIDATION_PAIRS_PATH
    validation_groups_manifest: Path = VALIDATION_GROUPS_MANIFEST_PATH
    validation_raw_features: Path = VALIDATION_RAW_FEATURES_PATH
    validation_raw_features_manifest: Path = VALIDATION_RAW_FEATURES_MANIFEST_PATH
    recall_validation: Path = RECALL_VALIDATION_PATH
    inference_validation: Path = INFERENCE_VALIDATION_PATH

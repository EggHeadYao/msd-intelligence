# Audio Preprocessing

- `fill_segment_missing_values(df)`: Replaces segment aggregate placeholders for songs with `has_segments == 0` using means computed from songs with available segment arrays.
- `add_key_circular_features(df)`: Converts raw musical key into `key_sin` and `key_cos` so adjacent keys remain close on a circle.
- `add_time_signature_one_hot(df)`: Converts observed `time_signature` categories into one-hot numeric columns.
- `add_log_clipped_features(df)`: Clips `tempo`, `duration`, and `loudness` at approximate 1st/99th percentiles, then applies `log1p` to tempo and duration.
- `collect_scalar_medians_and_feature_ranges(df, feature_columns)`: Computes scalar fill medians and finite feature ranges in one Spark aggregation.
- `select_non_constant_features(ranges, columns)`: Removes constant features before StandardScaler and PCA.
- `preprocess_audio_features(df)`: Runs the full preprocessing sequence and returns the transformed DataFrame, selected feature columns, and metadata.
- `validate_frozen_preprocess_contract(feature_columns, metadata)`: Validates saved medians, clip bounds, category columns, dropped features, and the fixed near-zero threshold without running Spark transformations.
- `preprocess.apply_frozen_preprocess(...)`: Replays the saved training transform without recomputing medians, quantiles, or dropped features; used by L1-1 validation and C3 Set-B group construction.
- `metadata.clip_bounds`: Stores clipping thresholds for reproducibility.
- `metadata.segment_medians`: Stores segment fill values used for missing segment aggregates.
- `metadata.time_signature_values`: Stores the raw time signature categories used for one-hot encoding.
- `metadata.time_signature_columns`: Stores the generated one-hot column names.
- `metadata.dropped_features`: Stores features removed by the zero-variance filter.

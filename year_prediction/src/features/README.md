# Year Prediction Feature Contract

This module fits one reusable audio preprocessing contract on the training artists only. The frozen contract is then applied unchanged to training, validation, and test rows.

## Build

```bash
spark-submit --master 'local[*]' --driver-memory 4g \
  p1team02/year_prediction/src/features/build_features.py
```

## Validate

```bash
spark-submit --master 'local[*]' --driver-memory 4g \
  p1team02/year_prediction/src/features/validate_features.py
```

The validator independently refits the preprocessing state from the training split and reconstructs both output views before comparing their complete contents.

## Outputs

- `engineered_features.parquet`: identifiers, raw year, normalized year, cleaned categorical values, unscaled engineered predictor columns, and `split`. This is the auditable input view for table models such as LightGBM.
- `linear_vectors.parquet`: identifiers, raw year, normalized year, a fixed-order standardized `array<double>` named `features`, and `split`. This is the shared input for Ridge, PCA, RFF, and Huber SGD.
- `preprocessing_metadata.json`: source fingerprint, fit scope, feature order, category mapping, clipping bounds, imputation values, variance filtering, scaler statistics, schemas, and row counts.

## Contract

- Non-positive tempo is missing; tempo, duration, and loudness use training-only 1st/99th percentile clipping.
- Tempo and duration use `log1p`; loudness remains in clipped dB units.
- Missing segment aggregates use means fitted only from training rows with segment data, while `has_segments` preserves the missingness signal.
- Key uses circular sine/cosine encoding plus an unknown flag. Time signature uses training categories plus an unknown one-hot column.
- Constant and near-zero-variance candidates are removed using training statistics only.
- Retained columns are centered and scaled using training means and sample standard deviations.
- The target is `normalized_year = (year - 1922) / 89`.


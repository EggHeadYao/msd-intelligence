# Year Prediction Feature Contract

This module fits reusable audio preprocessing contracts on the training artists only. Each frozen contract is applied unchanged to training, validation, and test rows.

## Build

```bash
for contract in k0 k1 k2 k3; do
  spark-submit --master 'local[6]' --driver-memory 4g \
    p1team02/year_prediction/src/features/build_features.py \
    --key-contract "$contract"
done
```

## Validate

```bash
for contract in k0 k1 k2 k3; do
  spark-submit --master 'local[6]' --driver-memory 4g \
    p1team02/year_prediction/src/features/validate_features.py \
    --features "parquets/year_prediction/features/$contract"
done
```

The validator independently refits the preprocessing state from the training split and reconstructs both output views before comparing their complete contents.

## Outputs

- `engineered_features.parquet`: identifiers, raw year, normalized year, cleaned categorical values, unscaled engineered predictor columns, and `split`. This is the auditable input view for table models such as LightGBM.
- `linear_vectors.parquet`: identifiers, raw year, normalized year, a fixed-order standardized `array<double>` named `features`, and `split`. This is the shared input for Ridge, PCA, RFF, and Huber SGD.
- `preprocessing_metadata.json`: source fingerprint, key contract, fit scope, feature order, category mapping, clipping bounds, imputation values, variance filtering, scaler statistics, schemas, and row counts.

## Contract

- K0 produces 110 retained dimensions, K1 produces 122, and K2/K3 each produce 112 on the current dataset snapshot.
- Non-positive tempo is missing; tempo, duration, and loudness use training-only 1st/99th percentile clipping.
- Tempo and duration use `log1p`; loudness remains in clipped dB units.
- Missing segment aggregates use means fitted only from training rows with segment data, while `has_segments` preserves the missingness signal.
- K0 removes key from the model, K1 uses one-hot encoding, K2 uses the chromatic circle, and K3 uses the circle of fifths. All applicable encodings preserve an unknown-key indicator.
- Constant and near-zero-variance candidates are removed using training statistics only.
- Retained columns are centered and scaled using training means and sample standard deviations.
- The target is `normalized_year = (year - 1922) / 89`.

# Ranker commands

These entry points orchestrate supervised C3 data preparation, model training,
and final evaluation.

## Entry points

- `build_split.py` publishes deterministic split assignments and manifest.
- `build_weak_labels.py` fits or applies frozen weak-label thresholds.
- `build_training_pairs.py` constructs Set-A tuning data or streams the
  A+B+Remaining retrain pairs and raw features.
- `build_validation_groups.py` builds frozen Set-B or Set-C validation groups.
- `export_ranker_features.py` exports training or validation raw features.
- `train_ranker.py` runs Spark LR training, Set-B selection, and frozen
  retraining.
- `freeze_set_c_protocol.py` binds all inputs before Set C is opened.
- `evaluate_set_c.py` scores Full/baseline/ablation rankings and publishes the
  evaluation report.

## Canonical order

```text
split
  -> weak labels
  -> Set-A candidates
  -> Set-A tuning pairs/features
  -> Set-B candidates and validation groups/features
  -> tuning model and selected regularization
  -> streamed A+B+Remaining pairs/features
  -> Full and no-hard-negative models
  -> frozen Set-C protocol
  -> Set-C candidates/groups/features
  -> evaluation report
```

Set-A tuning datasets live under `ranker/tuning/`. Root
`training_pairs.parquet` and `raw_pair_features.parquet` are reserved for the
canonical retrain datasets. The manifest stage remains `final_retrain` because
it identifies the frozen protocol; published filenames do not use a `final_`
prefix.

High-volume jobs support explicit scratch roots and minimum-free-space guards.
The streamed retrain command is resumable only when every input, output, and
behavioral option matches the checkpoint contract.

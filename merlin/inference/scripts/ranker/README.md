# Ranker commands

These are the canonical C3 commands after the current workflow redesign. Set A
builds tuning data, Set B selects and confirms the guarded LR/C1 blend, and the
retrain dataset streams Set A + Set B + Set C + Remaining. Set C is reusable
development data, not an unbiased holdout.

## Recommended pipeline entry

Run the complete workflow with one command:

```bash
./merlin/inference/scripts/run_c3_pipeline.sh
```

The script skips complete outputs and lets the streamed retrain step resume
from its checkpoint. It never deletes or overwrites artifacts automatically.
Limit a run to a section with `--from` and `--to`, for example:

```bash
./merlin/inference/scripts/run_c3_pipeline.sh --to tune-model
./merlin/inference/scripts/run_c3_pipeline.sh --from retrain-data --to ablation-model
./merlin/inference/scripts/run_c3_pipeline.sh --from development-protocol
```

Use `--list-steps` to show valid boundaries and `--dry-run` to print commands.
The individual commands below remain the reference for manual recovery and
debugging.

## Environment

Run commands from the repository root. The Spark module wrapper is deliberately
kept outside the repository; it contains only
`from merlin.inference.scripts.ranker.train_ranker import main` followed by the
usual `if __name__ == "__main__": main()` call.

```bash
export MERLIN_ROOT=/home/zjk/p1team02
export MERLIN_PYTHON=/home/zjk/.venvs/merlin-faiss/bin/python
export MERLIN_SPARK_ENTRY=/home/zjk/merlin_local_tests/c3/run_train_ranker_module.py
export MERLIN_RANKER_ROOT="$MERLIN_ROOT/parquets_new/merlin/ranker"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$MERLIN_ROOT"
export OPENBLAS_CORETYPE=generic
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
cd "$MERLIN_ROOT"
```

Before a Spark job, also export:

```bash
export PYSPARK_PYTHON="$MERLIN_PYTHON"
export PYSPARK_DRIVER_PYTHON="$MERLIN_PYTHON"
export SPARK_LOCAL_IP=127.0.0.1
export JAVA_TOOL_OPTIONS='-XX:UseAVX=0 -XX:UseSSE=2 -XX:-TieredCompilation'
```

Do not remove an active `.c3-scratch` directory. Set `--min-free-gb` according
to available disk; lowering it to zero disables only the reserve, not the
projected-work check.

## 1. Split and weak labels

Rebuild the split whenever its artifact version or split policy changes. Before
exporting candidates, also ensure the recall contract is current; rebuild it
with the command in [`../recall/README.md`](../recall/README.md) only when the
candidate-policy or Tag-IDF contract changed.

```bash
"$MERLIN_PYTHON" -m merlin.inference.scripts.ranker.build_split \
  --songs-metadata parquets_new/prepared/songs_metadata.parquet \
  --assignments parquets_new/merlin/ranker/split_assignments.parquet \
  --manifest parquets_new/merlin/ranker/split_manifest.json

"$MERLIN_PYTHON" -m merlin.inference.scripts.ranker.build_weak_labels \
  --split-assignments parquets_new/merlin/ranker/split_assignments.parquet \
  --split-manifest parquets_new/merlin/ranker/split_manifest.json \
  --query-splits set_a \
  --min-free-gb 8
```

## 2. Set-A tuning data

First export the Set-A pool with the recall command, then build pairs and raw
features:

```bash
"$MERLIN_PYTHON" -m merlin.inference.scripts.ranker.build_training_pairs \
  --candidate-pool parquets_new/merlin/ranker/candidate_pool.parquet \
  --candidate-pool-manifest parquets_new/merlin/ranker/candidate_pool_manifest.json \
  --weak-positives parquets_new/merlin/ranker/weak_positives.parquet \
  --weak-positives-manifest parquets_new/merlin/ranker/weak_positives_manifest.json \
  --thresholds parquets_new/merlin/ranker/weak_label_thresholds.json \
  --split-assignments parquets_new/merlin/ranker/split_assignments.parquet \
  --stage tuning \
  --scope formal \
  --output parquets_new/merlin/ranker/tuning/training_pairs.parquet \
  --manifest parquets_new/merlin/ranker/tuning/training_pairs_manifest.json

"$MERLIN_PYTHON" -m merlin.inference.scripts.ranker.export_ranker_features \
  --pair-kind training \
  --pairs parquets_new/merlin/ranker/tuning/training_pairs.parquet \
  --pairs-manifest parquets_new/merlin/ranker/tuning/training_pairs_manifest.json \
  --output parquets_new/merlin/ranker/tuning/raw_pair_features.parquet \
  --manifest parquets_new/merlin/ranker/tuning/raw_pair_features_manifest.json \
  --stage tuning \
  --scope formal \
  --min-free-gb 8
```

For a non-canonical proxy run, add `--limit-queries N` to pair construction and
provide explicit output and manifest paths outside the formal artifact tree.
The limited pair manifest is marked `scope=smoke`; its feature export and model
training must therefore also use `--scope smoke`.

## 3. Set-B validation data

Export the Set-B pool with the recall command, then build the three frozen query
groups with Spark:

```bash
/opt/spark/bin/spark-submit \
  --master 'local[6]' \
  --driver-memory 5g \
  --conf spark.eventLog.enabled=false \
  --conf spark.ui.enabled=false \
  --conf spark.sql.shuffle.partitions=64 \
  --conf spark.hadoop.fs.defaultFS=file:/// \
  --conf spark.driver.bindAddress=127.0.0.1 \
  "$MERLIN_ROOT/merlin/inference/scripts/ranker/build_validation_groups.py" \
  --candidate-pool "$MERLIN_RANKER_ROOT/set_b_candidate_pool.parquet" \
  --candidate-pool-manifest "$MERLIN_RANKER_ROOT/set_b_candidate_pool_manifest.json" \
  --apply-split set_b \
  --scope formal \
  --audio-pair-engine numpy \
  --audio-block-size 256 \
  --shuffle-partitions 64 \
  --scratch-root "$MERLIN_RANKER_ROOT/.c3-scratch" \
  --min-free-gb 4

"$MERLIN_PYTHON" -m merlin.inference.scripts.ranker.export_ranker_features \
  --pair-kind validation \
  --pairs parquets_new/merlin/ranker/validation_pairs.parquet \
  --pairs-manifest parquets_new/merlin/ranker/validation_groups_manifest.json \
  --validation-positives parquets_new/merlin/ranker/validation_group_positives.parquet \
  --validation-thresholds parquets_new/merlin/ranker/validation_group_thresholds.json \
  --output parquets_new/merlin/ranker/validation_raw_features.parquet \
  --manifest parquets_new/merlin/ranker/validation_raw_features_manifest.json \
  --stage tuning \
  --scope formal \
  --min-free-gb 8
```

## 4. Select the frozen model configuration

This trains the Set-A LR grid and uses the Set-B tune/confirm folds to select
regularization, Audio quota, and relation-evidence gate:

```bash
/opt/spark/bin/spark-submit \
  --master 'local[4]' \
  --driver-memory 5500m \
  --conf spark.eventLog.enabled=false \
  --conf spark.ui.enabled=false \
  --conf spark.sql.shuffle.partitions=64 \
  --conf spark.hadoop.fs.defaultFS=file:/// \
  --conf spark.driver.bindAddress=127.0.0.1 \
  --conf spark.memory.fraction=0.40 \
  "$MERLIN_SPARK_ENTRY" \
  --train-features "$MERLIN_RANKER_ROOT/tuning/raw_pair_features.parquet" \
  --train-features-manifest "$MERLIN_RANKER_ROOT/tuning/raw_pair_features_manifest.json" \
  --validation-features "$MERLIN_RANKER_ROOT/validation_raw_features.parquet" \
  --validation-features-manifest "$MERLIN_RANKER_ROOT/validation_raw_features_manifest.json" \
  --output "$MERLIN_RANKER_ROOT/tuning_model" \
  --stage tuning \
  --scope formal \
  --scratch-root "$MERLIN_RANKER_ROOT/.c3-scratch" \
  --min-free-gb 4 \
  --max-block-size-mb 8 \
  --parent audio_index_manifest="$MERLIN_ROOT/parquets_new/merlin/audio/index_audio_manifest.json" \
  --parent graph_index_manifest="$MERLIN_ROOT/parquets_new/merlin/graph/index_graph_manifest.json" \
  --parent candidate_policy_manifest="$MERLIN_RANKER_ROOT/candidate_policy_manifest.json" \
  --parent tag_idf="$MERLIN_RANKER_ROOT/tag_idf.json" \
  --parent validation_features_manifest="$MERLIN_RANKER_ROOT/validation_raw_features_manifest.json"
```

Read `selected_reg_param` from `tuning_model/training_manifest.json` and use
that exact value as `MERLIN_REG` below. Do not select it from Set C.

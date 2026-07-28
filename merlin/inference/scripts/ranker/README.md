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

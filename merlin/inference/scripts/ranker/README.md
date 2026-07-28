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

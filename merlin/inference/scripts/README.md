# Supported commands

This package contains the supported command-line entry points for C3. Scripts
parse arguments, validate artifact lineage, configure external engines, call
the reusable inference packages, and publish outputs. Core algorithms should
not be duplicated here.

`run_c3_pipeline.sh` is the resumable top-level entry point for the complete
formal workflow. It orchestrates the commands below without duplicating their
Python implementations.

## Command groups

- [`recall/`](recall/README.md) builds, exports, validates, and audits Stage-1
  candidate artifacts.
- [`ranker/`](ranker/README.md) builds supervised datasets, trains models, and
  runs reproducible Set-C development evaluation.
- [`support/`](support/README.md) contains shared operational helpers such as
  Spark scratch-space preparation.
- `run_c3_pipeline.sh` runs the canonical sequence from split construction
  through Set-C development evaluation.
- `validate_inference.py` validates a fully assembled runtime pipeline.

Formal commands default to canonical paths from `InferenceArtifactPaths` but
also accept explicit paths. Smoke and diagnostic outputs should be written
outside the formal artifact directory.

Spark commands on this host must disable event logging unless a writable event
log directory is configured. See the repository `AGENTS.md` for the local
Spark invocation constraints.

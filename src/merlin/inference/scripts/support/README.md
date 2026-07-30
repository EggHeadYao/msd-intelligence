# Command support

This package contains small operational helpers shared by inference command entry points.

`scratch.py` estimates ranker/Spark temporary storage, validates filesystem free-space reserves, creates a job-local scratch directory, and configures Spark to place shuffle data on the selected volume.

It does not own training algorithms or published artifacts. Callers remain responsible for cleaning only their job-local scratch directory after Spark has stopped; an active Spark block-manager directory must never be removed.

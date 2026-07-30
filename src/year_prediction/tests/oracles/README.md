# Mathematical Oracles

Each subdirectory defines an independent mathematical reference for one training objective. Oracle code must not import the corresponding production objective or optimizer implementation.

- `ridge/`: deterministic Ridge loss, gradient, finite-difference, update, and Spark partition checks.

Future objectives such as Huber should receive a separate oracle subdirectory before their distributed implementation is optimized.

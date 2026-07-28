# Development evaluation

This package binds a reproducible Set-C development contract and computes
ranking statistics after tuning and retraining are complete.

## Modules

- `protocol.py` binds Set C to the current splits, indexes, recall policy,
  preprocessing, Full model, ablation model, metric rules, and random seeds.
- `metrics.py` computes retrieval metrics, query-level ranking scores, macro
  aggregates, random baselines, and paired bootstrap confidence intervals.

Set C is known development data: it may be evaluated repeatedly, may guide
later model revisions, and enters final retraining. Its report is suitable for
comparing configurations on the current data, but not for an unbiased
generalization or production-accuracy claim. Every Set-C-producing stage binds
the development protocol and rejects a lineage mismatch.

The development report contains candidate-layer diagnostics, stratified nDCG/Recall,
paired query and artist-cluster comparisons, ablations, and robustness slices.
Canonical outputs live below `ranker/development_evaluation/`.

See [`scripts/ranker`](../scripts/ranker/README.md) for the prepare, export, and
evaluation command order.

# Set-C evaluation

This package freezes the final evaluation contract and computes ranking
statistics after tuning and retraining are complete.

## Modules

- `protocol.py` binds Set C to the frozen splits, indexes, recall policy,
  preprocessing, Full model, ablation model, metric rules, and random seeds.
- `metrics.py` computes retrieval metrics, query-level ranking scores, macro
  aggregates, random baselines, and paired bootstrap confidence intervals.

Set C remains unopened while Set-A training, Set-B selection, retraining, and
ablation construction are in progress. Every Set-C-producing stage must accept
the frozen protocol and reject a lineage mismatch.

The final report contains candidate-layer diagnostics, stratified nDCG/Recall,
paired query and artist-cluster comparisons, ablations, and robustness slices.
Canonical outputs live below `ranker/set_c_evaluation/`.

See [`scripts/ranker`](../scripts/ranker/README.md) for the freeze, export, and
evaluation command order.

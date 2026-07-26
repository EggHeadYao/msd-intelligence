# Production runtime

Runtime modules assemble validated artifacts into the online recommendation
pipeline. They do not train models or construct C1/C2 indexes.

## Modules

- `factory.py` loads canonical artifacts, validates their contracts, and
  assembles recall, feature, and ranker components.
- `pipeline.py` implements `MerlinPipeline` and the separate
  `ColdAudioPipeline`.
- `validation.py` performs end-to-end structural and deterministic pipeline
  checks.

## Standard request path

```text
query track
  -> four-source recall (at most 1,000 unique candidates)
  -> canonical pair features
  -> frozen LR raw margins
  -> deterministic sort
  -> top 20 recommendations
```

Assembly fails closed when an index, mapping, policy, feature schema, scaler,
coefficient file, model manifest, or parent hash is inconsistent.

Cold Audio is intentionally separate. It accepts a C1-compatible 128D query
embedding and ranks Audio neighbors directly; it never invokes Graph, BFS,
Tag, or LR.

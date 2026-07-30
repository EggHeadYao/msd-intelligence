# Production runtime

Runtime modules assemble validated artifacts into the online recommendation pipeline. They do not train models or construct C1/C2 indexes.

## Modules

- `factory.py` loads canonical artifacts, validates their contracts, and assembles recall, feature, and ranker components.
- `pipeline.py` implements `MerlinPipeline` and the separate `ColdAudioPipeline`.
- `validation.py` performs end-to-end structural and deterministic pipeline checks.

## Standard request path

```text
query track
  -> four-source primary recall + deterministic Tag/BFS backfill
  -> canonical pair features
  -> frozen LR raw margins and C1 order
  -> relation gate: C1 fallback or deterministic quota interleave
  -> top 20 recommendations
```

Assembly fails closed when an index, mapping, policy, feature schema, scaler, coefficient file, model manifest, or parent hash is inconsistent.

Cold Audio is intentionally separate. It accepts a C1-compatible 128D query embedding and ranks Audio neighbors directly; it never invokes Graph, BFS, Tag, or LR.

# Candidate retrieval

Retrieval modules implement one candidate source at a time. The recall package
combines these sources and applies the canonical quota policy.

## Modules

- `faiss.py` validates and loads the normalized Audio and Graph FAISS indexes,
  their row-to-track mappings, and their manifests.
- `retrievers.py` implements `VectorRetriever`, `BfsRetriever`, and
  `TagRetriever`, plus deterministic candidate merging.

## Candidate sources

- **Audio** searches the C1 128D normalized embedding index.
- **Graph** searches the C2 128D normalized embedding index.
- **BFS** traverses projected artist-similarity relationships.
- **Tag** retrieves tracks through sparse artist Tag similarity.

Each retriever emits `Candidate` values with source evidence, source score,
and source rank. `merge_candidates` unions duplicate track IDs without losing
that evidence and filters the query track.

FAISS construction belongs to C1/C2. This package only consumes published
indexes and rejects an index whose type, dimension, mapping, hash, or lineage
does not match its manifest.

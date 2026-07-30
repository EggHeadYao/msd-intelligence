# BFS

## BfsRules.java
- `initialVertex`: Builds the initial BFS state for one adjacency row.
- `isFrontier`: Checks whether a vertex should expand in the current BFS layer.
- `candidateDistance`: Computes the distance offered by a frontier vertex to its neighbors.
- `isBetterCandidate`: Selects the better candidate distance and deterministic parent.
- `nextStatus`: Updates the vertex status after one unweighted BFS iteration.
- `toArtistDistance`: Converts an internal BFS vertex to the final output record.

## BfsPaths.java
- `initial`: Returns the output path for the initial BFS state.
- `iteration`: Returns the output path for a numbered BFS iteration.
- `finalOutput`: Returns the final result output path.
- `child`: Joins a parent path and child name with one slash.

# Artist Distance Schemas

## edge.avsc
- `src`: source artist id from `similarity.target`.
- `dst`: destination artist id from `similarity.similar`.

## adjacency.avsc
- `id`: artist id for this vertex.
- `neighbors`: outgoing neighbor artist ids.
- `degree`: number of outgoing neighbors.

## bfs_vertex.avsc
- `id`: artist id for this BFS vertex.
- `neighbors`: outgoing neighbor artist ids.
- `degree`: number of outgoing neighbors.
- `distance`: shortest known distance from the input source artist, or `null`.
- `parent`: previous artist on the shortest path, or `null`.
- `status`: BFS state, one of `unvisited`, `frontier`, `visited`.

## artist_distance.avsc
- `id`: artist id in the final output.
- `distance`: shortest distance from the input source artist, or `null`.
- `parent`: previous artist on the shortest path, or `null`.

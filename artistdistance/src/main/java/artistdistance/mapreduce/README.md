# MapReduce BFS

- `BfsMessage`: Writable value used between the iteration mapper and reducer.
- `BfsInitMapper`: Converts adjacency rows into initial BFS vertices.
- `BfsIterationMapper`: Keeps each vertex and expands frontier vertices to candidate messages.
- `BfsIterationReducer`: Merges one vertex with candidate messages and counts new discoveries.
- `BfsFinalMapper`: Converts final BFS vertices into artist-distance output records.
- `MapReduceBfsRunner`: Runs init, iteration, and final MapReduce jobs.
- `avro/AvroMapReduceBfs`: CLI entry point for Avro input and output.

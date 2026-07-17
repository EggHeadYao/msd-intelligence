"""Stage-1 candidate merging and lightweight retriever adapters."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .interfaces import CandidateRetriever
from .types import Candidate


def merge_candidates(groups: Sequence[Sequence[Candidate]], query_track_id: str) -> list[Candidate]:
    """Union candidates by track ID while preserving all recall evidence."""
    merged: dict[str, dict[str, object]] = {}
    for group in groups:
        for candidate in group:
            if candidate.track_id == query_track_id:
                continue
            state = merged.setdefault(
                candidate.track_id,
                {"sources": set(), "scores": {}, "ranks": {}},
            )
            state["sources"].update(candidate.sources)  # type: ignore[union-attr]
            state["scores"].update(candidate.recall_scores)  # type: ignore[union-attr]
            state["ranks"].update(candidate.source_ranks)  # type: ignore[union-attr]
    return [
        Candidate(
            track_id=track_id,
            sources=frozenset(state["sources"]),  # type: ignore[arg-type]
            recall_scores=state["scores"],  # type: ignore[arg-type]
            source_ranks=state["ranks"],  # type: ignore[arg-type]
        )
        for track_id, state in merged.items()
    ]

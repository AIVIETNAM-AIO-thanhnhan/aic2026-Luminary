"""Reciprocal Rank Fusion across retrieval branches.

Four branches score frames on incomparable scales: cosine similarity from SigLIP,
BM25 from OCR and ASR, and a binary-ish object filter. Normalizing those into a
weighted sum needs a labelled set large enough to fit the weights, which we do not
have. RRF instead consumes only *ranks*:

.. math:: score(d) = \\sum_{b} w_b / (k + rank_b(d))

so a branch with a wildly different score scale cannot dominate, and a frame
ranked well by two weak branches beats one ranked well by a single branch. ``k=60``
is the value from Cormack et al. (2009); it flattens the tail so positions past
~60 contribute little.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field


@dataclass
class BranchResult:
    """One branch's ranked frames, best first."""

    name: str
    gids: Sequence[int]
    scores: Sequence[float] = field(default_factory=list)

    def as_ranking(self) -> list[int]:
        return list(self.gids)


@dataclass
class FusedHit:
    gid: int
    score: float
    #: Per-branch 1-based rank, for explaining *why* a frame surfaced in the UI.
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def branches(self) -> list[str]:
        return sorted(self.ranks)


def reciprocal_rank_fusion(
    branches: Iterable[BranchResult],
    weights: Mapping[str, float] | None = None,
    k: int = 60,
    top_n: int | None = None,
) -> list[FusedHit]:
    """Fuse ranked lists into a single ranking."""
    weights = weights or {}
    totals: dict[int, float] = defaultdict(float)
    ranks: dict[int, dict[str, int]] = defaultdict(dict)

    for branch in branches:
        weight = float(weights.get(branch.name, 1.0))
        if weight == 0.0:
            continue
        for position, gid in enumerate(branch.as_ranking(), start=1):
            gid = int(gid)
            totals[gid] += weight / (k + position)
            # Keep the best rank if a branch somehow lists a frame twice.
            previous = ranks[gid].get(branch.name)
            if previous is None or position < previous:
                ranks[gid][branch.name] = position

    fused = [FusedHit(gid=gid, score=score, ranks=ranks[gid]) for gid, score in totals.items()]
    # Tie-break on gid so the ordering is reproducible run to run.
    fused.sort(key=lambda hit: (-hit.score, hit.gid))
    return fused[:top_n] if top_n else fused

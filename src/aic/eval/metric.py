"""Official AIC 2026 preliminary-round scoring.

Implements section 2 of the rules exactly:

* R-Score per answer, with a different rule per query type.
* ``R@k = max`` R-Score over the first *k* answers, for k in {1, 5, 20, 50, 100}.
* ``Final Score = mean(R@k)`` over those five thresholds.

Two consequences drive :mod:`aic.submit.policy`:

1. Rank 1 alone contributes 1/5 of the final score, because it is the only answer
   counted by ``R@1``.
2. Ranks 6..100 only ever matter if they beat everything in the top 5, so they are
   nearly free and should be spent on coverage rather than repetition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

TOP_K_THRESHOLDS: tuple[int, ...] = (1, 5, 20, 50, 100)
MAX_ANSWERS = 100


@dataclass(frozen=True)
class KISAnswer:
    video_id: str
    frame_id: int


@dataclass(frozen=True)
class VQAAnswer:
    video_id: str
    frame_id: int
    answer: str


@dataclass(frozen=True)
class TRAKEAnswer:
    video_id: str
    frame_ids: tuple[int, ...]


@dataclass(frozen=True)
class KISGroundTruth:
    video_id: str
    start: int
    end: int

    def contains(self, frame_id: int) -> bool:
        return self.start <= frame_id <= self.end


@dataclass(frozen=True)
class VQAGroundTruth:
    video_id: str
    start: int
    end: int
    answers: frozenset[str]

    def contains(self, frame_id: int) -> bool:
        return self.start <= frame_id <= self.end


@dataclass(frozen=True)
class TRAKEGroundTruth:
    video_id: str
    #: One inclusive ``[s_j, e_j]`` window per event in the sequence.
    windows: tuple[tuple[int, int], ...]


def normalize_answer(text: str) -> str:
    """Casefold and collapse whitespace for semantic answer comparison.

    The organizers judge answers semantically ("5" and "Năm" both count), which no
    string comparison can fully reproduce. This performs the surface-level part;
    :class:`VQAGroundTruth` carries a set of accepted surface forms so the dev-set
    can enumerate the variants it cares about.
    """
    return " ".join(text.strip().casefold().split())


def kis_r_score(answer: KISAnswer, truth: KISGroundTruth) -> float:
    """1.0 when the video matches and the frame falls inside ``[s, e]``, else 0.0."""
    return float(answer.video_id == truth.video_id and truth.contains(answer.frame_id))


def vqa_r_score(answer: VQAAnswer, truth: VQAGroundTruth) -> float:
    """1.0 only when video, frame window, and answer text all match."""
    if answer.video_id != truth.video_id or not truth.contains(answer.frame_id):
        return 0.0
    accepted = {normalize_answer(a) for a in truth.answers}
    return float(normalize_answer(answer.answer) in accepted)


def trake_r_score(answer: TRAKEAnswer, truth: TRAKEGroundTruth) -> float:
    """Fraction of events landing inside their window; 0.0 if the video is wrong.

    Unlike KIS and VQA this awards partial credit, which is why the submission
    policy varies individual event frames across lower ranks instead of switching
    to a different video.
    """
    if answer.video_id != truth.video_id:
        return 0.0
    if not truth.windows:
        return 0.0
    hits = 0
    for index, (start, end) in enumerate(truth.windows):
        # A short submission simply misses the remaining events rather than erroring:
        # a partial answer is still worth submitting.
        if index < len(answer.frame_ids) and start <= answer.frame_ids[index] <= end:
            hits += 1
    return hits / len(truth.windows)


def r_at_k(scores: Sequence[float], k: int) -> float:
    """Best R-Score among the first *k* answers (0.0 when there are none)."""
    head = scores[:k]
    return max(head) if head else 0.0


def final_score(scores: Sequence[float], thresholds: Sequence[int] = TOP_K_THRESHOLDS) -> float:
    """Mean of ``R@k`` across the official thresholds."""
    if len(scores) > MAX_ANSWERS:
        raise ValueError(f"at most {MAX_ANSWERS} answers per query, got {len(scores)}")
    return sum(r_at_k(scores, k) for k in thresholds) / len(thresholds)


def score_query(answers: Sequence[object], truth: object) -> float:
    """Score a full ranked answer list against its ground truth."""
    if isinstance(truth, KISGroundTruth):
        scores = [kis_r_score(a, truth) for a in answers]  # type: ignore[arg-type]
    elif isinstance(truth, VQAGroundTruth):
        scores = [vqa_r_score(a, truth) for a in answers]  # type: ignore[arg-type]
    elif isinstance(truth, TRAKEGroundTruth):
        scores = [trake_r_score(a, truth) for a in answers]  # type: ignore[arg-type]
    else:  # pragma: no cover - developer error
        raise TypeError(f"unsupported ground truth type: {type(truth)!r}")
    return final_score(scores)

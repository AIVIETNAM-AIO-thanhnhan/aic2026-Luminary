"""Turn a ranked candidate list into the 100 rows actually submitted.

``Final Score = mean(R@1, R@5, R@20, R@50, R@100)`` has an exploitable shape, and
this module is where that is cashed in:

* **Rank 1 alone is 1/5 of the score**, since it is the only answer ``R@1`` sees.
* **Ranks 6-100 are nearly free.** They change nothing unless they beat everything
  above them, so spending them on coverage costs no expected score and can only add.
* **A wrong video scores 0** for every query type. Concentrating all 100 answers on
  one video is therefore an all-or-nothing bet.

The resulting policy: make ranks 1-5 hedge across *distinct videos*, then spend the
long tail on temporal coverage within the strongest candidates. TRAKE is handled
differently because it is the one type with partial credit — there, staying on the
best video and varying individual event frames is worth more than switching video.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from aic.eval.metric import MAX_ANSWERS


@dataclass
class Candidate:
    """A scored moment produced by search, before it becomes submission rows."""

    video_id: str
    frame_idx: int
    score: float
    #: Optional answer text for Q&A queries.
    answer: str | None = None
    #: Frames bounding the shot this came from, used to spread the tail.
    shot_start: int | None = None
    shot_end: int | None = None


@dataclass
class TrakeCandidate:
    video_id: str
    frame_ids: tuple[int, ...]
    score: float
    #: Per-event confidence, so jitter can target the least certain events first.
    per_event_scores: tuple[float, ...] = field(default_factory=tuple)


def _spread_frames(candidate: Candidate, count: int, spread: int) -> list[int]:
    """Frames around a candidate, to cover a ``[s, e]`` window we cannot see.

    The accepted window is unknown but contiguous, so offsetting around the chosen
    frame raises the chance one row lands inside it. Offsets are clipped to the
    shot's bounds where known: leaving the shot means leaving the event entirely.
    """
    if count <= 1:
        return [candidate.frame_idx]

    offsets = [0]
    step = 1
    while len(offsets) < count:
        offsets.extend([step * spread, -step * spread])
        step += 1
    offsets = offsets[:count]

    frames: list[int] = []
    for offset in offsets:
        frame = candidate.frame_idx + offset
        if candidate.shot_start is not None:
            frame = max(frame, candidate.shot_start)
        if candidate.shot_end is not None:
            frame = min(frame, candidate.shot_end)
        frame = max(0, frame)
        if frame not in frames:
            frames.append(frame)
    return frames


def build_kis_answers(
    candidates: list[Candidate],
    max_answers: int = MAX_ANSWERS,
    diversify_head: int = 5,
    frames_per_shot: int = 3,
    frame_spread: int = 8,
) -> list[tuple[str, int]]:
    """Build the ranked ``(video_id, frame_id)`` list for a Textual KIS query."""
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    if not ordered:
        return []

    rows: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    def full() -> bool:
        return len(rows) >= max_answers

    def push(video_id: str, frame_idx: int) -> None:
        """Append unless the row is a duplicate or the list is already full.

        Callers must test :func:`full` themselves: a rejected duplicate is not a
        reason to stop emitting the remaining offsets for this candidate.
        """
        key = (video_id, max(0, frame_idx))
        if key in seen or full():
            return
        seen.add(key)
        rows.append(key)

    # Head: best candidate from each distinct video, hedging the video choice.
    used_videos: set[str] = set()
    for candidate in ordered:
        if len(rows) >= diversify_head:
            break
        if candidate.video_id in used_videos:
            continue
        used_videos.add(candidate.video_id)
        push(candidate.video_id, candidate.frame_idx)

    # Tail: spread frames within each candidate shot, strongest candidates first.
    for candidate in ordered:
        if full():
            break
        for frame in _spread_frames(candidate, frames_per_shot, frame_spread):
            if full():
                break
            push(candidate.video_id, frame)

    return rows[:max_answers]


def build_vqa_answers(
    candidates: list[Candidate],
    fallback_answer: str = "",
    max_answers: int = MAX_ANSWERS,
    diversify_head: int = 5,
    frames_per_shot: int = 3,
    frame_spread: int = 8,
) -> list[tuple[str, int, str]]:
    """Same shape as KIS, carrying each candidate's answer text through.

    Q&A needs video, frame, *and* answer to be right, so a candidate whose answer is
    unknown inherits the most common answer among the confident candidates rather
    than being dropped: a row with a plausible answer can score, an absent row cannot.
    """
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    if not ordered:
        return []

    if not fallback_answer:
        answers = [c.answer for c in ordered[:10] if c.answer]
        fallback_answer = max(set(answers), key=answers.count) if answers else ""

    answer_by_video_frame = {(c.video_id, c.frame_idx): (c.answer or fallback_answer) for c in ordered}
    answer_by_video: dict[str, str] = {}
    for candidate in ordered:
        if candidate.answer:
            answer_by_video.setdefault(candidate.video_id, candidate.answer)

    pairs = build_kis_answers(
        ordered,
        max_answers=max_answers,
        diversify_head=diversify_head,
        frames_per_shot=frames_per_shot,
        frame_spread=frame_spread,
    )
    return [
        (
            video_id,
            frame_id,
            answer_by_video_frame.get((video_id, frame_id))
            or answer_by_video.get(video_id)
            or fallback_answer,
        )
        for video_id, frame_id in pairs
    ]


def build_trake_answers(
    candidates: list[TrakeCandidate],
    max_answers: int = MAX_ANSWERS,
    diversify_head: int = 5,
    jitter: tuple[int, ...] = (0, -4, 4, -8, 8, -12, 12),
) -> list[tuple[str, tuple[int, ...]]]:
    """Build the ranked ``(video_id, frame_ids...)`` list for a TRAKE query.

    Partial credit changes the calculus. Once the video is right, every additional
    event landing in its window adds ``1/N``, so the tail perturbs one event at a
    time around the best tuple — each variant keeps the events that were already
    correct and gives a different event another chance. The head still hedges across
    videos, because a wrong video is still a hard zero.
    """
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    if not ordered:
        return []

    rows: list[tuple[str, tuple[int, ...]]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()

    def full() -> bool:
        return len(rows) >= max_answers

    def push(video_id: str, frames: tuple[int, ...]) -> None:
        """Append unless duplicate, full, or out of chronological order.

        The submission format requires event frame ids to be strictly increasing.
        Jitter can break that when two events sit closer together than the offset,
        so such variants are dropped here — one lost variant is far cheaper than an
        invalid row, which would make the whole file unparseable.
        """
        clamped = tuple(max(0, f) for f in frames)
        if any(a >= b for a, b in itertools.pairwise(clamped)):
            return
        key = (video_id, clamped)
        if key in seen or full():
            return
        seen.add(key)
        rows.append(key)

    used_videos: set[str] = set()
    for candidate in ordered:
        if len(rows) >= diversify_head:
            break
        if candidate.video_id in used_videos:
            continue
        used_videos.add(candidate.video_id)
        push(candidate.video_id, candidate.frame_ids)

    # Perturb one event at a time, least-confident events first: those are the ones
    # most likely to be sitting outside their window.
    for candidate in ordered:
        if full():
            break
        n_events = len(candidate.frame_ids)
        if candidate.per_event_scores and len(candidate.per_event_scores) == n_events:
            event_order = sorted(range(n_events), key=lambda i: candidate.per_event_scores[i])
        else:
            event_order = list(range(n_events))

        for offset in jitter:
            if offset == 0:
                continue
            for event_index in event_order:
                if full():
                    break
                frames = list(candidate.frame_ids)
                frames[event_index] += offset
                push(candidate.video_id, tuple(frames))
            if full():
                break

    return rows[:max_answers]

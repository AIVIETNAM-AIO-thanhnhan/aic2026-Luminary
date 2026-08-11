"""TRAKE: retrieve the video, then align each event to a single frame.

The two stages mirror the task definition, and the second exists because of a hard
numeric fact: TRAKE's accepted window per event is *typically under 10 frames*,
while the organizers' I-frames sit seconds apart. No amount of keyframe retrieval
can hit a 10-frame target from a 50-frame-spaced grid, so the winning frame has to
be found by decoding the video densely around the coarse hit.

Stage 1 (:func:`rank_videos`) — score whole videos on how well they contain the
*sequence*, not the individual events. A video showing all four moments in the
wrong order is the wrong video, so ordering is enforced rather than just summed.

Stage 2 (:func:`align_events`) — decode every frame in a window around each coarse
hit, score them against that event's own description, and pick the best assignment
subject to ``t_1 < t_2 < ... < t_n``. Only runs on a handful of candidate videos,
which is what keeps it feasible on CPU.
"""

from __future__ import annotations

import io
import itertools
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from aic.query.expand import ExpandedQuery
from aic.submit.policy import TrakeCandidate


@dataclass
class EventHit:
    """Where stage 1 thinks one event happens, before dense refinement."""

    event_index: int
    frame_idx: int
    pts_time: float | None
    score: float


@dataclass
class VideoCandidate:
    video_id: str
    score: float
    event_hits: list[EventHit] = field(default_factory=list)

    @property
    def in_order(self) -> bool:
        frames = [hit.frame_idx for hit in self.event_hits]
        return all(a < b for a, b in itertools.pairwise(frames))


def rank_videos(
    engine,
    per_event_queries: list[ExpandedQuery],
    max_videos: int = 5,
    per_event_top_n: int = 300,
) -> list[VideoCandidate]:
    """Rank videos by how well they contain the whole event sequence.

    A video's score is the mean of its best per-event scores, multiplied by an
    ordering bonus. Requiring monotonically increasing times is what distinguishes
    "contains these moments" from "contains this sequence" — the training material
    calls this out as a core failure mode of bag-of-concepts matching.
    """
    per_video: dict[str, dict[int, EventHit]] = {}

    for event_index, expanded in enumerate(per_event_queries):
        result = engine.search(expanded.original, top_n=per_event_top_n, expanded=expanded)
        for hit in result.hits:
            if hit.frame_idx is None:
                continue
            events = per_video.setdefault(hit.video_id, {})
            existing = events.get(event_index)
            if existing is None or hit.score > existing.score:
                events[event_index] = EventHit(
                    event_index=event_index,
                    frame_idx=hit.frame_idx,
                    pts_time=hit.pts_time,
                    score=hit.score,
                )

    n_events = len(per_event_queries)
    candidates: list[VideoCandidate] = []
    for video_id, events in per_video.items():
        hits = [events[i] for i in sorted(events)]
        # Penalize videos that only match some events: a missing event caps the
        # achievable R-Score anyway, so it should not outrank a complete match.
        coverage = len(hits) / n_events
        mean_score = float(np.mean([h.score for h in hits])) if hits else 0.0

        ordered = sorted(hits, key=lambda h: h.event_index)
        frames = [h.frame_idx for h in ordered]
        monotonic = sum(a < b for a, b in itertools.pairwise(frames))
        order_bonus = 1.0 + (monotonic / max(1, len(frames) - 1)) if len(frames) > 1 else 1.0

        candidates.append(
            VideoCandidate(
                video_id=video_id,
                score=mean_score * coverage * order_bonus,
                event_hits=ordered,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_videos]


# -- stage 2: dense alignment --------------------------------------------------------


def extract_frame_range(
    video_path: Path, start_frame: int, end_frame: int, step: int = 1
) -> list[tuple[int, Image.Image]]:
    """Decode frames ``[start_frame, end_frame]`` as ``(frame_idx, image)`` pairs.

    Selects on decoded frame number (``between(n,...)``) rather than timestamps, so
    the returned ``frame_idx`` is exactly the number that will be submitted and
    stays correct on variable-frame-rate input.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg must be on PATH for TRAKE dense alignment")
    start_frame = max(0, start_frame)
    if end_frame < start_frame:
        return []

    select = f"between(n\\,{start_frame}\\,{end_frame})"
    if step > 1:
        select += f"*not(mod(n-{start_frame}\\,{step}))"

    result = subprocess.run(
        [
            ffmpeg, "-v", "error", "-i", str(video_path),
            "-vf", f"select='{select}'", "-vsync", "0",
            "-f", "image2pipe", "-vcodec", "png", "pipe:1",
        ],
        capture_output=True, check=False,
    )
    if result.returncode != 0 or not result.stdout:
        return []

    frames: list[tuple[int, Image.Image]] = []
    for offset, chunk in enumerate(_split_pngs(result.stdout)):
        frame_idx = start_frame + offset * step
        if frame_idx > end_frame:
            break
        frames.append((frame_idx, Image.open(io.BytesIO(chunk)).convert("RGB")))
    return frames


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _split_pngs(blob: bytes) -> list[bytes]:
    """Split ffmpeg's concatenated PNG stream into individual images."""
    positions = []
    start = blob.find(PNG_MAGIC)
    while start != -1:
        positions.append(start)
        start = blob.find(PNG_MAGIC, start + len(PNG_MAGIC))
    return [blob[a:b] for a, b in zip(positions, positions[1:] + [len(blob)])]


def _enforce_order(score_matrix: np.ndarray) -> list[int]:
    """Best strictly-increasing frame assignment, one frame per event.

    ``score_matrix[event, frame]``. Solved with a small DP over frame positions
    (events are few and frames per window number in the hundreds, so this is
    cheap) rather than taking a per-event argmax, which could otherwise return
    events out of chronological order.
    """
    n_events, n_frames = score_matrix.shape
    if n_events == 0 or n_frames == 0:
        return []

    NEG = -1e9
    best = np.full((n_events, n_frames), NEG, dtype=np.float64)
    back = np.full((n_events, n_frames), -1, dtype=np.int64)
    best[0] = score_matrix[0]

    for event in range(1, n_events):
        # running_max[j] = best score for the previous event at some frame < j
        running_best, running_arg = NEG, -1
        for frame in range(n_frames):
            if frame > 0 and best[event - 1, frame - 1] > running_best:
                running_best = best[event - 1, frame - 1]
                running_arg = frame - 1
            if running_arg >= 0:
                best[event, frame] = running_best + score_matrix[event, frame]
                back[event, frame] = running_arg

    last = int(np.argmax(best[n_events - 1]))
    if best[n_events - 1, last] <= NEG / 2:
        # No strictly increasing assignment exists (windows too tight); fall back to
        # independent argmax so we still return a usable, if unordered, answer.
        return [int(np.argmax(row)) for row in score_matrix]

    picks = [last]
    for event in range(n_events - 1, 0, -1):
        last = int(back[event, last])
        picks.append(last)
    return list(reversed(picks))


def align_events(
    video_path: Path,
    coarse_hits: list[EventHit],
    per_event_queries: list[ExpandedQuery],
    encoder,
    fps: float,
    window_seconds: float = 2.0,
    frame_step: int = 1,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Refine each coarse hit to a single frame by dense decoding and scoring.

    Returns the chosen frame per event and the per-event confidence, which
    :func:`aic.submit.policy.build_trake_answers` uses to decide which events to
    jitter first.
    """
    if not coarse_hits:
        return (), ()

    window_frames = max(1, round(window_seconds * fps))
    all_frames: list[tuple[int, Image.Image]] = []
    spans: list[tuple[int, int]] = []

    for hit in coarse_hits:
        start = max(0, hit.frame_idx - window_frames)
        end = hit.frame_idx + window_frames
        frames = extract_frame_range(video_path, start, end, step=frame_step)
        spans.append((len(all_frames), len(all_frames) + len(frames)))
        all_frames.extend(frames)

    if not all_frames:
        return tuple(h.frame_idx for h in coarse_hits), tuple(h.score for h in coarse_hits)

    image_vectors = encoder.encode_images([image for _, image in all_frames])
    texts = [
        (q.visual_en[0] if q.visual_en else q.original) for q in per_event_queries[: len(coarse_hits)]
    ]
    text_vectors = encoder.encode(texts)

    # scores[event, frame] over the union of all windows, so the ordering DP can
    # see every event's candidates on one shared timeline.
    scores = text_vectors @ image_vectors.T
    frame_numbers = np.array([idx for idx, _ in all_frames])

    # Mask each event to its own window: an event should not be aligned to a frame
    # from a different event's neighbourhood.
    masked = np.full_like(scores, -1e9)
    for event, (start, end) in enumerate(spans):
        if start < end:
            masked[event, start:end] = scores[event, start:end]

    order = np.argsort(frame_numbers, kind="stable")
    picks = _enforce_order(masked[:, order])

    chosen_frames = tuple(int(frame_numbers[order][p]) for p in picks)
    chosen_scores = tuple(float(masked[e, order][p]) for e, p in enumerate(picks))
    return chosen_frames, chosen_scores


def solve_trake(
    engine,
    query: str,
    event_descriptions: list[str],
    videos_dir: Path,
    config,
) -> list[TrakeCandidate]:
    """Full two-stage TRAKE pipeline, returning candidates ready for the policy."""
    from aic.data.verify import _find_video, probe_video
    from aic.query.encoder import get_encoder
    from aic.query.expand import expand_trake_query

    _, per_event = expand_trake_query(
        query, event_descriptions, model=str(config.query.llm_model)
    )
    video_candidates = rank_videos(
        engine, per_event, max_videos=int(config.trake.max_candidate_videos)
    )

    space = config.active_space
    encoder = get_encoder(space.model_id, int(space.dim))

    results: list[TrakeCandidate] = []
    for candidate in video_candidates:
        video_path = _find_video(Path(videos_dir), candidate.video_id)
        if video_path is None:
            continue
        try:
            fps = probe_video(video_path).fps or 25.0
        except Exception:  # noqa: BLE001 - a bad probe should not kill the query
            fps = 25.0

        frames, per_event_scores = align_events(
            video_path,
            candidate.event_hits,
            per_event,
            encoder,
            fps=fps,
            window_seconds=float(config.trake.window_seconds),
            frame_step=int(config.trake.frame_step),
        )
        if not frames:
            continue
        results.append(
            TrakeCandidate(
                video_id=candidate.video_id,
                frame_ids=frames,
                score=candidate.score,
                per_event_scores=per_event_scores,
            )
        )
    return results

"""Tests for submission-list construction and file writing."""

from __future__ import annotations

import pytest

from aic.eval.metric import (
    KISAnswer,
    KISGroundTruth,
    TRAKEAnswer,
    TRAKEGroundTruth,
    final_score,
    kis_r_score,
    trake_r_score,
)
from aic.submit.policy import (
    Candidate,
    TrakeCandidate,
    build_kis_answers,
    build_trake_answers,
    build_vqa_answers,
)
from aic.submit.writer import SubmissionError, validate_rows, write_submission


def test_kis_head_spans_distinct_videos() -> None:
    """Ranks 1-5 must hedge across videos: a wrong video scores zero."""
    candidates = [
        Candidate(video_id="L01_V001", frame_idx=100, score=0.9),
        Candidate(video_id="L01_V001", frame_idx=140, score=0.88),
        Candidate(video_id="L02_V002", frame_idx=200, score=0.85),
        Candidate(video_id="L03_V003", frame_idx=300, score=0.80),
    ]
    rows = build_kis_answers(candidates, diversify_head=3, frames_per_shot=2, frame_spread=5)
    head_videos = [video for video, _ in rows[:3]]
    assert len(set(head_videos)) == 3
    assert head_videos[0] == "L01_V001"  # strongest candidate still leads


def test_kis_emits_no_duplicate_rows() -> None:
    candidates = [Candidate(video_id="L01_V001", frame_idx=100, score=0.9)]
    rows = build_kis_answers(candidates, frames_per_shot=5, frame_spread=0)
    assert len(rows) == len(set(rows))


def test_kis_respects_shot_bounds_and_the_100_cap() -> None:
    candidates = [
        Candidate(video_id=f"L{i:02d}_V001", frame_idx=500, score=1.0 - i / 100, shot_start=498, shot_end=502)
        for i in range(60)
    ]
    rows = build_kis_answers(candidates, frames_per_shot=5, frame_spread=8)
    assert len(rows) <= 100
    for _, frame in rows:
        assert 498 <= frame <= 502  # clipped into the shot rather than drifting out


def test_kis_never_emits_negative_frames() -> None:
    candidates = [Candidate(video_id="L01_V001", frame_idx=2, score=0.9)]
    rows = build_kis_answers(candidates, frames_per_shot=5, frame_spread=10)
    assert all(frame >= 0 for _, frame in rows)


def test_frame_spreading_recovers_a_near_miss() -> None:
    """The point of spreading: a slightly-off frame still lands in the window."""
    truth = KISGroundTruth(video_id="L01_V001", start=500, end=510)
    candidate = Candidate(video_id="L01_V001", frame_idx=496, score=0.9)

    single = build_kis_answers([candidate], frames_per_shot=1)
    spread = build_kis_answers([candidate], frames_per_shot=3, frame_spread=8)

    single_scores = [kis_r_score(KISAnswer(v, f), truth) for v, f in single]
    spread_scores = [kis_r_score(KISAnswer(v, f), truth) for v, f in spread]
    assert final_score(single_scores) == 0.0
    assert final_score(spread_scores) > 0.0


def test_vqa_carries_answers_and_fills_gaps() -> None:
    candidates = [
        Candidate(video_id="L05_V005", frame_idx=888, score=0.9, answer="màu xanh"),
        Candidate(video_id="L06_V006", frame_idx=100, score=0.5),  # no answer of its own
    ]
    rows = build_vqa_answers(candidates, frames_per_shot=1)
    assert rows[0] == ("L05_V005", 888, "màu xanh")
    assert all(answer.strip() for _, _, answer in rows)  # never an empty answer


def test_trake_head_hedges_videos_then_jitters_the_best() -> None:
    candidates = [
        TrakeCandidate(video_id="L10_V010", frame_ids=(100, 150, 200, 250), score=0.9),
        TrakeCandidate(video_id="L11_V011", frame_ids=(90, 140, 190, 240), score=0.6),
    ]
    rows = build_trake_answers(candidates, diversify_head=2, jitter=(0, -4, 4))
    assert rows[0] == ("L10_V010", (100, 150, 200, 250))
    assert rows[1][0] == "L11_V011"
    # Every jittered variant differs from the base tuple in exactly one event.
    for video, frames in rows[2:]:
        base = (100, 150, 200, 250) if video == "L10_V010" else (90, 140, 190, 240)
        assert sum(a != b for a, b in zip(frames, base)) <= 1


def test_trake_jitter_targets_least_confident_events_first() -> None:
    candidates = [
        TrakeCandidate(
            video_id="L10_V010",
            frame_ids=(100, 150, 200, 250),
            score=0.9,
            per_event_scores=(0.9, 0.9, 0.1, 0.9),  # event 3 is the weak one
        )
    ]
    rows = build_trake_answers(candidates, diversify_head=1, jitter=(0, -4))
    changed_index = next(
        i for i, (a, b) in enumerate(zip(rows[1][1], (100, 150, 200, 250))) if a != b
    )
    assert changed_index == 2


def test_trake_partial_credit_is_actually_recovered() -> None:
    """End-to-end: jitter turns a 3/4 answer into a scoring 4/4 further down."""
    truth = TRAKEGroundTruth(
        video_id="L10_V010", windows=((95, 105), (145, 155), (195, 205), (245, 255))
    )
    # Event 2 is off by one frame past its window.
    candidate = TrakeCandidate(video_id="L10_V010", frame_ids=(101, 156, 203, 251), score=0.9)
    rows = build_trake_answers([candidate], diversify_head=1, jitter=(0, -4, 4))
    scores = [trake_r_score(TRAKEAnswer(v, f), truth) for v, f in rows]

    assert scores[0] == pytest.approx(0.75)  # the rules' worked example
    assert max(scores) == pytest.approx(1.0)  # a later row fixes event 2
    assert final_score(scores) > 0.75


def test_empty_candidates_produce_no_rows() -> None:
    assert build_kis_answers([]) == []
    assert build_vqa_answers([]) == []
    assert build_trake_answers([]) == []


# -- writer ------------------------------------------------------------------------


def test_validate_rejects_bad_submissions() -> None:
    with pytest.raises(SubmissionError, match="empty submission"):
        validate_rows([], "kis")
    with pytest.raises(SubmissionError, match="exceeds"):
        validate_rows([("L01_V001", i) for i in range(101)], "kis")
    with pytest.raises(SubmissionError, match="negative"):
        validate_rows([("L01_V001", -1)], "kis")
    with pytest.raises(SubmissionError, match="duplicate"):
        validate_rows([("L01_V001", 5), ("L01_V001", 5)], "kis")
    with pytest.raises(SubmissionError, match="empty answer"):
        validate_rows([("L01_V001", 5, "  ")], "vqa")


def test_validate_rejects_ragged_trake_tuples() -> None:
    rows = [("L10_V010", (1, 2, 3, 4)), ("L10_V010", (1, 2, 3))]
    with pytest.raises(SubmissionError, match="every answer must cover"):
        validate_rows(rows, "trake")


def test_write_submission_formats_each_task(tmp_path) -> None:
    kis = write_submission([("L01_V001", 505)], tmp_path / "kis.csv", "kis")
    assert kis.read_text(encoding="utf-8") == "L01_V001,505\n"

    vqa = write_submission([("L05_V005", 888, "màu xanh")], tmp_path / "vqa.csv", "vqa")
    assert vqa.read_text(encoding="utf-8") == "L05_V005,888,màu xanh\n"

    trake = write_submission([("L10_V010", (101, 156, 203, 251))], tmp_path / "t.csv", "trake")
    assert trake.read_text(encoding="utf-8") == "L10_V010,101,156,203,251\n"

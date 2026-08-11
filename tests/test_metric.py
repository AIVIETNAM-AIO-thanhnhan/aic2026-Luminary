"""Scoring tests taken verbatim from section 2 of the AIC 2026 rules document."""

from __future__ import annotations

import pytest

from aic.eval.metric import (
    KISAnswer,
    KISGroundTruth,
    TRAKEAnswer,
    TRAKEGroundTruth,
    VQAAnswer,
    VQAGroundTruth,
    final_score,
    kis_r_score,
    r_at_k,
    trake_r_score,
    vqa_r_score,
)

# Rules 2.1.1: "một người đang mở laptop" -> L01_V001, frames 500..510.
KIS_TRUTH = KISGroundTruth(video_id="L01_V001", start=500, end=510)


@pytest.mark.parametrize(
    ("video_id", "frame_id", "expected"),
    [
        ("L01_V001", 505, 1.0),  # inside the window
        ("L01_V001", 600, 0.0),  # right video, frame outside [500, 510]
        ("L02_V003", 505, 0.0),  # wrong video
        ("L01_V001", 500, 1.0),  # inclusive lower bound
        ("L01_V001", 510, 1.0),  # inclusive upper bound
        ("L01_V001", 499, 0.0),
        ("L01_V001", 511, 0.0),
    ],
)
def test_kis_r_score(video_id: str, frame_id: int, expected: float) -> None:
    assert kis_r_score(KISAnswer(video_id, frame_id), KIS_TRUTH) == expected


# Rules 2.1.2: party scene -> L05_V005, frames 800..900, answer "màu xanh".
VQA_TRUTH = VQAGroundTruth(
    video_id="L05_V005", start=800, end=900, answers=frozenset({"màu xanh", "blue"})
)


@pytest.mark.parametrize(
    ("video_id", "frame_id", "answer", "expected"),
    [
        ("L05_V005", 888, "màu xanh", 1.0),
        ("L05_V005", 888, "màu trắng", 0.0),  # wrong answer
        ("L06_V007", 888, "màu xanh", 0.0),  # wrong video
        ("L05_V005", 950, "màu xanh", 0.0),  # frame outside window
        ("L05_V005", 888, "  MÀU   XANH ", 1.0),  # normalization applies
    ],
)
def test_vqa_r_score(video_id: str, frame_id: int, answer: str, expected: float) -> None:
    assert vqa_r_score(VQAAnswer(video_id, frame_id, answer), VQA_TRUTH) == expected


# Rules 2.1.3: high-jump sequence -> L10_V010 with four event windows.
TRAKE_TRUTH = TRAKEGroundTruth(
    video_id="L10_V010",
    windows=((95, 105), (145, 155), (195, 205), (245, 255)),
)


def test_trake_worked_example_scores_three_of_four() -> None:
    """The document's own example: 101, 156, 203, 251 -> 3/4 = 0.75."""
    answer = TRAKEAnswer(video_id="L10_V010", frame_ids=(101, 156, 203, 251))
    assert trake_r_score(answer, TRAKE_TRUTH) == pytest.approx(0.75)


def test_trake_wrong_video_scores_zero_despite_perfect_frames() -> None:
    answer = TRAKEAnswer(video_id="L11_V011", frame_ids=(100, 150, 200, 250))
    assert trake_r_score(answer, TRAKE_TRUTH) == 0.0


def test_trake_all_events_correct() -> None:
    answer = TRAKEAnswer(video_id="L10_V010", frame_ids=(100, 150, 200, 250))
    assert trake_r_score(answer, TRAKE_TRUTH) == pytest.approx(1.0)


def test_trake_short_answer_scores_only_the_events_supplied() -> None:
    answer = TRAKEAnswer(video_id="L10_V010", frame_ids=(100, 150))
    assert trake_r_score(answer, TRAKE_TRUTH) == pytest.approx(0.5)


def test_r_at_k_takes_the_best_of_the_head() -> None:
    scores = [0.5, 0.0, 0.8, 0.0, 0.0]
    assert r_at_k(scores, 1) == 0.5
    assert r_at_k(scores, 5) == 0.8
    assert r_at_k([], 5) == 0.0


def test_final_score_worked_example() -> None:
    """Rules 2.2: 0.5 at rank 1, 0.8 at rank 3, 0.6 at rank 15 -> 0.74."""
    scores = [0.0] * 100
    scores[0] = 0.5
    scores[2] = 0.8
    scores[14] = 0.6
    assert final_score(scores) == pytest.approx(0.74)


def test_final_score_rejects_more_than_100_answers() -> None:
    with pytest.raises(ValueError, match="at most 100"):
        final_score([1.0] * 101)


def test_rank_one_is_worth_one_fifth_of_the_score() -> None:
    """Sanity-check the incentive the submission policy is built around."""
    only_rank_one = final_score([1.0] + [0.0] * 99)
    only_rank_two = final_score([0.0, 1.0] + [0.0] * 98)
    assert only_rank_one == pytest.approx(1.0)
    assert only_rank_two == pytest.approx(0.8)

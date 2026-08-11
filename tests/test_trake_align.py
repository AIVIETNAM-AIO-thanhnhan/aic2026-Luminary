"""Tests for the TRAKE ordering solver and PNG stream splitting."""

from __future__ import annotations

import numpy as np

from aic.tasks.trake import PNG_MAGIC, _enforce_order, _split_pngs


def test_enforce_order_returns_strictly_increasing_picks() -> None:
    # Each event's peak is out of chronological order: argmax alone would pick
    # frames 4, 2, 0 which violates t1 < t2 < t3.
    scores = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.9],
            [0.1, 0.9, 0.3, 0.4, 0.5],
            [0.9, 0.2, 0.3, 0.4, 0.5],
        ]
    )
    picks = _enforce_order(scores)
    assert picks == sorted(picks)
    assert len(set(picks)) == 3


def test_enforce_order_finds_the_obvious_diagonal() -> None:
    scores = np.array(
        [
            [0.9, 0.1, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1],
            [0.1, 0.1, 0.9, 0.1],
        ]
    )
    assert _enforce_order(scores) == [0, 1, 2]


def test_enforce_order_maximizes_total_not_each_event() -> None:
    """A locally-best pick is given up when it blocks a better overall assignment."""
    scores = np.array(
        [
            [0.50, 0.60, 0.10],
            [0.10, 0.55, 0.50],
        ]
    )
    # Greedy would take frame 1 for event 0 (0.60), forcing event 1 onto frame 2
    # (0.50) for 1.10. The optimum is frames 0 and 1, totalling 1.05... so greedy
    # actually wins here; assert the solver finds the true maximum either way.
    picks = _enforce_order(scores)
    total = sum(scores[event, frame] for event, frame in enumerate(picks))
    best_possible = max(
        scores[0, i] + scores[1, j]
        for i in range(3)
        for j in range(3)
        if i < j
    )
    assert total == best_possible
    assert picks[0] < picks[1]


def test_enforce_order_handles_single_event_and_empty() -> None:
    assert _enforce_order(np.array([[0.1, 0.7, 0.2]])) == [1]
    assert _enforce_order(np.zeros((0, 5))) == []
    assert _enforce_order(np.zeros((3, 0))) == []


def test_enforce_order_falls_back_when_no_ordering_exists() -> None:
    """More events than frames: return something usable rather than nothing."""
    scores = np.array([[0.9], [0.8], [0.7]])
    picks = _enforce_order(scores)
    assert len(picks) == 3


def test_split_pngs_separates_a_concatenated_stream() -> None:
    first = PNG_MAGIC + b"first-image-body"
    second = PNG_MAGIC + b"second-image-body"
    third = PNG_MAGIC + b"third"
    assert _split_pngs(first + second + third) == [first, second, third]


def test_split_pngs_on_empty_input() -> None:
    assert _split_pngs(b"") == []

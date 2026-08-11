"""Write submission CSV files.

One file per query, no header, one answer per line, ranked best-first — the format
implied by the rules' answer specification:

* KIS   ``video_id,frame_id``
* Q&A   ``video_id,frame_id,answer``
* TRAKE ``video_id,frame_id_1,...,frame_id_n``

:func:`validate_rows` runs before anything is written. A malformed submission is
worth zero regardless of how good the retrieval was, so the checks are strict:
frame numbers must be non-negative integers, rows must be unique and within the
100-answer cap, and TRAKE tuples must all have the same arity.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from aic.eval.metric import MAX_ANSWERS

VIDEO_ID_MAX_LEN = 64


class SubmissionError(ValueError):
    """Raised when rows would produce an invalid submission file."""


def _check_video_id(video_id: str, row_number: int) -> None:
    if not video_id or not video_id.strip():
        raise SubmissionError(f"row {row_number}: empty video_id")
    if len(video_id) > VIDEO_ID_MAX_LEN or "," in video_id:
        raise SubmissionError(f"row {row_number}: suspicious video_id {video_id!r}")


def _check_frame(frame: object, row_number: int) -> int:
    if isinstance(frame, bool) or not isinstance(frame, (int,)):
        raise SubmissionError(f"row {row_number}: frame_id must be an int, got {frame!r}")
    if frame < 0:
        raise SubmissionError(f"row {row_number}: negative frame_id {frame}")
    return int(frame)


def validate_rows(rows: Sequence[Sequence], task: str) -> None:
    """Raise :class:`SubmissionError` if ``rows`` would not be accepted."""
    if not rows:
        raise SubmissionError("refusing to write an empty submission")
    if len(rows) > MAX_ANSWERS:
        raise SubmissionError(f"{len(rows)} answers exceeds the {MAX_ANSWERS} limit")

    seen: set[tuple] = set()
    arity: int | None = None

    for row_number, row in enumerate(rows, start=1):
        if task == "kis":
            if len(row) != 2:
                raise SubmissionError(f"row {row_number}: KIS needs (video_id, frame_id)")
            _check_video_id(str(row[0]), row_number)
            _check_frame(row[1], row_number)
            key = (row[0], row[1])
        elif task == "vqa":
            if len(row) != 3:
                raise SubmissionError(f"row {row_number}: Q&A needs (video_id, frame_id, answer)")
            _check_video_id(str(row[0]), row_number)
            _check_frame(row[1], row_number)
            if not str(row[2]).strip():
                raise SubmissionError(f"row {row_number}: empty answer text")
            key = (row[0], row[1], str(row[2]))
        elif task == "trake":
            if len(row) != 2 or not isinstance(row[1], (tuple, list)):
                raise SubmissionError(
                    f"row {row_number}: TRAKE needs (video_id, (frame_1, ..., frame_n))"
                )
            _check_video_id(str(row[0]), row_number)
            frames = tuple(_check_frame(f, row_number) for f in row[1])
            if not frames:
                raise SubmissionError(f"row {row_number}: TRAKE answer has no frames")
            if arity is None:
                arity = len(frames)
            elif len(frames) != arity:
                raise SubmissionError(
                    f"row {row_number}: {len(frames)} events but earlier rows have {arity}; "
                    "every answer must cover the same event sequence"
                )
            key = (row[0], frames)
        else:
            raise SubmissionError(f"unknown task {task!r}; expected kis, vqa, or trake")

        if key in seen:
            raise SubmissionError(f"row {row_number}: duplicate answer {key!r} wastes a slot")
        seen.add(key)


def write_submission(
    rows: Sequence[Sequence],
    path: Path,
    task: str,
    validate: bool = True,
) -> Path:
    """Validate and write one query's submission file."""
    if validate:
        validate_rows(rows, task)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in rows:
            if task == "trake":
                writer.writerow([row[0], *row[1]])
            else:
                writer.writerow(list(row))
    return path


def submission_path(submissions_dir: Path, query_id: str, task: str) -> Path:
    """``submissions/<task>/<query_id>.csv``."""
    safe = "".join(c for c in query_id if c.isalnum() or c in "-_") or "query"
    return Path(submissions_dir) / task / f"{safe}.csv"

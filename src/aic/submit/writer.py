"""Write submission CSV files that satisfy the organizers' format spec.

One file per query, no header, one answer per line, ranked best-first:

* KIS   ``video_id,frame_id``
* Q&A   ``video_id,frame_id,answer``
* TRAKE ``video_id,frame_id_1,...,frame_id_n``

:func:`validate_rows` runs before anything is written, because a malformed file is
worth zero no matter how good the retrieval was — and a rejected upload still burns
one of the three attempts per query package. Every check below maps to a stated
rule in "Hướng dẫn nộp bài sơ tuyển":

* at most 100 rows per file;
* video names carry **no** ``.mp4`` extension (listed as a common mistake);
* frame ids are plain integers with no stray whitespace;
* Q&A answers are at most 100 characters;
* TRAKE rows all have exactly the number of events the query asked for, in
  chronological order.

Quoting is left to :mod:`csv` with its default ``QUOTE_MINIMAL``, which is exactly
the rule the guide describes: quote only when the value contains a comma, a quote
or a newline, and escape embedded quotes by doubling them.
"""

from __future__ import annotations

import csv
import itertools
from collections.abc import Sequence
from pathlib import Path

from aic.eval.metric import MAX_ANSWERS

VIDEO_ID_MAX_LEN = 64
ANSWER_MAX_LEN = 100
VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".avi")
TASKS = ("kis", "vqa", "trake")


class SubmissionError(ValueError):
    """Raised when rows would produce an invalid submission file."""


def _check_video_id(video_id: str, row_number: int) -> None:
    if not video_id or not video_id.strip():
        raise SubmissionError(f"row {row_number}: empty video_id")
    if video_id != video_id.strip():
        raise SubmissionError(f"row {row_number}: video_id has surrounding whitespace {video_id!r}")
    if video_id.lower().endswith(VIDEO_SUFFIXES):
        raise SubmissionError(
            f"row {row_number}: video name must not include the file extension "
            f"({video_id!r} -> {Path(video_id).stem!r})"
        )
    if len(video_id) > VIDEO_ID_MAX_LEN or "," in video_id:
        raise SubmissionError(f"row {row_number}: suspicious video_id {video_id!r}")


def _check_frame(frame: object, row_number: int) -> int:
    if isinstance(frame, bool) or not isinstance(frame, int):
        raise SubmissionError(f"row {row_number}: frame_id must be an int, got {frame!r}")
    if frame < 0:
        raise SubmissionError(f"row {row_number}: negative frame_id {frame}")
    return int(frame)


def _check_answer(answer: object, row_number: int) -> str:
    text = str(answer)
    if not text.strip():
        raise SubmissionError(f"row {row_number}: empty answer text")
    if len(text) > ANSWER_MAX_LEN:
        raise SubmissionError(
            f"row {row_number}: answer is {len(text)} characters, over the "
            f"{ANSWER_MAX_LEN}-character limit: {text[:60]!r}..."
        )
    return text


def validate_rows(
    rows: Sequence[Sequence],
    task: str,
    expected_events: int | None = None,
) -> None:
    """Raise :class:`SubmissionError` if ``rows`` would not be accepted.

    ``expected_events`` is the number of events the TRAKE query asked for. Pass it
    whenever it is known: the guide requires an exact match, and a count that is
    merely self-consistent across rows can still be uniformly wrong.
    """
    if task not in TASKS:
        raise SubmissionError(f"unknown task {task!r}; expected one of {TASKS}")
    if not rows:
        raise SubmissionError("refusing to write an empty submission")
    if len(rows) > MAX_ANSWERS:
        raise SubmissionError(f"{len(rows)} answers exceeds the {MAX_ANSWERS} limit")

    seen: set[tuple] = set()
    arity: int | None = expected_events

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
            key = (row[0], row[1], _check_answer(row[2], row_number))

        else:  # trake
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
                    f"row {row_number}: {len(frames)} frame ids but the query has {arity} "
                    "events; the counts must match exactly"
                )
            if any(a >= b for a, b in itertools.pairwise(frames)):
                raise SubmissionError(
                    f"row {row_number}: frame ids {frames} are not in chronological order; "
                    "events must be strictly increasing in time"
                )
            key = (row[0], frames)

        if key in seen:
            raise SubmissionError(f"row {row_number}: duplicate answer {key!r} wastes a slot")
        seen.add(key)


def write_submission(
    rows: Sequence[Sequence],
    path: Path,
    task: str,
    validate: bool = True,
    expected_events: int | None = None,
) -> Path:
    """Validate and write one query's submission CSV."""
    if validate:
        validate_rows(rows, task, expected_events=expected_events)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" lets csv control line endings; utf-8 without BOM, as required.
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in rows:
            writer.writerow([row[0], *row[1]] if task == "trake" else list(row))
    return path


def submission_path(submissions_dir: Path, query_id: str, task: str | None = None) -> Path:
    """Path for one query's CSV inside the flat ``submission/`` folder.

    The organizers require the CSV to be named after the query file it answers —
    ``query-1-kis.txt`` is answered by ``query-1-kis.csv`` — all sitting directly
    inside a single ``submission/`` directory. ``task`` is accepted for call-site
    convenience and appended only when the query id does not already carry it.
    """
    stem = Path(str(query_id)).stem
    safe = "".join(c for c in stem if c.isalnum() or c in "-_") or "query"
    if task and not safe.lower().endswith(f"-{task_suffix(task)}"):
        safe = f"{safe}-{task_suffix(task)}"
    return Path(submissions_dir) / f"{safe}.csv"


def task_suffix(task: str) -> str:
    """Map an internal task name to the organizers' filename suffix."""
    return {"kis": "kis", "vqa": "qa", "trake": "trake"}.get(task, task)

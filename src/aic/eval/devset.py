"""A self-labelled development set.

The organizers publish no ground truth for the preliminary round, so every tuning
decision in M2-M4 — which embedding model, which fusion weights, how wide to spread
frames — is unmeasurable without labels of our own. This module defines the file
format and loads it; the labelling itself is manual work done through the Streamlit
UI (find the moment, note the video and frame range).

``configs/devset.jsonl``, one JSON object per line:

.. code-block:: json

    {"id": "kis-01", "task": "kis", "query": "Tìm cảnh một người đang mở laptop",
     "video_id": "L01_V001", "start": 500, "end": 510}
    {"id": "vqa-01", "task": "vqa", "query": "...", "question": "Cầm ly màu gì?",
     "video_id": "L05_V005", "start": 800, "end": 900, "answers": ["màu xanh"]}
    {"id": "trake-01", "task": "trake", "query": "...",
     "events": ["giậm nhảy", "bay qua xà", "tiếp đất", "đứng dậy"],
     "video_id": "L10_V010", "windows": [[95,105],[145,155],[195,205],[245,255]]}

Aim for ~30 entries spread across the three types, written in the same register as
the organizers' examples — terse, third-person, one concrete visual detail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from aic.eval.metric import KISGroundTruth, TRAKEGroundTruth, VQAGroundTruth

DEFAULT_DEVSET = Path("configs/devset.jsonl")


@dataclass
class DevQuery:
    id: str
    task: str  # "kis" | "vqa" | "trake"
    query: str
    truth: object
    #: Q&A only: the question asked about the located moment.
    question: str | None = None
    #: TRAKE only: the ordered event descriptions.
    events: list[str] = field(default_factory=list)


def _parse(record: dict, line_number: int) -> DevQuery:
    task = record.get("task")
    query_id = record.get("id") or f"line-{line_number}"

    missing = [k for k in ("task", "query", "video_id") if not record.get(k)]
    if missing:
        raise ValueError(f"devset line {line_number}: missing {missing}")

    if task == "kis":
        truth = KISGroundTruth(
            video_id=record["video_id"], start=int(record["start"]), end=int(record["end"])
        )
    elif task == "vqa":
        answers = record.get("answers") or []
        if not answers:
            raise ValueError(f"devset line {line_number}: vqa entry needs a non-empty 'answers'")
        truth = VQAGroundTruth(
            video_id=record["video_id"],
            start=int(record["start"]),
            end=int(record["end"]),
            answers=frozenset(answers),
        )
    elif task == "trake":
        windows = record.get("windows") or []
        if not windows:
            raise ValueError(f"devset line {line_number}: trake entry needs 'windows'")
        truth = TRAKEGroundTruth(
            video_id=record["video_id"],
            windows=tuple((int(a), int(b)) for a, b in windows),
        )
    else:
        raise ValueError(f"devset line {line_number}: unknown task {task!r}")

    return DevQuery(
        id=query_id,
        task=task,
        query=record["query"],
        truth=truth,
        question=record.get("question"),
        events=record.get("events") or [],
    )


def load_devset(path: Path = DEFAULT_DEVSET) -> list[DevQuery]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no dev set at {path}. Label ~30 moments through the Streamlit UI first; "
            "see the format in aic/eval/devset.py."
        )
    queries: list[DevQuery] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            queries.append(_parse(json.loads(line), line_number))
    return queries


def save_devset(queries: list[dict], path: Path = DEFAULT_DEVSET) -> Path:
    """Append labelled entries, used by the UI's "save as dev query" action."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in queries)
    return path

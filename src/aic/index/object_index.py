"""SQLite index over the organizers' per-frame object detections.

Unlike OCR/ASR, detections are already per-frame (one JSON per keyframe, named
to match the keyframe file exactly), so there is no time-span-to-frame mapping
to do - a detection's ``gid`` is just the catalog row for that same image.

The class taxonomy is OpenImages, the same one :mod:`aic.query.expand` asks
Gemini to use for a query's ``objects`` field, so the two line up directly.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    gid      INTEGER NOT NULL,
    video_id TEXT NOT NULL,
    label    TEXT NOT NULL,
    score    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label);
CREATE INDEX IF NOT EXISTS idx_detections_gid ON detections(gid);
"""

DEFAULT_SCORE_THRESHOLD = 0.4

#: The OpenImages detector treats "Person" as mutually exclusive with the more
#: specific "Man"/"Woman"/"Boy"/"Girl" classes - it picks one label per detected
#: human, not "Person" plus a gendered/age refinement. Found live: a frame with
#: 5 people, all labeled "Man"/"Boy", had *zero* "Person" detections at all, so
#: every "Person" instance-count check (search_by_min_count, counts_by_gid, ...)
#: was silently blind to it despite the frame genuinely satisfying the count.
#: 30,991 of 177,321 frames (17.5% of the corpus) have a gendered label but no
#: "Person" label - this is not a rare edge case. "Human face"/"Human body" are
#: deliberately excluded: a detector can (and does) tag both a "Man" box and a
#: separate "Human face" box for the *same* individual, so folding those in
#: would double-count one person as two.
PERSON_SYNONYMS = ("person", "man", "woman", "boy", "girl")

#: A label present in more than this fraction of all detected frames (e.g.
#: "Person" at ~39%, "Clothing" at ~46% in this corpus) cannot discriminate
#: anything - it just becomes a new attractor for any query that mentions
#: people, food, etc. alongside a genuinely specific class.
DEFAULT_MAX_LABEL_FRACTION = 0.03


@dataclass
class ObjectHit:
    gid: int
    video_id: str
    label: str
    score: float


class ObjectIndex:
    """Read/write wrapper around the per-frame detection database."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: see the identical note in text_index.py -
        # a cached SearchEngine gets reused from a different thread on each
        # Streamlit rerun, but never concurrently.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._total_frames: int | None = None

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def clear(self) -> None:
        self.conn.execute("DELETE FROM detections")
        self.conn.commit()

    def add_detections(self, rows: Iterable[dict]) -> int:
        """Insert detections. Each row needs ``gid``, ``video_id``, ``label``, ``score``."""
        rows = list(rows)
        self.conn.executemany(
            "INSERT INTO detections(gid, video_id, label, score) VALUES (?, ?, ?, ?)",
            [(r["gid"], r["video_id"], r["label"], r["score"]) for r in rows],
        )
        self.conn.commit()
        return len(rows)

    def total_frames(self) -> int:
        """Frames with at least one detection - the denominator for rarity checks."""
        if self._total_frames is None:
            row = self.conn.execute("SELECT COUNT(DISTINCT gid) AS n FROM detections").fetchone()
            self._total_frames = int(row["n"]) or 1
        return self._total_frames

    def is_discriminative(self, label: str, max_fraction: float = DEFAULT_MAX_LABEL_FRACTION) -> bool:
        """False for a label so common (e.g. "Person") it cannot narrow anything down."""
        label = label.strip().lower()
        if not label:
            return False
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT gid) AS n FROM detections WHERE LOWER(label) = ?", (label,)
        ).fetchone()
        n = int(row["n"])
        return 0 < n <= self.total_frames() * max_fraction

    @staticmethod
    def _label_group(label: str) -> tuple[str, ...]:
        """Labels to count together for ``label`` - see PERSON_SYNONYMS."""
        label = label.strip().lower()
        return PERSON_SYNONYMS if label == "person" else (label,)

    def counts_by_gid(self, label: str) -> dict[int, int]:
        """Every gid's detected instance count for ``label``, for hard-filtering.

        Unlike :meth:`search`/:meth:`search_by_min_count`, this isn't itself a
        ranked branch - it's the raw lookup used to check whether a frame that
        surfaced via some *other* branch (visual, OCR, ...) actually satisfies a
        count constraint at all, since RRF fusion never enforces that on its own.
        """
        labels = self._label_group(label)
        if not labels[0]:
            return {}
        placeholders = ",".join("?" for _ in labels)
        rows = self.conn.execute(
            f"SELECT gid, COUNT(*) AS n FROM detections WHERE LOWER(label) IN ({placeholders}) GROUP BY gid",
            labels,
        ).fetchall()
        return {int(r["gid"]): int(r["n"]) for r in rows}

    def search_by_min_count(self, label: str, min_count: int, limit: int = 500) -> list[ObjectHit]:
        """Frames with at least ``min_count`` separate detections of ``label``.

        Each detection row is one detector instance (not a deduplicated presence
        flag), so ``COUNT(*)`` per gid is the detected instance count. Ranked by
        that count descending, which doubles as the RRF-fed score: a frame with
        more detected instances is a stronger match for a "more than N" query,
        regardless of any individual instance's confidence. Deliberately bypasses
        :meth:`is_discriminative` - a label like "Person" is too common to filter
        on presence alone, but requiring a high count is itself discriminative.
        """
        labels = self._label_group(label)
        if not labels[0] or min_count < 1:
            return []
        placeholders = ",".join("?" for _ in labels)
        rows = self.conn.execute(
            "SELECT gid, video_id, COUNT(*) AS n FROM detections "
            f"WHERE LOWER(label) IN ({placeholders}) GROUP BY gid HAVING COUNT(*) >= ? "
            "ORDER BY n DESC LIMIT ?",
            (*labels, min_count, limit),
        ).fetchall()
        return [
            ObjectHit(gid=r["gid"], video_id=r["video_id"], label=label.strip().lower(), score=float(r["n"]))
            for r in rows
        ]

    def search_by_target_count(self, label: str, target_count: int, limit: int = 500) -> list[ObjectHit]:
        """Frames whose detected ``label`` count is closest to an exact target.

        For an "exactly N" detail ("chỉ có MỘT người đeo kính" - only one person
        wears glasses; "BA người đội nón" - three people wear hats), more detected
        instances is a *worse* match once past N, not a better one - unlike
        :meth:`search_by_min_count`'s "more than N" semantics, ranking by raw
        count here would put a frame with a dozen glasses-wearers ahead of the
        one true frame with exactly one. Ranking by ``ABS(count - target)``
        instead puts exact matches first, then off-by-one, and so on - still
        biased towards the truth as detector noise (occlusion, a missed small
        object like glasses) skews real counts down more often than up.
        """
        labels = self._label_group(label)
        if not labels[0] or target_count < 1:
            return []
        placeholders = ",".join("?" for _ in labels)
        rows = self.conn.execute(
            "SELECT gid, video_id, COUNT(*) AS n FROM detections "
            f"WHERE LOWER(label) IN ({placeholders}) GROUP BY gid "
            "ORDER BY ABS(COUNT(*) - ?) ASC LIMIT ?",
            (*labels, target_count, limit),
        ).fetchall()
        return [
            ObjectHit(gid=r["gid"], video_id=r["video_id"], label=label.strip().lower(), score=float(r["n"]))
            for r in rows
        ]

    def search(self, labels: Sequence[str], limit: int = 500) -> list[ObjectHit]:
        """Best-scoring frame per matching label, ranked by that score."""
        wanted = [label.strip().lower() for label in labels if label.strip()]
        if not wanted:
            return []
        placeholders = ",".join("?" for _ in wanted)
        sql = (
            "SELECT gid, video_id, label, MAX(score) AS best_score "
            "FROM detections "
            f"WHERE LOWER(label) IN ({placeholders}) "
            "GROUP BY gid "
            "ORDER BY best_score DESC "
            "LIMIT ?"
        )
        rows = self.conn.execute(sql, [*wanted, limit]).fetchall()
        return [
            ObjectHit(gid=r["gid"], video_id=r["video_id"], label=r["label"], score=float(r["best_score"]))
            for r in rows
        ]

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM detections")
        return int(row.fetchone()["n"])


def build_object_index(
    catalog: pd.DataFrame,
    objects_dir: Path,
    index_path: Path,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> int:
    """Populate an :class:`ObjectIndex` from the organizers' per-frame JSON files.

    Each catalog row's keyframe file has a same-named detection JSON under
    ``objects_dir/{video_id}/{stem}.json`` - missing files are skipped rather
    than treated as an error, since detection coverage can lag keyframe coverage.
    """
    objects_dir = Path(objects_dir)
    written = 0
    with ObjectIndex(index_path) as index:
        index.clear()
        batch: list[dict] = []
        for row in catalog.itertuples():
            json_path = objects_dir / row.video_id / f"{Path(row.path).stem}.json"
            if not json_path.exists():
                continue
            with open(json_path, encoding="utf-8") as handle:
                detection = json.load(handle)
            for label, score in zip(
                detection.get("detection_class_entities", []),
                detection.get("detection_scores", []),
                strict=False,
            ):
                score = float(score)
                if score >= score_threshold:
                    batch.append({"gid": int(row.gid), "video_id": row.video_id, "label": label, "score": score})
            if len(batch) >= 5000:
                written += index.add_detections(batch)
                batch = []
        if batch:
            written += index.add_detections(batch)
    return written

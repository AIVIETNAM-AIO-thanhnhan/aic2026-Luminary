"""SQLite FTS5 full-text index over OCR and ASR text.

Replaces Elasticsearch from the reference architecture: FTS5 ships inside Python's
stdlib sqlite3, needs no daemon or Docker, and BM25 over a few hundred thousand
short news captions is instant. One less moving part to keep alive during a contest.

Vietnamese needs care. FTS5's default tokenizer splits on ASCII boundaries and does
not fold diacritics, so *"họp báo"* would not match a query typed *"hop bao"*. Each
row is therefore stored twice: the original text, and a diacritic-stripped copy.
Queries are matched against both, so accented and unaccented typing both work.
"""

from __future__ import annotations

import sqlite3
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL,          -- 'ocr' | 'asr'
    video_id   TEXT NOT NULL,
    start_frame INTEGER,               -- inclusive, in original-video frame numbers
    end_frame   INTEGER,               -- inclusive
    start_time  REAL,
    end_time    REAL,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_id, start_frame);

CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    text, text_ascii, content=''
);
"""


def strip_diacritics(text: str) -> str:
    """Fold Vietnamese diacritics to ASCII so 'họp báo' also matches 'hop bao'."""
    # đ/Đ carry no combining mark, so NFD alone leaves them; map them explicitly.
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def escape_fts_query(terms: Sequence[str]) -> str:
    """Build an FTS5 query matching ANY of ``terms``, each as an exact phrase.

    Each term is quoted as a single FTS5 phrase - its words must appear
    adjacent, in order - rather than split into individual words each OR'd
    separately. Splitting on words was the previous behavior, and it silently
    degrades a specific multi-word phrase into a bag of words: "chạm mũi chân"
    (touching toes) would match any segment containing just "chân" (foot) on
    its own, such as an unrelated "chân gà" (chicken feet) recipe line. The
    different *terms* in the list are still OR'd against each other - expand_query
    hands over several independent candidate phrases, and any one of them
    matching is still a hit - only the words *within* one term are now required
    to appear together.
    """
    quoted = []
    for term in terms:
        cleaned = term.replace('"', " ").strip()
        if cleaned:
            quoted.append(f'"{cleaned}"')
    return " OR ".join(quoted)


@dataclass
class TextHit:
    kind: str
    video_id: str
    start_frame: int | None
    end_frame: int | None
    start_time: float | None
    end_time: float | None
    text: str
    score: float

    @property
    def mid_frame(self) -> int | None:
        if self.start_frame is None:
            return None
        if self.end_frame is None:
            return self.start_frame
        return (self.start_frame + self.end_frame) // 2


class TextIndex:
    """Read/write wrapper around the FTS5 database."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: Streamlit reruns a cached SearchEngine (and the
        # TextIndex it lazily opens) from a different thread on each interaction,
        # which the sqlite3 default would reject even though usage here is never
        # concurrent - each call runs to completion before the next one starts.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def clear(self, kind: str | None = None) -> None:
        """Drop rows so a source can be re-indexed without rebuilding everything."""
        if kind:
            self.conn.execute("DELETE FROM segments WHERE kind = ?", (kind,))
        else:
            self.conn.execute("DELETE FROM segments")
        self.conn.execute("DELETE FROM segments_fts")
        self.conn.commit()
        if kind:
            self._reindex_fts()

    def _reindex_fts(self) -> None:
        self.conn.execute("DELETE FROM segments_fts")
        self.conn.execute(
            "INSERT INTO segments_fts(rowid, text, text_ascii) "
            "SELECT id, text, text FROM segments"
        )
        # text_ascii needs folding, which SQL cannot do; rewrite it row by row.
        rows = self.conn.execute("SELECT id, text FROM segments").fetchall()
        self.conn.executemany(
            "UPDATE segments_fts SET text_ascii = ? WHERE rowid = ?",
            [(strip_diacritics(r["text"]), r["id"]) for r in rows],
        )
        self.conn.commit()

    def add_segments(self, rows: Iterable[dict]) -> int:
        """Insert OCR/ASR segments. Each row needs ``kind``, ``video_id``, ``text``."""
        count = 0
        for row in rows:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            cursor = self.conn.execute(
                "INSERT INTO segments(kind, video_id, start_frame, end_frame, start_time, "
                "end_time, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    row["kind"], row["video_id"],
                    row.get("start_frame"), row.get("end_frame"),
                    row.get("start_time"), row.get("end_time"),
                    text,
                ),
            )
            self.conn.execute(
                "INSERT INTO segments_fts(rowid, text, text_ascii) VALUES (?, ?, ?)",
                (cursor.lastrowid, text, strip_diacritics(text)),
            )
            count += 1
        self.conn.commit()
        return count

    def search(self, terms: Sequence[str], kind: str | None = None, limit: int = 500) -> list[TextHit]:
        """BM25 search over both the accented and folded columns."""
        query = escape_fts_query(terms)
        if not query:
            return []
        folded = escape_fts_query([strip_diacritics(term) for term in terms])

        # The join is 1:1 (segments_fts.rowid == segments.id), so no GROUP BY is
        # needed - and bm25() is rejected outright in an aggregate context.
        sql = (
            "SELECT s.kind, s.video_id, s.start_frame, s.end_frame, s.start_time, "
            "       s.end_time, s.text, bm25(segments_fts) AS score "
            "FROM segments_fts JOIN segments s ON s.id = segments_fts.rowid "
            "WHERE segments_fts MATCH ? "
        )
        params: list[object] = [f"text:({query}) OR text_ascii:({folded})"]
        if kind:
            sql += "AND s.kind = ? "
            params.append(kind)
        sql += "ORDER BY score LIMIT ?"
        params.append(limit)

        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            # A malformed *user query* is expected and yields no results. Anything
            # else is our bug and must surface loudly rather than looking like
            # "the OCR branch found nothing" for the rest of the contest.
            if "fts5" in str(exc).lower() and "syntax" in str(exc).lower():
                return []
            raise

        # bm25() returns negative numbers with better matches more negative; flip so
        # callers can treat larger as better like every other branch.
        return [
            TextHit(
                kind=r["kind"], video_id=r["video_id"],
                start_frame=r["start_frame"], end_frame=r["end_frame"],
                start_time=r["start_time"], end_time=r["end_time"],
                text=r["text"], score=-float(r["score"]),
            )
            for r in rows
        ]

    def count(self, kind: str | None = None) -> int:
        if kind:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM segments WHERE kind = ?", (kind,))
        else:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM segments")
        return int(row.fetchone()["n"])

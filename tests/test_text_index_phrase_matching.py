"""Regression test: multi-word ASR/OCR terms must match as an exact phrase,
not degrade into an OR-of-individual-words search.

Found live: searching "chạm mũi chân" (touching toes) was matching any segment
containing just "chân" (foot) on its own - e.g. a "chân gà" (chicken feet)
cooking line - because the old escape_fts_query split every term on whitespace
before OR'ing the pieces together.
"""

from __future__ import annotations

from aic.index.text_index import TextIndex


def test_multiword_term_requires_adjacent_words_not_any_single_word() -> None:
    with TextIndex(":memory:") as index:
        index.add_segments(
            [
                {
                    "kind": "asr", "video_id": "L01_V001",
                    "text": "mọi người cùng chạm mũi chân theo nhịp nhạc",
                },
                {
                    "kind": "asr", "video_id": "L02_V001",
                    "text": "chân gà nấu với sả và ớt",
                },
            ]
        )
        hits = index.search(["chạm mũi chân"], kind="asr")
        assert [h.video_id for h in hits] == ["L01_V001"]


def test_multiple_terms_are_still_ored_against_each_other() -> None:
    with TextIndex(":memory:") as index:
        index.add_segments(
            [
                {"kind": "asr", "video_id": "L01_V001", "text": "xếp hàng tập thể dục"},
                {"kind": "asr", "video_id": "L02_V001", "text": "chạm mũi chân hai tay"},
                {"kind": "asr", "video_id": "L03_V001", "text": "không liên quan gì cả"},
            ]
        )
        hits = index.search(["xếp hàng", "chạm mũi chân"], kind="asr")
        assert {h.video_id for h in hits} == {"L01_V001", "L02_V001"}

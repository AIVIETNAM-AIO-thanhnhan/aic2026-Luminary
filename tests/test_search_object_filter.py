"""Tests for SearchEngine._filter_by_object_counts (aic.query.search)."""

from __future__ import annotations

from aic.index.object_index import ObjectIndex
from aic.query.fusion import FusedHit
from aic.query.search import SearchEngine


def _engine_with_index(index: ObjectIndex) -> SearchEngine:
    engine = SearchEngine.__new__(SearchEngine)
    engine._objects = index
    return engine


def test_drops_hits_that_fail_the_count_constraint() -> None:
    with ObjectIndex(":memory:") as index:
        index.add_detections(
            [{"gid": 1, "video_id": "L01_V001", "label": "Person", "score": 0.9}] * 6
            + [{"gid": 2, "video_id": "L01_V002", "label": "Person", "score": 0.9}] * 2
        )
        engine = _engine_with_index(index)
        fused = [FusedHit(gid=1, score=0.5), FusedHit(gid=2, score=0.9)]

        result = engine._filter_by_object_counts(fused, {"Person": 5})
        assert [h.gid for h in result] == [1]


def test_falls_back_to_a_smaller_subset_when_the_full_set_matches_nothing() -> None:
    with ObjectIndex(":memory:") as index:
        # gid 1 has 6 people but no glasses at all - the full {Person>=5, Glasses>=1}
        # constraint matches nothing, so the filter should fall back to Person alone
        # rather than return an empty result.
        index.add_detections([{"gid": 1, "video_id": "L01_V001", "label": "Person", "score": 0.9}] * 6)
        engine = _engine_with_index(index)
        fused = [FusedHit(gid=1, score=0.5)]

        result = engine._filter_by_object_counts(fused, {"Person": 5, "Glasses": 1})
        assert [h.gid for h in result] == [1]


def test_fallback_prefers_person_over_a_later_attribute() -> None:
    with ObjectIndex(":memory:") as index:
        # gid 1: 6 people, 0 glasses. gid 2: 1 person, 1 glasses.
        # {Person>=5, Glasses>=1} together matches nothing, so the filter must fall
        # back to a smaller subset - and it must keep "Person" (inserted first,
        # the query's primary countable subject) rather than degrade to "Glasses"
        # alone, which would hand back gid 2 despite it having almost no people.
        index.add_detections(
            [{"gid": 1, "video_id": "L01_V001", "label": "Person", "score": 0.9}] * 6
            + [{"gid": 2, "video_id": "L01_V002", "label": "Person", "score": 0.9}]
            + [{"gid": 2, "video_id": "L01_V002", "label": "Glasses", "score": 0.9}]
        )
        engine = _engine_with_index(index)
        fused = [FusedHit(gid=1, score=0.5), FusedHit(gid=2, score=0.9)]

        result = engine._filter_by_object_counts(fused, {"Person": 5, "Glasses": 1})
        assert [h.gid for h in result] == [1]


def test_no_object_index_leaves_fused_hits_unchanged() -> None:
    engine = SearchEngine.__new__(SearchEngine)
    engine._objects = None
    # objects_db points at a path that does not exist, so the `objects` property
    # falls through to None exactly as it does when `aic build-objects` never ran.
    engine.config = type("FakeConfig", (), {"derived_path": lambda self, key: "/nonexistent/objects.sqlite"})()
    fused = [FusedHit(gid=1, score=0.5)]
    assert engine._filter_by_object_counts(fused, {"Person": 5}) == fused


def test_no_min_counts_leaves_fused_hits_unchanged() -> None:
    with ObjectIndex(":memory:") as index:
        engine = _engine_with_index(index)
        fused = [FusedHit(gid=1, score=0.5)]
        assert engine._filter_by_object_counts(fused, {}) == fused

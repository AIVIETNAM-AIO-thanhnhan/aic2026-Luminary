"""Tests for the per-frame object detection index."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from aic.index.object_index import ObjectIndex, build_object_index


def test_add_and_search_returns_best_score_per_frame(tmp_path) -> None:
    with ObjectIndex(tmp_path / "objects.sqlite") as index:
        index.add_detections(
            [
                {"gid": 1, "video_id": "L01_V001", "label": "Lion", "score": 0.6},
                {"gid": 1, "video_id": "L01_V001", "label": "Lion", "score": 0.9},
                {"gid": 2, "video_id": "L01_V002", "label": "Car", "score": 0.5},
            ]
        )
        hits = index.search(["Lion"])
        assert len(hits) == 1
        assert hits[0].gid == 1
        assert hits[0].score == pytest.approx(0.9)


def test_search_is_case_insensitive() -> None:
    with ObjectIndex(":memory:") as index:
        index.add_detections([{"gid": 1, "video_id": "L01_V001", "label": "Car", "score": 0.7}])
        assert [h.gid for h in index.search(["car"])] == [1]
        assert [h.gid for h in index.search(["CAR"])] == [1]


def test_search_matches_any_of_several_labels() -> None:
    with ObjectIndex(":memory:") as index:
        index.add_detections(
            [
                {"gid": 1, "video_id": "L01_V001", "label": "Lion", "score": 0.7},
                {"gid": 2, "video_id": "L01_V002", "label": "Umbrella", "score": 0.8},
                {"gid": 3, "video_id": "L01_V003", "label": "Car", "score": 0.6},
            ]
        )
        gids = {h.gid for h in index.search(["Lion", "Umbrella"])}
        assert gids == {1, 2}


def test_search_with_no_labels_or_no_matches_returns_empty() -> None:
    with ObjectIndex(":memory:") as index:
        index.add_detections([{"gid": 1, "video_id": "L01_V001", "label": "Car", "score": 0.7}])
        assert index.search([]) == []
        assert index.search(["Airplane"]) == []


def test_is_discriminative_false_for_a_ubiquitous_label() -> None:
    with ObjectIndex(":memory:") as index:
        # "Person" in 4/5 frames, "Lion" in 1/5 - a 3% cutoff should keep Lion
        # and drop Person, matching the real corpus's ~39% vs ~6-frame split.
        for gid in range(4):
            index.add_detections([{"gid": gid, "video_id": "L01_V001", "label": "Person", "score": 0.9}])
        index.add_detections([{"gid": 4, "video_id": "L01_V001", "label": "Lion", "score": 0.9}])

        assert index.is_discriminative("Lion", max_fraction=0.5) is True
        assert index.is_discriminative("Person", max_fraction=0.5) is False


def test_is_discriminative_false_for_a_label_with_no_detections() -> None:
    with ObjectIndex(":memory:") as index:
        index.add_detections([{"gid": 0, "video_id": "L01_V001", "label": "Lion", "score": 0.9}])
        assert index.is_discriminative("Airplane") is False


def test_search_by_min_count_filters_and_ranks_by_instance_count() -> None:
    with ObjectIndex(":memory:") as index:
        # gid 1: 3 people, gid 2: 6 people, gid 3: 1 person.
        index.add_detections(
            [{"gid": 1, "video_id": "L01_V001", "label": "Person", "score": 0.9}] * 3
            + [{"gid": 2, "video_id": "L01_V002", "label": "Person", "score": 0.9}] * 6
            + [{"gid": 3, "video_id": "L01_V003", "label": "Person", "score": 0.9}]
        )
        hits = index.search_by_min_count("Person", min_count=5)
        assert [h.gid for h in hits] == [2]

        hits_low_bar = index.search_by_min_count("Person", min_count=1)
        assert [h.gid for h in hits_low_bar] == [2, 1, 3]


def test_person_count_includes_gendered_labels() -> None:
    # Found live: the OpenImages detector labels a human as EITHER "Person" or a
    # more specific "Man"/"Woman"/"Boy"/"Girl" - never both - so a frame full of
    # people the detector happened to tag by gender/age had zero "Person"
    # detections and was invisible to every "Person"-count check. 17.5% of this
    # corpus's frames were affected.
    with ObjectIndex(":memory:") as index:
        index.add_detections(
            [{"gid": 1, "video_id": "L01_V001", "label": "Man", "score": 0.9}] * 3
            + [{"gid": 1, "video_id": "L01_V001", "label": "Boy", "score": 0.9}] * 2
            + [{"gid": 2, "video_id": "L01_V002", "label": "Woman", "score": 0.9}] * 2
        )
        assert index.counts_by_gid("Person") == {1: 5, 2: 2}
        assert [h.gid for h in index.search_by_min_count("Person", min_count=5)] == [1]
        assert index.search_by_target_count("Person", target_count=2)[0].gid == 2


def test_gendered_label_counting_does_not_apply_to_other_labels() -> None:
    # The "Person" -> gendered-label expansion must not leak into an unrelated
    # label search (e.g. "Man" as in "Spider-Man" merchandise is nonsensical
    # here, but the point stands generally: only "person" gets the synonym set).
    with ObjectIndex(":memory:") as index:
        index.add_detections([{"gid": 1, "video_id": "L01_V001", "label": "Man", "score": 0.9}] * 3)
        assert index.counts_by_gid("Hat") == {}


def test_search_by_target_count_ranks_exact_match_first() -> None:
    with ObjectIndex(":memory:") as index:
        # gid 1: 1 glasses-wearer (the true "only one" match), gid 2: 5, gid 3: 3.
        index.add_detections(
            [{"gid": 1, "video_id": "L01_V001", "label": "Glasses", "score": 0.9}]
            + [{"gid": 2, "video_id": "L01_V002", "label": "Glasses", "score": 0.9}] * 5
            + [{"gid": 3, "video_id": "L01_V003", "label": "Glasses", "score": 0.9}] * 3
        )
        hits = index.search_by_target_count("Glasses", target_count=1, limit=10)
        # Exact match (gid 1) must rank first even though gid 2 has more instances -
        # for an "exactly N" constraint, more is a worse match, not a better one.
        assert [h.gid for h in hits] == [1, 3, 2]


def test_search_by_min_count_with_no_matches_returns_empty() -> None:
    with ObjectIndex(":memory:") as index:
        index.add_detections([{"gid": 1, "video_id": "L01_V001", "label": "Person", "score": 0.9}])
        assert index.search_by_min_count("Person", min_count=5) == []
        assert index.search_by_min_count("Airplane", min_count=1) == []


def test_count() -> None:
    with ObjectIndex(":memory:") as index:
        assert index.count() == 0
        index.add_detections([{"gid": 1, "video_id": "L01_V001", "label": "Car", "score": 0.7}])
        assert index.count() == 1


def _write_detection(path, labels_scores) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels, scores = zip(*labels_scores) if labels_scores else ((), ())
    path.write_text(
        json.dumps({"detection_class_entities": list(labels), "detection_scores": list(scores)}),
        encoding="utf-8",
    )


def test_build_object_index_applies_score_threshold(tmp_path) -> None:
    objects_dir = tmp_path / "objects"
    _write_detection(
        objects_dir / "L01_V001" / "0000.json",
        [("Lion", "0.9"), ("Car", "0.1")],
    )
    catalog = pd.DataFrame(
        [{"gid": 0, "video_id": "L01_V001", "path": str(tmp_path / "keyframes/L01_V001/0000.jpg")}]
    )

    index_path = tmp_path / "objects.sqlite"
    written = build_object_index(catalog, objects_dir, index_path, score_threshold=0.4)

    assert written == 1  # "Car" at 0.1 is below threshold
    with ObjectIndex(index_path) as index:
        assert index.count() == 1
        assert [h.label for h in index.search(["Lion"])] == ["Lion"]


def test_build_object_index_skips_frames_with_no_detection_file(tmp_path) -> None:
    objects_dir = tmp_path / "objects"
    catalog = pd.DataFrame(
        [{"gid": 0, "video_id": "L01_V001", "path": str(tmp_path / "keyframes/L01_V001/0000.jpg")}]
    )
    index_path = tmp_path / "objects.sqlite"
    written = build_object_index(catalog, objects_dir, index_path)
    assert written == 0

"""Tests for Vietnamese person-count extraction (aic.query.expand)."""

from __future__ import annotations

import aic.query.expand as expand_mod
from aic.query.expand import _extract_person_count, _rule_based


def test_more_than_n_people_uses_n_as_the_search_floor() -> None:
    # "hơn 5" (more than 5) literally means a true count of 6+, but the search
    # floor stays at 5: a detector is far more likely to undercount an occluded
    # person than invent one, so requiring 6+ risks excluding the true video
    # outright if it only detected 5. RRF still ranks higher counts first.
    query = "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục."
    assert _extract_person_count(query) == {"Person": 5}


def test_plain_n_people_is_treated_as_a_lower_bound() -> None:
    assert _extract_person_count("5 người đang đứng") == {"Person": 5}


def test_no_count_mention_returns_empty() -> None:
    assert _extract_person_count("một người đàn ông đang nói chuyện") == {}


def test_rule_based_fallback_populates_min_object_counts() -> None:
    expanded = _rule_based("hơn 10 người tham gia diễu hành", num_expansions=3)
    assert expanded.min_object_counts == {"Person": 10}


def test_hat_count_extracted_from_word_number(monkeypatch) -> None:
    # EXTRACT_ATTRIBUTE_COUNTS is off by default (see its docstring) - this
    # verifies the underlying regex/mechanism still works when re-enabled,
    # independent of that toggle.
    monkeypatch.setattr(expand_mod, "EXTRACT_ATTRIBUTE_COUNTS", True)
    # "ba người" here genuinely describes 3 people, so a Person floor alongside
    # the Hat count is correct, not a false positive - see
    # test_word_number_headcount_beats_a_later_digit_subset for the case this
    # word-number support exists to fix (a subset digit shadowing the real total).
    assert _extract_person_count("ba người đội nón màu đỏ") == {"Person": 3, "Hat": 3}


def test_word_number_headcount_beats_a_later_digit_subset() -> None:
    # Found live: "Ba người đang đi bộ..." states the true group size (3) in
    # word form, then a later clause restates a *subset* with a digit ("có 2
    # người cầm dù" - 2 of them hold umbrellas). Excluding word-numbers entirely
    # meant the digit-only regex skipped "Ba người" and grabbed "2 người"
    # instead - a materially wrong count, not just a redundant one.
    query = "Ba người đang đi bộ xuống một con dốc trong cơn mưa, có 2 người cầm dù."
    assert _extract_person_count(query) == {"Person": 3}


def test_glasses_count_extracted_even_at_one(monkeypatch) -> None:
    monkeypatch.setattr(expand_mod, "EXTRACT_ATTRIBUTE_COUNTS", True)
    assert _extract_person_count("trong nhóm chỉ có một người đeo kính") == {"Glasses": 1}


def test_attribute_counts_are_off_by_default() -> None:
    # Temporarily disabled at the user's request - extensive checking (per-frame,
    # per-video, broadened taxonomy, pixel-color verification) never found a real
    # match for a "1 glasses + 3 red hats" constraint, only coincidental ones.
    assert _extract_person_count("trong nhóm chỉ có một người đeo kính") == {}


def test_full_query_extracts_person_and_both_attribute_counts(monkeypatch) -> None:
    monkeypatch.setattr(expand_mod, "EXTRACT_ATTRIBUTE_COUNTS", True)
    query = (
        "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện "
        "động tác hai tay chạm mũi chân. Trong nhóm chỉ có một người đeo kính và "
        "ba người đội nón có màu đỏ."
    )
    assert _extract_person_count(query) == {"Person": 5, "Glasses": 1, "Hat": 3}


def test_full_query_extracts_only_person_when_attributes_disabled() -> None:
    query = (
        "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện "
        "động tác hai tay chạm mũi chân. Trong nhóm chỉ có một người đeo kính và "
        "ba người đội nón có màu đỏ."
    )
    assert _extract_person_count(query) == {"Person": 5}

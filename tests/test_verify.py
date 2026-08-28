"""Tests for frame-alignment verification.

``extract_frame``/``probe_video`` shell out to ffmpeg/ffprobe, so
:func:`verify_frame_alignment` is exercised with ``extract_frame`` monkeypatched
to a deterministic stand-in rather than decoding a real video — the thing under
test is the offset-voting/PASS-FAIL logic, not ffmpeg itself.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from aic.data.verify import (
    FrameSample,
    VerifyReport,
    _find_video,
    _mean_absolute_error,
    _parse_frame_rate,
    verify_frame_alignment,
)

WHITE = (255, 255, 255)


def _catalog_row(gid, video_id, path, frame_idx) -> dict:
    return {
        "gid": gid, "video_id": video_id, "path": str(path),
        "frame_idx": float(frame_idx), "pts_time": 0.0, "source": "btc_iframe",
    }


@pytest.fixture
def videos_dir(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    for video_id in ("L01_V001", "L02_V001"):
        (videos / f"{video_id}.mp4").write_bytes(b"not a real video, just needs to exist")
    return videos


def _make_scenario(tmp_path, videos_dir, video_ids, true_offset):
    """Build a catalog + stub so the "true" match sits at ``frame_idx + true_offset``."""
    keyframes: dict[str, Path] = {}
    rows = []
    correct: dict[str, int] = {}
    for i, video_id in enumerate(video_ids):
        frame_idx = 100 + i * 50
        stored = tmp_path / f"{video_id}.jpg"
        Image.new("RGB", (4, 4), (10 * (i + 1), 20 * (i + 1), 30 * (i + 1))).save(stored)
        keyframes[video_id] = stored
        correct[video_id] = frame_idx + true_offset
        rows.append(_catalog_row(i, video_id, stored, frame_idx))

    def extract(video_path, frame_idx):
        video_id = Path(video_path).stem
        if frame_idx == correct[video_id]:
            return Image.open(keyframes[video_id]).convert("RGB")
        return Image.new("RGB", (4, 4), WHITE)

    return pd.DataFrame(rows), extract


def test_parse_frame_rate_handles_fractions_and_missing_values() -> None:
    assert _parse_frame_rate("25/1") == 25.0
    assert _parse_frame_rate("30000/1001") == pytest.approx(29.97, abs=0.01)
    assert _parse_frame_rate("25") == 25.0
    assert _parse_frame_rate("0/0") is None
    assert _parse_frame_rate(None) is None


def test_mean_absolute_error_is_zero_for_identical_images() -> None:
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    assert _mean_absolute_error(image, image.copy()) == 0.0


def test_mean_absolute_error_matches_the_constant_channel_delta() -> None:
    a = Image.new("RGB", (4, 4), (10, 10, 10))
    b = Image.new("RGB", (4, 4), (30, 30, 30))
    assert _mean_absolute_error(a, b) == pytest.approx(20.0)


def test_mean_absolute_error_resizes_mismatched_images_instead_of_crashing() -> None:
    a = Image.new("RGB", (4, 4), (0, 0, 0))
    b = Image.new("RGB", (8, 8), (0, 0, 0))
    assert _mean_absolute_error(a, b) == pytest.approx(0.0)


def test_find_video_matches_by_stem_directly_under_the_dir(videos_dir) -> None:
    found = _find_video(videos_dir, "L01_V001")
    assert found is not None
    assert found.name == "L01_V001.mp4"


def test_find_video_searches_nested_directories(tmp_path) -> None:
    videos = tmp_path / "videos"
    (videos / "batch1").mkdir(parents=True)
    (videos / "batch1" / "L09_V009.mp4").write_bytes(b"x")
    found = _find_video(videos, "L09_V009")
    assert found == videos / "batch1" / "L09_V009.mp4"


def test_find_video_returns_none_when_absent_or_dir_missing(tmp_path) -> None:
    assert _find_video(tmp_path / "videos", "L01_V001") is None
    (tmp_path / "videos").mkdir()
    assert _find_video(tmp_path / "videos", "L99_V999") is None


def test_verify_frame_alignment_passes_when_frame_idx_is_correct(tmp_path, videos_dir, monkeypatch) -> None:
    catalog, extract = _make_scenario(tmp_path, videos_dir, ["L01_V001", "L02_V001"], true_offset=0)
    monkeypatch.setattr("aic.data.verify.extract_frame", extract)

    report = verify_frame_alignment(
        catalog, videos_dir, sample_size=10, mae_threshold=1.0, offsets_to_try=(0, -1, 1)
    )

    assert report.ok
    assert report.recommended_offset == 0
    assert len(report.samples) == 2
    assert all(s.best_offset == 0 and s.best_mae == pytest.approx(0.0) for s in report.samples)
    assert "PASS" in report.summary()


def test_verify_frame_alignment_detects_a_systematic_offset(tmp_path, videos_dir, monkeypatch) -> None:
    catalog, extract = _make_scenario(tmp_path, videos_dir, ["L01_V001", "L02_V001"], true_offset=-1)
    monkeypatch.setattr("aic.data.verify.extract_frame", extract)

    report = verify_frame_alignment(
        catalog, videos_dir, sample_size=10, mae_threshold=1.0, offsets_to_try=(0, -1, 1)
    )

    assert not report.ok
    assert report.recommended_offset == -1
    assert all(s.best_offset == -1 for s in report.samples)
    summary = report.summary()
    assert "FAIL" in summary
    assert "-1" in summary


def test_verify_frame_alignment_majority_offset_wins_over_one_outlier(tmp_path, videos_dir, monkeypatch) -> None:
    video_ids = ["L01_V001", "L02_V001", "L03_V003"]
    (videos_dir / "L03_V003.mp4").write_bytes(b"x")

    keyframes: dict[str, Path] = {}
    rows = []
    correct: dict[str, int] = {}
    for i, video_id in enumerate(video_ids):
        frame_idx = 100 + i * 50
        stored = tmp_path / f"{video_id}.jpg"
        Image.new("RGB", (4, 4), (10 * (i + 1), 0, 0)).save(stored)
        keyframes[video_id] = stored
        # Two videos agree the real frame sits one earlier; one is a noisy outlier at 0.
        correct[video_id] = frame_idx + (0 if video_id == "L03_V003" else -1)
        rows.append(_catalog_row(i, video_id, stored, frame_idx))

    def extract(video_path, frame_idx):
        video_id = Path(video_path).stem
        if frame_idx == correct[video_id]:
            return Image.open(keyframes[video_id]).convert("RGB")
        return Image.new("RGB", (4, 4), WHITE)

    monkeypatch.setattr("aic.data.verify.extract_frame", extract)
    catalog = pd.DataFrame(rows)

    report = verify_frame_alignment(
        catalog, videos_dir, sample_size=10, mae_threshold=1.0, offsets_to_try=(0, -1, 1)
    )
    assert report.recommended_offset == -1
    assert not report.ok


def test_verify_frame_alignment_skips_videos_not_found_on_disk(tmp_path, videos_dir, monkeypatch) -> None:
    catalog, extract = _make_scenario(tmp_path, videos_dir, ["L01_V001"], true_offset=0)
    missing_row = _catalog_row(1, "L99_V999", tmp_path / "L99_V999.jpg", 100)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(missing_row["path"])
    catalog = pd.concat([catalog, pd.DataFrame([missing_row])], ignore_index=True)
    monkeypatch.setattr("aic.data.verify.extract_frame", extract)

    report = verify_frame_alignment(catalog, videos_dir, sample_size=10, offsets_to_try=(0, -1, 1))

    assert report.skipped_videos == ["L99_V999"]
    assert [s.video_id for s in report.samples] == ["L01_V001"]


def test_verify_frame_alignment_fails_cleanly_with_no_frame_idx_data(tmp_path, videos_dir) -> None:
    catalog = pd.DataFrame(
        [_catalog_row(0, "L01_V001", tmp_path / "x.jpg", float("nan"))]
    )
    catalog["frame_idx"] = float("nan")

    report = verify_frame_alignment(catalog, videos_dir)

    assert not report.ok
    assert report.samples == []
    assert "FAIL" in report.summary()


def test_verify_report_summary_mentions_skipped_videos() -> None:
    report = VerifyReport(
        samples=[FrameSample(video_id="L01_V001", frame_idx=100, best_offset=0, best_mae=0.0)],
        mae_threshold=8.0,
        recommended_offset=0,
        ok=True,
        skipped_videos=["L02_V001"],
    )
    summary = report.summary()
    assert "PASS" in summary
    assert "1" in summary and "skipped" in summary

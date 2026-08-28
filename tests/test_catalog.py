"""Tests for the catalog: keyframe scanning, frame_idx joining, and I/O."""

from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image

from aic.data.catalog import build_catalog, catalog_summary, load_catalog, save_catalog


def _write_keyframe(path, color=(0, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path)


def test_build_catalog_joins_keyframe_number_to_frame_idx(tmp_path) -> None:
    keyframes = tmp_path / "keyframes"
    metadata = tmp_path / "metadata"
    metadata.mkdir()

    for frame in range(3):
        _write_keyframe(keyframes / "L01_V001" / f"{frame:04d}.jpg")
    pd.DataFrame(
        {"keyframe": ["0000", "0001", "0002"], "frame_idx": [0, 25, 50], "pts_time": [0.0, 1.0, 2.0]}
    ).to_csv(metadata / "L01_V001.csv", index=False)

    catalog = build_catalog(keyframes, metadata, repo_root=tmp_path)

    assert len(catalog) == 3
    assert list(catalog["gid"]) == [0, 1, 2]
    assert list(catalog.sort_values("path")["frame_idx"]) == [0, 25, 50]
    assert (catalog["source"] == "btc_iframe").all()
    # Stored relative to repo_root, not as an absolute filesystem path.
    assert all(not p.startswith("/") for p in catalog["path"])


def test_build_catalog_uses_the_given_source_label(tmp_path) -> None:
    keyframes = tmp_path / "keyframes"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    _write_keyframe(keyframes / "L01_V001" / "0000.jpg")

    catalog = build_catalog(keyframes, metadata, source="transnet", repo_root=tmp_path)
    assert list(catalog["source"]) == ["transnet"]


def test_build_catalog_marks_missing_map_as_nan_without_dropping_the_frame(tmp_path) -> None:
    keyframes = tmp_path / "keyframes"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    _write_keyframe(keyframes / "L02_V002" / "0000.jpg")  # no matching .csv at all

    catalog = build_catalog(keyframes, metadata, repo_root=tmp_path)

    assert len(catalog) == 1
    assert catalog["frame_idx"].isna().all()

    summary = catalog_summary(catalog)
    assert summary["missing_frame_idx"] == 1
    assert summary["videos_missing_map"] == ["L02_V002"]


def test_build_catalog_leaves_unmatched_keyframes_as_nan(tmp_path) -> None:
    keyframes = tmp_path / "keyframes"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    _write_keyframe(keyframes / "L03_V003" / "0000.jpg")
    _write_keyframe(keyframes / "L03_V003" / "0001.jpg")
    # Map only covers keyframe 0; keyframe 1 has no row.
    pd.DataFrame({"keyframe": ["0000"], "frame_idx": [10]}).to_csv(metadata / "L03_V003.csv", index=False)

    catalog = build_catalog(keyframes, metadata, repo_root=tmp_path)
    by_path = catalog.sort_values("path")
    assert list(by_path["frame_idx"].isna()) == [False, True]


def test_build_catalog_ignores_non_image_and_non_numeric_files(tmp_path) -> None:
    keyframes = tmp_path / "keyframes"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    video_dir = keyframes / "L04_V004"
    video_dir.mkdir(parents=True)
    _write_keyframe(video_dir / "0000.jpg")
    (video_dir / "notes.txt").write_text("not a keyframe")
    (video_dir / "thumb.jpg").write_bytes(b"")  # non-numeric stem, same suffix

    catalog = build_catalog(keyframes, metadata, repo_root=tmp_path)
    assert len(catalog) == 1


def test_build_catalog_orders_rows_by_video_id_then_path(tmp_path) -> None:
    keyframes = tmp_path / "keyframes"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for video_id in ("L02_V001", "L01_V001"):
        for frame in (1, 0):
            _write_keyframe(keyframes / video_id / f"{frame:04d}.jpg")

    catalog = build_catalog(keyframes, metadata, repo_root=tmp_path)
    assert list(catalog["video_id"]) == ["L01_V001", "L01_V001", "L02_V001", "L02_V001"]
    assert list(catalog["gid"]) == [0, 1, 2, 3]


def test_build_catalog_raises_when_keyframes_dir_is_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        build_catalog(tmp_path / "nope", tmp_path / "metadata")


def test_build_catalog_raises_on_map_without_keyframe_column(tmp_path) -> None:
    keyframes = tmp_path / "keyframes"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    _write_keyframe(keyframes / "L05_V005" / "0000.jpg")
    pd.DataFrame({"frame_idx": [0]}).to_csv(metadata / "L05_V005.csv", index=False)

    with pytest.raises(ValueError, match="keyframe"):
        build_catalog(keyframes, metadata, repo_root=tmp_path)


def test_catalog_summary_counts_sources_and_videos(tmp_path) -> None:
    keyframes = tmp_path / "keyframes"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for video_id in ("L01_V001", "L02_V001"):
        _write_keyframe(keyframes / video_id / "0000.jpg")
        pd.DataFrame({"keyframe": ["0000"], "frame_idx": [0]}).to_csv(metadata / f"{video_id}.csv", index=False)

    catalog = build_catalog(keyframes, metadata, source="btc_iframe", repo_root=tmp_path)
    summary = catalog_summary(catalog)

    assert summary["frames"] == 2
    assert summary["videos"] == 2
    assert summary["missing_frame_idx"] == 0
    assert summary["videos_missing_map"] == []
    assert summary["sources"] == {"btc_iframe": 2}


def test_save_and_load_catalog_round_trips(tmp_path) -> None:
    keyframes = tmp_path / "keyframes"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    _write_keyframe(keyframes / "L01_V001" / "0000.jpg")
    pd.DataFrame({"keyframe": ["0000"], "frame_idx": [7]}).to_csv(metadata / "L01_V001.csv", index=False)
    catalog = build_catalog(keyframes, metadata, repo_root=tmp_path)

    out_path = tmp_path / "derived" / "catalog.parquet"
    save_catalog(catalog, out_path)
    assert out_path.exists()

    reloaded = load_catalog(out_path)
    pd.testing.assert_frame_equal(reloaded, catalog)


def test_load_catalog_raises_a_helpful_error_when_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="build-catalog"):
        load_catalog(tmp_path / "missing.parquet")

"""Tests for the gid<->embedding-row invariant that the whole index depends on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aic.index.embed import embed_catalog, load_images, pending_shards, plan_shards
from aic.index.vector_index import load_embeddings


def _catalog(n: int, image_dir) -> pd.DataFrame:
    """Build a catalog of solid-colour images whose red channel encodes the gid.

    PNG, not JPEG: the marker value has to survive the round trip exactly for the
    row-alignment assertions to mean anything.
    """
    rows = []
    for gid in range(n):
        path = image_dir / f"{gid:04d}.png"
        Image.new("RGB", (8, 8), color=(gid % 256, 0, 0)).save(path)
        rows.append(
            {
                "gid": gid,
                "video_id": f"L{gid // 10:02d}_V001",
                "frame_idx": gid * 25,
                "pts_time": gid * 1.0,
                "source": "btc_iframe",
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def _fake_encoder(dim: int):
    """Encode each image to a vector carrying its red channel, so rows are traceable."""

    def encode(images):
        return np.array(
            [[float(np.asarray(img)[0, 0, 0])] * dim for img in images], dtype=np.float32
        )

    return encode


def test_plan_shards_covers_every_row_exactly_once(tmp_path) -> None:
    shards = plan_shards(250, shard_size=100, output_dir=tmp_path)
    assert [(s.start_gid, s.end_gid) for s in shards] == [(0, 100), (100, 200), (200, 250)]
    assert sum(s.size for s in shards) == 250


def test_shard_names_sort_in_gid_order(tmp_path) -> None:
    """load_embeddings concatenates by sorted filename, so the two orders must agree."""
    shards = plan_shards(30_000, shard_size=1_000, output_dir=tmp_path)
    names = [s.path.name for s in shards]
    assert names == sorted(names)


def test_pending_shards_enables_resume(tmp_path) -> None:
    shards = plan_shards(300, shard_size=100, output_dir=tmp_path)
    np.save(shards[0].path, np.zeros((100, 4), dtype=np.float32))
    assert [s.start_gid for s in pending_shards(shards)] == [100, 200]


def test_embed_catalog_preserves_gid_row_alignment(tmp_path) -> None:
    images_dir = tmp_path / "img"
    images_dir.mkdir()
    catalog = _catalog(25, images_dir)

    out = tmp_path / "emb"
    embed_catalog(catalog, _fake_encoder(4), out, dim=4, batch_size=4, shard_size=10)

    matrix = load_embeddings(out, expected_rows=25)
    assert matrix.shape == (25, 4)
    # Row i must carry the marker planted in image i.
    for gid in range(25):
        assert matrix[gid, 0] == pytest.approx(float(gid % 256))


def test_embed_catalog_resumes_without_reencoding(tmp_path) -> None:
    images_dir = tmp_path / "img"
    images_dir.mkdir()
    catalog = _catalog(20, images_dir)
    out = tmp_path / "emb"

    embed_catalog(catalog, _fake_encoder(4), out, dim=4, batch_size=4, shard_size=10)

    calls = {"n": 0}

    def counting_encoder(images):
        calls["n"] += 1
        return _fake_encoder(4)(images)

    embed_catalog(catalog, counting_encoder, out, dim=4, batch_size=4, shard_size=10)
    assert calls["n"] == 0  # everything already on disk


def test_unreadable_image_zeroes_its_row_without_shifting_others(tmp_path) -> None:
    images_dir = tmp_path / "img"
    images_dir.mkdir()
    catalog = _catalog(6, images_dir)
    # Corrupt frame 2; frames 3..5 must keep their own vectors.
    (images_dir / "0002.png").write_bytes(b"not an image")

    out = tmp_path / "emb"
    embed_catalog(catalog, _fake_encoder(4), out, dim=4, batch_size=3, shard_size=6)
    matrix = load_embeddings(out, expected_rows=6)

    assert np.all(matrix[2] == 0.0)
    for gid in (0, 1, 3, 4, 5):
        assert matrix[gid, 0] == pytest.approx(float(gid))


def test_embed_catalog_rejects_a_gappy_catalog(tmp_path) -> None:
    images_dir = tmp_path / "img"
    images_dir.mkdir()
    catalog = _catalog(5, images_dir)
    catalog.loc[3, "gid"] = 99  # break the 0..n-1 invariant

    with pytest.raises(ValueError, match="gids must be 0..n-1"):
        embed_catalog(catalog, _fake_encoder(4), tmp_path / "emb", dim=4)


def test_load_embeddings_rejects_a_catalog_size_mismatch(tmp_path) -> None:
    np.save(tmp_path / "a.npy", np.zeros((10, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="embedding/catalog mismatch"):
        load_embeddings(tmp_path, expected_rows=11)


def test_load_images_reports_positions_not_just_images(tmp_path) -> None:
    good = tmp_path / "ok.jpg"
    Image.new("RGB", (4, 4)).save(good)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope")

    images, positions = load_images([good, bad, good])
    assert len(images) == 2
    assert positions == [0, 2]

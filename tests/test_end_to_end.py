"""End-to-end test over a synthetic corpus.

Builds a tiny fake dataset, runs catalog -> embeddings -> FAISS -> search ->
policy -> submission file, and scores the result with the official metric. Uses a
stub text encoder so no model download is needed, which keeps the wiring under
test rather than the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

faiss = pytest.importorskip("faiss", reason="faiss-cpu not installed")

from aic.config import Config
from aic.data.catalog import (
    build_catalog,
    catalog_summary,
    save_catalog,
)
from aic.eval.metric import KISAnswer, KISGroundTruth, final_score, kis_r_score
from aic.index.text_index import TextIndex, strip_diacritics
from aic.index.vector_index import (
    build_index,
    load_embeddings,
    save_index,
    search_pooled,
)
from aic.query.search import SearchEngine
from aic.submit.policy import Candidate, build_kis_answers
from aic.submit.writer import write_submission

N_VIDEOS, FRAMES_PER_VIDEO, DIM = 4, 12, 8


@pytest.fixture
def corpus(tmp_path):
    """A fake dataset: 4 videos x 12 keyframes, with a frame-index map per video."""
    keyframes = tmp_path / "raw" / "keyframes"
    metadata = tmp_path / "raw" / "metadata"
    keyframes.mkdir(parents=True)
    metadata.mkdir(parents=True)

    for video in range(N_VIDEOS):
        video_id = f"L{video:02d}_V001"
        (keyframes / video_id).mkdir()
        rows = []
        for frame in range(FRAMES_PER_VIDEO):
            Image.new("RGB", (16, 16), (video * 60 % 256, frame * 20 % 256, 0)).save(
                keyframes / video_id / f"{frame:04d}.jpg"
            )
            # Keyframes sit 25 frames (1 s at 25 fps) apart, like real I-frames.
            rows.append({"keyframe": f"{frame:04d}", "frame_idx": frame * 25, "pts_time": frame * 1.0})
        pd.DataFrame(rows).to_csv(metadata / f"{video_id}.csv", index=False)

    return {"root": tmp_path, "keyframes": keyframes, "metadata": metadata}


def _config(corpus) -> Config:
    root = corpus["root"]
    return Config(
        raw={
            "paths": {
                "raw": {
                    "videos": str(root / "raw/videos"),
                    "keyframes": str(corpus["keyframes"]),
                    "objects": str(root / "raw/objects"),
                    "clip_features": str(root / "raw/clip"),
                    "metadata": str(corpus["metadata"]),
                },
                "derived": {
                    "root": str(root / "derived"),
                    "catalog": str(root / "derived/catalog.parquet"),
                    "embeddings": str(root / "derived/embeddings"),
                    "asr": str(root / "derived/asr.parquet"),
                    "ocr": str(root / "derived/ocr.parquet"),
                    "index": str(root / "derived/index"),
                    "text_db": str(root / "derived/text.sqlite"),
                    "objects_db": str(root / "derived/objects.sqlite"),
                },
                "submissions": str(root / "submissions"),
            },
            "embedding": {
                "active": "test",
                "spaces": {"test": {"dim": DIM, "model_id": "stub", "source": str(root / "derived/embeddings")}},
            },
            "fusion": {
                "rrf_k": 60,
                "weights": {"visual": 1.0, "ocr": 1.0, "asr": 0.8},
                "candidates_per_branch": 100,
            },
            "submission": {
                "max_answers": 100, "diversify_head": 5,
                "frames_per_shot": 3, "frame_spread": 8,
                "trake_jitter": [0, -4, 4],
            },
            "trake": {"window_seconds": 2.0, "max_candidate_videos": 3, "frame_step": 1},
            "query": {"translate_to_english": False, "llm_model": "stub", "num_expansions": 2},
            "verify": {"sample_size": 5, "mae_threshold": 8.0, "offsets_to_try": [0, -1, 1]},
        }
    )


def test_catalog_maps_keyframes_to_video_frame_numbers(corpus) -> None:
    catalog = build_catalog(corpus["keyframes"], corpus["metadata"], repo_root=corpus["root"])

    assert len(catalog) == N_VIDEOS * FRAMES_PER_VIDEO
    assert list(catalog["gid"]) == list(range(len(catalog)))  # gid is the row index

    summary = catalog_summary(catalog)
    assert summary["videos"] == N_VIDEOS
    assert summary["missing_frame_idx"] == 0

    first = catalog[catalog["video_id"] == "L00_V001"].sort_values("path")
    assert list(first["frame_idx"])[:3] == [0, 25, 50]


def test_full_pipeline_produces_a_scoring_submission(corpus, monkeypatch) -> None:
    config = _config(corpus)

    # 1. catalog
    catalog = build_catalog(corpus["keyframes"], corpus["metadata"], repo_root=corpus["root"])
    save_catalog(catalog, config.catalog_path)

    # 2. embeddings: one-hot-ish vectors so a crafted query targets a known frame
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(len(catalog), DIM)).astype(np.float32)
    target_gid = 20
    embeddings[target_gid] = np.eye(DIM, dtype=np.float32)[0] * 10
    embeddings_dir = config.derived_path("embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_dir / "emb_000.npy", embeddings)

    # 3. index
    matrix = load_embeddings(embeddings_dir, expected_rows=len(catalog))
    index = build_index(matrix, expected_dim=DIM)
    save_index(index, config.index_path)
    assert index.ntotal == len(catalog)

    # 4. search with a stub encoder aimed at the planted vector
    query_vector = np.eye(DIM, dtype=np.float32)[0][None, :]
    monkeypatch.setattr(
        "aic.query.search.encode_texts", lambda texts, model_id, dim: np.repeat(query_vector, len(texts), axis=0)
    )
    engine = SearchEngine(config)
    result = engine.search("một người đang mở laptop", top_n=20)

    assert result.hits, "search returned nothing"
    assert result.hits[0].gid == target_gid
    top = result.hits[0]
    expected = catalog.loc[catalog.gid == target_gid].iloc[0]
    assert top.video_id == expected["video_id"]
    assert top.frame_idx == expected["frame_idx"]

    # 5. policy -> submission rows
    candidates = [
        Candidate(video_id=h.video_id, frame_idx=h.frame_idx, score=h.score)
        for h in result.hits if h.frame_idx is not None
    ]
    rows = build_kis_answers(candidates, diversify_head=5, frames_per_shot=3, frame_spread=8)
    assert 0 < len(rows) <= 100
    # The head hedges across every distinct video available (4 in this fixture).
    head = rows[:N_VIDEOS]
    assert len({video for video, _ in head}) == N_VIDEOS

    # 6. write and re-read the submission
    path = write_submission(rows, config.submissions_dir / "kis" / "q1.csv", "kis")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(rows)
    assert lines[0] == f"{rows[0][0]},{rows[0][1]}"

    # 7. score it: the planted frame is the ground truth, so rank 1 must be right
    truth = KISGroundTruth(
        video_id=str(expected["video_id"]),
        start=int(expected["frame_idx"]) - 5,
        end=int(expected["frame_idx"]) + 5,
    )
    scores = [kis_r_score(KISAnswer(v, f), truth) for v, f in rows]
    assert final_score(scores) == pytest.approx(1.0)


def test_search_reports_disabled_branches_instead_of_crashing(corpus, monkeypatch) -> None:
    """A dead encoder must degrade the search, not kill it."""
    config = _config(corpus)
    catalog = build_catalog(corpus["keyframes"], corpus["metadata"], repo_root=corpus["root"])
    save_catalog(catalog, config.catalog_path)

    from aic.query.encoder import EncoderUnavailableError

    def boom(texts, model_id, dim):
        raise EncoderUnavailableError("no text tower installed")

    monkeypatch.setattr("aic.query.search.encode_texts", boom)
    result = SearchEngine(config).search("bất kỳ", top_n=10)

    assert result.hits == []
    assert "visual" in result.disabled_branches


def test_text_index_matches_with_and_without_diacritics(tmp_path) -> None:
    with TextIndex(tmp_path / "text.sqlite") as index:
        index.add_segments(
            [
                {
                    "kind": "ocr", "video_id": "L01_V001", "text": "Họp báo Chính phủ tại Hà Nội",
                    "start_frame": 100, "end_frame": 120, "start_time": 4.0, "end_time": 4.8,
                },
                {
                    "kind": "ocr", "video_id": "L02_V001", "text": "Dự báo thời tiết",
                    "start_frame": 10, "end_frame": 20, "start_time": 0.4, "end_time": 0.8,
                },
            ]
        )
        assert index.count("ocr") == 2
        assert [h.video_id for h in index.search(["họp báo"])][:1] == ["L01_V001"]
        # Unaccented typing must still find the accented text.
        assert [h.video_id for h in index.search(["hop bao"])][:1] == ["L01_V001"]
        # Punctuation must not be parsed as FTS5 syntax.
        assert index.search(['"quoted" AND (weird)']) is not None


def test_strip_diacritics_handles_vietnamese_d() -> None:
    assert strip_diacritics("Đường Hồ Chí Minh") == "Duong Ho Chi Minh"


def test_search_pooled_keeps_the_best_score_per_frame() -> None:
    matrix = np.eye(4, dtype=np.float32)
    index = build_index(matrix, expected_dim=4)
    queries = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

    pooled = search_pooled(index, queries, top_k=4)
    gids = [gid for gid, _ in pooled]
    assert set(gids[:2]) == {0, 1}          # both query variants surface their target
    assert len(gids) == len(set(gids))      # each frame appears once
    assert pooled[0][1] == pytest.approx(1.0)

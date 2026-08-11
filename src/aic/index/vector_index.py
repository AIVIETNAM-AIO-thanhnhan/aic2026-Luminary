"""FAISS index over keyframe embeddings.

Row *i* of the embedding matrix is the frame with ``gid == i`` in
``catalog.parquet``; that invariant is what lets search results be turned back
into ``(video_id, frame_idx)`` pairs, and :func:`build_index` refuses to proceed
if the counts disagree.

Uses ``IndexFlatIP`` (exact inner product on L2-normalized vectors, i.e. cosine).
For batch-1 scale — roughly 300k keyframes — exact search costs a few hundred MB
and tens of milliseconds on CPU, so an approximate index would add tuning risk
and recall loss for no useful gain. Revisit if the corpus passes a few million.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # faiss is optional so the metric/policy modules import without it
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


def _require_faiss():
    if faiss is None:
        raise ImportError("faiss-cpu is required for vector search: pip install faiss-cpu")
    return faiss


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so inner product equals cosine similarity."""
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)  # guard all-zero rows
    return matrix / norms


def load_embeddings(source: Path, expected_rows: int | None = None) -> np.ndarray:
    """Load embeddings from a single ``.npy`` or a directory of per-video ``.npy``.

    A directory is concatenated in sorted filename order, which matches the
    ``(video_id, path)`` ordering :func:`aic.data.catalog.build_catalog` assigns.
    """
    source = Path(source)
    if source.is_dir():
        parts = sorted(source.glob("*.npy"))
        if not parts:
            raise FileNotFoundError(f"no .npy files under {source}")
        matrix = np.concatenate([np.load(p) for p in parts], axis=0)
    else:
        matrix = np.load(source)

    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D embedding matrix, got shape {matrix.shape}")
    if expected_rows is not None and len(matrix) != expected_rows:
        raise ValueError(
            f"embedding/catalog mismatch: {len(matrix)} vectors vs {expected_rows} catalog rows. "
            "gid must equal the embedding row index, so these must be equal. "
            "Rebuild the catalog and embeddings from the same keyframe set."
        )
    return matrix


def build_index(embeddings: np.ndarray, expected_dim: int | None = None):
    """Build an exact cosine-similarity index."""
    faiss_mod = _require_faiss()
    if expected_dim is not None and embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"embedding dim {embeddings.shape[1]} != configured dim {expected_dim}; "
            "check embedding.active in configs/default.yaml"
        )
    index = faiss_mod.IndexFlatIP(embeddings.shape[1])
    index.add(l2_normalize(embeddings))
    return index


def save_index(index, path: Path) -> None:
    faiss_mod = _require_faiss()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss_mod.write_index(index, str(path))


def load_index(path: Path):
    faiss_mod = _require_faiss()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"index not found at {path}. Run `aic build-index` first.")
    return faiss_mod.read_index(str(path))


def search(index, query_vectors: np.ndarray, top_k: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(scores, gids)``, each shaped ``(n_queries, top_k)``."""
    queries = l2_normalize(np.atleast_2d(query_vectors))
    scores, gids = index.search(queries, min(top_k, index.ntotal))
    return scores, gids


def search_pooled(index, query_vectors: np.ndarray, top_k: int = 500) -> list[tuple[int, float]]:
    """Search with several query variants and keep each frame's best score.

    Query expansion produces multiple phrasings of the same intent; taking the max
    per frame rewards a frame that matches any one phrasing strongly, rather than
    diluting it by averaging over phrasings it was never meant to match.
    """
    scores, gids = search(index, query_vectors, top_k)
    best: dict[int, float] = {}
    for row_scores, row_gids in zip(scores, gids):
        for score, gid in zip(row_scores, row_gids):
            if gid < 0:  # faiss pads with -1 when fewer than top_k results exist
                continue
            gid = int(gid)
            if score > best.get(gid, -np.inf):
                best[gid] = float(score)
    return sorted(best.items(), key=lambda item: item[1], reverse=True)

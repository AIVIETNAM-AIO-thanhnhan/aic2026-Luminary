"""Rank one video's own frames against a query, bypassing the global top-k cutoff.

`SearchEngine.search`'s `video_filter` filters *after* taking the global top-500
candidates per branch, so a video that the whole-corpus search ranked outside
that window never gets a chance even if it is a perfect match for one specific
query. This computes raw cosine similarity directly against just that video's
embeddings instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aic.config import load_config  # noqa: E402
from aic.query.encoder import encode_texts  # noqa: E402


def rank_within_video(video_id: str, texts: list[str], top_n: int = 15) -> None:
    config = load_config()
    catalog = pd.read_parquet(config.catalog_path)
    embeddings = np.load(config.active_space.path("source"))

    rows = catalog[catalog["video_id"] == video_id].sort_values("gid")
    if rows.empty:
        print(f"no rows for {video_id}")
        return

    space = config.active_space
    query_vecs = encode_texts(texts, space.model_id, int(space.dim))

    vecs = embeddings[rows["gid"].to_numpy()]
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    sims = (query_vecs @ vecs.T).max(axis=0)  # best score across query variants, per frame

    order = np.argsort(-sims)[:top_n]
    for i in order:
        row = rows.iloc[i]
        print(f"  frame_idx={int(row['frame_idx'])!s:>7}  sim={sims[i]:.4f}  path={row['path']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("texts", nargs="+")
    args = parser.parse_args()
    rank_within_video(args.video_id, args.texts)

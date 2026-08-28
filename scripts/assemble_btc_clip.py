"""Assemble one embeddings.npy matching the catalog's current gid order.

``clip-features-32/*.npy`` ships one file per video, covering the whole batch.
The catalog only covers whichever keyframe zips have landed on disk so far -
this script builds the subset+order that matches it exactly, so
``aic build-index`` (which requires ``rows(embeddings) == rows(catalog)``) has
something valid to load. This is a stopgap for racing the fetch against a
deadline; once the full keyframe corpus is down, `embedding.spaces.btc_clip.source`
can point straight at the clip-features directory instead.

Re-run this after every `aic build-catalog` and before `aic build-index`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from aic.config import load_config  # noqa: E402
from aic.data.catalog import load_catalog  # noqa: E402


def main() -> None:
    config = load_config()
    catalog = load_catalog(config.catalog_path).sort_values("gid", kind="stable").reset_index(drop=True)
    clip_dir = config.raw_path("clip_features")

    blocks = []
    for video_id, group in catalog.groupby("video_id", sort=False):
        npy_path = clip_dir / f"{video_id}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"no clip feature file for {video_id} at {npy_path}")
        vectors = np.load(npy_path).astype(np.float32)
        if len(vectors) != len(group):
            raise ValueError(
                f"{video_id}: catalog has {len(group)} keyframes but {npy_path.name} has "
                f"{len(vectors)} rows - re-extract this video's keyframes zip"
            )
        blocks.append(vectors)

    matrix = np.concatenate(blocks, axis=0)
    assert len(matrix) == len(catalog), f"{len(matrix)} vectors vs {len(catalog)} catalog rows"

    out_path = config.active_space.path("source")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, matrix)
    print(f"wrote {out_path} {matrix.shape}")


if __name__ == "__main__":
    main()

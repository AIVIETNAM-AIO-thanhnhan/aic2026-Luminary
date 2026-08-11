"""Encode keyframes into an embedding matrix (runs on Colab/Kaggle GPU).

Kept in the package rather than inside a notebook so it can be imported, reviewed,
and tested. The notebooks in ``notebooks/`` are thin drivers that clone the repo
and call :func:`embed_catalog`.

The contract that matters: **row *i* of the output is the frame with ``gid == i``
in ``catalog.parquet``.** :func:`embed_catalog` iterates the catalog in ``gid``
order and writes shards tagged with their gid range, so a run interrupted after
three hours resumes instead of restarting.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class ShardPlan:
    start_gid: int
    end_gid: int  # exclusive
    path: Path

    @property
    def size(self) -> int:
        return self.end_gid - self.start_gid


def plan_shards(n_rows: int, shard_size: int, output_dir: Path) -> list[ShardPlan]:
    """Split the catalog into fixed-size shards named by their gid range.

    Zero-padded names keep sorted-filename order identical to gid order, which is
    what :func:`aic.index.vector_index.load_embeddings` relies on when it
    concatenates a directory of shards.
    """
    output_dir = Path(output_dir)
    width = max(8, len(str(max(0, n_rows - 1))))
    return [
        ShardPlan(
            start_gid=start,
            end_gid=min(start + shard_size, n_rows),
            path=output_dir / f"emb_{start:0{width}d}_{min(start + shard_size, n_rows):0{width}d}.npy",
        )
        for start in range(0, n_rows, shard_size)
    ]


def pending_shards(shards: list[ShardPlan]) -> list[ShardPlan]:
    """Shards not yet written, so a re-run resumes where it stopped."""
    return [s for s in shards if not s.path.exists()]


def load_images(paths: list[Path], repo_root: Path | None = None) -> tuple[list[Image.Image], list[int]]:
    """Load images, returning the successfully loaded ones and their positions.

    A corrupt JPEG must not shift every later row by one — that would silently
    misalign the entire index — so callers get back the positions that loaded and
    fill the rest with zero vectors.
    """
    root = repo_root or Path.cwd()
    images: list[Image.Image] = []
    positions: list[int] = []
    for position, path in enumerate(paths):
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = root / resolved
        try:
            images.append(Image.open(resolved).convert("RGB"))
            positions.append(position)
        except Exception:  # noqa: BLE001 - skip unreadable frames, keep alignment
            continue
    return images, positions


def embed_catalog(
    catalog: pd.DataFrame,
    encode_fn: Callable[[list[Image.Image]], np.ndarray],
    output_dir: Path,
    dim: int,
    batch_size: int = 64,
    shard_size: int = 20_000,
    repo_root: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Encode every catalogued frame, writing resumable shards.

    ``encode_fn`` takes PIL images and returns an ``(n, dim)`` float array; the
    notebooks pass a GPU SigLIP/CLIP image tower. Rows whose image failed to load
    are left as zeros — they simply never match a query, rather than corrupting
    the gid alignment of everything after them.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = catalog.sort_values("gid", kind="stable").reset_index(drop=True)
    if not (catalog["gid"].to_numpy() == np.arange(len(catalog))).all():
        raise ValueError(
            "catalog gids must be 0..n-1 with no gaps: gid is used directly as the "
            "embedding row index. Rebuild it with `aic build-catalog`."
        )

    shards = plan_shards(len(catalog), shard_size, output_dir)
    todo = pending_shards(shards)

    for shard_number, shard in enumerate(todo, start=1):
        block = catalog.iloc[shard.start_gid : shard.end_gid]
        vectors = np.zeros((shard.size, dim), dtype=np.float32)

        for offset in range(0, len(block), batch_size):
            batch = block.iloc[offset : offset + batch_size]
            images, positions = load_images(batch["path"].tolist(), repo_root=repo_root)
            if images:
                encoded = np.asarray(encode_fn(images), dtype=np.float32)
                for row, position in enumerate(positions):
                    vectors[offset + position] = encoded[row]
            if progress:
                progress(shard.start_gid + offset + len(batch), len(catalog))

        # Write to a temp name first: an interrupted write must not leave a
        # truncated shard that the resume logic would mistake for finished.
        # The name deliberately does not end in .npy so a leftover temp file is
        # never picked up by load_embeddings' *.npy glob; np.save is handed an
        # open file so it does not re-append the extension.
        temporary = shard.path.with_name(shard.path.name + ".tmp")
        with open(temporary, "wb") as handle:
            np.save(handle, vectors)
        temporary.replace(shard.path)
        print(f"[{shard_number}/{len(todo)}] wrote {shard.path.name} {vectors.shape}")

    return [shard.path for shard in shards]


def iter_batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
